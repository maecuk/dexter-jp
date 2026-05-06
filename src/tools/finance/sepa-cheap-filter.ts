import { DynamicStructuredTool } from '@langchain/core/tools';
import { z } from 'zod';
import { formatToolResult } from '../types.js';
import { logger } from '../../utils/logger.js';

const JQUANTS_BASE = 'https://api.jquants.com/v2';

type DailyBar = {
  Code: string;
  Date: string;
  AdjO?: number;
  AdjH?: number;
  AdjL?: number;
  AdjC?: number;
  AdjVo?: number;
  O?: number;
  H?: number;
  L?: number;
  C?: number;
  Vo?: number;
  Va?: number;
};

type SepaCandidate = {
  code: string;
  ticker: string;
  date: string;
  close: number;
  volume: number;
  turnoverYen: number;
  rsScore: number;
  trendTemplateScore: number;
  stage: 'Stage 2' | 'Watch' | 'Fail';
  sma50: number | null;
  sma150: number | null;
  sma200: number | null;
  pctFrom52wHigh: number | null;
  pctFrom52wLow: number | null;
  return3m: number | null;
  return6m: number | null;
  return12m: number | null;
  avgTurnover20d: number | null;
  avgTurnover50d: number | null;
  vcpHint: 'Possible' | 'Weak/Unknown';
  pivotCandidate: number | null;
  stopLossCandidate: number | null;
  notes: string[];
};

export const SEPA_CHEAP_FILTER_DESCRIPTION = `
Screens Japanese equities for SEPA cheap-filter candidates using J-Quants daily OHLCV.

Use this for broad universe SEPA screening where the criteria are price, volume, moving averages,
52-week high/low position, relative strength, and liquidity. Do not use company_screener for SEPA
cheap filters because EDINET DB does not provide market trend or RS data.

Returns top candidates only. Use get_financials afterward for Code 33, accounting quality, earnings
reaction, and fundamental deep analysis on the shortlisted names.
`.trim();

function getJQuantsApiKey(): string {
  return process.env.JQUANTS_API_KEY || '';
}

function compactDate(date: Date): string {
  return date.toISOString().slice(0, 10).replaceAll('-', '');
}

function daysAgo(base: Date, days: number): Date {
  const date = new Date(base);
  date.setUTCDate(date.getUTCDate() - days);
  return date;
}

function asNumber(value: unknown): number | undefined {
  return typeof value === 'number' && Number.isFinite(value) ? value : undefined;
}

function closeOf(bar: DailyBar | undefined): number | undefined {
  if (!bar) return undefined;
  return asNumber(bar.AdjC) ?? asNumber(bar.C);
}

function highOf(bar: DailyBar | undefined): number | undefined {
  if (!bar) return undefined;
  return asNumber(bar.AdjH) ?? asNumber(bar.H);
}

function lowOf(bar: DailyBar | undefined): number | undefined {
  if (!bar) return undefined;
  return asNumber(bar.AdjL) ?? asNumber(bar.L);
}

function volumeOf(bar: DailyBar | undefined): number | undefined {
  if (!bar) return undefined;
  return asNumber(bar.AdjVo) ?? asNumber(bar.Vo);
}

function average(values: number[]): number | null {
  if (values.length === 0) return null;
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

async function jquantsGetAll(
  endpoint: string,
  params: Record<string, string | undefined>,
): Promise<Record<string, unknown>[]> {
  const apiKey = getJQuantsApiKey();
  if (!apiKey) {
    throw new Error('JQUANTS_API_KEY not set');
  }

  const rows: Record<string, unknown>[] = [];
  let paginationKey: string | undefined;

  for (let page = 0; page < 20; page += 1) {
    const url = new URL(`${JQUANTS_BASE}${endpoint}`);
    for (const [key, value] of Object.entries(params)) {
      if (value !== undefined) url.searchParams.set(key, value);
    }
    if (paginationKey) url.searchParams.set('pagination_key', paginationKey);

    const response = await fetch(url.toString(), {
      headers: { 'x-api-key': apiKey },
    });

    if (!response.ok) {
      const detail = `${response.status} ${response.statusText}`;
      logger.error(`[J-Quants] API error: ${detail}`);
      throw new Error(`J-Quants API error: ${detail}`);
    }

    const payload = await response.json() as Record<string, unknown>;
    const data = (payload.data ?? payload.daily_quotes) as Record<string, unknown>[] | undefined;
    if (Array.isArray(data)) rows.push(...data);

    paginationKey = typeof payload.pagination_key === 'string'
      ? payload.pagination_key
      : typeof payload.paginationKey === 'string'
        ? payload.paginationKey
        : undefined;

    if (!paginationKey) break;
  }

  return rows;
}

async function fetchMarketBarsForDate(date: Date): Promise<DailyBar[]> {
  for (let offset = 0; offset < 10; offset += 1) {
    const rows = await jquantsGetAll('/equities/bars/daily', {
      date: compactDate(daysAgo(date, offset)),
    });
    const bars = rows as DailyBar[];
    if (bars.length > 0) return bars;
  }
  return [];
}

function barMap(bars: DailyBar[]): Map<string, DailyBar> {
  return new Map(bars.filter((bar) => bar.Code).map((bar) => [bar.Code, bar]));
}

function pctReturn(current: number, past?: number): number | null {
  if (!past || past <= 0) return null;
  return ((current / past) - 1) * 100;
}

function percentileRank(values: Array<{ code: string; value: number }>): Map<string, number> {
  const sorted = [...values].sort((a, b) => a.value - b.value);
  const ranks = new Map<string, number>();
  const denom = Math.max(sorted.length - 1, 1);
  sorted.forEach((item, index) => {
    ranks.set(item.code, Math.round((index / denom) * 100));
  });
  return ranks;
}

function trendTemplateScore(history: DailyBar[], latest: DailyBar, rsScore: number): number {
  const closes = history.map(closeOf).filter((value): value is number => value !== undefined);
  if (closes.length < 200) return 0;

  const close = closeOf(latest);
  if (!close) return 0;

  const sma50 = average(closes.slice(-50));
  const sma150 = average(closes.slice(-150));
  const sma200 = average(closes.slice(-200));
  const prevSma200 = average(closes.slice(-220, -20));
  const high52 = Math.max(...history.map(highOf).filter((value): value is number => value !== undefined));
  const low52 = Math.min(...history.map(lowOf).filter((value): value is number => value !== undefined));

  let score = 0;
  if (sma150 && sma200 && close > sma150 && close > sma200) score += 1;
  if (sma150 && sma200 && sma150 > sma200) score += 1;
  if (sma200 && prevSma200 && sma200 > prevSma200) score += 1;
  if (sma50 && sma150 && sma200 && sma50 > sma150 && sma50 > sma200) score += 1;
  if (sma50 && close > sma50) score += 1;
  if (low52 > 0 && ((close / low52) - 1) * 100 >= 25) score += 1;
  if (high52 > 0 && ((close / high52) - 1) * 100 >= -25) score += 1;
  if (rsScore >= 70) score += 1;
  return score;
}

function buildCandidate(history: DailyBar[], latest: DailyBar, rsScore: number): SepaCandidate | null {
  const close = closeOf(latest);
  const volume = volumeOf(latest);
  if (!close || !volume) return null;

  const closes = history.map(closeOf).filter((value): value is number => value !== undefined);
  if (closes.length < 200) return null;

  const highs = history.map(highOf).filter((value): value is number => value !== undefined);
  const lows = history.map(lowOf).filter((value): value is number => value !== undefined);
  const turnovers = history
    .map((bar) => {
      const c = closeOf(bar);
      const v = volumeOf(bar);
      return c && v ? c * v : undefined;
    })
    .filter((value): value is number => value !== undefined);

  const sma50 = average(closes.slice(-50));
  const sma150 = average(closes.slice(-150));
  const sma200 = average(closes.slice(-200));
  const high52 = Math.max(...highs);
  const low52 = Math.min(...lows);
  const recentHigh = Math.max(...highs.slice(-25));
  const recentLow = Math.min(...lows.slice(-25));
  const priorHigh = Math.max(...highs.slice(-75, -25));
  const priorLow = Math.min(...lows.slice(-75, -25));
  const recentRange = recentHigh > 0 ? (recentHigh - recentLow) / recentHigh : null;
  const priorRange = priorHigh > 0 ? (priorHigh - priorLow) / priorHigh : null;
  const vcpHint = recentRange !== null && priorRange !== null && recentRange < priorRange * 0.75
    ? 'Possible'
    : 'Weak/Unknown';

  const trendScore = trendTemplateScore(history, latest, rsScore);
  const avgTurnover20d = average(turnovers.slice(-20));
  const avgTurnover50d = average(turnovers.slice(-50));
  const turnoverYen = close * volume;
  const stage = trendScore >= 7 ? 'Stage 2' : trendScore >= 5 ? 'Watch' : 'Fail';
  const pivotCandidate = recentHigh || null;
  const stopLossCandidate = recentLow || (sma50 ? Math.min(sma50, close * 0.93) : null);

  return {
    code: latest.Code,
    ticker: latest.Code.endsWith('0') ? latest.Code.slice(0, -1) : latest.Code,
    date: latest.Date,
    close,
    volume,
    turnoverYen,
    rsScore,
    trendTemplateScore: trendScore,
    stage,
    sma50,
    sma150,
    sma200,
    pctFrom52wHigh: high52 > 0 ? ((close / high52) - 1) * 100 : null,
    pctFrom52wLow: low52 > 0 ? ((close / low52) - 1) * 100 : null,
    return3m: pctReturn(close, closes.at(-64)),
    return6m: pctReturn(close, closes.at(-127)),
    return12m: pctReturn(close, closes.at(-253)),
    avgTurnover20d,
    avgTurnover50d,
    vcpHint,
    pivotCandidate,
    stopLossCandidate,
    notes: [
      'Cheap filter only: price/volume/RS/SMA-based. Fundamentals require get_financials.',
      'RS is an in-universe percentile approximation from 3m/6m/12m returns, not IBD RS.',
    ],
  };
}

const SepaCheapFilterInputSchema = z.object({
  limit: z.number().optional().describe('Number of final candidates to return. Default 10.'),
  min_turnover_yen: z.number().optional().describe('Minimum latest-day turnover in JPY. Default 100,000,000.'),
  max_deep_candidates: z.number().optional().describe('Number of preliminary names to fetch full history for. Default 40, max 80.'),
});

export const sepaCheapFilter = new DynamicStructuredTool({
  name: 'sepa_cheap_filter',
  description: 'J-Quants based cheap filter for SEPA candidates. Screens Japanese equities by OHLCV, liquidity, approximate RS, SMA trend template, Stage 2, 52-week position, and VCP hints. Use before deep SEPA analysis.',
  schema: SepaCheapFilterInputSchema,
  func: async (input) => {
    const limit = Math.min(input.limit ?? 10, 50);
    const minTurnoverYen = input.min_turnover_yen ?? 100_000_000;
    const maxDeepCandidates = Math.min(input.max_deep_candidates ?? 40, 80);

    const latestBars = await fetchMarketBarsForDate(new Date());
    if (latestBars.length === 0) {
      return formatToolResult({ error: 'No latest market bars returned from J-Quants' }, []);
    }

    const latestDate = latestBars[0].Date ? new Date(`${latestBars[0].Date}T00:00:00Z`) : new Date();
    const latestMap = barMap(latestBars);
    const bars3m = barMap(await fetchMarketBarsForDate(daysAgo(latestDate, 90)));
    const bars6m = barMap(await fetchMarketBarsForDate(daysAgo(latestDate, 180)));
    const bars12m = barMap(await fetchMarketBarsForDate(daysAgo(latestDate, 370)));

    const momentum = [...latestMap.values()].flatMap((bar) => {
      const close = closeOf(bar);
      const volume = volumeOf(bar);
      if (!close || !volume) return [];
      const turnoverYen = close * volume;
      if (turnoverYen < minTurnoverYen) return [];

      const r3 = pctReturn(close, closeOf(bars3m.get(bar.Code) as DailyBar));
      const r6 = pctReturn(close, closeOf(bars6m.get(bar.Code) as DailyBar));
      const r12 = pctReturn(close, closeOf(bars12m.get(bar.Code) as DailyBar));
      const composite = (r3 ?? 0) * 0.5 + (r6 ?? 0) * 0.3 + (r12 ?? 0) * 0.2;
      return [{ code: bar.Code, value: composite, latest: bar, turnoverYen }];
    });

    const ranks = percentileRank(momentum.map(({ code, value }) => ({ code, value })));
    const preliminary = momentum
      .map((item) => ({ ...item, rsScore: ranks.get(item.code) ?? 0 }))
      .filter((item) => item.rsScore >= 70)
      .sort((a, b) => (b.rsScore - a.rsScore) || (b.turnoverYen - a.turnoverYen))
      .slice(0, maxDeepCandidates);

    const candidates: SepaCandidate[] = [];
    const from = compactDate(daysAgo(latestDate, 390));
    const to = compactDate(latestDate);

    for (const item of preliminary) {
      const rows = await jquantsGetAll('/equities/bars/daily', {
        code: item.code,
        from,
        to,
      });
      const history = (rows as DailyBar[]).sort((a, b) => a.Date.localeCompare(b.Date));
      const latest = history.at(-1) ?? item.latest;
      const candidate = buildCandidate(history, latest, item.rsScore);
      if (candidate && candidate.trendTemplateScore >= 7 && candidate.stage === 'Stage 2') {
        candidates.push(candidate);
      }
    }

    const ranked = candidates
      .sort((a, b) =>
        (b.trendTemplateScore - a.trendTemplateScore)
        || (b.rsScore - a.rsScore)
        || ((b.avgTurnover20d ?? b.turnoverYen) - (a.avgTurnover20d ?? a.turnoverYen))
      )
      .slice(0, limit);

    return formatToolResult({
      asOf: latestBars[0].Date,
      universeSize: latestBars.length,
      preliminaryCount: preliminary.length,
      criteria: {
        minTurnoverYen,
        rsScore: '>= 70 approximate in-universe percentile',
        trendTemplateScore: '>= 7/8',
        stage: 'Stage 2',
      },
      candidates: ranked,
    }, []);
  },
});
