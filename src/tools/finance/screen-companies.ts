import { DynamicStructuredTool } from '@langchain/core/tools';
import type { RunnableConfig } from '@langchain/core/runnables';
import { z } from 'zod';
import { callLlm } from '../../model/llm.js';
import { formatToolResult } from '../types.js';
import { getCurrentDate } from '../../agent/prompts.js';
import { api } from './api.js';

/**
 * Rich description for the screen_companies tool.
 */
export const SCREEN_COMPANIES_DESCRIPTION = `
Screens for Japanese listed companies matching financial criteria. Takes a natural language query describing the screening criteria and returns matching companies with their metric values.

## When to Use

- Finding companies by financial criteria (e.g., "ROE 15%以上 and equity ratio above 50%")
- Screening for value, growth, dividend, or quality stocks
- Filtering by valuation ratios, profitability metrics, or growth rates
- Filtering by industry (e.g., "情報・通信業", "医薬品")
- Finding companies matching a specific investment thesis

## When NOT to Use

- Looking up a specific company's financials (use get_financials)
- Reading securities reports (use read_filings)
- SEPA / technical screening by price, volume, SMA, RS, Stage 2, 52-week position, VCP, or pivot candidates (use sepa_cheap_filter if available)
- General web searches (use web_search)

## Usage Notes

- Call ONCE with the complete natural language query describing your screening criteria
- The tool translates your criteria into exact API filters automatically
- Returns matching companies with the metric values used for screening
- Supports 100+ metrics including ROE, ROIC, operating margin, equity ratio, dividend yield, revenue CAGR, etc.
`.trim();

// Available metrics for the EDINET DB screener
const AVAILABLE_METRICS = `
Supported metrics (use exact keys):
- roe: Return on Equity (%)
- roic: Return on Invested Capital (%)
- roa: Return on Assets (%)
- operating-margin: Operating profit margin (%)
- net-margin: Net profit margin (%)
- equity-ratio: Equity to assets ratio (%)
- per: Price-to-Earnings Ratio
- pbr: Price-to-Book Ratio
- eps: Earnings Per Share (JPY)
- bps: Book Value Per Share (JPY)
- dividend-yield: Dividend yield (%)
- payout-ratio: Dividend payout ratio (%)
- revenue: Total revenue (millions of JPY)
- revenue-growth: Revenue YoY growth (%)
- ni-growth: Net income YoY growth (%)
- eps-growth: EPS YoY growth (%)
- revenue-cagr-3y: Revenue 3-year CAGR (%)
- oi-cagr-3y: Operating income 3-year CAGR (%)
- ni-cagr-3y: Net income 3-year CAGR (%)
- eps-cagr-3y: EPS 3-year CAGR (%)
- health-score: Financial health score (0-100)
- current-ratio: Current ratio
- de-ratio: Debt-to-Equity ratio
- free-cf: Free cash flow (millions of JPY)
- ebitda: EBITDA (millions of JPY)
- financial-leverage: Financial leverage ratio

Operators: gte (>=), lte (<=), gt (>), lt (<), eq (=)
Values use display units: ROE in %, revenue in millions of JPY.

Industries (Japanese, exact match):
情報・通信業, 卸売業, 電気機器, 輸送用機器, 医薬品, 銀行業, 小売業, サービス業, 化学, 機械, 建設業, 不動産業, 食料品, 鉄鋼, 証券・商品先物取引業, 保険業, etc.
`.trim();

const ScreenerConditionSchema = z.object({
  conditions: z.array(z.object({
    metric: z.string().describe('Metric key from the available metrics list'),
    operator: z.enum(['gte', 'lte', 'gt', 'lt', 'eq']).describe('Comparison operator'),
    value: z.number().describe('Threshold value in display units'),
  })).describe('Array of screening conditions to apply (AND logic)'),
  industry: z.string().nullable().describe('Filter by industry (Japanese name, exact match), or null for no industry filter'),
  limit: z.number().nullable().describe('Maximum number of results (default: 25, max: 200), or null for default'),
  sort_by: z.string().nullable().describe('Sort results by this metric key, or null for default sort'),
});

type ScreenerConditions = z.infer<typeof ScreenerConditionSchema>;

const INDUSTRIES = [
  '情報・通信業',
  '卸売業',
  '電気機器',
  '輸送用機器',
  '医薬品',
  '銀行業',
  '小売業',
  'サービス業',
  '化学',
  '機械',
  '建設業',
  '不動産業',
  '食料品',
  '鉄鋼',
  '証券・商品先物取引業',
  '保険業',
  '非鉄金属',
  'ガラス・土石製品',
] as const;

function parseLimit(query: string): number | null {
  const topMatch = query.match(/(?:上位|top)\s*(\d{1,3})/i);
  if (topMatch) return Math.min(Number(topMatch[1]), 200);

  const countMatch = query.match(/(\d{1,3})\s*(?:件|銘柄)/);
  if (countMatch) return Math.min(Number(countMatch[1]), 200);

  return null;
}

function parseIndustry(query: string): string | null {
  const mentioned = INDUSTRIES.filter((industry) => query.includes(industry));
  if (mentioned.length === 1) return mentioned[0];

  const explicit = query.match(/(?:業種|industry)\s*[:：]\s*([^\s,、。]+)/i);
  if (explicit) {
    const value = explicit[1].trim();
    return INDUSTRIES.find((industry) => industry === value) ?? null;
  }

  return null;
}

function parseNumericCondition(
  query: string,
  labels: string[],
  metric: string,
): ScreenerConditions['conditions'][number] | null {
  const escapedLabels = labels.map((label) => label.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'));
  const pattern = new RegExp(`(?:${escapedLabels.join('|')})\\s*(\\d+(?:\\.\\d+)?)\\s*(以上|以下|超|未満|>|<|>=|<=)?`);
  const match = query.match(pattern);
  if (!match) return null;

  const value = Number(match[1]);
  const opText = match[2] ?? '以上';
  const operator = opText === '以下' || opText === '未満' || opText === '<' || opText === '<='
    ? opText === '未満' || opText === '<' ? 'lt' : 'lte'
    : opText === '超' || opText === '>' ? 'gt' : 'gte';

  return { metric, operator, value };
}

function buildHeuristicConditions(query: string): ScreenerConditions {
  const conditions = [
    parseNumericCondition(query, ['ROE', 'roe'], 'roe'),
    parseNumericCondition(query, ['ROIC', 'roic'], 'roic'),
    parseNumericCondition(query, ['自己資本比率', 'equity ratio'], 'equity-ratio'),
    parseNumericCondition(query, ['PER', 'per'], 'per'),
    parseNumericCondition(query, ['PBR', 'pbr'], 'pbr'),
    parseNumericCondition(query, ['営業利益率', 'operating margin'], 'operating-margin'),
    parseNumericCondition(query, ['売上成長率', 'revenue growth'], 'revenue-growth'),
  ].filter((condition): condition is ScreenerConditions['conditions'][number] => condition !== null);

  const lower = query.toLowerCase();
  const isStageOne = query.includes('Stage 1') || query.includes('Stage1') || query.includes('一次') || query.includes('軽量');
  const isMultibagger = lower.includes('multibagger') || query.includes('マルチバガー');

  if (conditions.length === 0) {
    if (isStageOne || isMultibagger) {
      conditions.push(
        { metric: 'revenue-growth', operator: 'gte', value: 0 },
        { metric: 'equity-ratio', operator: 'gte', value: 30 },
        { metric: 'health-score', operator: 'gte', value: 40 },
      );
    } else if (query.includes('割安')) {
      conditions.push({ metric: 'per', operator: 'lte', value: 15 });
    } else if (query.includes('高ROE')) {
      conditions.push({ metric: 'roe', operator: 'gte', value: 15 });
    } else if (query.includes('高配当')) {
      conditions.push({ metric: 'dividend-yield', operator: 'gte', value: 3 });
    } else {
      conditions.push({ metric: 'health-score', operator: 'gte', value: 0 });
    }
  }

  return {
    conditions,
    industry: parseIndustry(query),
    limit: parseLimit(query),
    sort_by: isStageOne || isMultibagger ? 'revenue-growth' : null,
  };
}

function buildScreenerPrompt(): string {
  return `You are a Japanese stock screening assistant.
Current date: ${getCurrentDate()}

Given a user's natural language query about stock screening criteria, produce the structured screening conditions.

## Available Metrics

${AVAILABLE_METRICS}

## Guidelines

1. Map user criteria to exact metric keys from the list above
2. Choose the correct operator:
   - "以下", "below", "under", "less than" → lte
   - "以上", "above", "over", "greater than", "more than" → gte
   - "equal to", "exactly" → eq
3. Use reasonable defaults:
   - If the user says "高ROE" without a number, use gte 15
   - If the user says "高配当" without a number, use gte 3
   - If the user says "割安" consider PER lte 15 or PBR lte 1
4. Set limit to 25 unless the user specifies otherwise
5. For industry filters, use Japanese industry names (exact match)
6. If the user mentions sorting, set sort_by to the relevant metric

Return only the structured output fields.`;
}

const ScreenCompaniesInputSchema = z.object({
  query: z.string().describe('Natural language query describing company screening criteria'),
});

/**
 * Create a screen_companies tool configured with the specified model.
 */
export function createScreenCompanies(model: string): DynamicStructuredTool {
  return new DynamicStructuredTool({
    name: 'company_screener',
    description: `Screens for Japanese listed companies matching financial criteria. Takes a natural language query and returns matching companies with metric values. Use for:
- Finding companies by valuation (PER, PBR)
- Screening by profitability (margins, ROE, ROA, ROIC)
- Filtering by growth rates (revenue, earnings, EPS growth)
- Dividend screening (yield, payout ratio)
- Filtering by industry (e.g., "情報・通信業", "医薬品")
Do not use for SEPA, RS, SMA, price/volume, Stage 2, VCP, pivot, or chart-pattern screening; use sepa_cheap_filter instead when available.`,
    schema: ScreenCompaniesInputSchema,
    func: async (input, _runManager, config?: RunnableConfig) => {
      const onProgress = config?.metadata?.onProgress as ((msg: string) => void) | undefined;

      // LLM structured output — translate natural language → screening conditions
      onProgress?.('Building screening criteria...');
      let conditions: ScreenerConditions;
      try {
        const { response } = await callLlm(input.query, {
          model,
          systemPrompt: buildScreenerPrompt(),
          outputSchema: ScreenerConditionSchema,
        });
        conditions = ScreenerConditionSchema.parse(response);
      } catch (error) {
        onProgress?.('Using heuristic screening criteria...');
        conditions = buildHeuristicConditions(input.query);
      }

      // GET /screener with conditions as JSON query param
      onProgress?.('Screening companies...');
      try {
        const params: Record<string, string | number | undefined> = {
          conditions: JSON.stringify(conditions.conditions),
          limit: conditions.limit ?? 25,
        };
        if (conditions.industry) params.industry = conditions.industry;
        if (conditions.sort_by) params.sort = conditions.sort_by;

        const { data, url } = await api.get('/screener', params);
        return formatToolResult(data, [url]);
      } catch (error) {
        return formatToolResult(
          {
            error: 'Screener request failed',
            details: error instanceof Error ? error.message : String(error),
            conditions: conditions.conditions,
          },
          [],
        );
      }
    },
  });
}
