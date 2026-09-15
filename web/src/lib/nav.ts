/**
 * Every mode and step in the app. The shell's tabs, step row, command search
 * and keyboard shortcuts, and the generated routes, all read from this list.
 */

export type Step = {
  slug: string;
  label: string;
  /** What the step shows once it is built (step 4 of the rebuild). */
  summary: string;
  /** API operations that feed it. */
  endpoints: string[];
};

export type Mode = {
  slug: string;
  label: string;
  steps: Step[];
};

export const MODES: Mode[] = [
  {
    slug: "deal",
    label: "Deal",
    steps: [
      {
        slug: "inputs",
        label: "Deal inputs",
        summary: "Entry and exit multiples, operations and capital structure, with sources and uses.",
        endpoints: ["POST /api/deal/sources-and-uses"],
      },
      {
        slug: "debt",
        label: "Debt & cash flow",
        summary: "Tranche schedule, cash sweep, working capital and minimum cash.",
        endpoints: ["POST /api/deal/run"],
      },
      {
        slug: "returns",
        label: "Returns",
        summary: "IRR, MOIC, equity bridge, exit sensitivity and debt paydown.",
        endpoints: ["POST /api/deal/run"],
      },
      {
        slug: "summary",
        label: "Summary",
        summary: "P&L, cash flow, debt schedule and balance sheet for the whole hold.",
        endpoints: ["POST /api/deal/run"],
      },
    ],
  },
  {
    slug: "monte-carlo",
    label: "Monte Carlo",
    steps: [
      {
        slug: "distribution",
        label: "Distribution",
        summary: "IRR and MOIC distributions, percentiles, and the chance of clearing the hurdle.",
        endpoints: ["POST /api/montecarlo/run"],
      },
      {
        slug: "scenarios",
        label: "Scenarios",
        summary: "Recession, stagflation, base and bull cases side by side.",
        endpoints: ["POST /api/montecarlo/scenarios"],
      },
      {
        slug: "drivers",
        label: "Drivers",
        summary: "Which inputs move IRR most, and the correlations between them.",
        endpoints: ["POST /api/montecarlo/run"],
      },
      {
        slug: "heatmap",
        label: "Heatmap",
        summary: "IRR across growth and exit multiple.",
        endpoints: ["POST /api/montecarlo/run"],
      },
    ],
  },
  {
    slug: "backtest",
    label: "Backtest",
    steps: [
      {
        slug: "predicted",
        label: "Predicted vs actual",
        summary: "The model run on a real deal's entry assumptions, against what happened.",
        endpoints: ["GET /api/backtesting/deals", "POST /api/backtesting/run"],
      },
      {
        slug: "attribution",
        label: "Error attribution",
        summary: "Where the prediction missed: growth, margin, cash conversion and paydown.",
        endpoints: ["POST /api/backtesting/run"],
      },
      {
        slug: "years",
        label: "Year by year",
        summary: "Predicted and actual EBITDA, revenue, free cash flow and debt per year.",
        endpoints: ["POST /api/backtesting/run"],
      },
    ],
  },
  {
    slug: "forecast",
    label: "Forecast",
    steps: [
      {
        slug: "historicals",
        label: "Historicals",
        summary: "Three years of statements, entered by hand or autofilled from SEC EDGAR.",
        endpoints: ["GET /api/forecasting/defaults", "GET /api/edgar/{ticker}"],
      },
      {
        slug: "assumptions",
        label: "Assumptions",
        summary: "Growth, margins and working capital for each forecast year.",
        endpoints: ["POST /api/forecasting/seed"],
      },
      {
        slug: "statements",
        label: "Statements",
        summary: "Income statement, balance sheet and cash flow, with the balance check.",
        endpoints: ["POST /api/forecasting/run"],
      },
      {
        slug: "simulation",
        label: "Simulation",
        summary: "Revenue and EBITDA fans, and the chance of reaching each target.",
        endpoints: ["POST /api/forecasting/run"],
      },
    ],
  },
  {
    slug: "settings",
    label: "Settings",
    steps: [
      {
        slug: "deal",
        label: "Deal defaults",
        summary: "Starting values for every deal input.",
        endpoints: ["GET /api/settings/defaults"],
      },
      {
        slug: "fees",
        label: "Fees",
        summary: "Transaction and financing fees, and other uses of funds.",
        endpoints: ["GET /api/settings/defaults"],
      },
      {
        slug: "monte-carlo",
        label: "Monte Carlo",
        summary: "Simulation defaults and hurdle rate.",
        endpoints: ["GET /api/settings/defaults"],
      },
      {
        slug: "correlations",
        label: "Correlations",
        summary: "The driver correlation matrix, checked for validity as you edit.",
        endpoints: ["POST /api/settings/validate"],
      },
      {
        slug: "presets",
        label: "Scenario presets",
        summary: "Multipliers behind the bull, recession and stagflation cases.",
        endpoints: ["GET /api/settings/defaults"],
      },
    ],
  },
];

export const DEFAULT_HREF = stepHref("deal", "inputs");

export function stepHref(mode: string, step: string): string {
  return `/${mode}/${step}`;
}

export function findMode(slug: string | undefined): Mode | undefined {
  return MODES.find((m) => m.slug === slug);
}

export function findStep(modeSlug: string, stepSlug: string): { mode: Mode; step: Step } | undefined {
  const mode = findMode(modeSlug);
  const step = mode?.steps.find((s) => s.slug === stepSlug);
  return mode && step ? { mode, step } : undefined;
}

/** Mode and step slugs from a pathname like "/deal/returns". */
export function parsePath(pathname: string): { mode?: Mode; step?: Step } {
  const [modeSlug, stepSlug] = pathname.split("/").filter(Boolean);
  const mode = findMode(modeSlug);
  return { mode, step: mode?.steps.find((s) => s.slug === stepSlug) };
}
