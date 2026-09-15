/**
 * Every mode and step in the app. The shell's tabs, step row, command search
 * and keyboard shortcuts, and the generated routes, all read from this list.
 */

export type Step = {
  slug: string;
  label: string;
  /** One line on what the step shows (used by search). */
  summary: string;
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
      },
      {
        slug: "debt",
        label: "Debt & cash flow",
        summary: "Tranche schedule, cash sweep, working capital and minimum cash.",
      },
      {
        slug: "returns",
        label: "Returns",
        summary: "IRR, MOIC, equity bridge, exit sensitivity and debt paydown.",
      },
      {
        slug: "summary",
        label: "Summary",
        summary: "P&L, cash flow, debt schedule and balance sheet for the whole hold.",
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
      },
      {
        slug: "scenarios",
        label: "Scenarios",
        summary: "Recession, stagflation, base and bull cases side by side.",
      },
      {
        slug: "drivers",
        label: "Drivers",
        summary: "Which inputs move IRR most, and the correlations between them.",
      },
      {
        slug: "heatmap",
        label: "Heatmap",
        summary: "IRR across growth and exit multiple.",
      },
      {
        slug: "live",
        label: "Live",
        summary: "Instant IRR estimates from a neural network trained on the simulation, as you drag the assumptions.",
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
      },
      {
        slug: "attribution",
        label: "Error attribution",
        summary: "Where the prediction missed: growth, margin, cash conversion and paydown.",
      },
      {
        slug: "years",
        label: "Year by year",
        summary: "Predicted and actual EBITDA, revenue, free cash flow and debt per year.",
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
      },
      {
        slug: "assumptions",
        label: "Assumptions",
        summary: "Growth, margins and working capital for each forecast year.",
      },
      {
        slug: "statements",
        label: "Statements",
        summary: "Income statement, balance sheet and cash flow, with the balance check.",
      },
      {
        slug: "schedules",
        label: "Schedules",
        summary: "PP&E, retained earnings, working capital, interest and revolver schedules, and the EBITDA to net income bridge.",
      },
      {
        slug: "simulation",
        label: "Simulation",
        summary: "Revenue and EBITDA fans, and the chance of reaching each target.",
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
      },
      {
        slug: "fees",
        label: "Fees",
        summary: "Transaction and financing fees, and other uses of funds.",
      },
      {
        slug: "monte-carlo",
        label: "Monte Carlo",
        summary: "Simulation defaults and hurdle rate.",
      },
      {
        slug: "correlations",
        label: "Correlations",
        summary: "The driver correlation matrix, checked for validity as you edit.",
      },
      {
        slug: "presets",
        label: "Scenario presets",
        summary: "Multipliers behind the bull, recession and stagflation cases.",
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

/** Mode and step slugs from a pathname like "/deal/returns". */
export function parsePath(pathname: string): { mode?: Mode; step?: Step } {
  const [modeSlug, stepSlug] = pathname.split("/").filter(Boolean);
  const mode = findMode(modeSlug);
  return { mode, step: mode?.steps.find((s) => s.slug === stepSlug) };
}
