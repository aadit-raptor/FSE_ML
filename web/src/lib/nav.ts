/**
 * Every mode and step in the app. The shell's tabs, step row, command search
 * and keyboard shortcuts, and the generated routes, all read from this list.
 *
 * The words live in the translation files (PLAN.md 2.3b): each entry carries
 * the key of its label and of the line search matches on, under the `nav`
 * namespace of `web/messages/<language>.json`.
 *
 * i18n-keys: nav.*
 */

export type Step = {
  slug: string;
  labelKey: string;
  /** Key of the one line on what the step shows (used by search). */
  summaryKey: string;
};

export type Mode = {
  slug: string;
  labelKey: string;
  steps: Step[];
  /** Hidden when switched off: the reference library (`useModes`). */
  optional?: boolean;
};

export const MODES: Mode[] = [
  {
    slug: "deal",
    labelKey: "modeDeal",
    steps: [
      { slug: "inputs", labelKey: "dealInputs", summaryKey: "dealInputsSummary" },
      { slug: "debt", labelKey: "dealDebt", summaryKey: "dealDebtSummary" },
      { slug: "returns", labelKey: "dealReturns", summaryKey: "dealReturnsSummary" },
      { slug: "summary", labelKey: "dealSummary", summaryKey: "dealSummarySummary" },
      { slug: "saved", labelKey: "dealSaved", summaryKey: "dealSavedSummary" },
    ],
  },
  {
    slug: "monte-carlo",
    labelKey: "modeMonteCarlo",
    steps: [
      { slug: "distribution", labelKey: "mcDistribution", summaryKey: "mcDistributionSummary" },
      { slug: "scenarios", labelKey: "mcScenarios", summaryKey: "mcScenariosSummary" },
      { slug: "drivers", labelKey: "mcDrivers", summaryKey: "mcDriversSummary" },
      { slug: "heatmap", labelKey: "mcHeatmap", summaryKey: "mcHeatmapSummary" },
      { slug: "live", labelKey: "mcLive", summaryKey: "mcLiveSummary" },
    ],
  },
  {
    slug: "backtest",
    labelKey: "modeBacktest",
    steps: [
      { slug: "actuals", labelKey: "backtestActuals", summaryKey: "backtestActualsSummary" },
      { slug: "predicted", labelKey: "backtestPredicted", summaryKey: "backtestPredictedSummary" },
      { slug: "attribution", labelKey: "backtestAttribution", summaryKey: "backtestAttributionSummary" },
      { slug: "years", labelKey: "backtestYears", summaryKey: "backtestYearsSummary" },
    ],
  },
  {
    slug: "forecast",
    labelKey: "modeForecast",
    steps: [
      { slug: "historicals", labelKey: "forecastHistoricals", summaryKey: "forecastHistoricalsSummary" },
      { slug: "assumptions", labelKey: "forecastAssumptions", summaryKey: "forecastAssumptionsSummary" },
      { slug: "statements", labelKey: "forecastStatements", summaryKey: "forecastStatementsSummary" },
      { slug: "schedules", labelKey: "forecastSchedules", summaryKey: "forecastSchedulesSummary" },
      { slug: "simulation", labelKey: "forecastSimulation", summaryKey: "forecastSimulationSummary" },
    ],
  },
  {
    slug: "settings",
    labelKey: "modeSettings",
    steps: [
      { slug: "deal", labelKey: "settingsDeal", summaryKey: "settingsDealSummary" },
      { slug: "fees", labelKey: "settingsFees", summaryKey: "settingsFeesSummary" },
      { slug: "monte-carlo", labelKey: "settingsMonteCarlo", summaryKey: "settingsMonteCarloSummary" },
      { slug: "correlations", labelKey: "settingsCorrelations", summaryKey: "settingsCorrelationsSummary" },
      { slug: "presets", labelKey: "settingsPresets", summaryKey: "settingsPresetsSummary" },
    ],
  },
  {
    // The optional reference library (PLAN.md 4.5): last, so hiding it moves no other mode's shortcut
    slug: "library",
    labelKey: "modeLibrary",
    optional: true,
    steps: [
      { slug: "base-rates", labelKey: "libraryBaseRates", summaryKey: "libraryBaseRatesSummary" },
      { slug: "references", labelKey: "libraryReferences", summaryKey: "libraryReferencesSummary" },
      { slug: "examples", labelKey: "libraryExamples", summaryKey: "libraryExamplesSummary" },
      { slug: "coverage", labelKey: "libraryCoverage", summaryKey: "libraryCoverageSummary" },
      { slug: "review", labelKey: "libraryReview", summaryKey: "libraryReviewSummary" },
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
