import type { Mode, Step } from "@/lib/nav";

/**
 * Stand-in for a screen that step 4 of the rebuild has not built yet.
 * Says plainly what will be here and which API feeds it; no fake controls.
 */
export function StepPending({ mode, step }: { mode: Mode; step: Step }) {
  return (
    <div className="grid h-full grid-cols-[270px_1fr]">
      <aside className="border-r border-line bg-panel px-3.5 py-2.5" aria-label="Inputs">
        <p className="type-input-group">Inputs</p>
        <p className="type-body mt-2">This step&apos;s inputs will appear here.</p>
      </aside>
      <section className="grid content-start gap-4 p-6">
        <div className="grid gap-2">
          <p className="type-control">{mode.label}</p>
          <h1 className="type-result-title text-[18px]">{step.label}</h1>
          <p className="type-body max-w-[70ch] text-[10.5px]">{step.summary}</p>
        </div>
        <div className="grid max-w-[640px] gap-2 border border-line bg-panel p-4">
          <p className="type-alert">Not built yet</p>
          <p className="type-body">
            The shell, navigation and API client are in place. This screen is rebuilt in step 4, reading from:
          </p>
          <ul className="grid gap-1">
            {step.endpoints.map((e) => (
              <li key={e} className="font-mono text-[11px] text-ink">
                {e}
              </li>
            ))}
          </ul>
        </div>
      </section>
    </div>
  );
}
