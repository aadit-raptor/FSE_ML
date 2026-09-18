import { api, type Schemas } from "@/lib/api/client";

/**
 * Long runs go through the API's job queue (PLAN.md 1.9): submit, poll the
 * job until it ends, return its result. The screen stays usable meanwhile,
 * and a run survives the free API restarting (the server resumes it).
 * Aborting the signal cancels the job on the server too.
 */
export type JobSubmit = Schemas["MonteCarloJob"] | Schemas["ScenariosJob"] | Schemas["BacktestJob"] | Schemas["ForecastJob"];
export type Job = Schemas["JobOut"];

export class JobFailed extends Error {}

// Poll quickly at first (most runs take a few seconds), then ease off
const FIRST_POLL_MS = 400;
const MAX_POLL_MS = 2000;
// Polls that may fail in a row (the API waking, a network blip) before giving up
const MAX_POLL_FAILURES = 15;

function sleep(ms: number, signal: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    const t = setTimeout(resolve, ms);
    signal.addEventListener("abort", () => {
      clearTimeout(t);
      reject(new DOMException("Aborted", "AbortError"));
    }, { once: true });
  });
}

export function problem(err: unknown, fallback: string): string {
  const d = (err as { detail?: unknown })?.detail;
  if (typeof d === "string") return d;
  if (Array.isArray(d)) return d.map((x: { loc?: unknown[]; msg?: string }) => `${String(x.loc?.at(-1))}: ${x.msg}`).join("; ");
  return fallback;
}

export async function runJob<T>(job: JobSubmit, opts: { signal: AbortSignal; onUpdate?: (job: Job) => void }): Promise<T> {
  const { signal, onUpdate } = opts;
  const submitted = await api.POST("/api/jobs", { body: job, signal });
  if (!submitted.data) throw new JobFailed(problem(submitted.error, "The run couldn't start with these inputs."));
  const id = submitted.data.id;
  onUpdate?.(submitted.data);

  const cancel = () => void api.POST("/api/jobs/{job_id}/cancel", { params: { path: { job_id: id } } }).catch(() => undefined);
  signal.addEventListener("abort", cancel, { once: true });
  try {
    let delay = FIRST_POLL_MS;
    let failures = 0;
    for (;;) {
      await sleep(delay, signal);
      delay = Math.min(MAX_POLL_MS, delay * 1.4);
      let res;
      try {
        res = await api.GET("/api/jobs/{job_id}", { params: { path: { job_id: id } }, signal });
      } catch (e) {
        if (signal.aborted) throw e;
        if (++failures > MAX_POLL_FAILURES) throw e;
        continue;
      }
      if (!res.data) {
        if (res.response.status >= 500 && ++failures <= MAX_POLL_FAILURES) continue;
        throw new JobFailed(problem(res.error, "Lost track of the run."));
      }
      failures = 0;
      const j = res.data;
      if (j.status === "succeeded") {
        if (!j.result) throw new JobFailed("The result has expired. Run it again.");
        return j.result as T;
      }
      if (j.status === "failed" || j.status === "cancelled") throw new JobFailed(j.error ?? "The run didn't finish.");
      onUpdate?.(j);
    }
  } finally {
    signal.removeEventListener("abort", cancel);
  }
}

/** A line for people: where a job is. */
export function describeJob(job: Job | undefined): string {
  if (!job) return "Starting";
  if (job.status === "queued") {
    if (job.attempts > 0) return "Resuming after a server restart";
    return job.ahead ? `Waiting: ${job.ahead} run${job.ahead === 1 ? "" : "s"} ahead` : "Waiting to start";
  }
  return job.stage ?? "Running";
}
