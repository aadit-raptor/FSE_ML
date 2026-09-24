import { fmtAxis, fmtDelta, fmtMoney } from "@/lib/format";

import { niceScale } from "./scale";

export type WaterfallStep = { label: string; value: number; isTotal: boolean };

const W = 560;
const H = 236;
const L = 44;
const R = 8;
const TOP = 22;
const BOTTOM = 38;

/** Each step's span: totals from zero, deltas from the running level. */
function stackSteps(steps: WaterfallStep[]) {
  return steps.reduce<(WaterfallStep & { lo: number; hi: number; end: number })[]>((acc, s) => {
    const from = s.isTotal ? 0 : (acc.at(-1)?.end ?? 0);
    const to = s.isTotal ? s.value : from + s.value;
    return [...acc, { ...s, lo: Math.min(from, to), hi: Math.max(from, to), end: to }];
  }, []);
}

/** Totals in accent, gains green, losses red, dashed connectors between steps. */
export function Waterfall({ steps, label }: { steps: WaterfallStep[]; label: string }) {
  const bars = stackSteps(steps);
  const { max, ticks } = niceScale(Math.max(...bars.map((b) => b.hi), 1));
  const y = (v: number) => TOP + (H - TOP - BOTTOM) * (1 - v / max);
  const band = (W - L - R) / Math.max(steps.length, 1);
  const w = band * 0.56;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={label} className="block h-auto w-full">
      {ticks.map((t) => (
        <g key={t}>
          <line x1={L} x2={W - R} y1={y(t)} y2={y(t)} stroke="var(--color-grid)" />
          <text x={L - 6} y={y(t) + 4} textAnchor="end" className="chart-tick">
            {fmtAxis(t)}
          </text>
        </g>
      ))}
      {bars.map((b, i) => {
        const x = L + i * band + (band - w) / 2;
        const top = y(b.hi);
        const fill = b.isTotal ? "var(--color-accent)" : b.value < 0 ? "var(--color-loss)" : "var(--color-gain)";
        const words = b.label.split(" ");
        return (
          <g key={b.label}>
            <rect x={x} y={top} width={w} height={Math.max(1, y(b.lo) - top)} fill={fill} />
            <text x={x + w / 2} y={top - 6} textAnchor="middle" className={b.isTotal ? "chart-total" : "chart-value"}>
              {b.isTotal ? fmtMoney(b.value) : fmtDelta(b.value)}
            </text>
            {i < bars.length - 1 && (
              <line x1={x + w} x2={x + band} y1={y(b.end)} y2={y(b.end)} stroke="var(--color-muted)" strokeDasharray="2 2" />
            )}
            {[words.slice(0, 1).join(" "), words.slice(1).join(" ")].filter(Boolean).map((ln, k) => (
              <text key={k} x={x + w / 2} y={H - BOTTOM + 16 + k * 12} textAnchor="middle" className="chart-category">
                {ln}
              </text>
            ))}
          </g>
        );
      })}
    </svg>
  );
}
