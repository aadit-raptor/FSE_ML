import { fmtAxis, fmtMoney } from "@/lib/format";

import { niceScale } from "./scale";

export type Series = { name: string; values: number[]; color: string };

const W = 560;
const H = 210;
const L = 44;
const R = 8;
const TOP = 22;
const BOTTOM = 24;

/** Stacked columns with the stack total labelled on top. */
export function StackedBars({ categories, series, label }: { categories: string[]; series: Series[]; label: string }) {
  const totals = categories.map((_, i) => series.reduce((s, x) => s + (x.values[i] ?? 0), 0));
  const { max, ticks } = niceScale(Math.max(...totals, 1));
  const y = (v: number) => TOP + (H - TOP - BOTTOM) * (1 - v / max);
  const band = (W - L - R) / Math.max(categories.length, 1);
  const w = band * 0.5;

  return (
    <div className="grid gap-2">
      <div className="flex gap-4">
        {series.map((s) => (
          <span key={s.name} className="type-input-label flex items-center gap-1.5 text-[8.5px] text-soft">
            <i className="inline-block size-2.5" style={{ background: s.color }} aria-hidden />
            {s.name}
          </span>
        ))}
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={label} className="block h-auto w-full">
        {ticks.map((t) => (
          <g key={t}>
            <line x1={L} x2={W - R} y1={y(t)} y2={y(t)} stroke="var(--color-grid)" />
            <text x={L - 6} y={y(t) + 4} textAnchor="end" className="chart-tick">
              {fmtAxis(t)}
            </text>
          </g>
        ))}
        {categories.map((c, i) => {
          const x = L + i * band + (band - w) / 2;
          const values = series.map((s) => s.values[i] ?? 0);
          return (
            <g key={c}>
              {series.map((s, k) => {
                const base = values.slice(0, k).reduce((a, b) => a + b, 0);
                const v = values[k];
                return <rect key={s.name} x={x} y={y(base + v)} width={w} height={Math.max(0, y(base) - y(base + v))} fill={s.color} />;
              })}
              <text x={x + w / 2} y={y(totals[i]) - 6} textAnchor="middle" className="chart-value">
                {fmtMoney(totals[i])}
              </text>
              <text x={x + w / 2} y={H - 7} textAnchor="middle" className="chart-category">
                {c}
              </text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}
