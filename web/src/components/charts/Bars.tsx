import { niceScale } from "./scale";

/** Horizontal bars from a zero line; negative left in loss, positive right in gain. */
export function DivergingBars({
  rows,
  format,
  label,
  domain,
}: {
  rows: { label: string; value: number }[];
  format: (v: number) => string;
  label: string;
  /** Symmetric limit; defaults to the largest magnitude */
  domain?: number;
}) {
  const W = 460;
  const rowH = 36;
  const L = 132;
  const R = 64;
  const H = rowH * rows.length + 8;
  const lim = domain ?? Math.max(...rows.map((r) => Math.abs(r.value)), 1e-9);
  const hasNeg = rows.some((r) => r.value < 0);
  const lo = hasNeg ? -lim : 0;
  const x = (v: number) => L + ((W - L - R) * (v - lo)) / (lim - lo);
  const x0 = x(0);
  return (
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={label} className="block h-auto w-full">
      <line x1={x0} x2={x0} y1={2} y2={H - 4} stroke="var(--color-line)" />
      {rows.map((r, i) => {
        const cy = 4 + i * rowH + rowH / 2;
        return (
          <g key={r.label}>
            <text x={0} y={cy + 3} className="chart-category">
              {r.label}
            </text>
            <rect
              x={Math.min(x0, x(r.value))}
              y={cy - 6}
              width={Math.abs(x(r.value) - x0)}
              height={12}
              fill={r.value < 0 ? "var(--color-loss)" : "var(--color-gain)"}
            />
            <text x={W} y={cy + 4} textAnchor="end" className="chart-value">
              {format(r.value)}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

/** Low-to-high range per row with a mean tick and a threshold line. */
export function RangeRows({
  rows,
  threshold,
  format,
  label,
}: {
  rows: { label: string; low: number; high: number; mid: number; note: string }[];
  threshold?: { value: number; label: string };
  format: (v: number) => string;
  label: string;
}) {
  const W = 600;
  const rowH = 42;
  const L = 110;
  const R = 70;
  const TOP = 18;
  const H = TOP + rowH * rows.length + 22;
  const vals = rows.flatMap((r) => [r.low, r.high]).concat(threshold ? [threshold.value] : []);
  const span = Math.max(...vals) - Math.min(...vals) || 1;
  const lo = Math.min(...vals) - span * 0.05;
  const hi = Math.max(...vals) + span * 0.05;
  const x = (v: number) => L + ((W - L - R) * (v - lo)) / (hi - lo);
  const ticks = Array.from({ length: 5 }, (_, i) => lo + ((hi - lo) * (i + 0.5)) / 5);
  return (
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={label} className="block h-auto w-full">
      {ticks.map((t) => (
        <g key={t}>
          <line x1={x(t)} x2={x(t)} y1={TOP} y2={H - 20} stroke="var(--color-grid)" />
          <text x={x(t)} y={H - 6} textAnchor="middle" className="chart-tick">
            {format(t)}
          </text>
        </g>
      ))}
      {threshold && (
        <g>
          <line x1={x(threshold.value)} x2={x(threshold.value)} y1={TOP - 4} y2={H - 20} stroke="var(--color-attention)" strokeDasharray="3 3" />
          <text x={x(threshold.value) + 5} y={TOP - 6} className="chart-reference" style={{ fill: "var(--color-attention)" }}>
            {threshold.label}
          </text>
        </g>
      )}
      {rows.map((r, i) => {
        const cy = TOP + i * rowH + rowH / 2;
        return (
          <g key={r.label}>
            <text x={0} y={cy + 3} className="chart-category">
              {r.label}
            </text>
            <line x1={x(r.low)} x2={x(r.high)} y1={cy} y2={cy} stroke={r.mid < 0 ? "var(--color-loss)" : "var(--color-neutral-bar)"} strokeWidth={7} />
            <rect x={x(r.mid) - 1.5} y={cy - 9} width={3} height={18} fill="var(--color-accent)" />
            <text x={W} y={cy + 4} textAnchor="end" className="chart-value">
              {r.note}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

/** Side-by-side columns per category (e.g. predicted vs actual). */
export function GroupedBars({
  categories,
  series,
  format,
  label,
}: {
  categories: string[];
  series: { name: string; values: number[]; color: string; labelled?: boolean }[];
  format: (v: number) => string;
  label: string;
}) {
  const W = 620;
  const H = 250;
  const L = 48;
  const R = 12;
  const TOP = 22;
  const BOTTOM = 26;
  const { max, ticks } = niceScale(Math.max(...series.flatMap((s) => s.values), 1));
  const y = (v: number) => TOP + (H - TOP - BOTTOM) * (1 - Math.max(v, 0) / max);
  const band = (W - L - R) / Math.max(categories.length, 1);
  const w = (band * 0.62) / series.length;
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
              {t.toLocaleString("en-US")}
            </text>
          </g>
        ))}
        {categories.map((c, i) => {
          const start = L + i * band + band * 0.19;
          return (
            <g key={c}>
              {series.map((s, k) => {
                const v = s.values[i] ?? 0;
                const bx = start + k * w;
                return (
                  <g key={s.name}>
                    <rect x={bx + 1} y={y(v)} width={w - 2} height={H - BOTTOM - y(v)} fill={s.color} />
                    {s.labelled && (
                      <text x={bx + w / 2} y={y(v) - 6} textAnchor="middle" className="chart-total">
                        {format(v)}
                      </text>
                    )}
                  </g>
                );
              })}
              <text x={L + i * band + band / 2} y={H - 8} textAnchor="middle" className="chart-tick">
                {c}
              </text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}

/** Point cloud with an optional least-squares line. */
export function Scatter({
  xs,
  ys,
  fit,
  xFormat,
  yFormat,
  label,
}: {
  xs: number[];
  ys: number[];
  /** y = intercept + slope * x, in the same units as the plotted values */
  fit?: { slope: number; intercept: number };
  xFormat: (v: number) => string;
  yFormat: (v: number) => string;
  label: string;
}) {
  const W = 640;
  const H = 280;
  const L = 52;
  const R = 12;
  const TOP = 12;
  const BOTTOM = 26;
  const pts = xs.map((x, i) => [x, ys[i]] as const).filter(([a, b]) => Number.isFinite(a) && Number.isFinite(b));
  if (!pts.length) return <p className="type-body">No sample to plot.</p>;
  const sortX = pts.map((p) => p[0]).sort((a, b) => a - b);
  const sortY = pts.map((p) => p[1]).sort((a, b) => a - b);
  // Clip the outer 1% so a few extreme paths don't flatten the cloud
  const q = (arr: number[], p: number) => arr[Math.min(arr.length - 1, Math.max(0, Math.round((arr.length - 1) * p)))];
  const [x0, x1] = [q(sortX, 0.005), q(sortX, 0.995)];
  const [y0, y1] = [q(sortY, 0.005), q(sortY, 0.995)];
  const sx = (v: number) => L + ((W - L - R) * (v - x0)) / (x1 - x0 || 1);
  const sy = (v: number) => TOP + (H - TOP - BOTTOM) * (1 - (v - y0) / (y1 - y0 || 1));
  const xt = Array.from({ length: 5 }, (_, i) => x0 + ((x1 - x0) * (i + 0.5)) / 5);
  const yt = Array.from({ length: 5 }, (_, i) => y0 + ((y1 - y0) * i) / 4);
  return (
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={label} className="block h-auto w-full">
      <defs>
        <clipPath id="scatter-plot">
          <rect x={L} y={TOP} width={W - L - R} height={H - TOP - BOTTOM} />
        </clipPath>
      </defs>
      {yt.map((t) => (
        <g key={t}>
          <line x1={L} x2={W - R} y1={sy(t)} y2={sy(t)} stroke="var(--color-grid)" />
          <text x={L - 6} y={sy(t) + 4} textAnchor="end" className="chart-tick">
            {yFormat(t)}
          </text>
        </g>
      ))}
      {xt.map((t) => (
        <text key={t} x={sx(t)} y={H - 8} textAnchor="middle" className="chart-tick">
          {xFormat(t)}
        </text>
      ))}
      <g clipPath="url(#scatter-plot)">
        {pts.map(([a, b], i) => (
          <circle key={i} cx={sx(a)} cy={sy(b)} r={1.6} fill="var(--color-accent)" fillOpacity={0.35} />
        ))}
        {fit && (
          <line x1={sx(x0)} x2={sx(x1)} y1={sy(fit.intercept + fit.slope * x0)} y2={sy(fit.intercept + fit.slope * x1)} stroke="var(--color-attention)" strokeWidth={2} />
        )}
      </g>
    </svg>
  );
}
