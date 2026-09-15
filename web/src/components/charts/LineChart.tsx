export type Line = { name: string; values: number[]; color: string; width?: number; dashed?: boolean };
export type Band = { lower: number[]; upper: number[]; color: string; opacity: number };
export type HLine = { value: number; label: string; tone?: "attention" | "ink" };

const W = 640;
const H = 250;
const L = 48;
const R = 88;
const TOP = 16;
const BOTTOM = 26;

/** Lines and shaded bands over categorical x labels; end values labelled on the right. */
export function LineChart({
  xLabels,
  lines,
  bands = [],
  hlines = [],
  yFormat,
  label,
  yMin,
  yMax,
  endLabels = true,
  xTickEvery,
}: {
  /** Label every nth x value (default: about 8 labels) */
  xTickEvery?: number;
  xLabels: string[];
  lines: Line[];
  bands?: Band[];
  hlines?: HLine[];
  yFormat: (v: number) => string;
  label: string;
  yMin?: number;
  yMax?: number;
  endLabels?: boolean;
}) {
  const all = [...lines.flatMap((l) => l.values), ...bands.flatMap((b) => [...b.lower, ...b.upper]), ...hlines.map((h) => h.value)].filter(Number.isFinite);
  let lo = yMin ?? Math.min(...all);
  let hi = yMax ?? Math.max(...all);
  if (hi === lo) hi = lo + 1;
  const pad = (hi - lo) * 0.06;
  if (yMin === undefined) lo -= pad;
  if (yMax === undefined) hi += pad;
  const n = xLabels.length;
  const x = (i: number) => L + ((W - L - R) * i) / Math.max(n - 1, 1);
  const y = (v: number) => TOP + (H - TOP - BOTTOM) * (1 - (v - lo) / (hi - lo));
  const ticks = Array.from({ length: 5 }, (_, i) => lo + ((hi - lo) * i) / 4);
  const xEvery = xTickEvery ?? Math.ceil(n / 8);

  return (
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={label} className="block h-auto w-full">
      {ticks.map((t) => (
        <g key={t}>
          <line x1={L} x2={W - R} y1={y(t)} y2={y(t)} stroke="var(--color-grid)" />
          <text x={L - 6} y={y(t) + 4} textAnchor="end" className="chart-tick">
            {yFormat(t)}
          </text>
        </g>
      ))}
      {xLabels.map((xl, i) =>
        i % xEvery === 0 || i === n - 1 ? (
          <text key={i} x={x(i)} y={H - 8} textAnchor="middle" className="chart-tick">
            {xl}
          </text>
        ) : null,
      )}
      {bands.map((b, k) => (
        <path
          key={k}
          d={`M${b.upper.map((v, i) => `${x(i)},${y(v)}`).join(" L")} L${[...b.lower].reverse().map((v, i) => `${x(b.lower.length - 1 - i)},${y(v)}`).join(" L")} Z`}
          fill={b.color}
          fillOpacity={b.opacity}
        />
      ))}
      {hlines.map((h) => (
        <g key={h.label}>
          <line x1={L} x2={W - R} y1={y(h.value)} y2={y(h.value)} stroke={h.tone === "attention" ? "var(--color-attention)" : "var(--color-ink)"} strokeDasharray="3 3" />
          <text x={L + 4} y={y(h.value) - 5} className="chart-reference" style={{ fill: h.tone === "attention" ? "var(--color-attention)" : undefined }}>
            {h.label}
          </text>
        </g>
      ))}
      {lines.map((l) => (
        <g key={l.name}>
          <polyline
            points={l.values.map((v, i) => `${x(i)},${y(v)}`).join(" ")}
            fill="none"
            stroke={l.color}
            strokeWidth={l.width ?? 2}
            strokeDasharray={l.dashed ? "4 3" : undefined}
          />
          {endLabels && Number.isFinite(l.values.at(-1)) && (
            <text x={x(l.values.length - 1) + 8} y={y(l.values.at(-1)!) + 4} className="chart-reference">
              {l.name} {yFormat(l.values.at(-1)!)}
            </text>
          )}
        </g>
      ))}
    </svg>
  );
}
