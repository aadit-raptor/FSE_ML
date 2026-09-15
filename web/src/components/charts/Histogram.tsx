export type Marker = { value: number; label: string; tone?: "ink" | "attention" | "accent"; dashed?: boolean };

const W = 640;
const H = 250;
const L = 12;
const R = 12;
const TOP = 34;
const BOTTOM = 26;

const TONE = { ink: "var(--color-ink)", attention: "var(--color-attention)", accent: "var(--color-accent)" };

/**
 * Distribution from bin edges and densities. Empty tails are trimmed; bins at
 * or above `highlightFrom` are drawn in accent (e.g. paths clearing the hurdle).
 */
export function Histogram({
  edges,
  density,
  markers = [],
  highlightFrom,
  format,
  label,
  tickCount = 6,
}: {
  edges: number[];
  density: number[];
  markers?: Marker[];
  highlightFrom?: number;
  format: (v: number) => string;
  label: string;
  tickCount?: number;
}) {
  const nonzero = density.map((d, i) => (d > 0 ? i : -1)).filter((i) => i >= 0);
  if (!nonzero.length) return <p className="type-body">No distribution to show.</p>;
  const first = nonzero[0];
  const last = nonzero.at(-1)!;
  let lo = edges[first];
  let hi = edges[last + 1];
  for (const m of markers) {
    lo = Math.min(lo, m.value);
    hi = Math.max(hi, m.value);
  }
  const pad = (hi - lo) * 0.03;
  lo -= pad;
  hi += pad;
  const dmax = Math.max(...density) * 1.05;
  const x = (v: number) => L + ((W - L - R) * (v - lo)) / (hi - lo);
  const y = (d: number) => TOP + (H - TOP - BOTTOM) * (1 - d / dmax);
  const ticks = Array.from({ length: tickCount }, (_, i) => lo + ((hi - lo) * (i + 0.5)) / tickCount);

  // Stagger marker labels onto rows so neighbours don't collide
  const sorted = [...markers].sort((a, b) => a.value - b.value);
  const rows: number[] = [];
  const placed = sorted.map((m) => {
    const px = x(m.value);
    let row = 0;
    while (rows[row] !== undefined && px - rows[row] < 90) row++;
    rows[row] = px;
    return { ...m, px, row };
  });

  return (
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={label} className="block h-auto w-full">
      <line x1={L} x2={W - R} y1={H - BOTTOM} y2={H - BOTTOM} stroke="var(--color-line)" />
      {ticks.map((t) => (
        <text key={t} x={x(t)} y={H - 8} textAnchor="middle" className="chart-tick">
          {format(t)}
        </text>
      ))}
      {density.slice(first, last + 1).map((d, k) => {
        const i = first + k;
        const a = edges[i];
        const b = edges[i + 1];
        const highlighted = highlightFrom !== undefined && (a + b) / 2 >= highlightFrom;
        return (
          <rect
            key={i}
            x={x(a) + 0.5}
            y={y(d)}
            width={Math.max(1, x(b) - x(a) - 1)}
            height={H - BOTTOM - y(d)}
            fill={highlighted ? "var(--color-accent)" : "var(--color-neutral-bar)"}
          />
        );
      })}
      {placed.map((m) => (
        <g key={m.label}>
          <line
            x1={m.px}
            x2={m.px}
            y1={TOP - 10 + m.row * 13}
            y2={H - BOTTOM}
            stroke={TONE[m.tone ?? "ink"]}
            strokeDasharray={m.dashed ? "3 3" : undefined}
            opacity={0.85}
          />
          <text
            x={m.px + (m.px > W * 0.75 ? -5 : 5)}
            y={TOP - 12 + m.row * 13}
            textAnchor={m.px > W * 0.75 ? "end" : "start"}
            className="chart-reference"
            style={{ fill: TONE[m.tone ?? "ink"] }}
          >
            {m.label}
          </text>
        </g>
      ))}
    </svg>
  );
}
