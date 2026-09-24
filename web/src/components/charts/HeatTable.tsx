import { fmtMultiple, fmtRate, isNum } from "@/lib/format";

/** Background mixed toward gain or loss by distance from `center`. */
export function heat(v: number, center: number, span: number): string {
  const k = Math.max(-1, Math.min(1, (v - center) / span));
  const color = k >= 0 ? "var(--color-gain)" : "var(--color-loss)";
  return `color-mix(in oklab, ${color} ${Math.round(Math.abs(k) * 46)}%, var(--color-panel))`;
}

/** IRR grid: rows = exit multiples, columns = holding periods. */
export function SensitivityTable({
  exitMultiples,
  holds,
  table,
  hurdle,
  baseRow,
  baseCol,
}: {
  exitMultiples: number[];
  holds: number[];
  table: (number | null)[][];
  hurdle: number;
  baseRow: number;
  baseCol: number;
}) {
  return (
    <table className="w-full border-separate border-spacing-px font-mono text-[11px]">
      <caption className="sr-only">IRR by exit multiple and holding period</caption>
      <thead>
        <tr>
          <th scope="col" className="px-2 py-1 text-right font-normal text-muted">
            Exit
          </th>
          {holds.map((h) => (
            <th key={h} scope="col" className="px-2 py-1 text-right font-normal text-muted">
              {h}y
            </th>
          ))}
        </tr>
      </thead>
      <tbody>
        {table.map((row, i) => (
          <tr key={exitMultiples[i]}>
            <th scope="row" className="px-2 py-1 text-right font-normal text-muted">
              {fmtMultiple(exitMultiples[i], 1)}
            </th>
            {row.map((v, j) => {
              const base = i === baseRow && j === baseCol;
              return (
                <td
                  key={holds[j]}
                  className={`px-2 py-1 text-right whitespace-nowrap ${base ? "font-semibold outline-[1.5px] -outline-offset-[1.5px] outline-ink outline-solid" : ""}`}
                  style={isNum(v) ? { background: heat(v, hurdle, 0.25) } : undefined}
                  aria-current={base ? "true" : undefined}
                >
                  {fmtRate(v)}
                </td>
              );
            })}
          </tr>
        ))}
      </tbody>
    </table>
  );
}
