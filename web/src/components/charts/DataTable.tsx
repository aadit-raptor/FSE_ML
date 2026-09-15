import { fmtMoney, fmtRate, isNum } from "@/lib/format";

export type Row = {
  label: string;
  values: (number | null | undefined)[];
  /** money (default), rate (fraction), or money shown as an outflow */
  kind?: "money" | "rate" | "outflow";
  total?: boolean;
};

function cell(v: number | null | undefined, kind: Row["kind"]) {
  if (!isNum(v)) return { text: "n/a", negative: false };
  if (kind === "rate") return { text: fmtRate(v), negative: v < 0 };
  const shown = kind === "outflow" ? -Math.abs(v) : v;
  return { text: fmtMoney(shown), negative: shown < 0 };
}

/** Year-by-year financial table. Labels in thickened Michroma, figures mono. */
export function DataTable({ columns, rows, caption }: { columns: string[]; rows: Row[]; caption: string }) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full border-collapse font-mono text-[11px]">
        <caption className="sr-only">{caption}</caption>
        <thead>
          <tr>
            <th scope="col" className="w-[30%]" />
            {columns.map((c) => (
              <th key={c} scope="col" className="border-b border-grid px-2 py-1 text-right font-normal text-muted">
                {c}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.label} className={r.total ? "text-bright" : "text-ink"}>
              <th scope="row" className="type-input-label border-b border-grid px-2 py-1 text-left text-[9px] font-normal text-soft">
                {r.label}
              </th>
              {r.values.map((v, i) => {
                const c = cell(v, r.kind);
                return (
                  <td key={i} className={`border-b border-grid px-2 py-1 text-right whitespace-nowrap ${c.negative ? "text-loss" : ""} ${r.total ? "font-semibold" : ""}`}>
                    {c.text}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
