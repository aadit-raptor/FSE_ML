"use client";

import { GroupedBars } from "@/components/charts/Bars";
import { plain, type ReportBlock, type ReportSegment } from "@/lib/report";

const SERIES_COLOURS = ["var(--color-accent)", "var(--color-soft)", "var(--color-gain)"];

function Segments({ parts }: { parts: ReportSegment[] | null | undefined }) {
  return (
    <>
      {(parts ?? []).map((p, i) =>
        p.kind === "figure" ? (
          <span key={i} className="font-mono text-ink" data-figure={p.source ?? ""}>{p.text}</span>
        ) : p.kind === "typed" ? (
          <span key={i} className="whitespace-pre-wrap text-ink" data-typed="">{p.text}</span>
        ) : (
          <span key={i}>{p.text}</span>
        ),
      )}
    </>
  );
}

/** One block of a checked draft: a results note's (PLAN.md R8) or an investment memo's (6.3). */
export function DraftBlock({ block, chartLabel }: { block: ReportBlock; chartLabel: string }) {
  if (block.type === "heading") {
    return block.level === 1 ? (
      <h2 className="type-result-title text-[15px] text-bright"><Segments parts={block.parts} /></h2>
    ) : (
      <h3 className="type-result-title mt-2 text-[11px] text-accent"><Segments parts={block.parts} /></h3>
    );
  }
  if (block.type === "paragraph") {
    const tone = block.kind === "note" ? "text-muted text-[10px]" : "text-soft text-[11px]";
    return (
      <p className={`type-body ${tone}`} data-kind={block.kind ?? ""}>
        <Segments parts={block.parts} />
      </p>
    );
  }
  if (block.type === "table") {
    return (
      <table className="w-full border-collapse text-[10.5px]">
        <thead>
          <tr className="border-b border-line">
            {(block.header ?? []).map((cell, i) => (
              <th key={i} scope="col" className={`type-input-label py-1 ${i ? "text-end" : "text-start"}`}>
                <Segments parts={cell} />
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {(block.rows ?? []).map((row, r) => (
            <tr key={r} className="border-b border-line/60">
              {row.map((cell, i) =>
                i === 0 ? (
                  <th key={i} scope="row" className="type-body py-0.5 text-start font-normal text-soft">
                    <Segments parts={cell} />
                  </th>
                ) : (
                  <td key={i} className="py-0.5 text-end"><Segments parts={cell} /></td>
                ),
              )}
            </tr>
          ))}
        </tbody>
      </table>
    );
  }
  const series = block.series ?? [];
  return (
    <figure className="grid gap-1">
      <figcaption className="type-input-label">{plain(block.title)}</figcaption>
      <GroupedBars
        categories={block.categories ?? []}
        series={series.map((s, k) => ({
          // a figure the model doesn't hold is a gap with its dash, never a bar of nought
          name: s.name, values: s.values.map((v) => v ?? 0), color: SERIES_COLOURS[k % SERIES_COLOURS.length],
          labelled: true, labels: s.texts,
        }))}
        format={() => ""}
        label={chartLabel}
      />
    </figure>
  );
}
