"use client";

import { useTranslations } from "next-intl";
import { useEffect, useRef, useState } from "react";

import { pdfFile } from "@/lib/files";
import { fmtCount } from "@/lib/format";

/**
 * One page of the stored results PDF, drawn in the browser with pdf.js (PLAN.md R2): the review shows the page
 * a figure was read from beside the figures. The file comes once from the API's own address (`pdfFile`); the
 * drawing happens here, in pdf.js's worker, so the free server never renders a page for viewing.
 *
 * pdf.js loads only on this screen (a dynamic import) and never evaluates code from a PDF
 * (`isEvalSupported: false`); its worker is a file of this site, which the content security policy's
 * `worker-src 'self'` allows.
 */
type PdfDocument = {
  numPages: number;
  getPage: (n: number) => Promise<PdfPage>;
  destroy: () => Promise<void>;
};
type PdfPage = {
  getViewport: (o: { scale: number }) => { width: number; height: number };
  render: (o: { canvasContext: CanvasRenderingContext2D; viewport: unknown }) => { promise: Promise<void>; cancel: () => void };
};

let library: Promise<{ getDocument: (o: object) => { promise: Promise<PdfDocument> } }> | null = null;

function pdfjs() {
  library ??= import("pdfjs-dist").then((lib) => {
    lib.GlobalWorkerOptions.workerSrc = new URL("pdfjs-dist/build/pdf.worker.min.mjs", import.meta.url).toString();
    return lib as unknown as { getDocument: (o: object) => { promise: Promise<PdfDocument> } };
  });
  return library;
}

type State = { status: "loading" | "ready" | "gone" | "failed"; pages: number };

export function PageView({ fileId, page, onPage }: { fileId: string | null; page: number | null; onPage: (page: number) => void }) {
  const t = useTranslations("coverage");
  const canvas = useRef<HTMLCanvasElement>(null);
  const [doc, setDoc] = useState<PdfDocument | null>(null);
  const [state, setState] = useState<State>({ status: "loading", pages: 0 });

  useEffect(() => {
    let cancelled = false;
    let opened: PdfDocument | null = null;
    if (!fileId) return;
    (async () => {
      const bytes = await pdfFile(fileId);
      if (cancelled) return;
      if (!bytes) {
        setState({ status: "gone", pages: 0 });
        return;
      }
      try {
        const lib = await pdfjs();
        // pdf.js takes over the buffer it is given: hand it a copy so the cached bytes stay usable
        opened = await lib.getDocument({ data: new Uint8Array(bytes.slice(0)), isEvalSupported: false }).promise;
        if (cancelled) {
          void opened.destroy();
          return;
        }
        setDoc(opened);
        setState({ status: "ready", pages: opened.numPages });
      } catch {
        if (!cancelled) setState({ status: "failed", pages: 0 });
      }
    })();
    return () => {
      cancelled = true;
      setDoc(null);
      setState({ status: "loading", pages: 0 });
      if (opened) void opened.destroy();
    };
  }, [fileId]);

  const shown = page !== null && doc && page >= 1 && page <= doc.numPages ? page : doc ? 1 : null;
  useEffect(() => {
    if (!doc || shown === null || !canvas.current) return;
    let task: { promise: Promise<void>; cancel: () => void } | null = null;
    let cancelled = false;
    (async () => {
      const found = await doc.getPage(shown);
      const target = canvas.current;
      if (cancelled || !target) return;
      const width = target.parentElement?.clientWidth || 480;
      const base = found.getViewport({ scale: 1 });
      const ratio = window.devicePixelRatio || 1;
      const viewport = found.getViewport({ scale: (width / base.width) * ratio });
      target.width = Math.floor(viewport.width);
      target.height = Math.floor(viewport.height);
      target.style.width = `${Math.floor(viewport.width / ratio)}px`;
      const context = target.getContext("2d");
      if (!context) return;
      task = found.render({ canvasContext: context, viewport });
      await task.promise.catch(() => undefined);
      if (!cancelled) target.dataset.drawn = String(shown);
    })();
    return () => {
      cancelled = true;
      task?.cancel();
    };
  }, [doc, shown]);

  if (!fileId || state.status === "gone") {
    return <p className="type-body text-[10.5px]" data-page-gone>{t("pageGone")}</p>;
  }
  if (state.status === "failed") return <p className="type-body text-[10.5px] text-attention">{t("pageFailed")}</p>;
  return (
    <div className="grid gap-1.5" data-page-view>
      <div className="flex items-center justify-between gap-2">
        <button
          type="button"
          className="type-action border border-line px-2 py-0.5 text-ink disabled:opacity-40"
          disabled={!shown || shown <= 1}
          onClick={() => shown && onPage(shown - 1)}
        >
          {t("pagePrevious")}
        </button>
        <span className="font-mono text-[10.5px] text-muted" data-page-number>
          {state.status === "ready" && shown ? t("pageOf", { page: fmtCount(shown), pages: fmtCount(state.pages) }) : t("pageLoading")}
        </span>
        <button
          type="button"
          className="type-action border border-line px-2 py-0.5 text-ink disabled:opacity-40"
          disabled={!shown || shown >= state.pages}
          onClick={() => shown && onPage(shown + 1)}
        >
          {t("pageNext")}
        </button>
      </div>
      <div className="min-w-0 overflow-hidden border border-line bg-white">
        <canvas ref={canvas} aria-label={t("pageAlt", { page: shown ? fmtCount(shown) : "" })} role="img" />
      </div>
    </div>
  );
}
