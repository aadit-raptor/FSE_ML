/**
 * Office.js for the Excel add-in (PLAN.md R3): loading it, and the few parts
 * of it the pane and its sign-in dialog use. Only those parts are typed here
 * (no @types/office-js), so what the add-in depends on is visible in one file.
 *
 * Office.js comes from Microsoft's CDN (its terms ask add-ins not to serve a
 * copy) and only on the add-in's pages, whose content security policy names
 * that host (lib/security/headers.ts). It is added by our own code, which
 * `'strict-dynamic'` lets load it. Office.js removes `history.pushState` and
 * `replaceState` (old Office webviews lacked them), which Next's router
 * needs, so both are put back once it has loaded.
 */
import { OFFICE_SCRIPT } from "@/lib/security/headers";

export type AsyncResult<T> = { status: "succeeded" | "failed"; value: T; error?: { code: number; message: string } };

export type OfficeDialog = {
  addEventHandler(event: string, handler: (arg: { message?: string; error?: number; origin?: string }) => void): void;
  close(): void;
};

export type ExcelRange = {
  values: unknown[][];
  formulas: unknown[][];
  numberFormat: unknown[][];
  format: {
    font: { bold: boolean; color: string; size: number };
    fill: { color: string };
    horizontalAlignment: string;
    wrapText: boolean;
    columnWidth: number;
    rowHeight: number;
  };
};

export type ExcelWorksheet = {
  name: string;
  getRange(address: string): ExcelRange;
  freezePanes: { freezeAt(range: string): void; unfreeze(): void };
  notes?: { add(cellAddress: string, content: string): void };
  names?: { add(name: string, reference: ExcelRange): void };
  activate(): void;
};

export type ExcelContext = {
  workbook: {
    worksheets: { items: ExcelWorksheet[]; add(name: string): ExcelWorksheet; load(what: string): void };
    comments?: { add(cellAddress: string, content: string): void };
  };
  sync(): Promise<void>;
};

type OfficeGlobal = {
  onReady(): Promise<{ host: string | null; platform: string | null }>;
  context: {
    requirements: { isSetSupported(name: string, version?: string): boolean };
    ui: {
      displayDialogAsync(url: string, options: { height: number; width: number; promptBeforeOpen?: boolean },
        callback: (result: AsyncResult<OfficeDialog>) => void): void;
      messageParent(message: string, options?: { targetOrigin: string }): void;
    };
    diagnostics?: { platform?: string };
  };
  EventType: { DialogMessageReceived: string; DialogEventReceived: string };
};

type ExcelBatch<T> = (context: ExcelContext) => Promise<T>;
type ExcelGlobal = { run<T>(batch: ExcelBatch<T>): Promise<T> }; // text-ok: a type, not words

declare global {
  interface Window {
    Office?: OfficeGlobal;
    Excel?: ExcelGlobal;
  }
}

let loading: Promise<OfficeGlobal | null> | null = null;

/**
 * Office.js, ready; null when it can't load, or when the page is not inside Office (an ordinary browser tab)
 * and `insideOffice` is asked for. The dialog doesn't ask: Office names no host to a dialog's page.
 */
export function loadOffice({ insideOffice = true }: { insideOffice?: boolean } = {}): Promise<OfficeGlobal | null> {
  if (typeof window === "undefined") return Promise.resolve(null);
  if (loading) return loading;
  loading = new Promise((resolve) => {
    const { pushState, replaceState } = window.history;
    const script = document.createElement("script");
    script.src = OFFICE_SCRIPT;
    script.async = true;
    script.onload = async () => {
      window.history.pushState = window.history.pushState ?? pushState;
      window.history.replaceState = window.history.replaceState ?? replaceState;
      const office = window.Office;
      if (!office) return resolve(null);
      const info = await office.onReady().catch(() => null);
      resolve(info && (info.host || !insideOffice) ? office : null);
    };
    script.onerror = () => resolve(null);
    document.head.appendChild(script);
  });
  return loading;
}

/** Which Office this is: "PC", "Mac", "OfficeOnline" ... (Office.js's own names), or "". */
export function officePlatform(office: OfficeGlobal): string {
  return office.context.diagnostics?.platform ?? "";
}

export type DialogOutcome = { message: string } | { closed: true } | { failed: number };

/**
 * Opens Office's dialog on one of this site's pages and waits for its first
 * message, or for the person to close it. The dialog's page must be on the
 * pane's own domain (Office's rule).
 */
export function openDialog(office: OfficeGlobal, path: string): Promise<DialogOutcome> {
  return new Promise((resolve) => {
    const url = new URL(path, window.location.origin).toString();
    office.context.ui.displayDialogAsync(url, { height: 70, width: 30, promptBeforeOpen: false }, (result) => {
      if (result.status !== "succeeded") return resolve({ failed: result.error?.code ?? 0 });
      const dialog = result.value;
      dialog.addEventHandler(office.EventType.DialogMessageReceived, (arg) => {
        dialog.close();
        resolve({ message: arg.message ?? "" });
      });
      dialog.addEventHandler(office.EventType.DialogEventReceived, (arg) => {
        // 12006: the person closed the dialog
        resolve(arg.error === 12006 ? { closed: true } : { failed: arg.error ?? 0 });
      });
    });
  });
}

/** From inside the dialog: hand the pane a message. Only this site's own pages receive it. */
export function messageParent(office: OfficeGlobal, message: string): void {
  office.context.ui.messageParent(message, { targetOrigin: window.location.origin });
}
