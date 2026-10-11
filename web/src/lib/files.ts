import { api, authHeaders, type Schemas } from "@/lib/api/client";
import { newRequestId, REQUEST_ID_HEADER } from "@/lib/monitoring";

/**
 * Uploading a file (PLAN.md 6.1): reserve room, send the bytes, then wait for
 * the scan. The file can't be downloaded until the scan says it is clean.
 *
 * The bytes go to the API's own address, not through this site's /api proxy:
 * a proxy between the two may cap how large a request can be, and the API
 * streams what arrives to disk. The API accepts this site's origin (CORS) and
 * the content security policy allows that one address (lib/security/headers.ts).
 */
export type StoredFile = Schemas["FileOut"];
export type FileUsage = Schemas["FileUsage"];
export type FileList = Schemas["FileList"];
export type RefusalCode = Schemas["FileRefusalDetail"]["code"];

/** Why an upload stopped: one of the API's codes, or a step that failed here. */
export type UploadProblem = RefusalCode | "send_failed" | "scan_slow" | "failed";

export type UploadResult = { ok: true; file: StoredFile } | { ok: false; problem: UploadProblem };

/** Where the API itself answers (next.config.ts works it out per environment). */
const API_ORIGIN = process.env.NEXT_PUBLIC_FSE_API_ORIGIN ?? "";

const POLL_MS = 1000;
/** A scan takes a second or two; behind a running simulation, up to a couple of minutes. */
const SCAN_WAIT_MS = 180_000;
/** The server takes in two uploads at a time, and says so when asked for a third. */
const SEND_TRIES = 6;
const SEND_WAIT_MS = 3000;
const SENT_AGAIN: string[] = ["busy", "storage_unavailable"];

/** The code inside an API refusal, when it carries one. */
export function refusalCode(error: unknown): RefusalCode | null {
  const detail = (error as { detail?: unknown } | undefined)?.detail;
  const code = (detail as { code?: unknown } | undefined)?.code;
  return typeof code === "string" ? (code as RefusalCode) : null;
}

const EXTENSION: Record<StoredFile["kind"], string> = { pdf: ".pdf", xlsx: ".xlsx", docx: ".docx", xls: ".xls", doc: ".doc" };

/** What the file picker offers: the extensions the server accepts. */
export function acceptedExtensions(kinds: StoredFile["kind"][]): string {
  return kinds.map((k) => EXTENSION[k]).join(",");
}

/** Checked here first so an oversized file is refused before anything is sent; the API checks again. */
export function refusedLocally(file: File, usage: FileUsage, kinds: StoredFile["kind"][]): UploadProblem | null {
  const name = file.name.toLowerCase();
  if (!kinds.some((k) => name.endsWith(EXTENSION[k]))) return "bad_type";
  if (file.size === 0) return "empty";
  if (file.size > usage.max_file_bytes) return "too_large";
  return null;
}

const wait = (ms: number) => new Promise((done) => setTimeout(done, ms));

type Sent = { file?: StoredFile; problem?: UploadProblem };

async function send(id: string, file: File): Promise<Sent> {
  let response: Response;
  try {
    response = await fetch(`${API_ORIGIN}/api/files/${id}/content`, {
      method: "PUT",
      headers: { ...(await authHeaders()), "Content-Type": "application/octet-stream", [REQUEST_ID_HEADER]: newRequestId() },
      body: file,
    });
  } catch {
    return { problem: "send_failed" };
  }
  const answer: unknown = await response.json().catch(() => null);
  if (response.ok && answer) return { file: answer as StoredFile };
  return { problem: refusalCode(answer) ?? "send_failed" };
}

export async function uploadFile(file: File, onStage: (stage: "sending" | "checking") => void): Promise<UploadResult> {
  const begun = await api.POST("/api/files", { body: { name: file.name, size: file.size } });
  if (!begun.data) return { ok: false, problem: refusalCode(begun.error) ?? "failed" };
  const path = { params: { path: { file_id: begun.data.id } } };

  onStage("sending");
  let sent = await send(begun.data.id, file);
  for (let attempt = 1; !sent.file && attempt < SEND_TRIES && SENT_AGAIN.includes(sent.problem ?? ""); attempt++) {
    await wait(SEND_WAIT_MS);
    sent = await send(begun.data.id, file);
  }
  if (!sent.file) {
    await api.DELETE("/api/files/{file_id}", path); // give the reserved room back; the notice says why it stopped
    return { ok: false, problem: sent.problem ?? "send_failed" };
  }

  onStage("checking");
  let current = sent.file;
  const deadline = Date.now() + SCAN_WAIT_MS;
  while (current.status === "scanning" && Date.now() < deadline) {
    await wait(POLL_MS);
    const { data } = await api.GET("/api/files/{file_id}", path);
    if (data) current = data;
  }
  if (current.status === "scanning") return { ok: false, problem: "scan_slow" };
  return { ok: true, file: current };
}

/** Megabytes, for showing a size: one decimal under 10 MB, none above. */
export function megabytes(bytes: number): { value: number; digits: number } {
  const value = bytes / (1024 * 1024);
  return { value, digits: value > 0 && value < 10 ? 1 : 0 };
}
