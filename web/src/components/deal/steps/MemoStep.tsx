"use client";

import { useTranslations } from "next-intl";
import { useCallback, useEffect, useRef, useState } from "react";

import { useDeal } from "@/components/deal/DealProvider";
import { DraftBlock } from "@/components/ui/DraftBlock";
import { Notice, PrimaryButton, RailGroup, Screen, SecondaryButton } from "@/components/ui/Screen";
import { Tile, Tiles } from "@/components/ui/Tile";
import { api, type Schemas } from "@/lib/api/client";
import { formatDateTime } from "@/lib/locale";
import {
  deleteMemo, downloadMemo, draftMemo, getMemo, listMemos, refusedKey, writeSentences,
  type Memo, type MemoSummary,
} from "@/lib/memo";
import { plain } from "@/lib/report";

/**
 * Deal -> Memo (PLAN.md 6.3): an investment memo of a saved version of the open deal. Drafting runs the version
 * through the model as a background job (the deal model, the simulation with its drivers, the scenarios and, when
 * actual results are saved, plan against actual); the memo is then drawn from that run, every figure checked
 * against it. The linking sentences are the templates' until the AI is asked to write them: it is sent nothing of
 * the deal but which way its figures point, and the app puts every figure in afterwards.
 *
 * i18n-keys: memo.refused_*
 * i18n-keys: memo.source_*
 * i18n-keys: memo.section_*
 */
type Version = Schemas["VersionSummary"];
type Lists = { versions: Version[]; memos: MemoSummary[] };

async function fetchLists(dealId: string): Promise<Lists> {
  const [kept, drawn] = await Promise.all([
    api.GET("/api/deals/{deal_id}/versions", { params: { path: { deal_id: dealId } } }), listMemos(dealId)]);
  return { versions: kept.data?.versions ?? [], memos: drawn?.memos ?? [] };
}
const field = "border border-line bg-canvas px-2 py-1 text-[11px] text-ink";

export function MemoStep() {
  const t = useTranslations("memo");
  const { current } = useDeal();
  const dealId = current?.id ?? null;
  const [versions, setVersions] = useState<Version[] | null>(null);
  const [chosen, setChosen] = useState<number | null>(null);
  const [memos, setMemos] = useState<MemoSummary[]>([]);
  const [memo, setMemo] = useState<Memo | null>(null);
  const [refused, setRefused] = useState<string | null>(null);
  const [stage, setStage] = useState<string | null>(null);
  const inflight = useRef<AbortController | null>(null);

  const show = useCallback((found: Lists) => {
    setVersions(found.versions);
    setMemos(found.memos);
    setChosen((now) => (now !== null && found.versions.some((v) => v.number === now) ? now : found.versions[0]?.number ?? null));
  }, []);
  const reload = useCallback(async (id: string) => show(await fetchLists(id)), [show]);

  useEffect(() => {
    if (!dealId) return;
    let cancelled = false;
    void fetchLists(dealId).then((found) => {
      if (!cancelled) show(found);
    });
    return () => {
      cancelled = true;
      inflight.current?.abort();
    };
  }, [dealId, show]);

  // the chosen version's memo, when it has one
  const hasMemo = chosen !== null && memos.some((m) => m.version === chosen);
  useEffect(() => {
    let cancelled = false;
    if (!dealId || chosen === null || !hasMemo) return;
    void getMemo(dealId, chosen).then((answer) => {
      if (cancelled) return;
      setMemo(answer.memo ?? null);
      setRefused(answer.refused ?? null);
    });
    return () => {
      cancelled = true;
    };
  }, [dealId, chosen, hasMemo]);
  const shown = memo && chosen === memo.version && hasMemo ? memo : null;

  const take = (answer: { memo?: Memo; refused?: string }) => {
    if (answer.memo) setMemo(answer.memo);
    setRefused(answer.refused ?? null);
  };

  const draft = async () => {
    if (!dealId || chosen === null) return;
    inflight.current?.abort();
    const ctrl = new AbortController();
    inflight.current = ctrl;
    setStage(t("drafting"));
    setRefused(null);
    try {
      const answer = await draftMemo(dealId, chosen, ctrl.signal, (job) => setStage(job.stage ?? t("drafting")));
      take(answer);
      await reload(dealId);
    } catch {
      if (!ctrl.signal.aborted) setRefused("draft_failed");
    } finally {
      if (inflight.current === ctrl) setStage(null);
    }
  };

  if (!current) {
    return (
      <Screen rail={null}>
        <Notice tone="attention" title={t("emptyTitle")} role="note">{t("noDeal")}</Notice>
      </Screen>
    );
  }

  return (
    <Screen
      rail={
        <>
          <RailGroup title={t("versionGroup")}>
            {versions === null ? (
              <p className="type-step">{t("loading")}</p>
            ) : versions.length === 0 ? (
              <p className="type-body text-[10.5px]">{t("noVersions")}</p>
            ) : (
              <>
                <label className="grid gap-1">
                  <span className="type-input-label">{t("versionLabel")}</span>
                  <select className={field} value={chosen ?? ""} onChange={(e) => setChosen(Number(e.target.value))}>
                    {versions.map((v) => (
                      <option key={v.number} value={String(v.number)}>
                        {v.label ? t("versionNamed", { number: String(v.number), label: v.label })
                          : t("version", { number: String(v.number) })}
                      </option>
                    ))}
                  </select>
                </label>
                <div className="mt-1.5">
                  <PrimaryButton onClick={() => void draft()} disabled={stage !== null || chosen === null}>
                    {hasMemo ? t("redraft") : t("draft")}
                  </PrimaryButton>
                </div>
                {stage && <p className="type-body mt-1 text-[10px]" role="status">{stage}</p>}
              </>
            )}
          </RailGroup>
          <MemosGroup memos={memos} chosen={chosen} onOpen={setChosen} onDelete={async (version) => {
            const ok = await deleteMemo(current.id, version);
            if (!ok) setRefused("delete_failed");
            await reload(current.id);
          }} />
          {shown && (
            <SentencesGroup memo={shown} onWrite={async () => take(await writeSentences(current.id, shown.version))} />
          )}
          {shown && (
            <RailGroup title={t("downloadGroup")}>
              <div className="flex flex-wrap gap-1.5">
                <PrimaryButton onClick={() => void downloadMemo(current.id, shown.deal_name, shown.version, "word")
                  .catch(() => setRefused("download_failed"))}>
                  {t("word")}
                </PrimaryButton>
                <SecondaryButton onClick={() => void downloadMemo(current.id, shown.deal_name, shown.version, "pdf")
                  .catch(() => setRefused("download_failed"))}>
                  {t("pdf")}
                </SecondaryButton>
              </div>
            </RailGroup>
          )}
        </>
      }
    >
      <Notice tone="info" title={t("toolTitle")} role="note">{t("tool")}</Notice>
      {refused && (
        <Notice tone="loss" title={t("emptyTitle")} role="alert">
          {refused === "limit" ? t("limit") : refused === "full" ? t("full") : refused === "draft_failed"
            ? t("draftFailed") : refused === "delete_failed" ? t("deleteFailed") : refused === "download_failed"
              ? t("downloadFailed") : t(refusedKey(refused))}
        </Notice>
      )}
      {shown && !shown.current && <Notice tone="attention" title={t("emptyTitle")} role="note">{t("changed")}</Notice>}
      <Tiles>
        {!shown && (
          <Tile span={12} title={t("emptyTitle")}>
            <p className="type-body text-[10.5px]">{t("emptyBody")}</p>
          </Tile>
        )}
        {shown && (
          <Tile span={12} title={t("previewTitle", { name: shown.deal_name })}>
            <article className="grid max-w-[760px] gap-2" aria-label={t("previewLabel")}>
              {shown.blocks.map((block, i) => (
                <DraftBlock key={i} block={block} chartLabel={t("chartLabel", { title: plain(block.title) })} />
              ))}
            </article>
          </Tile>
        )}
      </Tiles>
    </Screen>
  );
}

function MemosGroup({ memos, chosen, onOpen, onDelete }: {
  memos: MemoSummary[]; chosen: number | null; onOpen: (version: number) => void;
  onDelete: (version: number) => Promise<void>;
}) {
  const t = useTranslations("memo");
  return (
    <RailGroup title={t("memosGroup")}>
      {memos.length === 0 ? (
        <p className="type-body text-[10.5px]">{t("noMemos")}</p>
      ) : (
        <ul className="grid gap-1">
          {memos.map((m) => (
            <li key={m.version} className="grid gap-0.5 border-b border-line/60 pb-1" data-memo={m.version}>
              <span className={`type-body text-[10.5px] ${m.version === chosen ? "text-ink" : "text-soft"}`}>
                {t("version", { number: String(m.version) })}
              </span>
              <span className="type-body text-[10px] text-muted">{t("drafted", { date: formatDateTime(new Date(m.drafted_at)) })}</span>
              <span className="flex gap-1.5">
                <SecondaryButton onClick={() => onOpen(m.version)}>{t("open")}</SecondaryButton>
                <SecondaryButton onClick={() => void onDelete(m.version)}>{t("delete")}</SecondaryButton>
              </span>
            </li>
          ))}
        </ul>
      )}
    </RailGroup>
  );
}

function SentencesGroup({ memo, onWrite }: { memo: Memo; onWrite: () => Promise<void> }) {
  const t = useTranslations("memo");
  const [busy, setBusy] = useState(false);
  return (
    <RailGroup title={t("sentencesGroup")}>
      <p className="type-body mb-1.5 text-[10.5px]">{t("sentencesHint")}</p>
      <ul className="mb-1.5 grid gap-0.5">
        {memo.sentences.map((s) => (
          <li key={s.section} className="type-body flex justify-between gap-2 text-[10px]" data-section={s.section}>
            <span>{t(`section_${s.section}`)}</span>
            <span className={s.source === "ai" ? "text-accent" : "text-muted"}>{t(`source_${s.source}`)}</span>
          </li>
        ))}
      </ul>
      {memo.written_by && <p className="type-body mb-1 text-[10px] text-muted">{t("writtenBy", { model: memo.written_by })}</p>}
      <SecondaryButton disabled={busy} onClick={() => {
        setBusy(true);
        void onWrite().finally(() => setBusy(false));
      }}>
        {busy ? t("writing") : t("write")}
      </SecondaryButton>
    </RailGroup>
  );
}
