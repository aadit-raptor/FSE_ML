"use client";

import { useTranslations } from "next-intl";
import { useCallback, useEffect, useState } from "react";

import { useProfile } from "@/components/auth/ProfileProvider";
import { Notice, PrimaryButton, SecondaryButton, Switch } from "@/components/ui/Screen";
import { Kpi, Tile, Tiles } from "@/components/ui/Tile";
import { api, type Schemas } from "@/lib/api/client";
import { fmtMultiple, fmtRate } from "@/lib/format";

import { apiMessage, useDeal } from "../DealProvider";
import { DealScreen, RailGroup, useDealLabel } from "../DealScreen";

type Summary = Schemas["DealSummary"];
type Version = Schemas["VersionSummary"];
type AuditEntry = Schemas["AuditEntry"];

/** i18n-keys: deal.kind* */
const KIND_KEY: Record<Version["kind"], string> = {
  created: "kindCreated",
  saved: "kindSaved",
  auto: "kindAuto",
  restored: "kindRestored",
};

// `dir="auto"` so a name typed in Arabic or Hebrew reads the way it was typed,
// whichever way the interface runs (PLAN.md 2.3b)
const TEXT_INPUT = "w-full border border-line bg-field px-2 py-1.5 font-mono text-[11px] text-ink outline-none focus:border-accent";

/** Times are stored in UTC and shown in the account's time zone and format. */
function useWhen() {
  const { profile } = useProfile();
  return useCallback(
    (iso: string) => {
      try {
        return new Intl.DateTimeFormat(profile?.locale, {
          dateStyle: "medium",
          timeStyle: "short",
          timeZone: profile?.time_zone,
        }).format(new Date(iso));
      } catch {
        return new Date(iso).toISOString().slice(0, 16).replace("T", " ") + " UTC";
      }
    },
    [profile],
  );
}

/** Save, open, rename, duplicate, archive and delete deals; keep and restore versions (PLAN.md 1.5). */
export function SavedStep() {
  const t = useTranslations("deal");
  const [error, setError] = useState<string>();
  const [bump, setBump] = useState(0);
  const refresh = useCallback(() => setBump((n) => n + 1), []);

  return (
    <DealScreen rail={<ThisDeal onError={setError} onChange={refresh} />}>
      {error && (
        <Notice tone="loss" title={t("savedNotDone")} role="alert" actions={<SecondaryButton onClick={() => setError(undefined)}>{t("dismiss")}</SecondaryButton>}>
          {error}
        </Notice>
      )}
      <Tiles>
        <Headline />
        <DealList bump={bump} onError={setError} onChange={refresh} />
        <History bump={bump} onError={setError} onChange={refresh} />
        <Activity bump={bump} onError={setError} />
      </Tiles>
    </DealScreen>
  );
}

function Headline() {
  const { current, run, saveState } = useDeal();
  const t = useTranslations("deal");
  const r = run.result?.returns;
  return (
    <>
      <Kpi title={t("kpiIrr")} value={fmtRate(r?.irr)} lead />
      <Kpi title={t("kpiMoic")} value={fmtMultiple(r?.moic)} />
      <Kpi
        title={t("kpiSaved")}
        value={saveState === "saved" ? t("savedYes") : saveState === "saving" ? t("savedSaving") : saveState === "error" ? t("savedFailedShort") : t("savedNo")}
        tone={saveState === "error" ? "loss" : saveState === "unsaved" ? "attention" : undefined}
        sub={current ? t("versionNumber", { number: String(current.latestVersion) }) : t("saveToKeep")}
      />
      <Tile span={6} title={t("howSavingWorks")}>
        <p className="type-body text-[10.5px]">{t("howSavingWorksBody")}</p>
      </Tile>
    </>
  );
}

/** The rail: name and save the deal on screen, keep a version, or start a new one. */
function ThisDeal({ onError, onChange }: { onError: (e?: string) => void; onChange: () => void }) {
  const { current, saveAs, newDeal, patchCurrent, flush } = useDeal();
  const t = useTranslations("deal");
  const e = useTranslations("errors");
  const [name, setName] = useState("");
  const [label, setLabel] = useState("");
  const [busy, setBusy] = useState(false);

  // The name box follows the open deal
  const [shownFor, setShownFor] = useState<string | null>(null);
  if ((current?.id ?? null) !== shownFor) {
    setShownFor(current?.id ?? null);
    setName(current?.name ?? "");
  }

  const act = async (work: () => Promise<string | undefined>) => {
    setBusy(true);
    const problem = await work();
    setBusy(false);
    onError(problem);
    if (!problem) onChange();
  };

  const save = () =>
    act(async () => {
      const r = await saveAs(name.trim());
      return r.ok ? undefined : r.error;
    });

  const rename = () =>
    act(async () => {
      if (!current) return undefined;
      const { data, error } = await api.PATCH("/api/deals/{deal_id}", { params: { path: { deal_id: current.id } }, body: { name: name.trim() } });
      if (!data) return apiMessage(error, e("dealRenameFailed"));
      patchCurrent({ name: data.name });
      setName(data.name);
      return undefined;
    });

  const keepVersion = () =>
    act(async () => {
      if (!current) return undefined;
      if (!(await flush())) return e("versionFlushFailed");
      const body = label.trim() ? { label: label.trim() } : {};
      const { data, error } = await api.POST("/api/deals/{deal_id}/versions", { params: { path: { deal_id: current.id } }, body });
      if (!data) return apiMessage(error, e("versionFailed"));
      patchCurrent({ latestVersion: Math.max(current.latestVersion, data.number) });
      setLabel("");
      return undefined;
    });

  return (
    <>
      <RailGroup title={current ? t("thisDeal") : t("saveThisDeal")}>
        <label className="grid gap-1.5 py-1">
          <span className="type-input-label">{t("dealName")}</span>
          <input aria-label={t("dealName")} dir="auto" value={name} maxLength={120} onChange={(ev) => setName(ev.target.value)} className={TEXT_INPUT} />
        </label>
        <div className="flex gap-2 pt-1.5">
          {current ? (
            <SecondaryButton onClick={rename} disabled={busy || !name.trim() || name.trim() === current.name}>
              {t("rename")}
            </SecondaryButton>
          ) : (
            <PrimaryButton onClick={save} disabled={busy || !name.trim()}>
              {t("saveDeal")}
            </PrimaryButton>
          )}
        </div>
      </RailGroup>

      {current && (
        <RailGroup title={t("versions")}>
          <p className="type-body py-1 text-[10px]">{t("versionsNote")}</p>
          <label className="grid gap-1.5 py-1">
            <span className="type-input-label">{t("versionLabel")}</span>
            <input
              aria-label={t("versionLabel")}
              dir="auto"
              value={label}
              maxLength={120}
              placeholder={t("optional")}
              onChange={(ev) => setLabel(ev.target.value)}
              className={TEXT_INPUT}
            />
          </label>
          <div className="flex gap-2 pt-1.5">
            <PrimaryButton onClick={keepVersion} disabled={busy}>
              {t("keepVersion")}
            </PrimaryButton>
          </div>
        </RailGroup>
      )}

      <RailGroup title={t("startAgain")}>
        <p className="type-body py-1 text-[10px]">{current ? t("startAgainOpen") : t("startAgainNew")}</p>
        <div className="flex gap-2 pt-1.5">
          <SecondaryButton
            onClick={() => {
              newDeal();
              onChange();
            }}
            disabled={busy}
          >
            {t("newDeal")}
          </SecondaryButton>
        </div>
      </RailGroup>
    </>
  );
}

function DealList({ bump, onError, onChange }: { bump: number; onError: (e?: string) => void; onChange: () => void }) {
  const { current, openDeal, newDeal, patchCurrent } = useDeal();
  const t = useTranslations("deal");
  const e = useTranslations("errors");
  const when = useWhen();
  const [showArchived, setShowArchived] = useState(false);
  const [deals, setDeals] = useState<Summary[] | null>(null);
  const [confirming, setConfirming] = useState<string | null>(null);
  const dealsLoadFailed = e("dealsLoadFailed");

  useEffect(() => {
    let cancelled = false;
    api.GET("/api/deals", { params: { query: { archived: showArchived } } }).then(({ data, error }) => {
      if (cancelled) return;
      if (data) setDeals(data.deals);
      else onError(apiMessage(error, dealsLoadFailed));
    });
    return () => {
      cancelled = true;
    };
    // The open deal's name, archive flag and last edit also change the list
  }, [bump, showArchived, current?.id, current?.name, current?.archived, current?.latestVersion, onError, dealsLoadFailed]);

  const run = async (work: () => Promise<string | undefined>) => {
    const problem = await work();
    onError(problem);
    setConfirming(null);
    onChange();
  };

  const open = (id: string) =>
    run(async () => {
      const r = await openDeal(id);
      return r.ok ? undefined : r.error;
    });

  const duplicate = (d: Summary) =>
    run(async () => {
      const { data, error } = await api.POST("/api/deals/{deal_id}/duplicate", { params: { path: { deal_id: d.id } }, body: {} });
      return data ? undefined : apiMessage(error, e("dealDuplicateFailed"));
    });

  const archive = (d: Summary, archived: boolean) =>
    run(async () => {
      const { data, error } = await api.PATCH("/api/deals/{deal_id}", { params: { path: { deal_id: d.id } }, body: { archived } });
      if (!data) return apiMessage(error, e("dealArchiveFailed"));
      if (current?.id === d.id) patchCurrent({ archived: data.archived });
      return undefined;
    });

  const remove = (d: Summary) =>
    run(async () => {
      const { response, error } = await api.DELETE("/api/deals/{deal_id}", { params: { path: { deal_id: d.id } } });
      if (!response.ok) return apiMessage(error, e("dealDeleteFailed"));
      if (current?.id === d.id) newDeal();
      return undefined;
    });

  return (
    <Tile span={12} title={t("savedDeals")} aside={<Switch checked={showArchived} onChange={setShowArchived} label={t("showArchived")} />}>
      {deals === null ? (
        <p className="type-body" role="note">
          {t("loadingDeals")}
        </p>
      ) : deals.length === 0 ? (
        <p className="type-body" role="note">
          {showArchived ? t("noDealsArchived") : t("noDeals")}
        </p>
      ) : (
        <table className="w-full border-collapse font-mono text-[11px]" aria-label={t("savedDeals")}>
          <thead>
            <tr className="text-start text-muted">
              <th scope="col" className="border-b border-grid px-2 py-1 font-normal">
                {t("colName")}
              </th>
              <th scope="col" className="border-b border-grid px-2 py-1 font-normal">
                {t("colLastEdited")}
              </th>
              <th scope="col" className="border-b border-grid px-2 py-1 text-end font-normal">
                {t("colVersions")}
              </th>
              <th scope="col" className="border-b border-grid px-2 py-1 text-end font-normal">
                <span className="sr-only">{t("colActions")}</span>
              </th>
            </tr>
          </thead>
          <tbody>
            {deals.map((d) => {
              const isOpen = current?.id === d.id;
              return (
                <tr key={d.id} data-deal={d.name} aria-current={isOpen ? "true" : undefined} className={isOpen ? "bg-raised text-bright" : "text-ink"}>
                  <th scope="row" className="border-b border-grid px-2 py-1.5 text-start font-normal">
                    {d.name}
                    {isOpen && <span className="chip ms-2 text-accent">{t("chipOpen")}</span>}
                    {d.archived && <span className="chip ms-2 text-attention">{t("chipArchived")}</span>}
                  </th>
                  <td className="border-b border-grid px-2 py-1.5 text-muted">{when(d.updated_at)}</td>
                  <td className="border-b border-grid px-2 py-1.5 text-end">{d.latest_version}</td>
                  <td className="border-b border-grid px-2 py-1">
                    <span className="flex justify-end gap-1.5">
                      {confirming === d.id ? (
                        <>
                          <span className="type-body self-center text-loss">{t("deleteForGood")}</span>
                          <SecondaryButton onClick={() => void remove(d)}>{t("confirmDelete")}</SecondaryButton>
                          <SecondaryButton onClick={() => setConfirming(null)}>{t("cancel")}</SecondaryButton>
                        </>
                      ) : (
                        <>
                          {!isOpen && <SecondaryButton onClick={() => void open(d.id)}>{t("open")}</SecondaryButton>}
                          <SecondaryButton onClick={() => void duplicate(d)}>{t("duplicate")}</SecondaryButton>
                          <SecondaryButton onClick={() => void archive(d, !d.archived)}>{d.archived ? t("unarchive") : t("archive")}</SecondaryButton>
                          <SecondaryButton onClick={() => setConfirming(d.id)}>{t("delete")}</SecondaryButton>
                        </>
                      )}
                    </span>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      )}
    </Tile>
  );
}

function History({ bump, onError, onChange }: { bump: number; onError: (e?: string) => void; onChange: () => void }) {
  const { current, adopt, flush } = useDeal();
  const t = useTranslations("deal");
  const e = useTranslations("errors");
  const when = useWhen();
  const [versions, setVersions] = useState<Version[] | null>(null);
  const currentId = current?.id;
  const historyLoadFailed = e("historyLoadFailed");

  useEffect(() => {
    if (!currentId) return;
    let cancelled = false;
    api.GET("/api/deals/{deal_id}/versions", { params: { path: { deal_id: currentId } } }).then(({ data, error }) => {
      if (cancelled) return;
      if (data) setVersions(data.versions);
      else onError(apiMessage(error, historyLoadFailed));
    });
    return () => {
      cancelled = true;
    };
  }, [currentId, current?.latestVersion, bump, onError, historyLoadFailed]);

  if (!current) {
    return (
      <Tile span={12} title={t("history")}>
        <p className="type-body" role="note">
          {t("historyEmpty")}
        </p>
      </Tile>
    );
  }

  const restore = async (number: number) => {
    // Unsaved edits go to the server first, so the restore keeps them as a version
    if (!(await flush())) {
      onError(e("restoreFlushFailed"));
      return;
    }
    const { data, error } = await api.POST("/api/deals/{deal_id}/versions/{number}/restore", {
      params: { path: { deal_id: current.id, number } },
    });
    if (!data) {
      onError(apiMessage(error, e("restoreFailed")));
      return;
    }
    adopt(data);
    onError(undefined);
    onChange();
  };

  return (
    <Tile span={12} title={t("historyOf", { name: current.name })}>
      {versions === null ? (
        <p className="type-body" role="note">
          {t("loadingHistory")}
        </p>
      ) : (
        <table className="w-full border-collapse font-mono text-[11px]" aria-label={t("versionHistory")}>
          <thead>
            <tr className="text-start text-muted">
              <th scope="col" className="border-b border-grid px-2 py-1 text-end font-normal">
                {t("colVersion")}
              </th>
              <th scope="col" className="border-b border-grid px-2 py-1 font-normal">
                {t("colKind")}
              </th>
              <th scope="col" className="border-b border-grid px-2 py-1 font-normal">
                {t("colLabel")}
              </th>
              <th scope="col" className="border-b border-grid px-2 py-1 font-normal">
                {t("colSaved")}
              </th>
              <th scope="col" className="border-b border-grid px-2 py-1 font-normal">
                <span className="sr-only">{t("colActions")}</span>
              </th>
            </tr>
          </thead>
          <tbody>
            {versions.map((v) => (
              <tr key={v.number} data-version={v.number} className="text-ink">
                <th scope="row" className="border-b border-grid px-2 py-1.5 text-end font-normal">
                  {v.number}
                </th>
                <td className="border-b border-grid px-2 py-1.5 text-muted">{t(KIND_KEY[v.kind])}</td>
                <td className="border-b border-grid px-2 py-1.5">{v.label ?? ""}</td>
                <td className="border-b border-grid px-2 py-1.5 text-muted">{when(v.created_at)}</td>
                <td className="border-b border-grid px-2 py-1 text-end">
                  <SecondaryButton onClick={() => void restore(v.number)}>{t("restore")}</SecondaryButton>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </Tile>
  );
}

/**
 * The audit history (PLAN.md 3.3): what was done to the open deal, or with the switch to every deal and the
 * account's settings, newest first. Read only: the API offers no way to change an entry.
 * i18n-keys: deal.activity_*, deal.activityExport_*
 */
function Activity({ bump, onError }: { bump: number; onError: (e?: string) => void }) {
  const { current, saveState } = useDeal();
  const t = useTranslations("deal");
  const e = useTranslations("errors");
  const settingsText = useTranslations("settings");
  const dealLabel = useDealLabel();
  const when = useWhen();
  const [whole, setWhole] = useState(false);
  const [entries, setEntries] = useState<AuditEntry[] | null>(null);
  const account = whole || !current;
  const currentId = current?.id;
  const activityLoadFailed = e("activityLoadFailed");

  // Another deal, or the whole account: never show the last list under the new title
  const scope = account ? "account" : currentId;
  const [shownScope, setShownScope] = useState(scope);
  if (scope !== shownScope) {
    setShownScope(scope);
    setEntries(null);
  }

  useEffect(() => {
    // Autosave writes entries: read again once a save lands, not while one is pending
    if (currentId && saveState !== "saved") return;
    let cancelled = false;
    const request = account
      ? api.GET("/api/account/history")
      : api.GET("/api/deals/{deal_id}/history", { params: { path: { deal_id: currentId! } } });
    request.then(({ data, error }) => {
      if (cancelled) return;
      if (data) setEntries(data.entries);
      else onError(apiMessage(error, activityLoadFailed));
    });
    return () => {
      cancelled = true;
    };
  }, [account, currentId, current?.name, current?.archived, current?.latestVersion, saveState, bump, onError, activityLoadFailed]);

  // A deal's own fields by their screen names; its settings, and the account's, by Settings' names
  const fieldName = (name: string, setting: boolean) => {
    const key = setting ? name.replace(/^settings\./, "") : name;
    if (setting || name.startsWith("settings.")) return settingsText.has(key) ? settingsText(key) : key;
    return dealLabel(key as Parameters<typeof dealLabel>[0]);
  };
  const detail = (entry: AuditEntry) => {
    if (entry.fields?.length) return entry.fields.map((f) => fieldName(f, entry.action === "settings_changed")).join(", ");
    if (entry.export) return t(`activityExport_${entry.export}`);
    if (entry.source_deal) return t("activityCopy");
    return "";
  };
  const dealCell = (entry: AuditEntry) =>
    entry.deal_id === null || entry.deal_id === undefined ? t("activitySettingsRow") : (entry.deal_name ?? t("activityDeletedDeal"));

  return (
    <Tile
      span={12}
      title={account ? t("activityAccount") : t("activityOf", { name: current!.name })}
      aside={current ? <Switch checked={whole} onChange={setWhole} label={t("activityWholeAccount")} /> : undefined}
    >
      <p className="type-body pb-2 text-[10px]">{t("activityNote")}</p>
      {entries === null ? (
        <p className="type-body" role="note">
          {t("loadingActivity")}
        </p>
      ) : entries.length === 0 ? (
        <p className="type-body" role="note">
          {t("activityEmpty")}
        </p>
      ) : (
        <table className="w-full border-collapse font-mono text-[11px]" aria-label={t("activityTable")}>
          <thead>
            <tr className="text-start text-muted">
              <th scope="col" className="border-b border-grid px-2 py-1 font-normal">
                {t("colWhen")}
              </th>
              {account && (
                <th scope="col" className="border-b border-grid px-2 py-1 font-normal">
                  {t("colDeal")}
                </th>
              )}
              <th scope="col" className="border-b border-grid px-2 py-1 font-normal">
                {t("colAction")}
              </th>
              <th scope="col" className="border-b border-grid px-2 py-1 font-normal">
                {t("colDetail")}
              </th>
            </tr>
          </thead>
          <tbody>
            {entries.map((entry) => (
              <tr key={entry.id} data-action={entry.action} className="text-ink">
                <td className="border-b border-grid px-2 py-1.5 text-muted">
                  {entry.until ? t("activitySpan", { from: when(entry.at), until: when(entry.until) }) : when(entry.at)}
                </td>
                {account && (
                  <td dir="auto" className="border-b border-grid px-2 py-1.5">
                    {dealCell(entry)}
                  </td>
                )}
                <th scope="row" className="border-b border-grid px-2 py-1.5 text-start font-normal">
                  {t(`activity_${entry.action}`, { count: entry.count, version: String(entry.version ?? ""), enabled: String(entry.enabled ?? ""), verdict: String(entry.verdict ?? "") })}
                </th>
                <td className="border-b border-grid px-2 py-1.5 text-muted">{detail(entry)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </Tile>
  );
}
