"use client";

import { useCallback, useEffect, useState } from "react";

import { useProfile } from "@/components/auth/ProfileProvider";
import { Notice, PrimaryButton, SecondaryButton, Switch } from "@/components/ui/Screen";
import { Kpi, Tile, Tiles } from "@/components/ui/Tile";
import { api, type Schemas } from "@/lib/api/client";
import { fmtMultiple } from "@/lib/format";

import { apiMessage, useDeal } from "../DealProvider";
import { DealScreen, RailGroup } from "../DealScreen";

type Summary = Schemas["DealSummary"];
type Version = Schemas["VersionSummary"];

const KIND_LABEL: Record<Version["kind"], string> = {
  created: "Created",
  saved: "Saved",
  auto: "Autosave",
  restored: "Restore",
};

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
  const [error, setError] = useState<string>();
  const [bump, setBump] = useState(0);
  const refresh = useCallback(() => setBump((n) => n + 1), []);

  return (
    <DealScreen rail={<ThisDeal onError={setError} onChange={refresh} />}>
      {error && (
        <Notice tone="loss" title="Not done" role="alert" actions={<SecondaryButton onClick={() => setError(undefined)}>Dismiss</SecondaryButton>}>
          {error}
        </Notice>
      )}
      <Tiles>
        <Headline />
        <DealList bump={bump} onError={setError} onChange={refresh} />
        <History bump={bump} onError={setError} onChange={refresh} />
      </Tiles>
    </DealScreen>
  );
}

function Headline() {
  const { current, run, saveState } = useDeal();
  const r = run.result?.returns;
  return (
    <>
      <Kpi title="IRR" value={r?.irr == null ? "n/a" : `${(r.irr * 100).toFixed(1)}%`} lead />
      <Kpi title="MOIC" value={fmtMultiple(r?.moic)} />
      <Kpi
        title="Saved"
        value={saveState === "saved" ? "Yes" : saveState === "saving" ? "Saving" : saveState === "error" ? "Failed" : "No"}
        tone={saveState === "error" ? "loss" : saveState === "unsaved" ? "attention" : undefined}
        sub={current ? `Version ${current.latestVersion}` : "Save to keep it"}
      />
      <Tile span={6} title="How saving works">
        <p className="type-body text-[10.5px]">
          A saved deal keeps its inputs and the Settings in effect, so it opens with the same numbers on any device.
          Edits save by themselves; keep a version to mark a point you can restore. Settings follow your account.
        </p>
      </Tile>
    </>
  );
}

/** The rail: name and save the deal on screen, keep a version, or start a new one. */
function ThisDeal({ onError, onChange }: { onError: (e?: string) => void; onChange: () => void }) {
  const { current, saveAs, newDeal, patchCurrent, flush } = useDeal();
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
      if (!data) return apiMessage(error, "The deal couldn't be renamed.");
      patchCurrent({ name: data.name });
      setName(data.name);
      return undefined;
    });

  const keepVersion = () =>
    act(async () => {
      if (!current) return undefined;
      if (!(await flush())) return "The latest edits couldn't be saved, so no version was kept.";
      const body = label.trim() ? { label: label.trim() } : {};
      const { data, error } = await api.POST("/api/deals/{deal_id}/versions", { params: { path: { deal_id: current.id } }, body });
      if (!data) return apiMessage(error, "The version couldn't be kept.");
      patchCurrent({ latestVersion: Math.max(current.latestVersion, data.number) });
      setLabel("");
      return undefined;
    });

  return (
    <>
      <RailGroup title={current ? "This deal" : "Save this deal"}>
        <label className="grid gap-1.5 py-1">
          <span className="type-input-label">Deal name</span>
          <input aria-label="Deal name" value={name} maxLength={120} onChange={(e) => setName(e.target.value)} className={TEXT_INPUT} />
        </label>
        <div className="flex gap-2 pt-1.5">
          {current ? (
            <SecondaryButton onClick={rename} disabled={busy || !name.trim() || name.trim() === current.name}>
              Rename
            </SecondaryButton>
          ) : (
            <PrimaryButton onClick={save} disabled={busy || !name.trim()}>
              Save deal
            </PrimaryButton>
          )}
        </div>
      </RailGroup>

      {current && (
        <RailGroup title="Versions">
          <p className="type-body py-1 text-[10px]">
            Edits save by themselves. Keep a version to mark a point you can come back to.
          </p>
          <label className="grid gap-1.5 py-1">
            <span className="type-input-label">Version label</span>
            <input aria-label="Version label" value={label} maxLength={120} placeholder="Optional" onChange={(e) => setLabel(e.target.value)} className={TEXT_INPUT} />
          </label>
          <div className="flex gap-2 pt-1.5">
            <PrimaryButton onClick={keepVersion} disabled={busy}>
              Keep version
            </PrimaryButton>
          </div>
        </RailGroup>
      )}

      <RailGroup title="Start again">
        <p className="type-body py-1 text-[10px]">
          {current ? "Leaves this deal as saved and starts an unsaved deal from the default inputs." : "Clears the inputs back to the defaults."}
        </p>
        <div className="flex gap-2 pt-1.5">
          <SecondaryButton
            onClick={() => {
              newDeal();
              onChange();
            }}
            disabled={busy}
          >
            New deal
          </SecondaryButton>
        </div>
      </RailGroup>
    </>
  );
}

function DealList({ bump, onError, onChange }: { bump: number; onError: (e?: string) => void; onChange: () => void }) {
  const { current, openDeal, newDeal, patchCurrent } = useDeal();
  const when = useWhen();
  const [showArchived, setShowArchived] = useState(false);
  const [deals, setDeals] = useState<Summary[] | null>(null);
  const [confirming, setConfirming] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    api.GET("/api/deals", { params: { query: { archived: showArchived } } }).then(({ data, error }) => {
      if (cancelled) return;
      if (data) setDeals(data.deals);
      else onError(apiMessage(error, "Your deals couldn't be loaded."));
    });
    return () => {
      cancelled = true;
    };
    // The open deal's name, archive flag and last edit also change the list
  }, [bump, showArchived, current?.id, current?.name, current?.archived, current?.latestVersion, onError]);

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
      return data ? undefined : apiMessage(error, "The deal couldn't be duplicated.");
    });

  const archive = (d: Summary, archived: boolean) =>
    run(async () => {
      const { data, error } = await api.PATCH("/api/deals/{deal_id}", { params: { path: { deal_id: d.id } }, body: { archived } });
      if (!data) return apiMessage(error, "The deal couldn't be archived.");
      if (current?.id === d.id) patchCurrent({ archived: data.archived });
      return undefined;
    });

  const remove = (d: Summary) =>
    run(async () => {
      const { response, error } = await api.DELETE("/api/deals/{deal_id}", { params: { path: { deal_id: d.id } } });
      if (!response.ok) return apiMessage(error, "The deal couldn't be deleted.");
      if (current?.id === d.id) newDeal();
      return undefined;
    });

  return (
    <Tile span={12} title="Saved deals" aside={<Switch checked={showArchived} onChange={setShowArchived} label="Show archived" />}>
      {deals === null ? (
        <p className="type-body" role="note">Loading your deals</p>
      ) : deals.length === 0 ? (
        <p className="type-body" role="note">
          {showArchived ? "No saved deals yet." : "No saved deals yet. Name the deal on screen and save it; edits then save by themselves."}
        </p>
      ) : (
        <table className="w-full border-collapse font-mono text-[11px]" aria-label="Saved deals">
          <thead>
            <tr className="text-left text-muted">
              <th scope="col" className="border-b border-grid px-2 py-1 font-normal">Name</th>
              <th scope="col" className="border-b border-grid px-2 py-1 font-normal">Last edited</th>
              <th scope="col" className="border-b border-grid px-2 py-1 text-right font-normal">Versions</th>
              <th scope="col" className="border-b border-grid px-2 py-1 text-right font-normal">
                <span className="sr-only">Actions</span>
              </th>
            </tr>
          </thead>
          <tbody>
            {deals.map((d) => {
              const isOpen = current?.id === d.id;
              return (
                <tr key={d.id} data-deal={d.name} aria-current={isOpen ? "true" : undefined} className={isOpen ? "bg-raised text-bright" : "text-ink"}>
                  <th scope="row" className="border-b border-grid px-2 py-1.5 text-left font-normal">
                    {d.name}
                    {isOpen && <span className="chip ml-2 text-accent">open</span>}
                    {d.archived && <span className="chip ml-2 text-attention">archived</span>}
                  </th>
                  <td className="border-b border-grid px-2 py-1.5 text-muted">{when(d.updated_at)}</td>
                  <td className="border-b border-grid px-2 py-1.5 text-right">{d.latest_version}</td>
                  <td className="border-b border-grid px-2 py-1">
                    <span className="flex justify-end gap-1.5">
                      {confirming === d.id ? (
                        <>
                          <span className="type-body self-center text-loss">Delete for good?</span>
                          <SecondaryButton onClick={() => void remove(d)}>Confirm delete</SecondaryButton>
                          <SecondaryButton onClick={() => setConfirming(null)}>Cancel</SecondaryButton>
                        </>
                      ) : (
                        <>
                          {!isOpen && <SecondaryButton onClick={() => void open(d.id)}>Open</SecondaryButton>}
                          <SecondaryButton onClick={() => void duplicate(d)}>Duplicate</SecondaryButton>
                          <SecondaryButton onClick={() => void archive(d, !d.archived)}>{d.archived ? "Unarchive" : "Archive"}</SecondaryButton>
                          <SecondaryButton onClick={() => setConfirming(d.id)}>Delete</SecondaryButton>
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
  const when = useWhen();
  const [versions, setVersions] = useState<Version[] | null>(null);
  const currentId = current?.id;

  useEffect(() => {
    if (!currentId) return;
    let cancelled = false;
    api.GET("/api/deals/{deal_id}/versions", { params: { path: { deal_id: currentId } } }).then(({ data, error }) => {
      if (cancelled) return;
      if (data) setVersions(data.versions);
      else onError(apiMessage(error, "The history couldn't be loaded."));
    });
    return () => {
      cancelled = true;
    };
  }, [currentId, current?.latestVersion, bump, onError]);

  if (!current) {
    return (
      <Tile span={12} title="History">
        <p className="type-body" role="note">Save the deal to start its history.</p>
      </Tile>
    );
  }

  const restore = async (number: number) => {
    // Unsaved edits go to the server first, so the restore keeps them as a version
    if (!(await flush())) {
      onError("The latest edits couldn't be saved, so nothing was restored.");
      return;
    }
    const { data, error } = await api.POST("/api/deals/{deal_id}/versions/{number}/restore", {
      params: { path: { deal_id: current.id, number } },
    });
    if (!data) {
      onError(apiMessage(error, "That version couldn't be restored."));
      return;
    }
    adopt(data);
    onError(undefined);
    onChange();
  };

  return (
    <Tile span={12} title={`History · ${current.name}`}>
      {versions === null ? (
        <p className="type-body" role="note">Loading the history</p>
      ) : (
        <table className="w-full border-collapse font-mono text-[11px]" aria-label="Version history">
          <thead>
            <tr className="text-left text-muted">
              <th scope="col" className="border-b border-grid px-2 py-1 text-right font-normal">Version</th>
              <th scope="col" className="border-b border-grid px-2 py-1 font-normal">Kind</th>
              <th scope="col" className="border-b border-grid px-2 py-1 font-normal">Label</th>
              <th scope="col" className="border-b border-grid px-2 py-1 font-normal">Saved</th>
              <th scope="col" className="border-b border-grid px-2 py-1 font-normal">
                <span className="sr-only">Actions</span>
              </th>
            </tr>
          </thead>
          <tbody>
            {versions.map((v) => (
              <tr key={v.number} data-version={v.number} className="text-ink">
                <th scope="row" className="border-b border-grid px-2 py-1.5 text-right font-normal">{v.number}</th>
                <td className="border-b border-grid px-2 py-1.5 text-muted">{KIND_LABEL[v.kind]}</td>
                <td className="border-b border-grid px-2 py-1.5">{v.label ?? ""}</td>
                <td className="border-b border-grid px-2 py-1.5 text-muted">{when(v.created_at)}</td>
                <td className="border-b border-grid px-2 py-1 text-right">
                  <SecondaryButton onClick={() => void restore(v.number)}>Restore</SecondaryButton>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </Tile>
  );
}
