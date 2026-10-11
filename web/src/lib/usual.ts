"use client";

import { useEffect, useSyncExternalStore } from "react";

import { api, type Schemas } from "@/lib/api/client";

export type Usual = Schemas["SuggestionsResponse"];
export type UsualFigure = Schemas["UsualFigure"];

/**
 * The account's usual figures for the deal on screen (PLAN.md 5.9): the median of each starting
 * figure across the person's own saved deals in the same region and sector, and whether they want
 * them at all. One answer at a time, shared by the Starting point rail (the switch) and the
 * starting figures tile (the column): it is read whenever the deal's country, industry or id
 * changes and each time the Inputs step opens, so a deal saved in between counts.
 */
type State = { key: string | null; usual: Usual | null };

let state: State = { key: null, usual: null };
let asked = 0;
const listeners = new Set<() => void>();

function publish(next: State) {
  state = next;
  listeners.forEach((l) => l());
}

function subscribe(listener: () => void) {
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
}

const snapshot = () => state;

/** Whose answer, and for which deal: an answer read for one account is never shown to the next one signed in. */
export function usualKey(subject: string | null, country: string, industry: string, dealId: string | null): string {
  return `${subject ?? ""}|${country}|${industry}|${dealId ?? ""}`;
}

async function load(subject: string | null, country: string, industry: string, dealId: string | null) {
  const mine = ++asked;
  const key = usualKey(subject, country, industry, dealId);
  let usual: Usual | null = null;
  try {
    const { data } = await api.GET("/api/account/suggestions", {
      params: { query: { country, ...(industry ? { industry } : {}), ...(dealId ? { deal_id: dealId } : {}) } },
    });
    usual = data ?? null;
  } catch {
    usual = null;
  }
  // A later question was asked meanwhile: its answer is the one to show
  if (mine === asked) publish({ key, usual });
}

/** Reads the usual figures for this deal; mount it once per screen (the Starting point rail). */
export function useUsualLoader(subject: string | null, country: string, industry: string, dealId: string | null) {
  useEffect(() => {
    void load(subject, country, industry, dealId);
  }, [subject, country, industry, dealId]);
}

/** The answer for exactly this deal, or null while it is unread, unreadable or for another deal. */
export function useUsual(subject: string | null, country: string, industry: string, dealId: string | null): Usual | null {
  const current = useSyncExternalStore(subscribe, snapshot, snapshot);
  return current.key === usualKey(subject, country, industry, dealId) ? current.usual : null;
}

/** Switch the suggestions on or off, then read them again; false when the choice was not saved. */
export async function setUsualEnabled(
  enabled: boolean, subject: string | null, country: string, industry: string, dealId: string | null,
): Promise<boolean> {
  try {
    const { data } = await api.PUT("/api/account/suggestions", { body: { enabled } });
    if (!data) return false;
  } catch {
    return false;
  }
  await load(subject, country, industry, dealId);
  return true;
}
