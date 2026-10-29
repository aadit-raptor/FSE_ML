/**
 * "What's new" (PLAN.md 7.1): the releases people would notice, newest first, from `whats-new.json`;
 * the words are `whatsNew.<id>Title` and `<id>Body` in the translation files.
 *
 * Which entries this browser has seen is a convenience kept in localStorage (the newest date seen),
 * so the top bar's Help link can mark something new; blocked storage just means the mark shows.
 *
 * i18n-keys: whatsNew.*
 */
import { useSyncExternalStore } from "react";

import data from "./whats-new.json";

export type WhatsNewEntry = { id: string; date: string; href: string };

export const WHATS_NEW: WhatsNewEntry[] = data.entries;

const SEEN_KEY = "fse.whatsNewSeen.v1";
const CHANGED = "fse:whats-new-seen";
const newest = WHATS_NEW[0]?.date ?? "";

function seen(): string {
  try {
    return window.localStorage.getItem(SEEN_KEY) ?? "";
  } catch {
    return "";
  }
}

function subscribe(onChange: () => void): () => void {
  window.addEventListener(CHANGED, onChange);
  window.addEventListener("storage", onChange);
  return () => {
    window.removeEventListener(CHANGED, onChange);
    window.removeEventListener("storage", onChange);
  };
}

/** Whether a release came out after the newest one this browser has seen. */
export function useUnseenNews(): boolean {
  return useSyncExternalStore(subscribe, () => seen() < newest, () => false);
}

/** Mark every entry as seen (the What's new page does, once it is shown). */
export function markNewsSeen(): void {
  try {
    window.localStorage.setItem(SEEN_KEY, newest);
  } catch {
    // Blocked storage: the mark simply stays
  }
  window.dispatchEvent(new Event(CHANGED));
}
