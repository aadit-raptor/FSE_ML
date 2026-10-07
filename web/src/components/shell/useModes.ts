"use client";

import { useLibrary } from "@/components/library/LibraryProvider";
import { type Mode, MODES } from "@/lib/nav";

/** Every mode but the optional ones, built once so the list keeps its identity between renders. */
const REQUIRED_MODES = MODES.filter((m) => !m.optional);

/** The modes listed in the tabs, search and Alt shortcuts: an optional one only while it is shown. */
export function useModes(): Mode[] {
  const { visible } = useLibrary();
  return visible ? MODES : REQUIRED_MODES;
}
