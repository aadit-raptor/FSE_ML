"use client";

import { usePathname } from "next/navigation";
import { useEffect } from "react";

import { useLibrary } from "@/components/library/LibraryProvider";
import { lastScreen, rememberScreen } from "@/lib/lastScreen";
import { type Mode, MODES, parsePath, type Workspace, workspaceModes, workspaceOf } from "@/lib/nav";

/** Every mode but the optional ones, built once so the list keeps its identity between renders. */
const REQUIRED_MODES = MODES.filter((m) => !m.optional);

/** Every mode shown (search lists them all): an optional one only while it is shown. */
export function useModes(): Mode[] {
  const { visible } = useLibrary();
  return visible ? MODES : REQUIRED_MODES;
}

/**
 * The workspace on screen: the current mode's; on the account screen, the
 * one last worked in, so its tabs stay; on the launcher, none.
 */
export function useCurrentWorkspace(): Workspace | undefined {
  const pathname = usePathname();
  const fromPath = workspaceOf(parsePath(pathname).mode);
  if (fromPath || pathname !== "/account") return fromPath;
  const last = lastScreen();
  return last ? workspaceOf(parsePath(last).mode) : undefined;
}

/** The tabs and Alt shortcuts: the current workspace's modes, in its order. */
export function useWorkspaceModes(): Mode[] {
  const workspace = useCurrentWorkspace();
  const modes = useModes();
  return workspace ? workspaceModes(workspace, modes) : [];
}

/** Notes each screen shown, for the account screen's tabs and the next visit (lib/lastScreen.ts). */
export function useTrackPlace(): void {
  const pathname = usePathname();
  useEffect(() => rememberScreen(pathname), [pathname]);
}
