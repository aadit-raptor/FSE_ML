"use client";

import { useAuth as useClerkAuth, useUser as useClerkUser } from "@clerk/nextjs";
import { useRouter } from "next/navigation";
import { createContext, useCallback, useContext, useEffect, useMemo, useSyncExternalStore } from "react";

import { setTokenSource, setUnauthorizedHandler } from "@/lib/api/client";
import { clearDevUser, devToken, readDevUser, subscribeDevUser, writeDevUser } from "@/lib/auth/dev";
import { AUTH_MODE, type AuthMode } from "@/lib/auth/mode";

/**
 * Who is signed in, whichever sign-in the build runs with (lib/auth/mode.ts).
 *
 * Whoever it is, the API client gets a token source before any screen renders
 * -- registered during render, not in an effect, because a child's effects run
 * first and the screens start loading the moment they mount.
 */
export type Session = {
  mode: AuthMode;
  /** False until we know whether anyone is signed in (Clerk loads in the browser). */
  ready: boolean;
  signedIn: boolean;
  /** Email address, or the development user's name. Shown in the top bar. */
  label: string | null;
  signOut: () => Promise<void>;
  /** Development sign-in only; Clerk's own screens do this in clerk mode. */
  signInAsDeveloper?: (name: string) => void;
};

const Ctx = createContext<Session | null>(null);

export function useSession(): Session {
  const session = useContext(Ctx);
  if (!session) throw new Error("useSession must be used inside AuthProvider");
  return session;
}

function ClerkSession({ children }: { children: React.ReactNode }) {
  const { isLoaded, isSignedIn, getToken, signOut } = useClerkAuth();
  const { user } = useClerkUser();
  const router = useRouter();

  // Clerk's tokens are short-lived and refreshed by getToken() on demand
  setTokenSource(isSignedIn ? () => getToken() : null);

  const session = useMemo<Session>(
    () => ({
      mode: "clerk",
      ready: isLoaded,
      signedIn: !!isSignedIn,
      label: user?.primaryEmailAddress?.emailAddress ?? user?.username ?? null,
      signOut: async () => {
        setTokenSource(null);
        await signOut({ redirectUrl: "/sign-in" });
      },
    }),
    [isLoaded, isSignedIn, signOut, user],
  );

  useEffect(() => {
    setUnauthorizedHandler(() => router.replace("/sign-in"));
    return () => setUnauthorizedHandler(null);
  }, [router]);

  return <Ctx.Provider value={session}>{children}</Ctx.Provider>;
}

function DeveloperSession({ children }: { children: React.ReactNode }) {
  const router = useRouter();
  // The cookie is only readable in the browser: the server renders "not ready"
  // and the first client render replaces it with whoever is signed in
  const user = useSyncExternalStore(subscribeDevUser, readDevUser, () => null);
  const ready = useSyncExternalStore(subscribeDevUser, () => true, () => false);

  setTokenSource(user ? async () => devToken(user) : null);

  const signInAsDeveloper = useCallback((name: string) => writeDevUser(name), []);

  const signOut = useCallback(async () => {
    clearDevUser();
    setTokenSource(null);
    router.replace("/sign-in");
  }, [router]);

  const session = useMemo<Session>(
    () => ({ mode: "dev", ready, signedIn: !!user, label: user, signOut, signInAsDeveloper }),
    [ready, user, signOut, signInAsDeveloper],
  );

  return <Ctx.Provider value={session}>{children}</Ctx.Provider>;
}

export function AuthProvider({ children }: { children: React.ReactNode }) {
  return AUTH_MODE === "clerk" ? (
    <ClerkSession>{children}</ClerkSession>
  ) : (
    <DeveloperSession>{children}</DeveloperSession>
  );
}
