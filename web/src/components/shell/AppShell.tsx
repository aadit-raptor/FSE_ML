"use client";

import { usePathname, useRouter } from "next/navigation";
import { useEffect } from "react";

import { AuthProvider, useSession } from "@/components/auth/AuthProvider";
import { ProfileProvider, useProfile } from "@/components/auth/ProfileProvider";
import { BacktestProvider } from "@/components/backtest/BacktestProvider";
import { DealProvider } from "@/components/deal/DealProvider";
import { ForecastProvider } from "@/components/forecast/ForecastProvider";
import { MonteCarloProvider } from "@/components/montecarlo/MonteCarloProvider";
import { SettingsProvider } from "@/components/settings/SettingsProvider";
import { isPublicRoute } from "@/lib/auth/mode";

import { CommandSearch } from "./CommandSearch";
import { Shortcuts } from "./Shortcuts";
import { StatusBar } from "./StatusBar";
import { StepBar } from "./StepBar";
import { TopBar } from "./TopBar";
import { WorkspaceProvider } from "./workspace";

/** The Guided shell: mode tabs, step row, search, content, status bar. */
export function AppShell({ children }: { children: React.ReactNode }) {
  return (
    <WorkspaceProvider>
      <AuthProvider>
        <Shell>{children}</Shell>
      </AuthProvider>
    </WorkspaceProvider>
  );
}

/**
 * What the visitor sees depends on whether they are signed in (PLAN.md 1.4).
 * Sign-in and sign-up get the bare canvas; everything else waits for a
 * session, so no screen ever starts loading a deal it has no token for.
 */
function Shell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const router = useRouter();
  const { ready, signedIn } = useSession();
  const publicPage = isPublicRoute(pathname);

  useEffect(() => {
    if (ready && !signedIn && !publicPage) router.replace("/sign-in");
  }, [ready, signedIn, publicPage, router]);

  if (publicPage) {
    // No mode tabs: signed out, there is nothing behind them to open
    return (
      <div className="flex h-dvh flex-col overflow-hidden bg-canvas">
        <header className="flex min-h-[42px] flex-none items-center border-b border-line bg-panel px-4">
          <span className="type-brand">FSE/ML</span>
        </header>
        <main id="content" className="min-h-0 flex-1 overflow-auto">
          {children}
        </main>
        <StatusBar />
      </div>
    );
  }

  if (!ready || !signedIn) {
    return (
      <div className="flex h-dvh flex-col items-center justify-center bg-canvas">
        <p className="type-step" role="status">
          {ready ? "Taking you to sign-in" : "Checking your session"}
        </p>
      </div>
    );
  }

  return (
    <ProfileProvider>
      <ProfileGate />
      {/* Model state lives above the routes so it survives switching modes */}
      <SettingsProvider>
        <DealProvider>
          <MonteCarloProvider>
          <BacktestProvider>
          <ForecastProvider>
            <div className="flex h-dvh flex-col overflow-hidden bg-canvas">
              <TopBar />
              <StepBar />
              <main id="content" className="min-h-0 flex-1 overflow-auto">
                {children}
              </main>
              <StatusBar />
            </div>
            <CommandSearch />
            <Shortcuts />
          </ForecastProvider>
          </BacktestProvider>
          </MonteCarloProvider>
        </DealProvider>
      </SettingsProvider>
    </ProfileProvider>
  );
}

/**
 * A new account answers four questions before the deal screens open: country,
 * currency, number format and time zone (PLAN.md 1.4). Until then every route
 * leads back to the account screen.
 */
function ProfileGate() {
  const { needsProfile } = useProfile();
  const pathname = usePathname();
  const router = useRouter();

  useEffect(() => {
    if (needsProfile && pathname !== "/account") router.replace("/account");
  }, [needsProfile, pathname, router]);

  return null;
}
