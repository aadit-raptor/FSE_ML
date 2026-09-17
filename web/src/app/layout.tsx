import { ClerkProvider } from "@clerk/nextjs";
import type { Metadata } from "next";
import { connection } from "next/server";
import { JetBrains_Mono, Michroma, Orbitron } from "next/font/google";

import { AppShell } from "@/components/shell/AppShell";
import { AUTH_MODE } from "@/lib/auth/mode";

import "./globals.css";

// Closest free matches to Microgramma / Eurostile Extended (agreed in step 1)
const michroma = Michroma({ weight: "400", subsets: ["latin"], variable: "--font-michroma", display: "swap" });
const orbitron = Orbitron({ subsets: ["latin"], variable: "--font-orbitron", display: "swap" });
const jetbrainsMono = JetBrains_Mono({ subsets: ["latin"], variable: "--font-jetbrains-mono", display: "swap" });

export const metadata: Metadata = {
  title: { default: "FSE/ML", template: "%s · FSE/ML" },
  description: "LBO modelling: deal returns, Monte Carlo simulation, backtesting and forecasting.",
};

export default async function RootLayout({ children }: LayoutProps<"/">) {
  // Rendered per request, never prerendered: each page's scripts carry the
  // nonce from that request's content security policy (src/proxy.ts)
  await connection();
  const shell = (
    <html lang="en" className={`${michroma.variable} ${orbitron.variable} ${jetbrainsMono.variable}`}>
      <body>
        <AppShell>{children}</AppShell>
      </body>
    </html>
  );
  // Clerk's provider only goes in when there is an instance to talk to; without
  // one the app uses the development sign-in (lib/auth/mode.ts)
  // `dynamic` puts the nonce on Clerk's script tags
  return AUTH_MODE === "clerk" ? <ClerkProvider dynamic>{shell}</ClerkProvider> : shell;
}
