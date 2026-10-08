import { ClerkProvider } from "@clerk/nextjs";
import type { Metadata, Viewport } from "next";
import { connection } from "next/server";
import { JetBrains_Mono, Michroma, Orbitron } from "next/font/google";

import { AppShell } from "@/components/shell/AppShell";
import { AUTH_MODE, SIGN_IN_URL, SIGN_UP_URL } from "@/lib/auth/mode";
import { APP_BRAND, APP_DESCRIPTION } from "@/lib/i18n/titles";

import "./globals.css";

// Closest free matches to Microgramma / Eurostile Extended (agreed in step 1)
const michroma = Michroma({ weight: "400", subsets: ["latin"], variable: "--font-michroma", display: "swap" });
const orbitron = Orbitron({ subsets: ["latin"], variable: "--font-orbitron", display: "swap" });
const jetbrainsMono = JetBrains_Mono({ subsets: ["latin"], variable: "--font-jetbrains-mono", display: "swap" });

// The icons, apple-icon and link-preview image are files beside this one
// (icon.svg, favicon.ico, apple-icon.png, opengraph-image.png), drawn by
// scripts/brand.mjs from components/brand/mark.json
export const metadata: Metadata = {
  title: { default: APP_BRAND, template: `%s · ${APP_BRAND}` },
  description: APP_DESCRIPTION,
  // Link previews need absolute addresses; staging previews point at production's image
  metadataBase: new URL("https://variater.com"),
  openGraph: { type: "website", siteName: APP_BRAND, title: APP_BRAND, description: APP_DESCRIPTION },
  twitter: { card: "summary_large_image", title: APP_BRAND, description: APP_DESCRIPTION },
};

export const viewport: Viewport = { themeColor: "#0C1012" };

export default async function RootLayout({ children }: LayoutProps<"/">) {
  // Rendered per request, never prerendered: each page's scripts carry the
  // nonce from that request's content security policy (src/proxy.ts)
  await connection();
  // `lang` and `dir` start at the default and are set to the account's locale
  // by I18nScope, which is the first thing that knows who is signed in
  const shell = (
    <html lang="en" dir="ltr" className={`${michroma.variable} ${orbitron.variable} ${jetbrainsMono.variable}`}>
      <body>
        <AppShell>{children}</AppShell>
      </body>
    </html>
  );
  // Clerk's provider only goes in when there is an instance to talk to; without
  // one the app uses the development sign-in (lib/auth/mode.ts)
  // `dynamic` puts the nonce on Clerk's script tags
  return AUTH_MODE === "clerk" ? (
    <ClerkProvider dynamic signInUrl={SIGN_IN_URL} signUpUrl={SIGN_UP_URL}>
      {shell}
    </ClerkProvider>
  ) : shell;
}
