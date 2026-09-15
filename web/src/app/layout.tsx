import type { Metadata } from "next";
import { JetBrains_Mono, Michroma, Orbitron } from "next/font/google";

import { AppShell } from "@/components/shell/AppShell";

import "./globals.css";

// Closest free matches to Microgramma / Eurostile Extended (agreed in step 1)
const michroma = Michroma({ weight: "400", subsets: ["latin"], variable: "--font-michroma", display: "swap" });
const orbitron = Orbitron({ subsets: ["latin"], variable: "--font-orbitron", display: "swap" });
const jetbrainsMono = JetBrains_Mono({ subsets: ["latin"], variable: "--font-jetbrains-mono", display: "swap" });

export const metadata: Metadata = {
  title: { default: "FSE/ML", template: "%s · FSE/ML" },
  description: "LBO modelling: deal returns, Monte Carlo simulation, backtesting and forecasting.",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en" className={`${michroma.variable} ${orbitron.variable} ${jetbrainsMono.variable}`}>
      <body>
        <AppShell>{children}</AppShell>
      </body>
    </html>
  );
}
