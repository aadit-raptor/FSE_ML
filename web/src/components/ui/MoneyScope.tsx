"use client";

import { createContext, useContext, useMemo } from "react";

import { DEFAULT_MONEY, type Money, moneyLabel } from "@/lib/money";

type MoneyContext = { money: Money; label: string };

const Ctx = createContext<MoneyContext>({ money: DEFAULT_MONEY, label: moneyLabel(DEFAULT_MONEY) });

/**
 * What the money on screen is counted in. The deal provider sets the open
 * deal's currency and unit for every screen; Backtest and Forecast set their
 * own deal's or company's inside it.
 */
export function MoneyScope({ money, children }: { money: Money; children: React.ReactNode }) {
  const { currency, unit } = money;
  const value = useMemo(() => {
    const m = { currency, unit };
    return { money: m, label: moneyLabel(m) };
  }, [currency, unit]);
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

/** The money on screen and its label, e.g. "€k". */
export function useMoney(): MoneyContext {
  return useContext(Ctx);
}
