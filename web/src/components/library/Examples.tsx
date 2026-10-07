"use client";

import Link from "next/link";
import { useTranslations } from "next-intl";
import { useEffect, useState } from "react";

import { splitDealName } from "@/components/backtest/BacktestSteps";
import { EmptyState, LoadingTiles, Notice, RailGroup, Screen } from "@/components/ui/Screen";
import { Tile, Tiles } from "@/components/ui/Tile";
import { api, type Schemas } from "@/lib/api/client";
import { fmtMoney, fmtMultiple } from "@/lib/format";
import { useProvenance } from "@/lib/i18n/useProvenance";
import { moneyLabel } from "@/lib/money";
import { stepHref } from "@/lib/nav";

import { LibraryGate } from "./LibraryGate";

type Example = Schemas["ExampleDeal"];

/** Library -> Examples: the four inception-era deals, labelled as unsourced examples (PLAN.md 2.1, 4.5). */
export function ExamplesStep() {
  return (
    <LibraryGate>
      <ExamplesScreen />
    </LibraryGate>
  );
}

function ExamplesScreen() {
  const t = useTranslations("library");
  const provenance = useProvenance();
  const [examples, setExamples] = useState<Example[] | null>(null);
  useEffect(() => {
    let live = true;
    api
      .GET("/api/backtesting/examples")
      .then(({ data }) => live && setExamples(data?.examples ?? []))
      .catch(() => live && setExamples([]));
    return () => {
      live = false;
    };
  }, []);

  const rail = (
    <RailGroup title={t("groupExamples")}>
      <p className="type-body text-[9px]">{t("examplesRail")}</p>
      <Link href={stepHref("backtest", "actuals")} className="type-control text-accent underline">
        {t("openBacktest")}
      </Link>
    </RailGroup>
  );
  if (examples === null) return <Screen rail={rail}><LoadingTiles /></Screen>;
  if (!examples.length) return <Screen rail={rail}><EmptyState title={t("noExamplesTitle")}>{t("noExamples")}</EmptyState></Screen>;
  return (
    <Screen rail={rail}>
      <Notice title={t("examplesNoticeTitle")} role="note">
        {provenance.backtest(examples.map((x) => splitDealName(x.name).year))}
        {t("examplesNotice")}
      </Notice>
      <Tiles>
        {examples.map((x) => (
          <ExampleTile key={x.name} example={x} />
        ))}
      </Tiles>
    </Screen>
  );
}

function ExampleTile({ example }: { example: Example }) {
  const t = useTranslations("library");
  const { name, sponsor, year } = splitDealName(example.name);
  const plan = example.plan;
  const ebitda = plan.ebitda ?? 0;
  const multiple = plan.entry_mult ?? 0;
  const money = moneyLabel({ currency: plan.currency ?? "USD", unit: plan.unit ?? "millions" });
  const success = example.outcome === "SUCCESS";
  return (
    <Tile
      span={6}
      title={name}
      aside={
        <span className="flex gap-1.5">
          <span className="chip text-attention">{t("unsourced")}</span>
          <span className={`chip ${success ? "text-gain" : "text-loss"}`}>{success ? t("outcome_success") : t("outcome_distress")}</span>
        </span>
      }
    >
      <p className="font-mono text-[10.5px] text-muted">{t("exampleWho", { sponsor, year })}</p>
      <p className="type-body text-[9.5px]">{example.description}</p>
      <dl className="grid grid-cols-3 gap-2 font-mono text-[11px]" data-example={example.name}>
        <div>
          <dt className="type-input-label">{t("entryEv", { unit: money })}</dt>
          <dd className="text-ink">{fmtMoney(ebitda * multiple)}</dd>
        </div>
        <div>
          <dt className="type-input-label">{t("entryMultiple")}</dt>
          <dd className="text-ink">{fmtMultiple(multiple, 1)}</dd>
        </div>
        <div>
          <dt className="type-input-label">{t("sector")}</dt>
          <dd className="text-ink">{example.sector}</dd>
        </div>
      </dl>
    </Tile>
  );
}
