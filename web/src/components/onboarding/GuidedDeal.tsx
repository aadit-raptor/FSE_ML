"use client";

import { useTranslations } from "next-intl";
import { useRouter } from "next/navigation";
import { useEffect, useMemo, useState } from "react";

import { useProfile } from "@/components/auth/ProfileProvider";
import { useDeal } from "@/components/deal/DealProvider";
import { useDealLabel } from "@/components/deal/DealScreen";
import { SourceLine, useFigureText } from "@/components/deal/StartingPoint";
import { SELECT_CLASS } from "@/components/ui/MoneySelects";
import { PrimaryButton, SecondaryButton } from "@/components/ui/Screen";
import { TipLabel, TipText, useTip } from "@/components/ui/Tip";
import { fetchStarting, type Starting, useIndustries } from "@/lib/benchmarks";
import type { DealInputs } from "@/lib/deal/fields";
import { fmtInput } from "@/lib/format";
import { countryOptions, parseNumber, regionName } from "@/lib/locale";
import { currencyCodes, currencyName, MONEY_UNITS, type MoneyUnit } from "@/lib/money";
import { currencyOf, GUIDE_STEPS, guidedInputs, SAMPLES, type GuidedAnswers, type Sample } from "@/lib/onboarding";

const ALL = "all";
const FIELD = `${SELECT_CLASS} w-full max-w-[320px] py-1 text-[12px]`;

/**
 * The guided first deal (PLAN.md 7.1): region, sector, size and currency, one question at a time,
 * then the sourced figures those answers give, each with its source, before anything is saved.
 * "Create the deal" saves it under the name given and opens it on Deal -> Inputs. Beside it, the
 * sample deals: the same four answers given in advance, for several regions.
 *
 * i18n-keys: onboarding.step_*, onboarding.question_*, onboarding.why_*
 * i18n-keys: onboarding.unit_*, onboarding.sample_*
 */
export function GuidedDeal() {
  const t = useTranslations("onboarding");
  return (
    <div className="grid content-start gap-6 p-6 xl:grid-cols-[minmax(0,620px)_minmax(0,1fr)]">
      <Guide />
      <Samples />
      <p className="type-body max-w-[620px] text-[10px] xl:col-span-2">{t("figuresNote")}</p>
    </div>
  );
}

function Guide() {
  const t = useTranslations("onboarding");
  const { profile } = useProfile();
  const { createDeal } = useDeal();
  const router = useRouter();
  const industries = useIndustries();
  const countries = useMemo(() => countryOptions(), []);
  const [step, setStep] = useState(0);
  const [country, setCountry] = useState<string | null>(null);
  const [industry, setIndustry] = useState(ALL);
  const [size, setSize] = useState("");
  const [unit, setUnit] = useState<MoneyUnit>("millions");
  // null: follow the country (its own currency, else the account's)
  const [chosenCurrency, setChosenCurrency] = useState<string | null>(null);
  const [name, setName] = useState("");
  // The figures read for one set of answers (`for`), so a read for earlier answers is never shown or saved
  const [read, setRead] = useState<{ for: string; starting: Starting | "failed" } | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const countryTip = useTip("startCountry");
  const industryTip = useTip("startIndustry");
  const sizeTip = useTip("guidedSize");
  const unitTip = useTip("moneyUnit");
  const currencyTip = useTip("currency");
  const nameTip = useTip("guidedName");

  const shownCountry = country ?? profile?.country ?? "";
  const currency = chosenCurrency ?? (currencyOf(shownCountry) || profile?.preferred_currency || "USD");
  const ebitda = parseNumber(size);
  const sizeValid = Number.isFinite(ebitda) && ebitda > 0;
  const current = GUIDE_STEPS[step];
  const industryName = industries?.industries.find((i) => i.id === industry)?.name ?? "";
  const answers: GuidedAnswers = { country: shownCountry, industry, ebitda, currency, unit };
  const readFor = `${shownCountry}|${industry}|${currency}`;
  const starting = read?.for === readFor ? read.starting : null;
  const codes = useMemo(() => {
    const all = currencyCodes();
    return all.includes(currency) ? all : [currency, ...all];
  }, [currency]);

  // The last step shows the sourced figures for the answers, read again whenever they change
  useEffect(() => {
    if (current !== "figures") return;
    let live = true;
    void fetchStarting(shownCountry, industry, currency).then((s) => {
      if (live) setRead({ for: `${shownCountry}|${industry}|${currency}`, starting: s ?? "failed" });
    });
    return () => {
      live = false;
    };
  }, [current, shownCountry, industry, currency]);

  const canGoOn =
    (current === "region" && !!shownCountry) || current === "sector" || (current === "size" && sizeValid) || current === "currency";
  const dealName = name.trim() || t("defaultName", { industry: industry === ALL ? t("anyIndustry") : industryName, country: regionName(shownCountry) });

  const create = async () => {
    if (!starting || starting === "failed") return;
    setBusy(true);
    setError(null);
    const outcome = await createDeal(dealName, guidedInputs(answers, starting));
    setBusy(false);
    if (!outcome.ok) return setError(outcome.error ?? t("createFailed"));
    router.push("/deal/inputs");
  };

  return (
    <section aria-label={t("guideTitle")} className="grid content-start gap-4 border border-line bg-panel p-5">
      <header className="grid gap-1">
        <h1 className="type-result-title text-[16px]">{t("guideTitle")}</h1>
        <p className="type-body">{t("guideIntro")}</p>
      </header>
      <ol className="flex flex-wrap gap-x-4 gap-y-1" aria-label={t("progress")}>
        {GUIDE_STEPS.map((s, i) => (
          <li key={s} aria-current={i === step ? "step" : undefined} className={`type-step ${i === step ? "text-accent" : i < step ? "text-ink" : "text-dim"}`}>
            {t(`step_${s}`)}
          </li>
        ))}
      </ol>
      <p className="font-mono text-[10px] text-dim">{t("stepOf", { step: String(step + 1), total: String(GUIDE_STEPS.length) })}</p>
      <div className="grid gap-2">
        <h2 className="type-input-group text-[12px]">{t(`question_${current}`)}</h2>
        <p className="type-body text-[10.5px]">{t(`why_${current}`)}</p>

        {current === "region" && (
          <>
            <label className="grid gap-1">
              <span className="type-input-label">
                <TipLabel tip={countryTip}>{t("country")}</TipLabel>
              </span>
              <select aria-label={t("countryOf")} aria-describedby={countryTip.id} value={shownCountry} onChange={(e) => setCountry(e.target.value)} className={FIELD}>
                <option value="">{t("chooseCountry")}</option>
                {countries.map((c) => (
                  <option key={c.code} value={c.code}>
                    {c.name}
                  </option>
                ))}
              </select>
            </label>
            <TipText tip={countryTip} />
          </>
        )}

        {current === "sector" && (
          <>
            <label className="grid gap-1">
              <span className="type-input-label">
                <TipLabel tip={industryTip}>{t("industry")}</TipLabel>
              </span>
              <select aria-label={t("industryOf")} aria-describedby={industryTip.id} value={industry} onChange={(e) => setIndustry(e.target.value)} className={FIELD}>
                {(industries?.industries.length ? industries.industries : [{ id: ALL, name: "", firms: 0 }]).map((i) => (
                  <option key={i.id} value={i.id}>
                    {i.id === ALL ? t("anyIndustry") : i.name}
                  </option>
                ))}
              </select>
            </label>
            <TipText tip={industryTip} />
          </>
        )}

        {current === "size" && (
          <div className="grid grid-cols-[minmax(0,180px)_minmax(0,180px)] gap-3">
            <label className="grid gap-1">
              <span className="type-input-label">
                <TipLabel tip={sizeTip}>{t("ebitda")}</TipLabel>
              </span>
              <input
                aria-describedby={sizeTip.id}
                inputMode="decimal"
                autoComplete="off"
                value={size}
                aria-invalid={size !== "" && !sizeValid}
                onChange={(e) => setSize(e.target.value)}
                className={`${FIELD} text-end ${size !== "" && !sizeValid ? "border-loss text-loss" : ""}`}
              />
            </label>
            <label className="grid gap-1">
              <span className="type-input-label">
                <TipLabel tip={unitTip}>{t("unit")}</TipLabel>
              </span>
              <select aria-label={t("unitOf")} aria-describedby={unitTip.id} value={unit} onChange={(e) => setUnit(e.target.value as MoneyUnit)} className={FIELD}>
                {MONEY_UNITS.map((u) => (
                  <option key={u.value} value={u.value}>
                    {t(`unit_${u.value}`)}
                  </option>
                ))}
              </select>
            </label>
            <TipText tip={sizeTip} />
            <TipText tip={unitTip} />
            {size !== "" && !sizeValid && (
              <p role="alert" className="col-span-2 font-mono text-[10px] text-loss">
                {t("sizeInvalid")}
              </p>
            )}
          </div>
        )}

        {current === "currency" && (
          <>
            <label className="grid gap-1">
              <span className="type-input-label">
                <TipLabel tip={currencyTip}>{t("currency")}</TipLabel>
              </span>
              <select aria-label={t("currencyOf")} aria-describedby={currencyTip.id} value={currency} onChange={(e) => setChosenCurrency(e.target.value)} className={FIELD}>
                {codes.map((c) => (
                  <option key={c} value={c}>
                    {currencyName(c)}
                  </option>
                ))}
              </select>
            </label>
            <TipText tip={currencyTip} />
          </>
        )}

        {current === "figures" && (
          <>
            <Figures starting={starting} answers={answers} />
            <label className="grid gap-1 pt-2">
              <span className="type-input-label">
                <TipLabel tip={nameTip}>{t("dealName")}</TipLabel>
              </span>
              <input
                aria-label={t("dealName")}
                aria-describedby={nameTip.id}
                dir="auto"
                maxLength={120}
                placeholder={dealName}
                value={name}
                onChange={(e) => setName(e.target.value)}
                className={FIELD}
              />
            </label>
            <TipText tip={nameTip} />
          </>
        )}
      </div>

      <div className="flex flex-wrap items-center gap-2">
        {step > 0 && <SecondaryButton onClick={() => setStep(step - 1)}>{t("back")}</SecondaryButton>}
        {current === "figures" ? (
          <PrimaryButton onClick={() => void create()} disabled={busy || !starting || starting === "failed"}>
            {busy ? t("creating") : t("create")}
          </PrimaryButton>
        ) : (
          <PrimaryButton onClick={() => setStep(step + 1)} disabled={!canGoOn}>
            {t("next")}
          </PrimaryButton>
        )}
      </div>
      {error && (
        <p role="alert" className="font-mono text-[10px] text-loss">
          {error}
        </p>
      )}
    </section>
  );
}

/** The sourced figures the answers give, as Deal -> Inputs' Starting figures tile shows them. */
function Figures({ starting, answers }: { starting: Starting | null | "failed"; answers: GuidedAnswers }) {
  const t = useTranslations("onboarding");
  const { areaName, value, sample } = useFigureText();
  const label = useDealLabel();
  if (starting === null) return <p className="type-step">{t("loadingFigures")}</p>;
  if (starting === "failed") {
    return (
      <p role="alert" className="font-mono text-[10px] text-loss">
        {t("figuresFailed")}
      </p>
    );
  }
  return (
    <div className="grid gap-2">
      <p className="type-body text-[10.5px]">
        {t("figuresSummary", {
          ebitda: fmtInput(answers.ebitda, 1),
          unit: t(`unit_${answers.unit}`),
          currency: answers.currency,
          count: starting.figures.length,
        })}
      </p>
      <div className="overflow-x-auto">
        <table className="w-full border-collapse font-mono text-[11px]" aria-label={t("figuresTitle")}>
          <thead>
            <tr className="type-input-label">
              <th scope="col" className="px-2 py-1 text-start font-normal">{t("colFigure")}</th>
              <th scope="col" className="px-2 py-1 text-end font-normal">{t("colValue")}</th>
              <th scope="col" className="px-2 py-1 text-start font-normal">{t("colSource")}</th>
            </tr>
          </thead>
          <tbody>
            {starting.figures.map((f) => (
              <tr key={f.field} className="border-t border-line align-top" data-field={f.field}>
                <th scope="row" className="type-input-label px-2 py-1 text-start font-normal">{label(f.field as keyof DealInputs)}</th>
                <td className="px-2 py-1 text-end text-ink">{value(f.field, f.value)}</td>
                <td className="px-2 py-1 text-start text-muted">
                  <SourceLine f={f} area={areaName(f.area)} sample={sample(f)} />
                </td>
              </tr>
            ))}
            {starting.missing.map((m) => (
              <tr key={m.field} className="border-t border-line" data-field={m.field}>
                <th scope="row" className="type-input-label px-2 py-1 text-start font-normal">{label(m.field as keyof DealInputs)}</th>
                <td />
                <td className="px-2 py-1 text-start text-attention">{t("missingFigure")}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

/** Sample deals from several regions: the guide's answers given in advance, opened with sourced figures. */
function Samples() {
  const t = useTranslations("onboarding");
  const { createDeal } = useDeal();
  const router = useRouter();
  const [opening, setOpening] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const open = async (s: Sample) => {
    setOpening(s.id);
    setError(null);
    const starting = await fetchStarting(s.country, s.industry, s.currency);
    if (!starting) {
      setOpening(null);
      return setError(t("figuresFailed"));
    }
    const outcome = await createDeal(t("sampleDealName", { name: t(`sample_${s.id}`) }), guidedInputs(s, starting));
    setOpening(null);
    if (!outcome.ok) return setError(outcome.error ?? t("createFailed"));
    router.push("/deal/inputs");
  };

  return (
    <section aria-labelledby="samples-title" className="grid content-start gap-3">
      <h2 id="samples-title" className="type-input-group text-[12px]">{t("samplesTitle")}</h2>
      <p className="type-body text-[10.5px]">{t("samplesIntro")}</p>
      <ul aria-label={t("samplesTitle")} className="grid gap-px border border-line bg-line">
        {SAMPLES.map((s) => (
          <li key={s.id} className="grid grid-cols-[minmax(0,1fr)_auto] items-center gap-3 bg-panel px-3 py-2.5">
            <span className="grid gap-0.5">
              <span className="type-input-label text-ink">{t(`sample_${s.id}`)}</span>
              <span className="type-body text-[10px]">
                {t("sampleLine", { country: regionName(s.country), currency: currencyName(s.currency), ebitda: fmtInput(s.ebitda, 0), unit: t(`unit_${s.unit}`) })}
              </span>
              <span className="type-body text-[10px] text-dim">{t(`sample_${s.id}Note`)}</span>
            </span>
            <button
              type="button"
              disabled={opening !== null}
              onClick={() => void open(s)}
              aria-label={t("openSample", { name: t(`sample_${s.id}`) })}
              className="type-action-secondary px-2.5 py-1.5 text-accent shadow-[inset_0_0_0_1px_var(--color-accent)] disabled:opacity-50"
            >
              {opening === s.id ? t("opening") : t("open")}
            </button>
          </li>
        ))}
      </ul>
      {error && (
        <p role="alert" className="font-mono text-[10px] text-loss">
          {error}
        </p>
      )}
    </section>
  );
}
