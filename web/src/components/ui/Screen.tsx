"use client";

import { useTranslations } from "next-intl";
import type { ReactNode } from "react";

import { TipLabel, TipText, useTip } from "./Tip";

/** Input rail on the left, a status bar slot and results on the right. */
export function Screen({ rail, bar, children }: { rail: ReactNode; bar?: ReactNode; children: ReactNode }) {
  const t = useTranslations("ui");
  return (
    <div className="grid h-full min-h-0 grid-cols-[270px_1fr]">
      <aside className="flex min-h-0 flex-col overflow-y-auto border-e border-line bg-panel" aria-label={t("inputs")}>
        {rail}
      </aside>
      <div className="min-h-0 overflow-y-auto">
        {bar}
        {children}
      </div>
    </div>
  );
}

export function RailGroup({ title, children, aside }: { title: string; children: ReactNode; aside?: ReactNode }) {
  return (
    <fieldset className="grid gap-0.5 border-b border-line px-3.5 pt-2.5 pb-3">
      <legend className="type-input-group float-left mb-1.5 flex w-full items-center justify-between gap-2">
        {title}
        {aside}
      </legend>
      {children}
    </fieldset>
  );
}

const TONES = {
  attention: { bg: "bg-[#1b1710]", rule: "shadow-[inset_3px_0_0_var(--color-attention)]", titleClass: "type-alert" },
  loss: { bg: "bg-[#1d1413]", rule: "shadow-[inset_3px_0_0_var(--color-loss)]", titleClass: "type-alert text-loss" },
  info: { bg: "bg-raised", rule: "shadow-[inset_3px_0_0_var(--color-accent)]", titleClass: "type-alert text-accent" },
};

/** Full-width bar: changes waiting to run, errors, or an open model finding. */
export function Notice({
  tone = "attention",
  title,
  children,
  actions,
  role = "status",
  className = "",
}: {
  tone?: keyof typeof TONES;
  title: string;
  children?: ReactNode;
  actions?: ReactNode;
  role?: "status" | "alert" | "note";
  className?: string;
}) {
  const t = TONES[tone];
  return (
    <div className={`flex flex-wrap items-center gap-x-3.5 gap-y-1 px-3 py-2 ${t.bg} ${t.rule} ${className}`} role={role}>
      <span className={t.titleClass}>{title}</span>
      {children && <span className="type-body min-w-0 flex-1">{children}</span>}
      {actions && <span className="ms-auto flex items-center gap-2">{actions}</span>}
    </div>
  );
}

export function PrimaryButton({ children, onClick, disabled }: { children: ReactNode; onClick: () => void; disabled?: boolean }) {
  return (
    <button type="button" onClick={onClick} disabled={disabled} className="type-action bg-accent px-3 py-2 whitespace-nowrap text-bg disabled:opacity-50">
      {children}
    </button>
  );
}

export function SecondaryButton({
  children,
  onClick,
  disabled,
  describedBy,
}: {
  children: ReactNode;
  onClick: () => void;
  disabled?: boolean;
  /** A tip's id, for a button that stands in for a hidden file picker (PLAN.md 7.1) */
  describedBy?: string;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      aria-describedby={describedBy}
      className="type-action-secondary px-2.5 py-1.5 whitespace-nowrap text-muted shadow-[inset_0_0_0_1px_#2a343a] hover:text-ink disabled:opacity-50"
    >
      {children}
    </button>
  );
}

/** Two-state switch with a visible label and its tip (PLAN.md 7.1: `tip` is the key under `tips`). */
export function Switch({ checked, onChange, label, tip }: { checked: boolean; onChange: (v: boolean) => void; label: string; tip: string }) {
  const handle = useTip(tip);
  return (
    <>
      <button
        type="button"
        role="switch"
        aria-checked={checked}
        aria-describedby={handle.id}
        onClick={() => onChange(!checked)}
        className="flex items-center gap-2"
      >
        <span className={`relative inline-block h-3.5 w-[26px] flex-none ${checked ? "bg-accent" : "bg-line-strong"}`} aria-hidden>
          <span className={`absolute top-0.5 size-2.5 bg-bg transition-[left] duration-150 ${checked ? "left-[14px]" : "left-0.5"}`} />
        </span>
        <span className="type-input-label">
          <TipLabel tip={handle}>{label}</TipLabel>
        </span>
      </button>
      <TipText tip={handle} />
    </>
  );
}

/** Empty results area with the action that fills it. */
export function EmptyState({ title, children, action }: { title: string; children: ReactNode; action?: ReactNode }) {
  return (
    <div className="grid max-w-[640px] content-start gap-3 p-6">
      <h2 className="type-result-title text-[14px]">{title}</h2>
      <p className="type-body text-[10.5px]">{children}</p>
      {action && <div>{action}</div>}
    </div>
  );
}

/**
 * An empty screen's guide (PLAN.md 7.1): what belongs here, the steps that fill it and the links that
 * start them. `id` marks it for the browser tests.
 */
export function EmptyGuide({ id, title, steps, children, actions }: {
  id: string;
  title: string;
  steps: ReactNode[];
  children: ReactNode;
  actions?: ReactNode;
}) {
  return (
    <div className="grid max-w-[640px] content-start gap-3 py-2" data-guide={id}>
      <h2 className="type-result-title text-[14px]">{title}</h2>
      <p className="type-body text-[10.5px]">{children}</p>
      <ol className="grid list-decimal gap-1 ps-5 marker:font-mono marker:text-dim">
        {steps.map((step, i) => (
          <li key={i} className="type-body text-[10.5px]">
            {step}
          </li>
        ))}
      </ol>
      {actions && <div className="flex flex-wrap gap-2">{actions}</div>}
    </div>
  );
}

/** Tile grid placeholder while results load. */
export function LoadingTiles() {
  const t = useTranslations("ui");
  return (
    <div className="grid grid-cols-12 content-start gap-px bg-line" aria-busy="true" aria-label={t("loadingResults")}>
      {Array.from({ length: 6 }).map((_, i) => (
        <div key={i} className="col-span-2 grid h-[86px] content-start gap-2 bg-canvas px-3 py-2.5">
          <div className="h-2.5 w-16 bg-raised" />
          <div className="h-5 w-24 bg-raised" />
        </div>
      ))}
      <div className="col-span-6 h-[270px] bg-canvas" />
      <div className="col-span-6 h-[270px] bg-canvas" />
    </div>
  );
}
