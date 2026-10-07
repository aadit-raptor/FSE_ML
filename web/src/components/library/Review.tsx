"use client";

import { useTranslations } from "next-intl";
import { Fragment, useCallback, useEffect, useRef, useState } from "react";

import { EmptyState, LoadingTiles, Notice, PrimaryButton, RailGroup, Screen, SecondaryButton } from "@/components/ui/Screen";
import { Tile, Tiles } from "@/components/ui/Tile";
import { SELECT_CLASS } from "@/components/ui/MoneySelects";
import { type Schemas } from "@/lib/api/client";
import { fmtCount, fmtNumber } from "@/lib/format";
import {
  fetchReview,
  postProposal,
  postVerdict,
  REJECT_REASONS,
  type RejectReason,
  type ReviewAnswer,
  type ReviewProposal,
  type ReviewQueue,
} from "@/lib/library";
import { formatDateTime } from "@/lib/locale";

import { LibraryGate } from "./LibraryGate";
import { ReferenceDealTile } from "./References";

type Problem = Schemas["RuleProblem"];
type Outcome = { tone: "info" | "loss" | "attention"; text: string };

/**
 * Library -> Review (PLAN.md 4.5b): administrators decide the proposed reference transactions. Each
 * shows its evidence, what the inclusion rules find (any finding blocks approval) and what the balance
 * rules say; it joins the library on the second approval by an administrator other than its proposer.
 */
export function ReviewStep() {
  return (
    <LibraryGate>
      <ReviewScreen />
    </LibraryGate>
  );
}

function ReviewScreen() {
  const t = useTranslations("library");
  const [queue, setQueue] = useState<ReviewQueue | null>(null);
  const [status, setStatus] = useState<number | null>(null);
  const [outcome, setOutcome] = useState<Outcome | null>(null);

  const load = useCallback(async () => {
    const answer = await fetchReview();
    setQueue(answer.queue);
    setStatus(answer.status);
  }, []);
  useEffect(() => {
    let live = true;
    void fetchReview().then((answer) => {
      if (!live) return;
      setQueue(answer.queue);
      setStatus(answer.status);
    });
    return () => {
      live = false;
    };
  }, []);

  const report = (answer: ReviewAnswer, done: string) => {
    if (answer.proposal) setOutcome({ tone: "info", text: done });
    else if (answer.problems.length) setOutcome({ tone: "loss", text: t("reviewBlocked") });
    else setOutcome({ tone: "loss", text: t(`reviewRefused_${refusalKey(answer.status)}`) });
    void load();
  };

  const rail = (
    <>
      <RailGroup title={t("groupReview")}>
        <p className="type-body text-[9px]">{t("reviewRail")}</p>
        {queue && (
          <p className="font-mono text-[10.5px] text-ink" data-testid="review-counts">
            {t("reviewCounts", { open: fmtCount(queue.proposals.length), library: fmtCount(queue.library_size) })}
          </p>
        )}
      </RailGroup>
      {queue && <ProposeFromFile onAnswer={(a) => report(a, t("proposed"))} />}
    </>
  );

  if (status === null) return <Screen rail={rail}><LoadingTiles /></Screen>;
  if (!queue) {
    const key = status === 403 ? "reviewAdminsOnly" : status === 503 ? "reviewNoDatabase" : "unreachable";
    return (
      <Screen rail={rail}>
        <EmptyState title={t("reviewTitle")}>{t(key)}</EmptyState>
      </Screen>
    );
  }
  return (
    <Screen rail={rail}>
      {outcome && (
        <Notice tone={outcome.tone} title={t("reviewOutcomeTitle")} role={outcome.tone === "loss" ? "alert" : "status"}>
          {outcome.text}
        </Notice>
      )}
      <Tiles>
        {!queue.proposals.length && (
          <Tile span={12} title={t("reviewTitle")}>
            <p className="type-body text-[9.5px]">{t("reviewEmpty")}</p>
          </Tile>
        )}
        {queue.proposals.map((p) => (
          <Fragment key={p.id}>
            <ReferenceDealTile deal={p.deal} aside={<ApprovalChip proposal={p} />} />
            <Decision proposal={p} onAnswer={report} />
          </Fragment>
        ))}
        {queue.decided.length > 0 && <Decided proposals={queue.decided} />}
      </Tiles>
    </Screen>
  );
}

const refusalKey = (status: number) => (status === 403 ? "403" : status === 409 ? "409" : status === 422 ? "422" : "other");

function ApprovalChip({ proposal }: { proposal: ReviewProposal }) {
  const t = useTranslations("library");
  return (
    <span className="chip text-attention" data-approvals={proposal.key}>
      {t("approvalsOf", { have: String(proposal.approvals), need: String(proposal.approvals_needed) })}
    </span>
  );
}

/** i18n-keys: library.problem_*, library.dimension_*, library.reason_*, library.reviewRefused_*, library.reviewAdminsOnly, library.reviewNoDatabase */
function Decision({ proposal, onAnswer }: { proposal: ReviewProposal; onAnswer: (a: ReviewAnswer, done: string) => void }) {
  const t = useTranslations("library");
  const [reason, setReason] = useState<RejectReason>("figure_wrong");
  const [busy, setBusy] = useState(false);
  const target = proposal.deal.target;
  const blocked = proposal.problems.length > 0;
  const why = proposal.mine ? t("cannotOwn") : proposal.my_verdict ? t("alreadyReviewed") : blocked ? t("cannotBlocked") : null;
  const send = async (verdict: "approve" | "reject") => {
    setBusy(true);
    const answer = await postVerdict(proposal.id, verdict, verdict === "reject" ? reason : undefined);
    setBusy(false);
    onAnswer(answer, verdict === "approve" ? t("approvedOne", { target }) : t("rejectedOne", { target }));
  };
  return (
    <Tile span={12} title={t("decisionTitle", { target })}>
      <div className="grid grid-cols-2 gap-4" data-decision={proposal.key}>
        <div className="grid content-start gap-1">
          <h3 className="type-input-label">{t("rulesTitle")}</h3>
          {blocked ? (
            <ul className="grid gap-0.5" data-problems={proposal.key}>
              {proposal.problems.map((p: Problem, i: number) => (
                <li key={i} className="font-mono text-[10.5px] text-loss">
                  {t(`problem_${p.code}`, { at: p.at ?? "" })}
                </li>
              ))}
            </ul>
          ) : (
            <p className="font-mono text-[10.5px] text-gain">{t("rulesPass")}</p>
          )}
          <h3 className="type-input-label pt-1">{t("balanceTitle")}</h3>
          <BalanceNotes proposal={proposal} />
          {proposal.replaces_approved && <p className="type-body text-[9px] text-attention">{t("replacesApproved")}</p>}
        </div>
        <div className="grid content-start gap-1.5" aria-busy={busy}>
          <p className="type-body text-[9.5px]">
            {proposal.origin === "repository" ? t("proposedByRepository") : proposal.mine ? t("proposedByYou") : t("proposedByAdmin")}{" "}
            {t("proposedAt", { at: formatDateTime(new Date(proposal.proposed_at)) })}
          </p>
          {why ? (
            <p className="font-mono text-[10.5px] text-dim" data-testid="cannot-review">
              {why}
            </p>
          ) : (
            <>
              <PrimaryButton onClick={() => void send("approve")} disabled={busy}>
                {t("approve")}
              </PrimaryButton>
              <label className="grid grid-cols-[minmax(0,1fr)_160px] items-center gap-1.5">
                <span className="type-input-label">{t("rejectReason")}</span>
                <select value={reason} onChange={(e) => setReason(e.target.value as RejectReason)} className={SELECT_CLASS}>
                  {REJECT_REASONS.map((r) => (
                    <option key={r} value={r}>
                      {t(`reason_${r}`)}
                    </option>
                  ))}
                </select>
              </label>
              <SecondaryButton onClick={() => void send("reject")} disabled={busy}>
                {t("reject")}
              </SecondaryButton>
            </>
          )}
        </div>
      </div>
    </Tile>
  );
}

/** i18n-keys: library.region_*, library.bucket_*, library.sector_* */
function BalanceNotes({ proposal }: { proposal: ReviewProposal }) {
  const t = useTranslations("library");
  const bucket = (dimension: string, b: string | null | undefined) => {
    if (!b) return t("none");
    const key = dimension === "region" ? `region_${b}` : dimension === "sector" ? `sector_${b}` : `bucket_${b}`;
    return t.has(key) ? t(key) : b;
  };
  const { over, fills } = proposal.balance;
  if (!over.length && !fills.length) return <p className="type-body text-[9px]">{t("balanceNeutral")}</p>;
  return (
    <ul className="grid gap-0.5" data-balance={proposal.key}>
      {over.map((o, i) => (
        <li key={`o${i}`} className="font-mono text-[10.5px] text-attention">
          {t("balanceOver", { dimension: t(`dimension_${o.dimension}`), bucket: bucket(o.dimension, o.bucket), share: fmtNumber(o.share_pct, 0) })}
        </li>
      ))}
      {fills.map((f, i) => (
        <li key={`f${i}`} className="font-mono text-[10.5px] text-gain">
          {t("balanceFills", { dimension: t(`dimension_${f.dimension}`), bucket: bucket(f.dimension, f.bucket) })}
        </li>
      ))}
    </ul>
  );
}

/** i18n-keys: library.status_* */
function Decided({ proposals }: { proposals: ReviewProposal[] }) {
  const t = useTranslations("library");
  const title = t("decidedTitle");
  return (
    <Tile span={12} title={title}>
      <table className="w-full border-collapse font-mono text-[11px]" aria-label={title}>
        <tbody>
          {proposals.map((p) => (
            <tr key={p.id} data-decided={p.key}>
              <th scope="row" className="border-b border-grid px-1 py-1 text-start text-[10px] font-normal text-soft">
                {p.deal.target}
              </th>
              <td className={`border-b border-grid px-1 py-1 ${p.status === "approved" ? "text-gain" : "text-loss"}`}>{t(`status_${p.status}`)}</td>
              <td className="border-b border-grid px-1 py-1 text-end text-dim">
                {p.decided_at ? formatDateTime(new Date(p.decided_at)) : ""}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </Tile>
  );
}

/** An administrator proposes a transaction from a file in the repository's own format. */
function ProposeFromFile({ onAnswer }: { onAnswer: (a: ReviewAnswer, done: string) => void }) {
  const t = useTranslations("library");
  const file = useRef<HTMLInputElement>(null);
  const [failed, setFailed] = useState(false);
  const onFile = async (f: File | undefined) => {
    if (!f) return;
    setFailed(false);
    let deal: unknown;
    try {
      deal = JSON.parse(await f.text());
    } catch {
      setFailed(true);
      return;
    }
    onAnswer(await postProposal(deal), t("proposed"));
    if (file.current) file.current.value = "";
  };
  return (
    <RailGroup title={t("groupPropose")}>
      <p className="type-body text-[9px]">{t("proposeRail")}</p>
      <input ref={file} type="file" accept=".json,application/json" className="sr-only" aria-label={t("proposeFile")} onChange={(e) => void onFile(e.target.files?.[0])} />
      <div>
        <SecondaryButton onClick={() => file.current?.click()}>{t("proposeFile")}</SecondaryButton>
      </div>
      {failed && <p className="font-mono text-[10px] text-loss">{t("proposeNotJson")}</p>}
    </RailGroup>
  );
}
