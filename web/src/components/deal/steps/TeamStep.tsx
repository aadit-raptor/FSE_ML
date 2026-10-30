"use client";

import Link from "next/link";
import { useTranslations } from "next-intl";
import { useCallback, useEffect, useState } from "react";

import { useDeal } from "@/components/deal/DealProvider";
import { Notice, PrimaryButton, RailGroup, Screen, SecondaryButton, Switch } from "@/components/ui/Screen";
import { Tile, Tiles } from "@/components/ui/Tile";
import { TipLabel, TipText, useTip } from "@/components/ui/Tip";
import { formatDateTime } from "@/lib/locale";
import {
  addComment, type Comment, dealPeople, type DealPerson, type DealShare, dealShares, deleteComment, listComments, listTeams,
  problemKey, shareDeal, type TeamSummary, unshareDeal,
} from "@/lib/teams";

const MAX_COMMENT = 2000;

/** Everything the step shows of a deal, asked for at once. */
function fetchAll(id: string) {
  return Promise.all([dealPeople(id), listComments(id), listTeams(), dealShares(id)]);
}
type Found = Awaited<ReturnType<typeof fetchAll>>;

/** The deal a mention email's link names (`?deal=<id>`), read once when the step opens. */
function linkedDeal(): string | null {
  return typeof window === "undefined" ? null : new URLSearchParams(window.location.search).get("deal");
}

/**
 * Deal -> Team (PLAN.md 7.2): who can read the open deal, sharing it with the owner's teams, and its comments,
 * mentioning people by name. Opened with `?deal=<id>` (the link in a mention email), it opens that deal first.
 */
export function TeamStep() {
  const t = useTranslations("teams");
  const { current, openDeal } = useDeal();
  const dealId = current?.id ?? null;
  const [problem, setProblem] = useState<string | null>(null);
  const [people, setPeople] = useState<DealPerson[] | null>(null);
  const [comments, setComments] = useState<Comment[] | null>(null);
  const [shares, setShares] = useState<DealShare[]>([]);
  const [teams, setTeams] = useState<TeamSummary[] | null>(null);

  // A link from a mention email names the deal to open
  const [asked, setAsked] = useState<string | null>(linkedDeal);
  useEffect(() => {
    if (!asked || asked === dealId) return;
    void openDeal(asked).then((outcome) => {
      if (!outcome.ok) setProblem("not_found");
      setAsked(null);
    });
  }, [asked, dealId, openDeal]);

  const show = useCallback(([who, said, mine, shared]: Found) => {
    if (!who.ok) return setProblem(who.code);
    setProblem(null);
    setPeople(who.data.people);
    if (said.ok) setComments(said.data.comments);
    if (mine.ok) setTeams(mine.data.teams);
    if (shared.ok) setShares(shared.data.teams);
  }, []);
  const load = useCallback(async (id: string) => show(await fetchAll(id)), [show]);

  useEffect(() => {
    if (!dealId) return;
    let cancelled = false;
    void fetchAll(dealId).then((found) => {
      if (!cancelled) show(found);
    });
    return () => {
      cancelled = true;
    };
  }, [dealId, show]);

  if (!current) {
    return (
      <Screen rail={null}>
        <Notice tone="attention" title={t("dealTitle")} role="note">{t("noDeal")}</Notice>
      </Screen>
    );
  }
  const access = current.access;

  const toggleShare = async (teamId: string, on: boolean) => {
    const done = on ? await shareDeal(current.id, teamId) : await unshareDeal(current.id, teamId);
    if (!done.ok) return setProblem(done.code);
    setShares(done.data.teams);
    await load(current.id);
  };

  return (
    <Screen
      rail={
        <>
          <RailGroup title={t("accessGroup")}>
            <p className="type-body text-[10.5px]" data-access={access}>{t(`access_${access}`)}</p>
          </RailGroup>
          <RailGroup title={t("shareGroup")}>
            {access !== "owner" ? (
              <p className="type-body text-[10.5px]">{t("shareOwnerOnly")}</p>
            ) : teams === null ? (
              <p className="type-step">{t("loading")}</p>
            ) : teams.length === 0 ? (
              <p className="type-body text-[10.5px]">{t("shareNoTeams")}</p>
            ) : (
              <div className="grid gap-1.5" data-shares>
                {teams.map((team) => (
                  <Switch key={team.id} checked={shares.some((s) => s.team_id === team.id)}
                    onChange={(on) => void toggleShare(team.id, on)} label={team.name} tip="shareWithTeam" />
                ))}
              </div>
            )}
            <Link href="/teams" className="type-action-secondary mt-1.5 text-accent hover:text-ink">{t("manageTeams")}</Link>
          </RailGroup>
        </>
      }
    >
      {problem && <Notice tone="loss" title={t("dealTitle")} role="alert">{t(problemKey(problem))}</Notice>}
      {access === "viewer" && <Notice tone="info" title={t("viewOnlyTitle")} role="note">{t("viewOnly")}</Notice>}
      <Tiles>
        <Tile span={4} title={t("peopleTitle")}>
          {people === null ? (
            <p className="type-step">{t("loading")}</p>
          ) : (
            <ul className="grid gap-1" aria-label={t("peopleTitle")}>
              {people.map((p) => (
                <li key={p.member_id} className="type-body flex justify-between gap-2 text-[10.5px]" data-person={p.name ?? ""}>
                  <span className={p.you ? "text-bright" : "text-ink"}>{p.you ? t("you", { name: p.name ?? t("someone") }) : (p.name ?? t("someone"))}</span>
                  <span className="text-muted">{t(`dealAccess_${p.access}`)}</span>
                </li>
              ))}
            </ul>
          )}
          {people !== null && people.length === 1 && <p className="type-body text-[10px] text-muted">{t("onlyYou")}</p>}
        </Tile>
        <Tile span={8} title={t("commentsTitle")}>
          <Comments dealId={current.id} comments={comments} people={people ?? []}
            onChange={() => void load(current.id)} onProblem={setProblem} />
        </Tile>
      </Tiles>
    </Screen>
  );
}

function Comments({ dealId, comments, people, onChange, onProblem }: {
  dealId: string; comments: Comment[] | null; people: DealPerson[]; onChange: () => void;
  onProblem: (code: string) => void;
}) {
  const t = useTranslations("teams");
  const tip = useTip("commentBody");
  const [body, setBody] = useState("");
  const [mentions, setMentions] = useState<number[]>([]);
  const [busy, setBusy] = useState(false);
  const others = people.filter((p) => !p.you);

  const post = async () => {
    setBusy(true);
    const made = await addComment(dealId, body.trim(), mentions);
    setBusy(false);
    if (!made.ok) return onProblem(made.code);
    setBody("");
    setMentions([]);
    onChange();
  };
  const remove = async (id: string) => {
    const gone = await deleteComment(dealId, id);
    if (!gone.ok) return onProblem(gone.code);
    onChange();
  };

  return (
    <div className="grid gap-3">
      {comments === null ? (
        <p className="type-step">{t("loading")}</p>
      ) : comments.length === 0 ? (
        <p className="type-body text-[10.5px]" role="note">{t("noComments")}</p>
      ) : (
        <ol className="grid gap-2" aria-label={t("commentsTitle")}>
          {comments.map((c) => (
            <li key={c.id} className="grid gap-1 border-b border-line/60 pb-2" data-comment={c.id}>
              <span className="type-body flex flex-wrap items-baseline gap-x-2 text-[10px] text-muted">
                <span className="text-ink">{c.author.name ?? t("someone")}</span>
                <span>{formatDateTime(new Date(c.created_at))}</span>
                {c.mentions.length > 0 && (
                  <span>{t("mentioned", { names: c.mentions.map((m) => m.name ?? t("someone")).join(", ") })}</span>
                )}
              </span>
              <p dir="auto" className="type-body text-[11px] whitespace-pre-wrap text-ink">{c.body}</p>
              {c.can_delete && (
                <span><SecondaryButton onClick={() => void remove(c.id)}>{t("deleteComment")}</SecondaryButton></span>
              )}
            </li>
          ))}
        </ol>
      )}
      <div className="grid gap-1.5">
        <label className="grid gap-1">
          <span className="type-input-label"><TipLabel tip={tip}>{t("commentLabel")}</TipLabel></span>
          <textarea aria-describedby={tip.id} value={body} maxLength={MAX_COMMENT} rows={3}
            onChange={(e) => setBody(e.target.value)}
            className="border border-line bg-canvas px-2 py-1 text-[11px] text-ink" />
        </label>
        <TipText tip={tip} />
        {others.length > 0 && (
          <div className="flex flex-wrap items-center gap-1.5" role="group" aria-label={t("mentionGroup")}>
            <span className="type-body text-[10px] text-muted">{t("mentionGroup")}</span>
            {others.map((p) => {
              const on = mentions.includes(p.member_id);
              return (
                <button key={p.member_id} type="button" aria-pressed={on} data-mention={p.name ?? ""}
                  onClick={() => setMentions(on ? mentions.filter((m) => m !== p.member_id) : [...mentions, p.member_id])}
                  className={`type-action-secondary px-2 py-1 ${on ? "bg-accent text-bg" : "text-muted shadow-[inset_0_0_0_1px_#2a343a]"}`}>
                  {t("mentionName", { name: p.name ?? t("someone") })}
                </button>
              );
            })}
          </div>
        )}
        <span><PrimaryButton onClick={() => void post()} disabled={busy || !body.trim()}>{t("postComment")}</PrimaryButton></span>
      </div>
    </div>
  );
}
