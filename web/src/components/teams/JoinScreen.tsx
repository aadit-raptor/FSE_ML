"use client";

import Link from "next/link";
import { useTranslations } from "next-intl";
import { useEffect, useState } from "react";

import { Notice } from "@/components/ui/Screen";
import { joinTeam, problemKey, type TeamSummary } from "@/lib/teams";

/**
 * Joining a team by an invitation link (PLAN.md 7.2): `/teams/join#<token>`. The token is after `#`, so the
 * browser never sends it in a request for the page; it goes to the API once, in the body of the join.
 */
export function JoinScreen() {
  const t = useTranslations("teams");
  const [joined, setJoined] = useState<TeamSummary | null>(null);
  const [problem, setProblem] = useState<string | null>(null);

  useEffect(() => {
    const token = window.location.hash.slice(1);
    let cancelled = false;
    void (token ? joinTeam(token) : Promise.resolve({ ok: false as const, code: "invite_invalid" })).then((done) => {
      if (cancelled) return;
      if (done.ok) {
        setJoined(done.data);
        // The link has done its work: leave nothing reusable in the address bar or the history
        window.history.replaceState(null, "", window.location.pathname);
      } else {
        setProblem(done.code);
      }
    });
    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <div className="grid max-w-[640px] content-start gap-4 p-6">
      <h1 className="type-result-title text-[14px]">{t("joinTitle")}</h1>
      {!joined && !problem && <p className="type-step" role="status">{t("joining")}</p>}
      {problem && <Notice tone="loss" title={t("joinTitle")} role="alert">{t(problemKey(problem))}</Notice>}
      {joined && (
        <Notice tone="info" title={t("joinedTitle")} role="status">
          {t("joined", { name: joined.name, role: t(`role_${joined.role}`) })}
        </Notice>
      )}
      <p className="type-body"><Link href="/teams" className="text-accent hover:text-ink">{t("toTeams")}</Link></p>
    </div>
  );
}
