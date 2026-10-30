"use client";

import Link from "next/link";
import { useTranslations } from "next-intl";
import { useCallback, useEffect, useState } from "react";

import { Notice, PrimaryButton, SecondaryButton } from "@/components/ui/Screen";
import { TipLabel, TipText, useTip, useTips } from "@/components/ui/Tip";
import { formatDateTime } from "@/lib/locale";
import {
  createInvite, createTeam, deleteTeam, getTeam, type Invite, inviteLink, listInvites, listTeams, problemKey,
  removeMember, renameTeam, type Role, setRole, type TeamDetail, type TeamList, withdrawInvite,
} from "@/lib/teams";

const FIELD = "border border-line bg-field px-2 py-1.5 font-mono text-[11px] text-ink outline-none focus:border-accent";
const ROLES: Role[] = ["owner", "editor", "viewer"];

/**
 * Teams (PLAN.md 7.2): the caller's teams; for one team, its people and their roles, invitation links (owners),
 * renaming, leaving and deleting it. People are named by the sign-in provider through the API, never stored.
 */
export function TeamsScreen() {
  const t = useTranslations("teams");
  const nameTip = useTip("teamName");
  const [list, setList] = useState<TeamList | null>(null);
  const [open, setOpen] = useState<string | null>(null);
  const [name, setName] = useState("");
  const [problem, setProblem] = useState<string | null>(null);

  const show = useCallback((found: Awaited<ReturnType<typeof listTeams>>) => {
    if (!found.ok) return setProblem(found.code);
    setList(found.data);
    setOpen((now) => now ?? found.data.teams[0]?.id ?? null);
  }, []);
  const reload = useCallback(async () => show(await listTeams()), [show]);
  useEffect(() => {
    let cancelled = false;
    void listTeams().then((found) => {
      if (!cancelled) show(found);
    });
    return () => {
      cancelled = true;
    };
  }, [show]);

  const create = async () => {
    const made = await createTeam(name.trim());
    if (!made.ok) return setProblem(made.code);
    setName("");
    setProblem(null);
    setOpen(made.data.id);
    await reload();
  };

  return (
    <div className="grid max-w-[880px] content-start gap-4 p-6">
      <div className="grid gap-1.5">
        <h1 className="type-result-title text-[14px]">{t("title")}</h1>
        <p className="type-body">{t("intro")}</p>
      </div>
      {problem && <Notice tone="loss" title={t("title")} role="alert">{t(problemKey(problem))}</Notice>}
      {list && list.directory === "none" && <Notice tone="attention" title={t("namesOffTitle")} role="note">{t("namesOff")}</Notice>}
      {list && list.mail === "none" && <Notice tone="attention" title={t("mailOffTitle")} role="note">{t("mailOff")}</Notice>}

      <section className="grid gap-2" aria-label={t("yourTeams")}>
        <h2 className="type-input-group">{t("yourTeams")}</h2>
        {list === null ? (
          <p className="type-step">{t("loading")}</p>
        ) : list.teams.length === 0 ? (
          <p className="type-body text-[10.5px]" data-guide="teams">{t("noTeams")}</p>
        ) : (
          <ul className="flex flex-wrap gap-1.5">
            {list.teams.map((team) => (
              <li key={team.id}>
                <button type="button" aria-pressed={open === team.id} onClick={() => setOpen(team.id)} data-team={team.name}
                  className={`type-action-secondary px-2.5 py-1.5 ${open === team.id ? "bg-raised text-bright" : "text-muted shadow-[inset_0_0_0_1px_#2a343a]"}`}>
                  {t("teamChip", { name: team.name, role: t(`role_${team.role}`), count: team.members })}
                </button>
              </li>
            ))}
          </ul>
        )}
        <div className="flex flex-wrap items-end gap-2">
          <label className="grid gap-1">
            <span className="type-input-label"><TipLabel tip={nameTip}>{t("newTeamLabel")}</TipLabel></span>
            <input aria-describedby={nameTip.id} className={FIELD} value={name} maxLength={80}
              onChange={(e) => setName(e.target.value)} />
          </label>
          <TipText tip={nameTip} />
          <PrimaryButton onClick={() => void create()} disabled={!name.trim()}>{t("createTeam")}</PrimaryButton>
        </div>
      </section>

      {open && <Team key={open} id={open} onGone={() => { setOpen(null); void reload(); }} onChange={() => void reload()}
        onProblem={setProblem} />}
      <p className="type-body text-[10.5px]">{t("shareHint")} <Link href="/deal/team" className="text-accent hover:text-ink">{t("toDealTeam")}</Link></p>
    </div>
  );
}

function Team({ id, onGone, onChange, onProblem }: {
  id: string; onGone: () => void; onChange: () => void; onProblem: (code: string) => void;
}) {
  const t = useTranslations("teams");
  const tips = useTips(["memberRole", "inviteRole", "inviteUses", "teamRename"]);
  const [team, setTeam] = useState<TeamDetail | null>(null);
  const [invites, setInvites] = useState<Invite[]>([]);
  const [made, setMade] = useState<string | null>(null);
  const [inviteRole, setInviteRole] = useState<"editor" | "viewer">("editor");
  const [uses, setUses] = useState("10");
  const usesCount = Math.max(1, Math.min(50, Math.round(Number(uses)) || 1));
  const [rename, setRename] = useState<string | null>(null);
  const [confirm, setConfirm] = useState(false);

  const show = useCallback(([found, live]: [Awaited<ReturnType<typeof getTeam>>, Awaited<ReturnType<typeof listInvites>>]) => {
    if (!found.ok) return onProblem(found.code);
    setTeam(found.data);
    if (live.ok) setInvites(live.data.invites);
  }, [onProblem]);
  // Invitations answer only an owner; anyone else's answer is a refusal, and simply not shown
  const fetchTeam = useCallback(() => Promise.all([getTeam(id), listInvites(id)]), [id]);
  const load = useCallback(async () => show(await fetchTeam()), [fetchTeam, show]);
  useEffect(() => {
    let cancelled = false;
    void fetchTeam().then((found) => {
      if (!cancelled) show(found);
    });
    return () => {
      cancelled = true;
    };
  }, [fetchTeam, show]);

  if (!team) return <p className="type-step">{t("loading")}</p>;
  const owner = team.role === "owner";
  const act = async (call: Promise<{ ok: boolean; code?: string }>, after: () => void = () => void load()) => {
    const done = await call;
    if (!done.ok) return onProblem(done.code ?? "failed");
    after();
    onChange();
  };

  return (
    <section className="grid gap-3 border-t border-line pt-3" aria-label={team.name} data-team-open={team.name}>
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h2 className="type-result-title">{team.name}</h2>
        <span className="type-body text-[10px] text-muted">{t("yourRole", { role: t(`role_${team.role}`) })}</span>
      </div>

      <table className="w-full border-collapse font-mono text-[11px]" aria-label={t("peopleTitle")}>
        <thead>
          <tr className="text-start text-muted">
            <th scope="col" className="border-b border-grid px-2 py-1 text-start font-normal">{t("colPerson")}</th>
            <th scope="col" className="border-b border-grid px-2 py-1 text-start font-normal">{t("colRole")}</th>
            <th scope="col" className="border-b border-grid px-2 py-1 text-start font-normal">{t("colJoined")}</th>
            <th scope="col" className="border-b border-grid px-2 py-1 font-normal"><span className="sr-only">{t("colActions")}</span></th>
          </tr>
        </thead>
        <tbody>
          {team.people.map((p) => (
            <tr key={p.member_id} data-member={p.name ?? ""}>
              <th scope="row" className="border-b border-grid px-2 py-1.5 text-start font-normal">
                {p.you ? t("you", { name: p.name ?? t("someone") }) : (p.name ?? t("someone"))}
              </th>
              <td className="border-b border-grid px-2 py-1">
                {owner ? (
                  <select aria-label={t("roleOf", { name: p.name ?? t("someone") })} aria-describedby={tips.memberRole.id}
                    className={FIELD} value={p.role}
                    onChange={(e) => void act(setRole(id, p.member_id, e.target.value as Role))}>
                    {ROLES.map((r) => <option key={r} value={r}>{t(`role_${r}`)}</option>)}
                  </select>
                ) : t(`role_${p.role}`)}
              </td>
              <td className="border-b border-grid px-2 py-1.5 text-muted">{formatDateTime(new Date(p.joined_at), { dateStyle: "medium" })}</td>
              <td className="border-b border-grid px-2 py-1 text-end">
                {p.you ? (
                  <SecondaryButton onClick={() => void act(removeMember(id, p.member_id), onGone)}>{t("leave")}</SecondaryButton>
                ) : owner ? (
                  <SecondaryButton onClick={() => void act(removeMember(id, p.member_id))}>{t("remove")}</SecondaryButton>
                ) : null}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <TipText tip={tips.memberRole} />

      {owner && (
        <div className="grid gap-2" data-invites>
          <h3 className="type-input-group">{t("inviteTitle")}</h3>
          <p className="type-body text-[10.5px]">{t("inviteIntro")}</p>
          <div className="flex flex-wrap items-end gap-2">
            <label className="grid gap-1">
              <span className="type-input-label"><TipLabel tip={tips.inviteRole}>{t("inviteRoleLabel")}</TipLabel></span>
              <select aria-describedby={tips.inviteRole.id} className={FIELD} value={inviteRole}
                onChange={(e) => setInviteRole(e.target.value as "editor" | "viewer")}>
                <option value="editor">{t("role_editor")}</option>
                <option value="viewer">{t("role_viewer")}</option>
              </select>
            </label>
            <label className="grid gap-1">
              <span className="type-input-label"><TipLabel tip={tips.inviteUses}>{t("inviteUsesLabel")}</TipLabel></span>
              <input aria-describedby={tips.inviteUses.id} type="number" min={1} max={50} className={`${FIELD} w-20`}
                value={uses} onChange={(e) => setUses(e.target.value)} />
            </label>
            <TipText tip={tips.inviteRole} />
            <TipText tip={tips.inviteUses} />
            <PrimaryButton onClick={() => void act(createInvite(id, inviteRole, usesCount).then((r) => {
              if (r.ok) setMade(inviteLink(r.data.token ?? ""));
              return r;
            }))}>{t("makeInvite")}</PrimaryButton>
          </div>
          {made && (
            <Notice tone="info" title={t("inviteMadeTitle")} role="status"
              actions={<SecondaryButton onClick={() => void navigator.clipboard?.writeText(made)}>{t("copy")}</SecondaryButton>}>
              <span className="break-all font-mono" data-invite-link>{made}</span>
            </Notice>
          )}
          {invites.length > 0 && (
            <ul className="grid gap-1" aria-label={t("liveInvites")}>
              {invites.map((i) => (
                <li key={i.id} className="type-body flex flex-wrap items-center justify-between gap-2 text-[10.5px]">
                  <span>{t("inviteRow", { role: t(`role_${i.role}`), uses: i.uses_left,
                    until: formatDateTime(new Date(i.expires_at)) })}</span>
                  <SecondaryButton onClick={() => void act(withdrawInvite(id, i.id))}>{t("withdraw")}</SecondaryButton>
                </li>
              ))}
            </ul>
          )}
        </div>
      )}

      {owner && (
        <div className="flex flex-wrap items-end gap-2">
          <label className="grid gap-1">
            <span className="type-input-label"><TipLabel tip={tips.teamRename}>{t("renameLabel")}</TipLabel></span>
            <input aria-describedby={tips.teamRename.id} className={FIELD} maxLength={80} value={rename ?? team.name}
              onChange={(e) => setRename(e.target.value)} />
          </label>
          <TipText tip={tips.teamRename} />
          <SecondaryButton disabled={!rename?.trim() || rename.trim() === team.name}
            onClick={() => void act(renameTeam(id, (rename ?? "").trim()), () => { setRename(null); void load(); })}>
            {t("rename")}
          </SecondaryButton>
          {confirm ? (
            <>
              <span className="type-body self-center text-loss">{t("deleteTeamForGood")}</span>
              <SecondaryButton onClick={() => void act(deleteTeam(id), onGone)}>{t("confirmDelete")}</SecondaryButton>
              <SecondaryButton onClick={() => setConfirm(false)}>{t("cancel")}</SecondaryButton>
            </>
          ) : (
            <SecondaryButton onClick={() => setConfirm(true)}>{t("deleteTeam")}</SecondaryButton>
          )}
        </div>
      )}
    </section>
  );
}
