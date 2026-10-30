"use client";

import { api, type Schemas } from "@/lib/api/client";

/**
 * Teams, sharing a deal with one, and comments (PLAN.md 7.2). People are member ids; their names come from the
 * sign-in provider through the API on each request and are never stored. A refusal is a code: `teams.problem_<code>`
 * says it in words. The mention emails' words are `email.*`, read by the API (teams/notices.py).
 *
 * i18n-keys: email.*
 * i18n-keys: teams.role_*
 * i18n-keys: teams.access_*
 * i18n-keys: teams.dealAccess_*
 * i18n-keys: teams.problem_*
 */
export type TeamSummary = Schemas["TeamSummary"];
export type TeamDetail = Schemas["TeamDetail"];
export type TeamList = Schemas["TeamList"];
export type Invite = Schemas["InviteOut"];
export type Comment = Schemas["CommentOut"];
export type DealPerson = Schemas["DealPerson"];
export type DealShare = Schemas["DealShareOut"];
export type Role = TeamSummary["role"];

/** Refusal codes with words of their own (`teams.problem_<code>`); anything else is `problem_failed`. */
const PROBLEMS = ["not_found", "invite_invalid", "not_team_owner", "not_deal_owner", "not_comment", "last_owner",
  "full_teams", "full_members", "full_invites", "full_comments", "invalid_name", "invalid_body", "invalid_mentions",
  "no_account", "no_database"];

export function problemKey(code: string): string {
  return `problem_${PROBLEMS.includes(code) ? code : "failed"}`;
}

type Result<T> = { ok: true; data: T } | { ok: false; code: string };

function codeOf(error: unknown): string {
  const detail = (error as { detail?: unknown } | undefined)?.detail;
  if (detail && typeof detail === "object" && typeof (detail as { code?: unknown }).code === "string") {
    return (detail as { code: string }).code;
  }
  return "failed";
}

type Call<T> = Promise<{ data?: T; error?: unknown; response: Response }>;

async function attempt<T>(call: Call<T>): Promise<Result<T>> { // text-ok: generics, not words
  try {
    const { data, error, response } = await call;
    if (response.ok) return { ok: true, data: data as T };
    return { ok: false, code: codeOf(error) };
  } catch {
    return { ok: false, code: "failed" };
  }
}

const team = (id: string) => ({ params: { path: { team_id: id } } });
const deal = (id: string) => ({ params: { path: { deal_id: id } } });

export const listTeams = () => attempt(api.GET("/api/teams"));
export const createTeam = (name: string) => attempt(api.POST("/api/teams", { body: { name } }));
export const getTeam = (id: string) => attempt(api.GET("/api/teams/{team_id}", team(id)));
export const renameTeam = (id: string, name: string) =>
  attempt(api.PATCH("/api/teams/{team_id}", { ...team(id), body: { name } }));
export const deleteTeam = (id: string) => attempt(api.DELETE("/api/teams/{team_id}", team(id)));
export const setRole = (id: string, member: number, role: Role) =>
  attempt(api.PUT("/api/teams/{team_id}/members/{member_id}", {
    params: { path: { team_id: id, member_id: member } }, body: { role },
  }));
export const removeMember = (id: string, member: number) =>
  attempt(api.DELETE("/api/teams/{team_id}/members/{member_id}", { params: { path: { team_id: id, member_id: member } } }));
export const listInvites = (id: string) => attempt(api.GET("/api/teams/{team_id}/invites", team(id)));
export const createInvite = (id: string, role: "editor" | "viewer", uses: number) =>
  attempt(api.POST("/api/teams/{team_id}/invites", { ...team(id), body: { role, uses } }));
export const withdrawInvite = (id: string, invite: string) =>
  attempt(api.DELETE("/api/teams/{team_id}/invites/{invite_id}", { params: { path: { team_id: id, invite_id: invite } } }));
export const joinTeam = (token: string) => attempt(api.POST("/api/teams/join", { body: { token } }));

export const dealShares = (id: string) => attempt(api.GET("/api/deals/{deal_id}/shares", deal(id)));
export const shareDeal = (id: string, teamId: string) =>
  attempt(api.PUT("/api/deals/{deal_id}/shares", { ...deal(id), body: { team_id: teamId } }));
export const unshareDeal = (id: string, teamId: string) =>
  attempt(api.DELETE("/api/deals/{deal_id}/shares/{team_id}", { params: { path: { deal_id: id, team_id: teamId } } }));
export const dealPeople = (id: string) => attempt(api.GET("/api/deals/{deal_id}/people", deal(id)));
export const listComments = (id: string) => attempt(api.GET("/api/deals/{deal_id}/comments", deal(id)));
export const addComment = (id: string, body: string, mentions: number[]) =>
  attempt(api.POST("/api/deals/{deal_id}/comments", { ...deal(id), body: { body, mentions } }));
export const deleteComment = (id: string, comment: string) =>
  attempt(api.DELETE("/api/deals/{deal_id}/comments/{comment_id}", { params: { path: { deal_id: id, comment_id: comment } } }));

/** The address an invitation's token makes: the token after `#`, so it never reaches a server's logs. */
export function inviteLink(token: string): string {
  return `${window.location.origin}/teams/join#${token}`;
}
