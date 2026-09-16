"use client";

import { useRouter } from "next/navigation";
import { useMemo, useState } from "react";

import { useSession } from "@/components/auth/AuthProvider";
import { type Profile, useProfile } from "@/components/auth/ProfileProvider";
import { Notice, PrimaryButton, SecondaryButton } from "@/components/ui/Screen";
import { AFTER_SIGN_IN } from "@/lib/auth/mode";

/**
 * The account screen (PLAN.md 1.4): who is signed in, and the four answers
 * every figure in the app depends on -- country, currency, locale and time
 * zone. A new account is sent here by the shell and can't reach the deal
 * screens until it has answered.
 *
 * The lists come from the browser's own CLDR data (Intl), not a list of
 * "supported" countries: any country and any currency must work (PLAN.md
 * guiding principle 2).
 */

const FALLBACK_CURRENCIES = ["USD", "EUR", "GBP", "JPY", "CNY", "INR", "BRL", "ZAR", "AUD", "CAD"];
const FALLBACK_ZONES = ["UTC", "Europe/London", "America/New_York", "Asia/Tokyo", "Asia/Kolkata"];

function supportedValues(key: "currency" | "timeZone", fallback: string[]): string[] {
  const intl = Intl as typeof Intl & { supportedValuesOf?: (k: string) => string[] };
  try {
    const values = intl.supportedValuesOf?.(key);
    return values && values.length ? values : fallback;
  } catch {
    return fallback;
  }
}

/** Every ISO 3166-1 alpha-2 code the browser can name, with its name. */
function countries(): { code: string; name: string }[] {
  const names = new Intl.DisplayNames(undefined, { type: "region", fallback: "code" });
  const out: { code: string; name: string }[] = [];
  for (let first = 65; first <= 90; first++) {
    for (let second = 65; second <= 90; second++) {
      const code = String.fromCharCode(first, second);
      let name: string;
      try {
        name = names.of(code) ?? code;
      } catch {
        continue;
      }
      // An unassigned code comes back as itself
      if (name !== code) out.push({ code, name });
    }
  }
  return out.sort((a, b) => a.name.localeCompare(b.name));
}

/** What this browser suggests, used only to prefill a new account's answers. */
function suggested(): Profile {
  const resolved = Intl.DateTimeFormat().resolvedOptions();
  const language = typeof navigator === "undefined" ? "en" : navigator.language;
  let region = "";
  try {
    region = new Intl.Locale(language).region ?? "";
  } catch {
    region = "";
  }
  return {
    country: region,
    preferred_currency: "",  // no browser answer for this: the account chooses
    locale: language,
    time_zone: resolved.timeZone || "UTC",
  };
}

export function AccountScreen() {
  const { label, mode, signOut } = useSession();
  const { profile, subject, loaded, needsProfile, save, error } = useProfile();
  const router = useRouter();

  // Edits sit on top of what the account already has (or, for a new account,
  // what the browser suggests), so nothing has to be copied into state when
  // the API answers
  const [edits, setEdits] = useState<Partial<Profile>>({});
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  const countryList = useMemo(() => countries(), []);
  const currencyList = useMemo(() => supportedValues("currency", FALLBACK_CURRENCIES), []);
  const zoneList = useMemo(() => supportedValues("timeZone", FALLBACK_ZONES), []);
  const browser = useMemo(() => suggested(), []);

  if (!loaded) {
    return (
      <div className="grid h-full place-items-center">
        <p className="type-step" role="status">Loading your account</p>
      </div>
    );
  }

  const draft: Profile = { ...(profile ?? browser), ...edits };

  const set = (key: keyof Profile) => (value: string) => {
    setEdits({ ...edits, [key]: value });
    setSaved(false);
  };
  const complete = Object.values(draft).every((v) => v.trim() !== "");
  const first = needsProfile && !profile;

  return (
    <div className="grid max-w-[720px] content-start gap-4 p-6">
      <div className="grid gap-1.5">
        <h1 className="type-result-title text-[14px]">{first ? "Finish setting up your account" : "Account"}</h1>
        <p className="type-body">
          {first
            ? "Every figure in the app is shown in your currency, your number format and your time zone, and defaults are chosen for where you work. Tell us once; you can change it here any time."
            : "How figures are shown, and which defaults you start from."}
        </p>
      </div>

      {error && (
        <Notice tone="loss" title="Not saved" role="alert">
          {error}
        </Notice>
      )}

      <dl className="grid grid-cols-[120px_1fr] gap-x-4 gap-y-1 border border-line bg-panel px-3 py-2.5">
        <dt className="type-input-group">Signed in as</dt>
        <dd className="font-mono text-[11px] text-ink" data-account="label">{label ?? "—"}</dd>
        <dt className="type-input-group">Account id</dt>
        <dd className="font-mono text-[11px] text-muted" data-account="subject">{subject ?? "—"}</dd>
        <dt className="type-input-group">Sign-in</dt>
        <dd className="font-mono text-[11px] text-muted">{mode === "clerk" ? "Clerk" : "Development (local only)"}</dd>
      </dl>

      <div className="grid grid-cols-2 gap-3">
        <label className="grid gap-1.5">
          <span className="type-input-group">Country</span>
          <select
            aria-label="Country"
            value={draft.country}
            onChange={(e) => set("country")(e.target.value)}
            className="border border-line bg-field px-2 py-1.5 font-mono text-[11px] text-ink outline-none focus:border-accent"
          >
            <option value="">Choose a country</option>
            {countryList.map((c) => (
              <option key={c.code} value={c.code}>
                {c.name} ({c.code})
              </option>
            ))}
          </select>
        </label>

        <label className="grid gap-1.5">
          <span className="type-input-group">Currency</span>
          <select
            aria-label="Currency"
            value={draft.preferred_currency}
            onChange={(e) => set("preferred_currency")(e.target.value)}
            className="border border-line bg-field px-2 py-1.5 font-mono text-[11px] text-ink outline-none focus:border-accent"
          >
            <option value="">Choose a currency</option>
            {currencyList.map((code) => (
              <option key={code} value={code}>
                {code}
              </option>
            ))}
          </select>
        </label>

        <label className="grid gap-1.5">
          <span className="type-input-group">Number and date format</span>
          <input
            aria-label="Number and date format"
            list="locale-options"
            value={draft.locale}
            autoComplete="off"
            onChange={(e) => set("locale")(e.target.value)}
            className="border border-line bg-field px-2 py-1.5 font-mono text-[11px] text-ink outline-none focus:border-accent"
          />
          <datalist id="locale-options">
            {(typeof navigator === "undefined" ? [] : navigator.languages).map((tag) => (
              <option key={tag} value={tag} />
            ))}
            {["en-GB", "en-US", "de-DE", "fr-FR", "pt-BR", "ja-JP", "hi-IN"].map((tag) => (
              <option key={tag} value={tag} />
            ))}
          </datalist>
        </label>

        <label className="grid gap-1.5">
          <span className="type-input-group">Time zone</span>
          <select
            aria-label="Time zone"
            value={draft.time_zone}
            onChange={(e) => set("time_zone")(e.target.value)}
            className="border border-line bg-field px-2 py-1.5 font-mono text-[11px] text-ink outline-none focus:border-accent"
          >
            {!zoneList.includes(draft.time_zone) && draft.time_zone && (
              <option value={draft.time_zone}>{draft.time_zone}</option>
            )}
            {zoneList.map((zone) => (
              <option key={zone} value={zone}>
                {zone}
              </option>
            ))}
          </select>
        </label>
      </div>

      <p className="type-body" data-account="preview">
        {/* Proof the answers are in use: the same amount, shown their way */}
        {`1,234,567.89 in ${draft.preferred_currency || "your currency"} reads `}
        <b className="font-mono text-ink">{formatExample(draft)}</b>
        {` · today is ${formatToday(draft)}`}
      </p>

      <div className="flex items-center gap-3">
        <PrimaryButton
          disabled={!complete || saving}
          onClick={async () => {
            setSaving(true);
            const result = await save({
              ...draft,
              country: draft.country.toUpperCase(),
              preferred_currency: draft.preferred_currency.toUpperCase(),
            });
            setSaving(false);
            setSaved(result.ok);
            if (result.ok && first) router.replace(AFTER_SIGN_IN);
          }}
        >
          {saving ? "Saving" : first ? "Save and start" : "Save"}
        </PrimaryButton>
        {saved && (
          <span role="status" className="font-mono text-[10px] text-gain" data-account="saved">
            Saved
          </span>
        )}
        <div className="flex-1" />
        <SecondaryButton onClick={() => void signOut()}>Sign out</SecondaryButton>
      </div>
    </div>
  );
}

function formatExample(draft: Profile): string {
  if (!draft.preferred_currency || !draft.locale) return "—";
  try {
    return new Intl.NumberFormat(draft.locale, { style: "currency", currency: draft.preferred_currency })
      .format(1234567.89);
  } catch {
    return "—";
  }
}

function formatToday(draft: Profile): string {
  try {
    return new Intl.DateTimeFormat(draft.locale, { dateStyle: "medium", timeZone: draft.time_zone })
      .format(new Date());
  } catch {
    return "—";
  }
}
