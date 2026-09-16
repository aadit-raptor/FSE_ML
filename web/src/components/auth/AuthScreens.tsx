"use client";

import { SignIn, SignUp } from "@clerk/nextjs";
import { useRouter, useSearchParams } from "next/navigation";
import { useState } from "react";

import { useSession } from "@/components/auth/AuthProvider";
import { PrimaryButton } from "@/components/ui/Screen";
import { DEV_USER, isDevUserName } from "@/lib/auth/dev";
import { AFTER_SIGN_IN, AFTER_SIGN_UP, AUTH_MODE } from "@/lib/auth/mode";

/**
 * Sign-in and sign-up (PLAN.md 1.4).
 *
 * With a Clerk instance these are Clerk's own screens -- email, password
 * reset and Google -- dressed in the Tape palette. Without one (local runs,
 * browser tests) the same routes offer the development sign-in, which the API
 * only accepts outside production (api/auth.py).
 */

// Clerk renders inside our canvas: square corners, Tape colours, Michroma
const APPEARANCE = {
  variables: {
    colorPrimary: "#62b6cb",
    colorBackground: "#12181b",
    colorText: "#d5dde1",
    colorTextSecondary: "#7f8c94",
    colorInputBackground: "#0c1114",
    colorInputText: "#eef3f5",
    colorDanger: "#d8665a",
    colorSuccess: "#58b28a",
    colorWarning: "#d9a54a",
    borderRadius: "0px",
    fontSize: "13px",
  },
  elements: {
    card: { border: "1px solid #222b30", boxShadow: "none" },
    headerTitle: { letterSpacing: "0.06em", textTransform: "uppercase" as const },
    formButtonPrimary: { letterSpacing: "0.12em", textTransform: "uppercase" as const },
    footer: { background: "transparent" },
  },
};

function Frame({ title, hint, children }: { title: string; hint?: string; children: React.ReactNode }) {
  return (
    <div className="grid h-full place-items-center p-6">
      <div className="grid w-full max-w-[420px] gap-4">
        <div className="grid gap-1.5">
          <h1 className="type-result-title text-[14px]">{title}</h1>
          {hint && <p className="type-body">{hint}</p>}
        </div>
        {children}
      </div>
    </div>
  );
}

/** The development sign-in: pick a name, and the API sees `dev:<name>`. */
function DeveloperSignIn({ heading }: { heading: string }) {
  const { signInAsDeveloper } = useSession();
  const router = useRouter();
  const params = useSearchParams();
  const [name, setName] = useState(DEV_USER);
  const valid = isDevUserName(name);

  return (
    <Frame
      title={heading}
      hint="This build has no Clerk instance, so it signs in as a development user. Deployed copies use real accounts; the API refuses development sign-in in production."
    >
      <label className="grid gap-1.5">
        <span className="type-input-group">Development user</span>
        <input
          aria-label="Development user"
          value={name}
          autoComplete="off"
          onChange={(e) => setName(e.target.value)}
          className="border border-line bg-field px-2 py-1.5 font-mono text-[12px] text-ink outline-none focus:border-accent"
        />
      </label>
      {!valid && (
        <p role="alert" className="font-mono text-[10px] text-loss">
          Letters, digits, hyphen and underscore only.
        </p>
      )}
      <div>
        <PrimaryButton
          disabled={!valid}
          onClick={() => {
            signInAsDeveloper?.(name);
            const next = params.get("next");
            router.replace(next && next.startsWith("/") ? next : AFTER_SIGN_IN);
          }}
        >
          Sign in
        </PrimaryButton>
      </div>
    </Frame>
  );
}

export function SignInScreen() {
  if (AUTH_MODE !== "clerk") return <DeveloperSignIn heading="Sign in" />;
  return (
    <Frame title="Sign in">
      <SignIn
        routing="path"
        path="/sign-in"
        signUpUrl="/sign-up"
        fallbackRedirectUrl={AFTER_SIGN_IN}
        appearance={APPEARANCE}
      />
    </Frame>
  );
}

export function SignUpScreen() {
  if (AUTH_MODE !== "clerk") return <DeveloperSignIn heading="Sign up" />;
  return (
    <Frame title="Create an account">
      <SignUp
        routing="path"
        path="/sign-up"
        signInUrl="/sign-in"
        // New accounts land on the account screen, which asks for country,
        // currency, locale and time zone before the deal screens open
        forceRedirectUrl={AFTER_SIGN_UP}
        appearance={APPEARANCE}
      />
    </Frame>
  );
}
