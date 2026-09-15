"use client";

import { useEffect, useState } from "react";

import { api, type Schemas } from "@/lib/api/client";

export type Capabilities = Schemas["Capabilities"];

let cached: Promise<Capabilities | null> | null = null;

/** Which optional ML features this API server can run (fetched once per page load). */
export function useCapabilities(): Capabilities | null {
  const [caps, setCaps] = useState<Capabilities | null>(null);
  useEffect(() => {
    let alive = true;
    cached ??= api
      .GET("/api/capabilities")
      .then(({ data }) => data ?? null)
      .catch(() => null);
    cached.then((c) => alive && setCaps(c));
    return () => {
      alive = false;
    };
  }, []);
  return caps;
}
