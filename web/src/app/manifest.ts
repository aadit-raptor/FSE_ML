import type { MetadataRoute } from "next";

import { APP_BRAND, APP_DESCRIPTION } from "@/lib/i18n/titles";

/** What a phone shows when the site is added to its home screen. */
export default function manifest(): MetadataRoute.Manifest {
  return {
    name: APP_BRAND,
    short_name: APP_BRAND,
    description: APP_DESCRIPTION,
    start_url: "/",
    display: "browser",
    background_color: "#0C1012",
    theme_color: "#0C1012",
    icons: [
      { src: "/brand/icon-192.png", sizes: "192x192", type: "image/png" },
      { src: "/brand/icon-512.png", sizes: "512x512", type: "image/png" },
      { src: "/brand/icon-maskable-512.png", sizes: "512x512", type: "image/png", purpose: "maskable" },
    ],
  };
}
