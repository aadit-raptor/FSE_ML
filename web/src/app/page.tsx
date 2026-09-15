import { redirect } from "next/navigation";

import { DEFAULT_HREF } from "@/lib/nav";

export default function Home() {
  redirect(DEFAULT_HREF);
}
