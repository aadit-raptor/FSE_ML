import type { Metadata } from "next";
import { notFound } from "next/navigation";

import { type Article, HelpArticle } from "@/components/help/Help";
import articles from "@/lib/help/articles.json";
import { HELP_TITLE } from "@/lib/i18n/titles";

/**
 * One help article (PLAN.md 7.1). The articles are read here, on the server, and only the one asked
 * for is sent to the browser: the whole methodology would be a large download for every page.
 */
type Props = PageProps<"/help/[slug]">;

const find = (slug: string) => (articles.articles as Article[]).find((a) => a.slug === slug);

export async function generateMetadata({ params }: Props): Promise<Metadata> {
  const { slug } = await params;
  return { title: [find(slug)?.title ?? HELP_TITLE, HELP_TITLE].join(" · ") };
}

export default async function Page({ params }: Props) {
  const { slug } = await params;
  const article = find(slug);
  if (!article) notFound();
  return <HelpArticle article={article} />;
}
