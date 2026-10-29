"""The in-app help articles (PLAN.md 7.1), made from the written methodology.

    python -m ops.help_articles          writes web/src/lib/help/articles.json and index.json

Each ``##`` section of ``docs/methodology.md`` is one article, so the help says exactly what the
methodology says and changes when it does: ``tests/test_onboarding.py`` fails when the committed file
is not what this writes. The Markdown is turned into a few kinds of block (heading, paragraph, list,
code, table) holding inline pieces (text, code, bold, italic, link) that the web app draws with its
own components (``components/help/``), so no Markdown library and no HTML from a file reach the page.
Each article's one-line summary is written for people in ``web/messages/en.json`` (``help.summary_<slug>``).

Links: one to another section becomes a link to its article (and the heading inside it), one to a
file in the repository a link to it on GitHub (the repository is public), and one to a section
the help leaves out plain text.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "docs" / "methodology.md"
OUT = ROOT / "web" / "src" / "lib" / "help" / "articles.json"
#: Each article's address and title alone, for the help list and search (the articles load only on their own page)
INDEX = ROOT / "web" / "src" / "lib" / "help" / "index.json"
REPOSITORY = "https://github.com/aadit-raptor/FSE_ML"
#: Sections that are not for someone using the app: the table of contents (the help lists the
#: articles itself) and the list of functions that change no number
LEFT_OUT = {"Contents", "Appendix: functions that affect no number"}

INLINE = re.compile(r"\[([^\]]+)\]\(([^)\s]+)\)|`([^`]+)`|\*\*([^*]+)\*\*|(?<![\w*])\*([^*\s][^*]*?)\*(?![\w*])")


def anchor(heading: str) -> str:
    """GitHub's id for a heading: lower case, punctuation dropped, spaces to hyphens."""
    text = re.sub(r"[`*]", "", heading).strip().lower()
    text = re.sub(r"[^\w\- ]", "", text)
    return text.replace(" ", "-")


def article_slug(heading: str) -> str:
    """An article's address: its heading without the section number ("3. The deal model" -> the-deal-model)."""
    return anchor(re.sub(r"^\d+\.\s*", "", heading))


def sections(text: str) -> list[tuple[str, list[str]]]:
    found: list[tuple[str, list[str]]] = []
    for line in text.splitlines():
        if line.startswith("## "):
            found.append((line[3:].strip(), []))
        elif found:
            found[-1][1].append(line)
    return found


def anchors(found: list[tuple[str, list[str]]]) -> dict[str, tuple[str, str | None]]:
    """Every heading's GitHub id -> (the article it is in, the heading's id inside it, None for the article itself)."""
    where: dict[str, tuple[str, str | None]] = {}
    for title, lines in found:
        if title in LEFT_OUT:
            continue
        slug = article_slug(title)
        where[anchor(title)] = (slug, None)
        for line in lines:
            m = re.match(r"^(#{3,6})\s+(.*)", line)
            if m:
                where[anchor(m.group(2))] = (slug, anchor(m.group(2)))
    return where


def link(text: str, href: str, where: dict[str, tuple[str, str | None]]) -> dict:
    shown = text.replace("`", "")
    code = text.startswith("`") and text.endswith("`")
    if href.startswith("#"):
        target = where.get(href[1:])
        if not target:
            return {"text": shown, **({"code": True} if code else {})}
        return {"text": shown, "article": target[0], **({"anchor": target[1]} if target[1] else {}), **({"code": True} if code else {})}
    if href.startswith("http"):
        return {"text": shown, "href": href, **({"code": True} if code else {})}
    path, _, fragment = href.partition("#")
    # Relative to docs/; a directory opens as a tree
    resolved = (SOURCE.parent / path).resolve().relative_to(ROOT).as_posix()
    kind = "tree" if path.endswith("/") or (ROOT / resolved).is_dir() else "blob"
    url = f"{REPOSITORY}/{kind}/main/{resolved}" + (f"#{fragment}" if fragment else "")
    return {"text": shown, "href": url, **({"code": True} if code else {})}


def inlines(text: str, where: dict[str, tuple[str, str | None]]) -> list[dict]:
    parts: list[dict] = []
    at = 0
    for m in INLINE.finditer(text):
        if m.start() > at:
            parts.append({"text": text[at : m.start()]})
        label, href, code, strong, em = m.groups()
        if label is not None:
            parts.append(link(label, href, where))
        elif code is not None:
            parts.append({"text": code, "code": True})
        elif strong is not None:
            parts.append({"text": strong.replace("`", ""), "strong": True})
        else:
            parts.append({"text": em, "em": True})
        at = m.end()
    if at < len(text):
        parts.append({"text": text[at:]})
    # Neighbouring plain pieces as one
    merged: list[dict] = []
    for p in parts:
        if merged and set(p) == {"text"} and set(merged[-1]) == {"text"}:
            merged[-1]["text"] += p["text"]
        else:
            merged.append(p)
    return merged


LIST_ITEM = re.compile(r"^\s*(?:[-*]|\d+\.)\s+")


def blocks(lines: list[str], where: dict[str, tuple[str, str | None]]) -> list[dict]:
    out: list[dict] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if not stripped or stripped == "---":
            i += 1
        elif stripped.startswith("```"):
            body = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                body.append(lines[i])
                i += 1
            out.append({"type": "code", "text": "\n".join(body)})
            i += 1
        elif m := re.match(r"^(#{3,6})\s+(.*)", line):
            out.append({"type": "heading", "level": len(m.group(1)) - 1, "id": anchor(m.group(2)), "inlines": inlines(m.group(2), where)})
            i += 1
        elif stripped.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-+:?", c) for c in cells):
                    rows.append([inlines(c, where) for c in cells])
                i += 1
            out.append({"type": "table", "header": rows[0], "rows": rows[1:]})
        elif LIST_ITEM.match(line):
            ordered = bool(re.match(r"^\s*\d+\.", line))
            items: list[str] = []
            while i < len(lines) and lines[i].strip() and not lines[i].strip().startswith(("```", "|")):
                if LIST_ITEM.match(lines[i]):
                    items.append(LIST_ITEM.sub("", lines[i], count=1).strip())
                elif lines[i].startswith((" ", "\t")):
                    items[-1] += " " + lines[i].strip()
                else:
                    break
                i += 1
            out.append({"type": "list", "ordered": ordered, "items": [inlines(t, where) for t in items]})
        else:
            para = []
            while (
                i < len(lines) and lines[i].strip() and not lines[i].strip().startswith(("```", "|", "#"))
                and not LIST_ITEM.match(lines[i]) and lines[i].strip() != "---"
            ):
                para.append(lines[i].strip())
                i += 1
            out.append({"type": "paragraph", "inlines": inlines(" ".join(para), where)})
    return out


def build() -> dict:
    found = sections(SOURCE.read_text(encoding="utf-8"))
    where = anchors(found)
    articles = []
    for title, lines in found:
        if title in LEFT_OUT:
            continue
        b = blocks(lines, where)
        articles.append({"slug": article_slug(title), "title": title, "blocks": b})
    return {"source": "docs/methodology.md", "articles": articles}


def index(built: dict) -> dict:
    return {"articles": [{"slug": a["slug"], "title": a["title"]} for a in built["articles"]]}


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    built = build()
    OUT.write_text(json.dumps(built, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    INDEX.write_text(json.dumps(index(built), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)} and {INDEX.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
