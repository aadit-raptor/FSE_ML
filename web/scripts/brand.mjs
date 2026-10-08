/**
 * Draws every logo file from src/components/brand/mark.json.
 *
 *   PW_CHANNEL=msedge node scripts/brand.mjs      (from web/)
 *
 * Writes the browser icons Next.js serves from src/app (icon.svg,
 * favicon.ico, apple-icon.png, opengraph-image.png) and the package in
 * public/brand (app icons, logos for emails, Google's consent screen, Clerk,
 * avatars, one-colour marks). SVG marks are written directly; PNGs are
 * rendered in a browser so the wordmark uses the real Orbitron 900 (loaded
 * from Google Fonts while rendering). Run it after changing mark.json and
 * commit what it writes; tests/test_brand.py checks the SVGs still match.
 */
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import { chromium } from "@playwright/test";

const web = join(dirname(fileURLToPath(import.meta.url)), "..");
const mark = JSON.parse(readFileSync(join(web, "src/components/brand/mark.json"), "utf8"));
const en = JSON.parse(readFileSync(join(web, "messages/en.json"), "utf8"));
const NAME = en.app.brand.toUpperCase();
const TAGLINE = en.app.description;
const SITE = "variater.com";
const CANVAS = "#0C1012";
const LINE = "#222B30";
const CYAN = "#62B6CB";

const app = join(web, "src/app");
const out = join(web, "public/brand");
mkdirSync(out, { recursive: true });

/** The bars as SVG elements in one tone. */
function bars(toneName) {
  const tone = mark.tones[toneName];
  return mark.bars
    .map((b) => {
      const fill = tone[b.role];
      if (b.role === "dip" && tone.dipOutline) {
        return `<rect x="${b.x + 1.5}" y="${b.y + 1.5}" width="${b.w - 3}" height="${b.h - 3}" rx="${mark.radius}" fill="none" stroke="${fill}" stroke-width="3"/>`;
      }
      const opacity = b.role === "dip" && tone.dipOpacity ? ` fill-opacity="${tone.dipOpacity}"` : "";
      return `<rect x="${b.x}" y="${b.y}" width="${b.w}" height="${b.h}" rx="${mark.radius}" fill="${fill}"${opacity}/>`;
    })
    .join("");
}

/** The mark alone, transparent, tight to its bars. */
function markSvg(tone) {
  return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${mark.width} ${mark.height}" width="${mark.width * 4}" height="${mark.height * 4}">${bars(tone)}</svg>\n`;
}

/** Square with the mark centred, `fill` of the side taken by the mark's width. */
function squareSvg(tone, { background = null, fill = 0.9, corner = 0 } = {}) {
  const side = mark.width / fill;
  const x = (side - mark.width) / 2;
  const y = (side - mark.height) / 2;
  const bg = background ? `<rect width="${side}" height="${side}" rx="${side * corner}" fill="${background}"/>` : "";
  return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${side} ${side}">${bg}<g transform="translate(${x} ${y})">${bars(tone)}</g></svg>`;
}

/** The browser-tab icon: dark bars on light tabs, light bars on dark ones. */
function tabIconSvg() {
  const side = mark.width + 2;
  const y = (side - mark.height) / 2;
  const roles = ["start", "dip", "rise", "end"];
  const css = (tone) => roles.map((r) => `.${r}{fill:${mark.tones[tone][r]}}`).join("");
  const rects = mark.bars
    .map((b) => `<rect class="${b.role}" x="${b.x + 1}" y="${b.y + y}" width="${b.w}" height="${b.h}" rx="${mark.radius}"/>`)
    .join("");
  return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${side} ${side}"><style>${css("light")}@media (prefers-color-scheme: dark){${css("dark")}}</style>${rects}</svg>\n`;
}

/** The mark beside (or above) the name; `size` is the name's font size in px. */
function lockupHtml(tone, size, { stacked = false, background = "transparent", padded = true, id = "shot" } = {}) {
  const t = mark.tones[tone];
  const height = stacked ? size * 1.6 : size * 0.72; // beside the name: its capital height
  const svg = `<svg viewBox="0 0 ${mark.width} ${mark.height}" style="height:${height}px;width:${height * (mark.width / mark.height)}px;${stacked ? "" : `margin-right:${size * 0.35}px`}">${bars(tone)}</svg>`;
  // The trailing letter spacing is cancelled so the name ends at its last letter
  const word = `<span style="font:900 ${size}px/1 Orbitron;letter-spacing:0.12em;color:${t.text};margin-right:-0.12em">${NAME}</span>`;
  const padding = padded ? (stacked ? `${size * 0.5}px` : `${size * 0.4}px ${size * 0.5}px`) : "0";
  const layout = stacked ? `flex-direction:column;align-items:center;gap:${size * 0.7}px` : "align-items:baseline";
  return `<div id="${id}" style="display:inline-flex;${layout};padding:${padding};background:${background}">${svg}${word}</div>`;
}

function ogHtml() {
  return `<div id="shot" style="width:1200px;height:630px;box-sizing:border-box;background:${CANVAS};padding:88px;display:flex;flex-direction:column;justify-content:space-between;border:2px solid ${LINE}">
    <div>${lockupHtml("dark", 88, { padded: false, id: "lockup" })}</div>
    <div style="font:400 30px/1.5 Michroma;color:#B4BFC4;max-width:960px">${TAGLINE}</div>
    <div style="display:flex;justify-content:space-between;align-items:center"><span style="font:500 26px JetBrains Mono;color:${CYAN}">${SITE}</span><span style="width:240px;height:4px;background:${CYAN}"></span></div>
  </div>`;
}

const FONTS =
  '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Orbitron:wght@900&family=Michroma&family=JetBrains+Mono:wght@500&display=block">';

async function main() {
  // SVGs
  writeFileSync(join(app, "icon.svg"), tabIconSvg());
  for (const tone of ["dark", "light", "black", "white"]) writeFileSync(join(out, `mark-${tone}.svg`), markSvg(tone));

  const browser = await chromium.launch({ channel: process.env.PW_CHANNEL || undefined });
  const page = await browser.newPage();

  async function shoot(html, file, { width, height, transparent = false } = {}) {
    await page.setViewportSize({ width: width ?? 2000, height: height ?? 1200 });
    await page.setContent(
      `<!doctype html><html><head>${FONTS}<style>html,body{margin:0;background:transparent}</style></head><body>${html}</body></html>`,
    );
    await page.evaluate(() => document.fonts.ready);
    const target = width ? page : page.locator("#shot");
    const buffer = await target.screenshot({ omitBackground: transparent, scale: "css", ...(width ? { clip: { x: 0, y: 0, width, height } } : {}) });
    if (file) writeFileSync(file, buffer);
    return buffer;
  }

  const square = (tone, size, opts) =>
    shoot(`<img src="data:image/svg+xml;base64,${Buffer.from(squareSvg(tone, opts)).toString("base64")}" width="${size}" height="${size}" style="display:block">`, null, {
      width: size,
      height: size,
      // Always RGBA (the background, if any, is drawn in the SVG): Next.js
      // refuses an .ico whose PNGs have no alpha channel
      transparent: true,
    });

  // Browser and home-screen icons: an opaque dark square, so they show on any background
  const tile = { background: CANVAS, fill: 0.72 };
  writeFileSync(join(app, "apple-icon.png"), await square("dark", 180, tile));
  writeFileSync(join(out, "icon-192.png"), await square("dark", 192, tile));
  writeFileSync(join(out, "icon-512.png"), await square("dark", 512, tile));
  // Android crops a maskable icon to a circle of 80% of the side: keep the mark inside it
  writeFileSync(join(out, "icon-maskable-512.png"), await square("dark", 512, { background: CANVAS, fill: 0.5 }));
  writeFileSync(join(out, "logo-120.png"), await square("dark", 120, tile));
  writeFileSync(join(out, "avatar.png"), await square("dark", 400, { background: CANVAS, fill: 0.55 }));
  writeFileSync(join(out, "avatar-cyan.png"), await square("onCyan", 400, { background: mark.tones.onCyan.background, fill: 0.55 }));

  // favicon.ico for browsers that don't take the SVG: 16, 32 and 48 on a dark
  // square. Its rounded corners keep the PNGs RGBA (Chromium saves a fully
  // opaque screenshot as RGB, and Next.js refuses an .ico holding those)
  const icoSizes = [16, 32, 48];
  const pngs = [];
  for (const size of icoSizes) pngs.push(await square("dark", size, { background: CANVAS, fill: 0.94, corner: 0.15 }));
  writeFileSync(join(app, "favicon.ico"), ico(icoSizes, pngs));

  // Logos with the name (rendered at twice their display size)
  await shoot(lockupHtml("dark", 96), join(out, "logo-horizontal-dark.png"), { transparent: true });
  await shoot(lockupHtml("light", 96), join(out, "logo-horizontal-light.png"), { transparent: true });
  await shoot(lockupHtml("dark", 64, { stacked: true }), join(out, "logo-stacked-dark.png"), { transparent: true });
  await shoot(lockupHtml("light", 64, { stacked: true }), join(out, "logo-stacked-light.png"), { transparent: true });
  await shoot(lockupHtml("black", 96), join(out, "logo-black.png"), { transparent: true });
  await shoot(lockupHtml("white", 96), join(out, "logo-white.png"), { transparent: true });
  // Email apps show neither SVG nor transparency reliably: light, opaque, 2x of 48px tall
  await shoot(lockupHtml("light", 96, { background: "#FFFFFF" }), join(out, "logo-email.png"));

  // Link previews (WhatsApp, Slack, LinkedIn, X)
  await shoot(ogHtml(), join(app, "opengraph-image.png"), { width: 1200, height: 630 });
  writeFileSync(join(app, "opengraph-image.alt.txt"), `${en.app.brand}: ${TAGLINE}`);

  await browser.close();
  console.log("brand: wrote src/app/{icon.svg,favicon.ico,apple-icon.png,opengraph-image.png} and public/brand/");
}

/** An .ico holding PNG images (supported by every browser since IE Vista era). */
function ico(sizes, pngs) {
  const header = Buffer.alloc(6 + 16 * sizes.length);
  header.writeUInt16LE(0, 0);
  header.writeUInt16LE(1, 2);
  header.writeUInt16LE(sizes.length, 4);
  let offset = header.length;
  sizes.forEach((size, i) => {
    const entry = 6 + 16 * i;
    header.writeUInt8(size >= 256 ? 0 : size, entry);
    header.writeUInt8(size >= 256 ? 0 : size, entry + 1);
    header.writeUInt8(0, entry + 2);
    header.writeUInt8(0, entry + 3);
    header.writeUInt16LE(1, entry + 4);
    header.writeUInt16LE(32, entry + 6);
    header.writeUInt32LE(pngs[i].length, entry + 8);
    header.writeUInt32LE(offset, entry + 12);
    offset += pngs[i].length;
  });
  return Buffer.concat([header, ...pngs]);
}

await main();
