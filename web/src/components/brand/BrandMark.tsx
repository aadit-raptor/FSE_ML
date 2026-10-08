import mark from "./mark.json";

export type BrandTone = keyof typeof mark.tones;

type Tone = { start: string; dip: string; rise: string; end: string; dipOutline?: boolean; dipOpacity?: number };

/**
 * The Variater mark (a waterfall that dips, then climbs), drawn from
 * mark.json so the app and every exported icon are the same shape.
 * Decorative: the brand name always sits next to it as text.
 */
export function BrandMark({ tone = "dark", className }: { tone?: BrandTone; className?: string }) {
  const colours: Tone = mark.tones[tone];
  return (
    <svg viewBox={`0 0 ${mark.width} ${mark.height}`} className={className} aria-hidden focusable="false">
      {mark.bars.map((bar, i) => {
        const fill = colours[bar.role as "start" | "dip" | "rise" | "end"];
        const outline = bar.role === "dip" && colours.dipOutline;
        // An outlined bar is inset by half its stroke so it keeps the same size
        const inset = outline ? 1.5 : 0;
        return (
          <rect
            key={i}
            x={bar.x + inset}
            y={bar.y + inset}
            width={bar.w - 2 * inset}
            height={bar.h - 2 * inset}
            rx={mark.radius}
            fill={outline ? "none" : fill}
            stroke={outline ? fill : undefined}
            strokeWidth={outline ? 3 : undefined}
            fillOpacity={bar.role === "dip" ? colours.dipOpacity : undefined}
          />
        );
      })}
    </svg>
  );
}

/** The mark beside the brand name, sized to the name's capital height. */
export function BrandLockup({ name }: { name: string }) {
  return (
    <span className="type-brand inline-flex items-baseline whitespace-nowrap">
      <BrandMark className="me-[0.35em] h-[0.72em] w-[0.956em] flex-none self-baseline" />
      {name}
    </span>
  );
}
