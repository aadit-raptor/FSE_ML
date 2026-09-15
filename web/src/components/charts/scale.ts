/** A rounded axis maximum and evenly spaced ticks from 0. */
export function niceScale(max: number, divisions = 4): { max: number; ticks: number[] } {
  if (!(max > 0)) return { max: 1, ticks: [0, 1] };
  const raw = max / divisions;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw) ?? 10 * mag;
  const top = step * Math.ceil(max / step);
  const ticks: number[] = [];
  for (let t = 0; t <= top + 1e-9; t += step) ticks.push(Number(t.toFixed(6)));
  return { max: top, ticks };
}
