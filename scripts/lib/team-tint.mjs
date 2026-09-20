export const TEAM_TINTS = {
  teal: { hue: 170.7, sourceHue: 220 },
  orange: { hue: 33.4, sourceHue: 0 },
};

const clamp = (value) => Math.max(0, Math.min(1, value));
const smooth = (value) => {
  const t = clamp(value);
  return t * t * (3 - 2 * t);
};

/**
 * Paired authored skins identify team paint without recolouring shared ice,
 * gold, skin, or metal. Keep source value, saturation, alpha and pixel layout.
 * The soft edge accommodates antialiasing and small ETC1S compression errors.
 */
export function tintTeamPixels(blue, red, team) {
  if (blue.length !== red.length || blue.length % 4 !== 0) {
    throw new Error('Team skins must have matching RGBA buffers');
  }
  if (!Object.hasOwn(TEAM_TINTS, team)) throw new Error(`Unknown derived team: ${team}`);
  const tint = TEAM_TINTS[team];
  const source = team === 'teal' ? blue : red;
  const result = new Uint8Array(source);
  let changedPixels = 0;
  for (let i = 0; i < source.length; i += 4) {
    if (source[i + 3] === 0) continue;
    // Both channels must change in the authored blue -> red direction. Shared
    // coloured details and neutral ETC1S noise therefore do not become paint.
    const redChange = red[i] - blue[i];
    const blueChange = blue[i + 2] - red[i + 2];
    const blueChroma = blue[i + 2] - blue[i];
    const redChroma = red[i] - red[i + 2];
    const weight =
      smooth((redChange + blueChange - 16) / 40) *
      smooth((Math.min(redChange, blueChange) + 8) / 14) *
      smooth((Math.max(blueChroma, redChroma) - 5) / 19);
    if (weight === 0) continue;
    const [hue, saturation, value] = rgbToHSV(source[i], source[i + 1], source[i + 2]);
    const offset = ((hue - tint.sourceHue + 540) % 360) - 180;
    // Retain subtle cool/warm paint variation without drifting out of the team
    // colour family, including the strong cyan highlights in the blue source.
    const rgb = hsvToRGB(tint.hue + Math.max(-12, Math.min(12, offset * 0.22)), saturation, value);
    for (let c = 0; c < 3; c++)
      result[i + c] = Math.round(source[i + c] + (rgb[c] - source[i + c]) * weight);
    changedPixels++;
  }
  return { data: result, changedPixels };
}

function rgbToHSV(r, g, b) {
  const max = Math.max(r, g, b);
  const min = Math.min(r, g, b);
  const delta = max - min;
  let hue = 0;
  if (delta) {
    if (max === r) hue = ((g - b) / delta) % 6;
    else if (max === g) hue = (b - r) / delta + 2;
    else hue = (r - g) / delta + 4;
    hue = (hue * 60 + 360) % 360;
  }
  return [hue, max ? delta / max : 0, max];
}

function hsvToRGB(hue, saturation, value) {
  const h = (((hue % 360) + 360) % 360) / 60;
  const chroma = value * saturation;
  const x = chroma * (1 - Math.abs((h % 2) - 1));
  const rgb =
    h < 1
      ? [chroma, x, 0]
      : h < 2
        ? [x, chroma, 0]
        : h < 3
          ? [0, chroma, x]
          : h < 4
            ? [0, x, chroma]
            : h < 5
              ? [x, 0, chroma]
              : [chroma, 0, x];
  return rgb.map((channel) => channel + value - chroma);
}
