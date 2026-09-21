/**
 * The effect system's textures, generated rather than loaded.
 *
 * Every sprite the weapons draw — a glow, a flame puff, a smoke ball, the
 * cross-section of a beam, a shockwave ring, a scorch mark — is a small
 * greyscale mask built here into a `DataTexture` at startup.
 *
 * Two reasons it is done this way rather than with a painted sprite sheet or a
 * `CanvasTexture`. A canvas is a DOM object and `ProjectileRenderer` is
 * constructed under Node by `tests/projectiles.test.ts`, where there is no
 * `document`; and a handful of radial falloffs is not worth a download, a
 * cache-busting URL or a licence trail.
 *
 * The noise is a fixed integer hash rather than `Math.random`, so a flame puff
 * has the same shape on every machine and every reload. These are authored
 * assets that happen to be computed at load, not effects that vary per run —
 * the variation the player sees comes from how the particles move, which is
 * where variation reads as life rather than as flicker.
 *
 * Colour lives on the particle, not in the texture: every mask is white with
 * the shape in its alpha (the scorch is the one exception, being a darkening
 * rather than a light), so one texture serves every team and every weapon.
 */

import * as THREE from 'three';

/** Deterministic hash of an integer lattice point, as 0..1. */
function hash2(ix: number, iy: number): number {
  let h = Math.imul(ix, 374761393) + Math.imul(iy, 668265263);
  h = Math.imul(h ^ (h >>> 13), 1274126177);
  return ((h ^ (h >>> 16)) >>> 0) / 4294967295;
}

function ease(t: number): number {
  return t * t * (3 - 2 * t);
}

/** Bilinear value noise, smoothstepped between lattice points. */
function valueNoise(x: number, y: number): number {
  const ix = Math.floor(x);
  const iy = Math.floor(y);
  const fx = ease(x - ix);
  const fy = ease(y - iy);
  const a = hash2(ix, iy);
  const b = hash2(ix + 1, iy);
  const c = hash2(ix, iy + 1);
  const d = hash2(ix + 1, iy + 1);
  return (a + (b - a) * fx) * (1 - fy) + (c + (d - c) * fx) * fy;
}

/** Three octaves of value noise, renormalised to 0..1. */
function fbm(x: number, y: number): number {
  let sum = 0;
  let amp = 0.5;
  let norm = 0;
  let fx = x;
  let fy = y;
  for (let octave = 0; octave < 3; octave++) {
    sum += valueNoise(fx, fy) * amp;
    norm += amp;
    amp *= 0.5;
    // Irrational-ish lacunarity, so octaves do not line up into a visible grid.
    fx = fx * 2.07 + 13.1;
    fy = fy * 2.03 + 7.7;
  }
  return sum / norm;
}

function clamp01(v: number): number {
  return v < 0 ? 0 : v > 1 ? 1 : v;
}

/** Scratch texel, reused: a 128x128 build would otherwise make 16k objects. */
const texel = [0, 0, 0, 0];

/**
 * Run a shader-like function over a grid and wrap the result as a texture.
 *
 * `shade` receives pixel-centre UVs and writes 0..1 red, green, blue and alpha.
 * Mipmaps are generated because these sprites are drawn at every size from a
 * few pixels to most of the screen, and without them a distant explosion
 * sparkles.
 */
function build(
  width: number,
  height: number,
  shade: (u: number, v: number, out: number[]) => void,
): THREE.DataTexture {
  const data = new Uint8Array(width * height * 4);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      shade((x + 0.5) / width, (y + 0.5) / height, texel);
      const i = (y * width + x) * 4;
      data[i] = Math.round(clamp01(texel[0]!) * 255);
      data[i + 1] = Math.round(clamp01(texel[1]!) * 255);
      data[i + 2] = Math.round(clamp01(texel[2]!) * 255);
      data[i + 3] = Math.round(clamp01(texel[3]!) * 255);
    }
  }

  const texture = new THREE.DataTexture(data, width, height, THREE.RGBAFormat);
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.magFilter = THREE.LinearFilter;
  texture.minFilter = THREE.LinearMipmapLinearFilter;
  texture.wrapS = THREE.ClampToEdgeWrapping;
  texture.wrapT = THREE.ClampToEdgeWrapping;
  texture.generateMipmaps = true;
  texture.needsUpdate = true;
  return texture;
}

/** Distance from the sprite centre, where 1 is the edge of the inscribed disc. */
function radius(u: number, v: number): number {
  const dx = u - 0.5;
  const dy = v - 0.5;
  return Math.sqrt(dx * dx + dy * dy) * 2;
}

/**
 * The workhorse: a hot core inside a wide soft skirt.
 *
 * Two falloffs summed rather than one, because a single power curve either has
 * no centre worth calling a flash or a halo that stops dead. Used for muzzle
 * flashes, impact cores, embers and sparks.
 */
export function glowTexture(): THREE.DataTexture {
  return build(64, 64, (u, v, out) => {
    const d = radius(u, v);
    const skirt = Math.max(0, 1 - d);
    const core = Math.max(0, 1 - d * 2.6);
    out[0] = 1;
    out[1] = 1;
    out[2] = 1;
    out[3] = skirt * skirt * skirt * 0.55 + core * core * 0.95;
  });
}

/**
 * A flame puff: a blob whose edge is eaten away by noise.
 *
 * The low-frequency octave decides the silhouette and the high-frequency one
 * breaks up the interior, which is what stops a wall of fire reading as a row
 * of identical circles once a Firespout opens up.
 */
export function flameTexture(): THREE.DataTexture {
  return build(64, 64, (u, v, out) => {
    const d = radius(u, v);
    const shape = fbm(u * 3.1, v * 3.1);
    const detail = fbm(u * 8.5 + 11, v * 8.5 + 7);
    const edge = clamp01(1 - d * (0.7 + 0.62 * shape));
    out[0] = 1;
    out[1] = 1;
    out[2] = 1;
    out[3] = edge * edge * (0.55 + 0.45 * detail);
  });
}

/** A soft, rounder, lumpier puff for smoke and dust, tinted per particle. */
export function smokeTexture(): THREE.DataTexture {
  return build(64, 64, (u, v, out) => {
    const d = radius(u, v);
    const shape = fbm(u * 2.6 + 31, v * 2.6 + 17);
    const detail = fbm(u * 7 + 3, v * 7 + 5);
    const edge = clamp01(1 - d * (0.86 + 0.34 * shape));
    out[0] = 1;
    out[1] = 1;
    out[2] = 1;
    out[3] = Math.pow(edge, 1.7) * (0.6 + 0.4 * detail);
  });
}

/**
 * The cross-section of a beam, sampled across its width only.
 *
 * One row would do; four keeps the mipmap chain sane. The profile is a narrow
 * white core inside a wide halo, which is what makes a laser read as hot rather
 * than as a painted stripe — and it means a beam is one quad rather than the
 * two stacked ones this effect usually costs.
 */
export function beamTexture(): THREE.DataTexture {
  return build(64, 4, (u, _v, out) => {
    const across = Math.abs(u - 0.5) * 2;
    const halo = Math.pow(Math.max(0, 1 - across), 2.2);
    const core = Math.pow(Math.max(0, 1 - across * 3.4), 2);
    out[0] = 1;
    out[1] = 1;
    out[2] = 1;
    out[3] = Math.min(1, halo * 0.42 + core);
  });
}

/**
 * A shockwave ring: a bright band with a faint wash inside it.
 *
 * Deliberately not a perfect circle. A few percent of angular wobble is the
 * difference between a blast and a piece of UI, and at the radii these are
 * drawn at it is visible without being noticeable.
 */
export function ringTexture(): THREE.DataTexture {
  return build(128, 128, (u, v, out) => {
    const dx = u - 0.5;
    const dy = v - 0.5;
    const d = Math.sqrt(dx * dx + dy * dy) * 2;
    const angle = Math.atan2(dy, dx);
    const wobble = 1 + (fbm(Math.cos(angle) * 2 + 5, Math.sin(angle) * 2 + 9) - 0.5) * 0.1;
    const r = d / wobble;
    const offset = (r - 0.8) / 0.1;
    const band = Math.exp(-offset * offset);
    const wash = Math.pow(Math.max(0, 1 - r / 0.8), 3) * 0.1;
    out[0] = 1;
    out[1] = 1;
    out[2] = 1;
    out[3] = r > 1.02 ? 0 : band + wash;
  });
}

/**
 * A scorch mark, and the one texture that carries colour of its own.
 *
 * Everything else here is added to the frame; this is blended over it, so the
 * near-black is the effect. Warm rather than neutral, because a burn on this
 * palette's brown-grey ground reads as a hole in the terrain if it is pure
 * grey.
 */
export function scorchTexture(): THREE.DataTexture {
  return build(64, 64, (u, v, out) => {
    const d = radius(u, v);
    const shape = fbm(u * 3.4 + 61, v * 3.4 + 23);
    const detail = fbm(u * 9 + 13, v * 9 + 29);
    const edge = clamp01(1 - d * (0.92 + 0.5 * shape));
    out[0] = 0.1;
    out[1] = 0.085;
    out[2] = 0.078;
    out[3] = Math.pow(edge, 1.4) * (0.5 + 0.5 * detail);
  });
}
