/**
 * Terrain rendering.
 *
 * The ground is a single plane textured from a canvas painted once at load, and
 * cliffs are one `InstancedMesh` of boxes. A 128x128 map is 16,384 tiles; drawing
 * them as individual meshes would be 16,384 draw calls, so instead the whole
 * playfield costs two.
 *
 * Terrain is purely cosmetic — the simulation decides movement from the
 * walkability grid and never consults anything here.
 */

import * as THREE from 'three';
import { GameMap } from '../sim/map.js';
import { Tile } from '../sim/types.js';

const GROUND_BASE = '#3f5a44';
const GROUND_ALT = '#44614a';
const RESOURCE_TINT = '#3c5f60';
const CLIFF_TOP = 0x7c8595;
const CLIFF_SIDE = 0x424c5c;

/**
 * Height of a cliff tile one step in from open ground.
 *
 * The camera looks down at 52 degrees, so a wall of height H hides about 0.8H
 * tiles of ground behind it. These numbers top out around 3 units — enough for
 * terrain to have presence, little enough that a ridge never swallows the far
 * side of a lane.
 */
const CLIFF_BASE_HEIGHT = 1.0;
/** Extra height per step further from open ground. */
const CLIFF_STEP_HEIGHT = 0.45;
/** Matches `MAX_ELEVATION` in the generator; used only to normalise colour. */
const MAX_CLIFF_STEP = 6;
/** How much of its colour a cliff keeps once seen but no longer observed. */
const FOGGED_CLIFF_SHADE = 0.42;

export class TerrainRenderer {
  readonly group = new THREE.Group();
  private readonly disposables: { dispose(): void }[] = [];

  /** Cliff instances, and the map tile each one stands on. */
  private cliffs: THREE.InstancedMesh | null = null;
  private cliffTileX = new Int32Array(0);
  private cliffTileY = new Int32Array(0);
  /** Each cliff's unfogged colour, so shading can be reapplied from scratch. */
  private cliffTone = new Float32Array(0);
  private fogVersion = -1;

  constructor(map: GameMap) {
    this.group.add(this.buildGround(map));
    const cliffs = this.buildCliffs(map);
    if (cliffs) {
      this.cliffs = cliffs;
      this.group.add(cliffs);
    }
  }

  /**
   * Darken cliffs the player has not seen.
   *
   * The fog is a flat plane just above the ground, so anything with height
   * stands straight through it. With a map that is more than half cliff, that
   * handed the whole layout to the player on the first frame — every ridge and
   * every lane, before a single scout moved. Shading the instances instead keeps
   * fog to one draw call and makes unexplored rock read as the same void as
   * unexplored ground.
   */
  applyFog(fog: {
    version: number;
    isExploredAt(tx: number, tz: number): boolean;
    isVisibleAt(x: number, z: number): boolean;
  }): void {
    const mesh = this.cliffs;
    if (!mesh || fog.version === this.fogVersion) return;
    this.fogVersion = fog.version;

    const colour = new THREE.Color();
    const low = new THREE.Color(CLIFF_SIDE);
    const high = new THREE.Color(CLIFF_TOP);
    for (let k = 0; k < this.cliffTone.length; k++) {
      const tx = this.cliffTileX[k]!;
      const ty = this.cliffTileY[k]!;
      let shade = 0;
      if (fog.isVisibleAt(tx + 0.5, ty + 0.5)) shade = 1;
      else if (fog.isExploredAt(tx, ty)) shade = FOGGED_CLIFF_SHADE;
      colour.copy(low).lerp(high, this.cliffTone[k]!).multiplyScalar(shade);
      mesh.setColorAt(k, colour);
    }
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
  }

  /**
   * Ground plane, textured from a canvas painted per tile.
   *
   * A texture rather than vertex colours because it gives crisp tile edges at
   * any subdivision level, and costs one upload instead of a large vertex
   * buffer.
   */
  private buildGround(map: GameMap): THREE.Mesh {
    const canvas = document.createElement('canvas');
    canvas.width = map.width;
    canvas.height = map.height;
    const ctx = canvas.getContext('2d')!;

    for (let y = 0; y < map.height; y++) {
      for (let x = 0; x < map.width; x++) {
        const tile = map.tiles[map.index(x, y)];
        // Subtle checker so the eye can judge distance and unit speed; a flat
        // colour makes an RTS camera feel like it is not moving.
        const checker = (x + y) % 2 === 0;
        let colour = checker ? GROUND_BASE : GROUND_ALT;
        if (tile === Tile.Resource) colour = RESOURCE_TINT;
        ctx.fillStyle = colour;
        ctx.fillRect(x, y, 1, 1);
      }
    }

    const texture = new THREE.CanvasTexture(canvas);
    texture.magFilter = THREE.NearestFilter;
    texture.minFilter = THREE.LinearMipmapLinearFilter;
    texture.colorSpace = THREE.SRGBColorSpace;

    const geometry = new THREE.PlaneGeometry(map.width, map.height);
    geometry.rotateX(-Math.PI / 2);
    // The simulation's origin is a tile corner, so shift the plane to match.
    geometry.translate(map.width / 2, 0, map.height / 2);

    const material = new THREE.MeshLambertMaterial({ map: texture });
    this.disposables.push(geometry, material, texture);

    const mesh = new THREE.Mesh(geometry, material);
    mesh.receiveShadow = true;
    mesh.name = 'ground';
    return mesh;
  }

  /**
   * All cliff tiles as one instanced box mesh, scaled by elevation.
   *
   * `map.elevation` holds each cliff tile's distance from open ground, so
   * scaling height by it turns what would be a uniform grey wall into massifs
   * that rise away from the lanes. It costs nothing — the same instance count,
   * one scale per matrix — and it is what makes a corridor read as a canyon
   * from the game camera rather than as a gap in a hedge.
   */
  private buildCliffs(map: GameMap): THREE.InstancedMesh | null {
    const tiles: number[] = [];
    for (let i = 0; i < map.tiles.length; i++) {
      if (map.tiles[i] === Tile.Cliff) tiles.push(i);
    }
    if (tiles.length === 0) return null;

    // Unit cube sitting on the ground plane, so a Y scale raises the top face
    // without lifting the block off the floor.
    const geometry = new THREE.BoxGeometry(1, 1, 1);
    geometry.translate(0, 0.5, 0);
    const material = new THREE.MeshLambertMaterial({ color: CLIFF_TOP });
    this.disposables.push(geometry, material);

    const mesh = new THREE.InstancedMesh(geometry, material, tiles.length);
    const matrix = new THREE.Matrix4();
    const colour = new THREE.Color();
    const low = new THREE.Color(CLIFF_SIDE);
    const high = new THREE.Color(CLIFF_TOP);

    this.cliffTileX = new Int32Array(tiles.length);
    this.cliffTileY = new Int32Array(tiles.length);
    this.cliffTone = new Float32Array(tiles.length);

    for (let k = 0; k < tiles.length; k++) {
      const t = tiles[k]!;
      const tx = map.tileXOf(t);
      const ty = map.tileYOf(t);
      const step = Math.max(1, map.elevation[t]!);
      const height = CLIFF_BASE_HEIGHT + (step - 1) * CLIFF_STEP_HEIGHT;

      matrix.makeScale(1, height, 1);
      matrix.setPosition(tx + 0.5, 0, ty + 0.5);
      mesh.setMatrixAt(k, matrix);

      // Lighter with height, plus a faint per-tile break-up so a large massif
      // does not read as one flat-shaded lump.
      let tone = (step - 1) / (MAX_CLIFF_STEP - 1);
      if ((tx + ty) % 3 === 0) tone *= 0.82;
      tone = Math.min(1, tone);

      this.cliffTileX[k] = tx;
      this.cliffTileY[k] = ty;
      this.cliffTone[k] = tone;
      colour.copy(low).lerp(high, tone);
      mesh.setColorAt(k, colour);
    }

    mesh.instanceMatrix.needsUpdate = true;
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    mesh.name = 'cliffs';
    return mesh;
  }

  dispose(): void {
    for (const d of this.disposables) d.dispose();
    this.group.clear();
  }
}
