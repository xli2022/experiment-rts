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
const CLIFF_TOP = 0x6b7280;
const CLIFF_SIDE = 0x4b5563;

export class TerrainRenderer {
  readonly group = new THREE.Group();
  private readonly disposables: { dispose(): void }[] = [];

  constructor(map: GameMap) {
    this.group.add(this.buildGround(map));
    const cliffs = this.buildCliffs(map);
    if (cliffs) this.group.add(cliffs);
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

  /** All cliff tiles as one instanced box mesh. */
  private buildCliffs(map: GameMap): THREE.InstancedMesh | null {
    const tiles: number[] = [];
    for (let i = 0; i < map.tiles.length; i++) {
      if (map.tiles[i] === Tile.Cliff) tiles.push(i);
    }
    if (tiles.length === 0) return null;

    const geometry = new THREE.BoxGeometry(1, 1.4, 1);
    const material = new THREE.MeshLambertMaterial({ color: CLIFF_TOP });
    this.disposables.push(geometry, material);

    const mesh = new THREE.InstancedMesh(geometry, material, tiles.length);
    const matrix = new THREE.Matrix4();
    const colour = new THREE.Color();

    for (let k = 0; k < tiles.length; k++) {
      const t = tiles[k]!;
      const x = map.tileXOf(t) + 0.5;
      const z = map.tileYOf(t) + 0.5;
      matrix.makeTranslation(x, 0.7, z);
      mesh.setMatrixAt(k, matrix);
      // Alternate two greys so a rock field has some visual break-up.
      colour.setHex((map.tileXOf(t) + map.tileYOf(t)) % 3 === 0 ? CLIFF_SIDE : CLIFF_TOP);
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
