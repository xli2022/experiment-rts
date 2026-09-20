/**
 * Units and buildings assembled from primitives.
 *
 * Everything is boxes, cylinders and cones. That is a deliberate art direction
 * rather than a placeholder: chunky silhouettes with strong team colour read
 * clearly at RTS camera distance, where detailed models turn to mush anyway.
 *
 * Silhouettes are kept deliberately distinct — the worker is squat and round,
 * the Burstbot is upright and narrow, the Slicebot is wide and heavy — because in
 * a real fight players identify units by outline and colour long before they
 * make out any detail.
 */

import * as THREE from 'three';
import { EntityType } from '../../sim/types.js';
import { defOf } from '../../config/rules.js';
import { toFloat } from '../../sim/fixed.js';
import type { ModelPart, ModelProvider, ModelSpec } from './provider.js';

export class ProceduralModelProvider implements ModelProvider {
  private readonly cache = new Map<EntityType, ModelSpec>();
  private readonly geometries: THREE.BufferGeometry[] = [];

  get(type: EntityType): ModelSpec {
    const hit = this.cache.get(type);
    if (hit) return hit;
    const spec = this.build(type);
    this.cache.set(type, spec);
    for (const p of spec.parts) this.geometries.push(p.geometry);
    return spec;
  }

  dispose(): void {
    for (const g of this.geometries) g.dispose();
    this.geometries.length = 0;
    this.cache.clear();
  }

  private build(type: EntityType): ModelSpec {
    // Radius comes from the simulation so the selection ring always matches
    // where the unit actually collides.
    const radius = toFloat(defOf(type).radius);

    switch (type) {
      case EntityType.Worker:
        return {
          radius,
          height: 0.9,
          parts: [
            part(new THREE.CylinderGeometry(0.3, 0.34, 0.42, 8), 'player', [0, 0.21, 0]),
            part(new THREE.SphereGeometry(0.22, 8, 6), 'accent', [0, 0.55, 0]),
            // Forward-facing arm, so which way a worker is pointing is obvious.
            part(new THREE.BoxGeometry(0.12, 0.12, 0.34), 'dark', [0, 0.4, 0.28]),
          ],
        };

      case EntityType.Burstbot:
        return {
          radius,
          height: 1.15,
          parts: [
            part(new THREE.BoxGeometry(0.34, 0.5, 0.26), 'player', [0, 0.4, 0]),
            part(new THREE.SphereGeometry(0.17, 8, 6), 'accent', [0, 0.78, 0]),
            part(new THREE.BoxGeometry(0.08, 0.08, 0.55), 'dark', [0.16, 0.5, 0.24]),
            part(new THREE.BoxGeometry(0.4, 0.16, 0.3), 'dark', [0, 0.12, 0]),
          ],
        };

      case EntityType.Slicebot:
        return {
          radius,
          height: 1.3,
          parts: [
            part(new THREE.BoxGeometry(0.56, 0.54, 0.44), 'player', [0, 0.46, 0]),
            part(new THREE.BoxGeometry(0.3, 0.22, 0.3), 'accent', [0, 0.85, 0]),
            // Heavy shoulders read as "melee" at a glance.
            part(new THREE.BoxGeometry(0.2, 0.3, 0.2), 'dark', [0.36, 0.5, 0]),
            part(new THREE.BoxGeometry(0.2, 0.3, 0.2), 'dark', [-0.36, 0.5, 0]),
            part(new THREE.BoxGeometry(0.5, 0.2, 0.4), 'dark', [0, 0.14, 0]),
          ],
        };

      case EntityType.Beamdrone:
        return {
          radius,
          height: 1.0,
          // Wide, flat and swept — a silhouette nothing on the ground shares, so
          // air is identifiable at a glance even in a crowded fight.
          parts: [
            part(new THREE.ConeGeometry(0.3, 0.95, 4), 'player', [0, 0, 0], [Math.PI / 2, 0, 0]),
            part(new THREE.BoxGeometry(1.5, 0.09, 0.34), 'player', [0, 0.04, -0.05]),
            part(new THREE.BoxGeometry(0.3, 0.1, 0.3), 'accent', [0, 0.13, 0.1]),
            part(new THREE.BoxGeometry(0.1, 0.1, 0.42), 'dark', [0.5, -0.04, 0.12]),
            part(new THREE.BoxGeometry(0.1, 0.1, 0.42), 'dark', [-0.5, -0.04, 0.12]),
          ],
        };

      case EntityType.Boomwalker:
        return {
          radius,
          height: 1.1,
          // Almost all payload, on legs that barely show. The silhouette is the
          // warning: a player who sees this shape coming should read "get out
          // of the way" before they read the name on the panel.
          parts: [
            part(new THREE.CylinderGeometry(0.34, 0.34, 0.62, 8), 'accent', [0, 0.62, 0]),
            part(new THREE.CylinderGeometry(0.38, 0.38, 0.1, 8), 'dark', [0, 0.95, 0]),
            part(new THREE.BoxGeometry(0.1, 0.32, 0.1), 'player', [0.14, 0.16, 0]),
            part(new THREE.BoxGeometry(0.1, 0.32, 0.1), 'player', [-0.14, 0.16, 0]),
          ],
        };

      case EntityType.Fixomatic:
        return {
          radius,
          height: 1.2,
          // A lens on a stalk with two working arms out front, and nothing that
          // reads as a weapon — which is the point.
          parts: [
            part(new THREE.BoxGeometry(0.26, 0.4, 0.26), 'player', [0, 0.34, 0]),
            part(new THREE.SphereGeometry(0.2, 10, 8), 'accent', [0, 0.78, 0.06]),
            part(new THREE.BoxGeometry(0.07, 0.07, 0.5), 'dark', [0.2, 0.5, 0.24], [0.3, 0, 0]),
            part(new THREE.BoxGeometry(0.07, 0.07, 0.5), 'dark', [-0.2, 0.5, 0.24], [0.3, 0, 0]),
          ],
        };

      case EntityType.Firespout:
        return {
          radius,
          height: 1.25,
          // A barrel with one wide muzzle out the front, low on crab legs.
          parts: [
            part(new THREE.CylinderGeometry(0.34, 0.38, 0.72, 8), 'player', [0, 0.52, 0]),
            part(
              new THREE.CylinderGeometry(0.12, 0.17, 0.5, 8),
              'dark',
              [0, 0.6, 0.4],
              [1.57, 0, 0],
            ),
            part(new THREE.BoxGeometry(0.72, 0.14, 0.5), 'dark', [0, 0.14, 0]),
          ],
        };

      case EntityType.Arclight:
        return {
          radius,
          height: 1.35,
          // Three coils, because three is the number the panel promises.
          parts: [
            part(new THREE.SphereGeometry(0.42, 10, 8), 'player', [0, 0.52, 0]),
            part(new THREE.CylinderGeometry(0.2, 0.2, 0.1, 8), 'accent', [0, 0.98, -0.1]),
            part(new THREE.CylinderGeometry(0.17, 0.17, 0.1, 8), 'accent', [0.32, 0.84, -0.2]),
            part(new THREE.CylinderGeometry(0.17, 0.17, 0.1, 8), 'accent', [-0.32, 0.84, -0.2]),
            part(new THREE.BoxGeometry(0.6, 0.18, 0.4), 'dark', [0, 0.12, 0]),
          ],
        };

      case EntityType.Piercebot:
        return {
          radius,
          height: 0.9,
          // Long, low and mostly rail. Nothing else on the field is this flat.
          parts: [
            part(new THREE.BoxGeometry(0.5, 0.26, 0.8), 'player', [0, 0.3, 0]),
            part(new THREE.BoxGeometry(0.09, 0.09, 1.2), 'accent', [0.16, 0.52, 0.3]),
            part(new THREE.BoxGeometry(0.09, 0.09, 1.2), 'accent', [-0.16, 0.52, 0.3]),
            part(new THREE.BoxGeometry(0.8, 0.12, 0.5), 'dark', [0, 0.12, -0.1]),
          ],
        };

      case EntityType.Sentry:
        return {
          radius,
          height: 1.4,
          // The barrel points straight up, which is the whole explanation of
          // both its range and its minimum range.
          parts: [
            part(new THREE.BoxGeometry(0.56, 0.36, 0.56), 'player', [0, 0.32, 0]),
            part(new THREE.CylinderGeometry(0.19, 0.22, 0.66, 8), 'dark', [0, 0.82, -0.04]),
            part(new THREE.CylinderGeometry(0.24, 0.24, 0.1, 8), 'accent', [0, 1.18, -0.04]),
            part(new THREE.BoxGeometry(0.74, 0.14, 0.62), 'dark', [0, 0.1, 0]),
          ],
        };

      case EntityType.DarkGolem:
        return {
          radius,
          height: 1.8,
          // Shoulders first: the widest ground silhouette either side fields,
          // with two lit stacks behind it.
          parts: [
            part(new THREE.BoxGeometry(0.8, 0.82, 0.62), 'player', [0, 0.78, 0]),
            part(new THREE.BoxGeometry(1.14, 0.3, 0.5), 'dark', [0, 1.1, -0.02]),
            part(new THREE.CylinderGeometry(0.11, 0.11, 0.5, 6), 'accent', [0.24, 1.44, -0.24]),
            part(new THREE.CylinderGeometry(0.11, 0.11, 0.5, 6), 'accent', [-0.24, 1.44, -0.24]),
            part(new THREE.BoxGeometry(0.26, 0.6, 0.26), 'dark', [0.52, 0.5, 0.16]),
            part(new THREE.BoxGeometry(0.26, 0.6, 0.26), 'dark', [-0.52, 0.5, 0.16]),
            part(new THREE.BoxGeometry(0.78, 0.3, 0.6), 'player', [0, 0.18, 0]),
          ],
        };

      case EntityType.IceGolem:
        return {
          radius,
          height: 1.75,
          // The same bulk, but carrying two barrels instead of two fists.
          parts: [
            part(new THREE.BoxGeometry(0.78, 0.78, 0.6), 'player', [0, 0.74, 0]),
            part(
              new THREE.CylinderGeometry(0.14, 0.14, 0.72, 8),
              'accent',
              [0.42, 1.2, -0.1],
              [0.5, 0, 0.35],
            ),
            part(
              new THREE.CylinderGeometry(0.14, 0.14, 0.72, 8),
              'accent',
              [-0.42, 1.2, -0.1],
              [0.5, 0, -0.35],
            ),
            part(new THREE.BoxGeometry(0.24, 0.54, 0.24), 'dark', [0.5, 0.46, 0.14]),
            part(new THREE.BoxGeometry(0.24, 0.54, 0.24), 'dark', [-0.5, 0.46, 0.14]),
            part(new THREE.BoxGeometry(0.76, 0.3, 0.58), 'player', [0, 0.18, 0]),
          ],
        };

      case EntityType.Plasmodrone:
        return {
          radius,
          height: 1.15,
          // Wider than the Beamdrone and carrying two pods, so the two flyers
          // are told apart by width in the one glance a fight allows.
          parts: [
            part(new THREE.ConeGeometry(0.36, 1.1, 6), 'player', [0, 0, 0.05], [Math.PI / 2, 0, 0]),
            part(new THREE.BoxGeometry(1.9, 0.11, 0.42), 'player', [0, 0.05, -0.08]),
            part(new THREE.SphereGeometry(0.22, 10, 8), 'accent', [0.72, 0.0, 0.02]),
            part(new THREE.SphereGeometry(0.22, 10, 8), 'accent', [-0.72, 0.0, 0.02]),
            part(new THREE.BoxGeometry(0.34, 0.14, 0.34), 'dark', [0, 0.16, -0.2]),
          ],
        };

      case EntityType.CommandPost:
        return {
          radius,
          height: 2.6,
          parts: [
            part(new THREE.BoxGeometry(3.4, 1.1, 3.4), 'player', [0, 0.55, 0]),
            part(new THREE.BoxGeometry(2.2, 0.7, 2.2), 'accent', [0, 1.45, 0]),
            part(new THREE.CylinderGeometry(0.24, 0.24, 1.5, 6), 'dark', [1.3, 1.5, 1.3]),
            part(new THREE.CylinderGeometry(0.24, 0.24, 1.5, 6), 'dark', [-1.3, 1.5, -1.3]),
            part(new THREE.CylinderGeometry(0.5, 0.7, 0.5, 8), 'accent', [0, 2.05, 0]),
          ],
        };

      case EntityType.Depot:
        return {
          radius,
          height: 1.1,
          parts: [
            part(new THREE.BoxGeometry(1.7, 0.7, 1.7), 'player', [0, 0.35, 0]),
            part(new THREE.CylinderGeometry(0.55, 0.55, 0.45, 8), 'accent', [0, 0.9, 0]),
          ],
        };

      case EntityType.Barracks:
        return {
          radius,
          height: 1.9,
          parts: [
            part(new THREE.BoxGeometry(2.6, 1.2, 2.6), 'player', [0, 0.6, 0]),
            // Angled roof, so barracks never read as a bigger depot.
            part(new THREE.BoxGeometry(2.2, 0.5, 1.2), 'dark', [0, 1.4, 0]),
            part(new THREE.BoxGeometry(0.8, 0.9, 0.25), 'accent', [0, 0.45, 1.35]),
          ],
        };

      case EntityType.Foundry:
        return {
          radius,
          height: 2.2,
          // Reads as a Barracks that grew a chimney. Deliberately the same
          // family — it is the same idea one tier up — but a player scouting a
          // base has to be able to tell at a glance which one they are looking
          // at, and the stack does that from any angle.
          parts: [
            part(new THREE.BoxGeometry(2.6, 1.3, 2.6), 'player', [0, 0.65, 0]),
            part(new THREE.BoxGeometry(2.0, 0.4, 2.0), 'dark', [0, 1.5, 0]),
            part(new THREE.CylinderGeometry(0.38, 0.46, 1.0, 8), 'dark', [-0.7, 2.05, -0.7]),
            part(new THREE.CylinderGeometry(0.42, 0.42, 0.16, 8), 'accent', [-0.7, 2.6, -0.7]),
            part(new THREE.BoxGeometry(1.1, 0.8, 0.25), 'accent', [0, 0.45, 1.35]),
          ],
        };

      case EntityType.Turret:
        return {
          radius,
          height: 1.5,
          parts: [
            part(new THREE.CylinderGeometry(0.75, 0.9, 0.5, 8), 'dark', [0, 0.25, 0]),
            part(new THREE.SphereGeometry(0.5, 10, 8), 'player', [0, 0.7, 0]),
            part(new THREE.BoxGeometry(0.14, 0.14, 1.1), 'accent', [0, 0.75, 0.55]),
          ],
        };

      case EntityType.MineralPatch:
        return {
          radius,
          height: 0.9,
          parts: [
            // Irregular crystal cluster. Cones at varied tilts avoid the
            // "three identical spikes" look while staying cheap.
            part(new THREE.ConeGeometry(0.32, 1.0, 5), 'resource', [0, 0.5, 0]),
            part(
              new THREE.ConeGeometry(0.24, 0.72, 5),
              'resource',
              [0.42, 0.36, 0.2],
              [0, 0, 0.32],
            ),
            part(
              new THREE.ConeGeometry(0.2, 0.6, 5),
              'resource',
              [-0.36, 0.3, -0.26],
              [0, 0, -0.4],
            ),
            part(
              new THREE.ConeGeometry(0.18, 0.5, 5),
              'resource',
              [0.05, 0.25, -0.45],
              [0.3, 0, 0],
            ),
          ],
        };

      default:
        return {
          radius,
          height: 1,
          parts: [part(new THREE.BoxGeometry(0.6, 0.6, 0.6), 'player', [0, 0.3, 0])],
        };
    }
  }
}

function part(
  geometry: THREE.BufferGeometry,
  role: ModelPart['role'],
  offset: [number, number, number],
  rotation?: [number, number, number],
): ModelPart {
  return rotation ? { geometry, role, offset, rotation } : { geometry, role, offset };
}

/**
 * Player colours, plus the neutral palette used for resources and terrain.
 *
 * The table is **team-major**: entries 0 and 1 are one side's, 2 and 3 the
 * other's. Two hue families, cool against warm, because which *player* a unit
 * belongs to matters far less in play than which *side* it is on — the question
 * a glance at a battle has to answer is "may I shoot that", and hue answers it
 * from across the map where a shade could not. Telling a partner's army from
 * your own is what the selection ring and the minimap are for.
 *
 * Index it through `colourSlotFor`, never with a raw player id. See there.
 */
export const PLAYER_COLOURS = [0x4a9eff, 0x35d6bd, 0xff5a4a, 0xffa93d] as const;
export const ACCENT_COLOURS = [0xa8d4ff, 0x9ff0e4, 0xffb0a4, 0xffd9a0] as const;
export const DARK_COLOUR = 0x2a3140;
/**
 * Neutral resources: mineral crystals, everywhere they are drawn.
 *
 * Purple, and deliberately nowhere near a team hue. It was teal, one shade off
 * the second player colour above, and a mineral line behind a teal army read as
 * more army — a patch, a worker and a tracer all the same colour is the one
 * thing the palette has to avoid, because "may I shoot that" is answered by
 * hue. Violet sits a clear 60 degrees off the blue end of the cool family and
 * on the far side of every warm one, so it belongs to nobody at any player
 * count, at minimap scale as much as up close.
 */
export const RESOURCE_COLOUR = 0xb45cff;

/**
 * Which palette entry a player's colour comes from.
 *
 * The table above is laid out by side, and a roster is split down the middle —
 * the first half is one team, the second the other — so a four-player match
 * happens to read straight off the player id and a *duel does not*. Indexing a
 * duel by raw id paints the lone opponent in the local player's own hue family:
 * measured, the enemy's buildings, workers, tracers and minimap dots all went
 * teal, one shade from your own blue, while their combat units still wore the
 * red team skin. Mapping through the side is what keeps blue against red at two
 * players and changes nothing at four.
 */
export function colourSlotFor(owner: number, playerCount: number): number {
  const half = Math.max(1, playerCount >> 1);
  return owner < half ? owner : 2 + (owner - half);
}

/** Resolve a part role to a concrete colour for a palette slot. */
export function colourFor(role: ModelPart['role'], slot: number): number {
  switch (role) {
    case 'player':
      return PLAYER_COLOURS[slot] ?? 0x9aa4b2;
    case 'accent':
      return ACCENT_COLOURS[slot] ?? 0xc8d0dc;
    case 'resource':
      return RESOURCE_COLOUR;
    case 'dark':
      return DARK_COLOUR;
  }
}
