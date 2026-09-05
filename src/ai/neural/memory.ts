/**
 * What a player remembers having seen.
 *
 * Fog hides what is not being watched now, but a person does not forget the
 * Barracks they scouted a minute ago. This keeps a last-seen entry for every
 * enemy entity and mineral patch the side has ever had in view, with the
 * rules a person would apply: a building stays where it was until its ground
 * is seen again without it; a unit is assumed to have moved on after a while,
 * or the moment its tile is seen and it is not there.
 *
 * Nothing here reads the state of an entity it cannot currently see. That is
 * the whole contract — a memory that peeked would be a fog leak — and
 * `tests/memory.test.ts` proves it with a proxy on the pool.
 */

import { defOf } from '../../config/rules.js';
import type { Fix } from '../../sim/fixed.js';
import { EntityType, NEUTRAL, type EntityId, type PlayerId } from '../../sim/types.js';
import type { World } from '../../sim/world.js';
import type { Visibility } from '../../vision/visibility.js';
import { UNIT_MEMORY_TICKS } from './spec.js';

export interface Remembered {
  readonly id: EntityId;
  readonly type: EntityType;
  readonly owner: PlayerId;
  readonly serial: number;
  readonly isBuilding: boolean;
  tileX: number;
  tileY: number;
  posX: Fix;
  posY: Fix;
  hp: number;
  buildState: number;
  resourceAmount: number;
  lastSeen: number;
}

export class EntityMemory {
  /** Every remembered entity, in the order it was first seen. Pruned in place. */
  readonly entries: Remembered[] = [];
  private readonly byId = new Map<EntityId, Remembered>();

  constructor(readonly viewer: PlayerId) {}

  /** Call every tick, after the visibility has been updated for the same tick. */
  update(world: World, vis: Visibility): void {
    const pool = world.pool;
    const map = world.map;
    const tick = world.tick;
    const flip = world.flipOf(this.viewer);

    // Everything in view right now is refreshed. Allies are not remembered —
    // they are always in view, and never in the table as memories.
    for (let i = 0; i < pool.count; i++) {
      if (pool.alive[i] !== 1) continue;
      const owner = pool.owner[i]!;
      if (owner !== NEUTRAL && world.areAllied(owner, this.viewer)) continue;
      // Which tile an entity is "on" is decided in the viewer's canonical frame,
      // so that a building's centre — a corner between four tiles — is tested
      // against the same one of them from either seat. Sampled in the map's
      // frame it would be the bottom-right tile for one seat and that tile's
      // rotation, the top-left, for the other, and a sight disc that covers
      // one but not the other would make the two seats remember different
      // things about a mirrored world.
      const tile = map.tileOfPosFor(pool.posX[i]!, pool.posY[i]!, flip);
      if (!vis.isVisibleTile(tile)) continue;

      const id = pool.idAt(i);
      let entry = this.byId.get(id);
      if (!entry) {
        const type = pool.type[i]! as EntityType;
        entry = {
          id,
          type,
          owner,
          serial: pool.serial[i]!,
          isBuilding: defOf(type).isBuilding || type === EntityType.MineralPatch,
          tileX: 0,
          tileY: 0,
          posX: 0,
          posY: 0,
          hp: 0,
          buildState: 0,
          resourceAmount: 0,
          lastSeen: tick,
        };
        this.byId.set(id, entry);
        this.entries.push(entry);
      }
      // A building's tile is its top-left, which is where its footprint is
      // measured from; a unit's is simply the one under it.
      entry.tileX = entry.isBuilding ? pool.tileX[i]! : map.tileXOf(tile);
      entry.tileY = entry.isBuilding ? pool.tileY[i]! : map.tileYOf(tile);
      entry.posX = pool.posX[i]!;
      entry.posY = pool.posY[i]!;
      entry.hp = pool.hp[i]!;
      entry.buildState = pool.buildState[i]!;
      entry.resourceAmount = pool.resourceAmount[i]!;
      entry.lastSeen = tick;
    }

    // Anything not refreshed is out of view. Forget it if its ground is being
    // watched and it is not there, or — for a unit — if it has simply been
    // long enough that it will have moved on.
    let kept = 0;
    for (let k = 0; k < this.entries.length; k++) {
      const entry = this.entries[k]!;
      let forget = false;
      if (entry.lastSeen !== tick) {
        if (!entry.isBuilding && tick - entry.lastSeen > UNIT_MEMORY_TICKS) forget = true;
        else if (this.groundSeen(world, vis, entry)) forget = true;
      }
      if (forget) this.byId.delete(entry.id);
      else this.entries[kept++] = entry;
    }
    this.entries.length = kept;
  }

  /** Is any tile the entity was last seen standing on currently in view? */
  private groundSeen(world: World, vis: Visibility, entry: Remembered): boolean {
    const footprint = entry.isBuilding ? Math.max(1, defOf(entry.type).footprint) : 1;
    const map = world.map;
    for (let y = entry.tileY; y < entry.tileY + footprint; y++) {
      for (let x = entry.tileX; x < entry.tileX + footprint; x++) {
        if (map.inBounds(x, y) && vis.isVisibleTile(map.index(x, y))) return true;
      }
    }
    return false;
  }

  get(id: EntityId): Remembered | undefined {
    return this.byId.get(id);
  }
}
