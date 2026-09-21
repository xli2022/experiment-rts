/** Cached routes to every usable edge of a construction footprint. */
import { BUILD_REACH, defOf } from '../../config/rules.js';
import { fromInt, FIX_HALF, FIX_ONE, sqRange } from '../fixed.js';
import { distanceSqTo } from '../systems/economy.js';
import { EntityType } from '../types.js';
import type { World } from '../world.js';
import { FlowField } from './flowfield.js';

interface Entry {
  site: number;
  field: FlowField;
}

/**
 * A nearest walkable approach can be disconnected from the worker, even when
 * another edge is reachable. One multi-source field finds the closest reachable
 * approach without A*'s expansion limit. Builders share it until occupancy
 * changes; every Build request uses this same algorithm, warm or cold.
 *
 * Pure derived scratch: cache order/warmth cannot affect the returned path.
 */
export class ConstructionPaths {
  private readonly entries: Entry[] = [];
  private readonly seeds: number[] = [];

  constructor(private readonly capacity = 12) {}

  find(world: World, worker: number, site: number, out: number[]): number[] {
    out.length = 0;
    const { map, pool } = world;
    if (!pool.isAlive(site)) return out;
    const at = this.entries.findIndex((entry) => entry.site === site);
    let entry: Entry;
    if (at >= 0) {
      entry = this.entries.splice(at, 1)[0]!;
    } else {
      entry =
        this.entries.length >= this.capacity
          ? this.entries.shift()!
          : { site, field: new FlowField(map.width * map.height) };
      entry.site = site;
      entry.field.builtVersion = -1;
    }
    this.entries.push(entry);

    const target = site & 0xffff;
    const reach = defOf(EntityType.Worker).radius + BUILD_REACH;
    if (entry.field.builtVersion !== map.occupancyVersion) {
      this.seeds.length = 0;
      const footprint = defOf(pool.type[target]! as EntityType).footprint;
      const rings = Math.ceil(reach / FIX_ONE + 0.5);
      for (let y = pool.tileY[target]! - rings; y < pool.tileY[target]! + footprint + rings; y++) {
        for (
          let x = pool.tileX[target]! - rings;
          x < pool.tileX[target]! + footprint + rings;
          x++
        ) {
          if (!map.isWalkable(x, y)) continue;
          if (
            distanceSqTo(world, target, fromInt(x) + FIX_HALF, fromInt(y) + FIX_HALF) >
            sqRange(reach)
          )
            continue;
          this.seeds.push(map.index(x, y));
        }
      }
      entry.field.build(map, map.index(pool.tileX[target]!, pool.tileY[target]!), this.seeds);
    }

    const flip = world.flipOf(pool.owner[worker]!);
    let tile = map.tileOfPosFor(pool.posX[worker]!, pool.posY[worker]!, flip);
    if (tile < 0 || entry.field.isStranded(tile)) return out;
    // The tile center may be in construction range while the worker's actual
    // fractional position is not. Walking to this seed is still a useful path.
    if (entry.field.dist[tile] === 0) out.push(tile);
    // Distance strictly falls at every step; at most one visit per map tile.
    while (entry.field.dist[tile]! > 0) {
      const next = entry.field.stepFromCentre(map, tile, flip);
      if (next < 0) {
        out.length = 0;
        return out;
      }
      out.push(next);
      tile = next;
    }
    return out;
  }
}
