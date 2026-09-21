/**
 * Flow fields for group movement.
 *
 * ## Why this exists
 *
 * A* answers "how do I get from A to B". An RTS almost never asks that — it asks
 * "how do *these thirty units* get to B", and running thirty independent
 * searches to one destination is thirty times the work for essentially one
 * answer. Measured on a 128x128 map, a single army-wide attack-move cost ~60ms
 * of pathfinding per tick, which in lockstep stalls every peer, not just the
 * machine that computed it.
 *
 * A flow field inverts the problem: one Dijkstra sweep outward from the
 * destination produces the distance-to-goal for *every* tile at once. Each unit
 * then just steps to whichever neighbouring tile has the lowest remaining
 * distance. Cost becomes one sweep per destination instead of one search per
 * unit, and the field is reused by everyone heading there, across many ticks.
 *
 * A* is still the right tool for a single unit chasing a single moving target,
 * so both live side by side — see `movement.ts` for which is used when.
 *
 * ## Determinism
 *
 * Unlike A*, this needs no tie-breaking rule at all. Dijkstra assigns every tile
 * a *unique* shortest distance, so the finished `dist` array is identical no
 * matter what order equal-cost nodes were expanded in — which is why the bucket
 * queue below can pop LIFO without consequence. Steering then picks among
 * equally good neighbours by a strict order that reads the same from both
 * halves of the map (see `stepFrom`), so two peers read the same field and
 * take the same step, and two mirrored units take mirrored steps.
 */

import { FIX_HALF, fromInt, vecLenSqRaw, type Fix } from '../fixed.js';
import type { GameMap } from '../map.js';

const COST_STRAIGHT = 10;
const COST_DIAGONAL = 14;

/** Distance value meaning "no route to the goal from here". */
export const UNREACHABLE = 0x7fffffff;

/**
 * Cache slot meaning "not worked out yet".
 *
 * Distinct from -1, which `stepFrom` returns for a tile with nowhere better to
 * go and which is worth remembering like any other answer.
 */
const UNCACHED = -2;

const NEIGHBOURS: readonly (readonly [number, number, number])[] = [
  [1, 0, COST_STRAIGHT],
  [-1, 0, COST_STRAIGHT],
  [0, 1, COST_STRAIGHT],
  [0, -1, COST_STRAIGHT],
  [1, 1, COST_DIAGONAL],
  [1, -1, COST_DIAGONAL],
  [-1, 1, COST_DIAGONAL],
  [-1, -1, COST_DIAGONAL],
];

/**
 * Number of buckets in the priority queue.
 *
 * Dial's algorithm needs one more bucket than the largest edge weight. Every
 * node inserted while processing cost `c` lands in `[c, c + 14]`, so 15 buckets
 * are enough for those to occupy distinct slots modulo the bucket count.
 */
const BUCKET_COUNT = COST_DIAGONAL + 1; // 15

/**
 * How far out to look for walkable ground around an unwalkable goal.
 *
 * A Command Post is the largest footprint and its centre is two tiles from its
 * edge; anything further than this is a goal deep inside a cliff mass, which
 * `standableTarget` has already refused.
 */
const SEED_RINGS = 12;

export class FlowField {
  /** Cost-to-goal per tile, or UNREACHABLE. */
  readonly dist: Int32Array;
  goalTile = -1;
  /** Map occupancy version this field was built against. */
  builtVersion = -1;

  /**
   * Bucket queue (Dial's algorithm) rather than a binary heap.
   *
   * Edge weights here take exactly two values, 10 and 14, which is precisely
   * the situation bucket queues are for: insert and extract-min are O(1) with
   * no comparisons at all, against O(log n) and ~1.8M comparisons for a heap
   * sweep of a 128x128 map.
   *
   * It also removes a whole class of bug. A binary heap sized to the tile count
   * is *too small* — Dijkstra re-inserts a tile on every successful relaxation,
   * up to eight times per tile — and because writing past the end of a typed
   * array fails silently rather than throwing, the overflow corrupted the heap
   * instead of crashing. That cost 20ms per tick and was invisible. Buckets are
   * plain arrays that grow as needed, so the failure mode does not exist.
   */
  private readonly buckets: number[][] = Array.from({ length: BUCKET_COUNT }, () => []);

  /** Memoised `stepFromCentre` answers, per frame. See that method. */
  private centreStep: Int32Array | null = null;
  private centreStepFlipped: Int32Array | null = null;

  constructor(tileCount: number) {
    this.dist = new Int32Array(tileCount);
  }

  /** Sweep outward from `goalTile`, filling `dist` for the whole map. */
  build(map: GameMap, goalTile: number, seeds?: readonly number[]): void {
    this.goalTile = goalTile;
    this.builtVersion = map.occupancyVersion;
    this.dist.fill(UNREACHABLE);
    this.centreStep?.fill(UNCACHED);
    this.centreStepFlipped?.fill(UNCACHED);
    for (let b = 0; b < BUCKET_COUNT; b++) this.buckets[b]!.length = 0;

    if (goalTile < 0) return;

    const w = map.width;
    const h = map.height;

    // Seed from walkable tiles, even when asked for an unwalkable one.
    //
    // Attack-move targets a building's *centre*, which is occupied — and for
    // anything bigger than 2x2 every neighbour of that centre is inside the
    // same footprint. The sweep would then never escape the building and would
    // report the entire map unreachable, so every unit ordered to attack a
    // Command Post immediately cancelled and stood still. Armies grew to
    // hundreds without a single shot fired.
    //
    // *Every* walkable tile on the nearest ring that has one, all at distance
    // zero, rather than a single substitute. One tile has to be chosen by some
    // rule, and any rule that reads absolute map directions sends the two
    // halves' armies to different corners of a building; a whole ring is its
    // own rotation, so the field for the mirrored building is the mirrored
    // field. It also happens to be better behaviour: units approach the face
    // nearest them instead of funnelling round to one corner.
    let queued = 0;
    const gx = goalTile % w;
    const gy = (goalTile / w) | 0;
    if (seeds !== undefined) {
      // Entity interactions can finish at any usable footprint edge. Seed all
      // of them so a disconnected near edge cannot hide a reachable far edge.
      for (const tile of seeds) {
        if (!map.isWalkable(map.tileXOf(tile), map.tileYOf(tile))) continue;
        if (this.dist[tile] === 0) continue;
        this.dist[tile] = 0;
        this.buckets[0]!.push(tile);
        queued++;
      }
    } else if (map.isWalkable(gx, gy)) {
      this.dist[goalTile] = 0;
      this.buckets[0]!.push(goalTile);
      queued = 1;
    } else {
      for (let r = 1; r <= SEED_RINGS && queued === 0; r++) {
        for (let dy = -r; dy <= r; dy++) {
          for (let dx = -r; dx <= r; dx++) {
            if (dx > -r && dx < r && dy > -r && dy < r) continue;
            if (!map.isWalkable(gx + dx, gy + dy)) continue;
            const tile = (gy + dy) * w + gx + dx;
            this.dist[tile] = 0;
            this.buckets[0]!.push(tile);
            queued++;
          }
        }
      }
      if (queued === 0) return;
    }
    let cost = 0;

    while (queued > 0) {
      const bucket = this.buckets[cost % BUCKET_COUNT]!;
      if (bucket.length === 0) {
        // Nothing at this cost; the next non-empty bucket is within 14 steps
        // because that is the largest edge weight, so this always terminates.
        cost++;
        continue;
      }

      const cur = bucket.pop()!;
      queued--;
      // Stale entry: this tile was reached more cheaply after being queued.
      if (this.dist[cur]! !== cost) continue;

      const cx = cur % w;
      const cy = (cur / w) | 0;

      for (let n = 0; n < NEIGHBOURS.length; n++) {
        const [dx, dy, step] = NEIGHBOURS[n]!;
        const nx = cx + dx;
        const ny = cy + dy;
        if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
        if (!map.isWalkable(nx, ny)) continue;
        // Same anti-corner-cutting rule as A*, so units cannot slip diagonally
        // between two blocked tiles.
        if (dx !== 0 && dy !== 0) {
          if (!map.isWalkable(cx + dx, cy)) continue;
          if (!map.isWalkable(cx, cy + dy)) continue;
        }
        const ni = ny * w + nx;
        const next = cost + step;
        if (next < this.dist[ni]!) {
          this.dist[ni] = next;
          this.buckets[next % BUCKET_COUNT]!.push(ni);
          queued++;
        }
      }
    }
  }

  /**
   * Best neighbouring tile to step to from `tile`, or -1 when already at the
   * goal or stranded.
   *
   * Among neighbours with the same remaining distance, the one whose centre is
   * nearest the unit's own position `(px, py)` wins, and after that the lowest
   * tile index in the unit's canonical frame (`flip`). Both are the same for a
   * mirrored unit reading the mirrored field, which "lowest tile index" was
   * not: index inverts under the map's rotation, so one half stepped to the
   * top-left of two equal tiles and the other to what is, from its side, the
   * bottom-right. Open ground offers such ties every tick, and the two armies
   * traced visibly different lines to the same objective.
   */
  stepFrom(map: GameMap, tile: number, flip = false, px?: Fix, py?: Fix): number {
    const here = this.dist[tile];
    if (here === undefined || here === UNREACHABLE) return -1;
    if (here === 0) return -1;

    const w = map.width;
    const h = map.height;
    const cx = tile % w;
    const cy = (tile / w) | 0;
    const fromX = px ?? fromInt(cx) + FIX_HALF;
    const fromY = py ?? fromInt(cy) + FIX_HALF;

    let bestTile = -1;
    let bestDist = here;
    let bestNear = 0;
    let bestKey = 0;

    for (let n = 0; n < NEIGHBOURS.length; n++) {
      const [dx, dy] = NEIGHBOURS[n]!;
      const nx = cx + dx;
      const ny = cy + dy;
      if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
      if (!map.isWalkable(nx, ny)) continue;
      if (dx !== 0 && dy !== 0) {
        if (!map.isWalkable(cx + dx, cy)) continue;
        if (!map.isWalkable(cx, cy + dy)) continue;
      }
      const ni = ny * w + nx;
      const d = this.dist[ni]!;
      if (d === UNREACHABLE || d > bestDist) continue;
      const near = vecLenSqRaw(fromInt(nx) + FIX_HALF - fromX, fromInt(ny) + FIX_HALF - fromY);
      const key = map.canonicalIndex(ni, flip);
      if (
        d < bestDist ||
        bestTile === -1 ||
        near < bestNear ||
        (near === bestNear && key < bestKey)
      ) {
        bestDist = d;
        bestTile = ni;
        bestNear = near;
        bestKey = key;
      }
    }

    return bestTile;
  }

  /**
   * `stepFrom` asked from a tile's own centre, memoised.
   *
   * Steering looks several tiles down the field every tick, and every unit on a
   * tile gets the same answer for every step past the first — the question
   * depends on the tile, not on who is asking. Without this, an army of 120
   * marching units re-derived the same handful of answers some 800 times a
   * tick, which was most of what looking ahead cost.
   *
   * One cache per frame: `flip` decides the order equally good neighbours are
   * compared in, so the two halves of the map genuinely have different answers.
   * The flipped half's cache is allocated only if something asks for it, which
   * in an ordinary match is never — two players share a field only when they
   * order moves onto the very same tile.
   *
   * Scratch, like the field itself: derived from the map and the goal, and
   * excluded from the world checksum along with the rest of the cache.
   */
  stepFromCentre(map: GameMap, tile: number, flip: boolean): number {
    let cache = flip ? this.centreStepFlipped : this.centreStep;
    if (cache === null) {
      cache = new Int32Array(this.dist.length).fill(UNCACHED);
      if (flip) this.centreStepFlipped = cache;
      else this.centreStep = cache;
    }
    // Same guard `stepFrom` keeps on `dist`: an out-of-range tile has no answer
    // to remember, and reading one out of the cache would hand back undefined
    // where the caller is promised a tile index or -1.
    if (tile < 0 || tile >= cache.length) return this.stepFrom(map, tile, flip);
    const hit = cache[tile]!;
    if (hit !== UNCACHED) return hit;
    const step = this.stepFrom(map, tile, flip);
    cache[tile] = step;
    return step;
  }

  /** True when this tile has no route to the goal. */
  isStranded(tile: number): boolean {
    return this.dist[tile] === UNREACHABLE;
  }
}

/**
 * Small cache of recently used flow fields.
 *
 * Keyed by destination tile. Evicts least-recently-used, and rebuilds any field
 * whose map occupancy version is stale — a new building can invalidate routes,
 * and a unit walking into a wall because it followed a field built before the
 * wall existed would be both a visible bug and, since it depends on cache
 * contents, a desync.
 *
 * The cache is scratch: it is derived entirely from the map and the requested
 * goal, so it is deliberately excluded from the world checksum. Two peers with
 * differently-warmed caches still compute identical fields.
 */
export class FlowFieldCache {
  private readonly fields: FlowField[] = [];
  private readonly order: number[] = [];

  constructor(
    private readonly tileCount: number,
    private readonly capacity = 12,
  ) {}

  get(map: GameMap, goalTile: number): FlowField {
    for (let i = 0; i < this.fields.length; i++) {
      const f = this.fields[i]!;
      if (f.goalTile === goalTile) {
        if (f.builtVersion !== map.occupancyVersion) f.build(map, goalTile);
        this.touch(i);
        return f;
      }
    }

    let field: FlowField;
    if (this.fields.length < this.capacity) {
      field = new FlowField(this.tileCount);
      this.fields.push(field);
      this.order.push(this.fields.length - 1);
    } else {
      // Evict least recently used.
      const lru = this.order.shift()!;
      field = this.fields[lru]!;
      this.order.push(lru);
    }
    field.build(map, goalTile);
    return field;
  }

  private touch(index: number): void {
    const at = this.order.indexOf(index);
    if (at >= 0) this.order.splice(at, 1);
    this.order.push(index);
  }
}
