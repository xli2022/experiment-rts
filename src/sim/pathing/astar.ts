/**
 * Deterministic A* over the tile grid.
 *
 * ## What makes this deterministic
 *
 * A* is only as reproducible as its tie-breaking. When two nodes have the same
 * f-score — which happens constantly on a uniform grid — the order they come out
 * of the open set decides the path. A binary heap that leaves ties to
 * implementation detail will hand two peers different-but-equally-valid routes,
 * and the simulations separate immediately.
 *
 * So the heap comparator is a *total* order: f-score, then cost so far, then
 * tile index. Two nodes can never compare equal, because no two nodes share a
 * tile index.
 *
 * ## What makes it rotation-equivariant
 *
 * The tile index is compared in the requesting player's *canonical frame*, and
 * neighbours are enumerated in that frame too. Row-major index inverts under
 * the map's 180-degree rotation, so "lowest index wins" picked the top-left
 * route for one player and the top-left route for their opposite number as
 * well — which, seen from the second player's side, is the bottom-right one.
 * Equal-cost routes therefore came out differently shaped for the two halves,
 * and a differently shaped polyline is walked at a different real speed. With
 * both decisions made in the canonical frame, the mirrored request returns the
 * mirrored route, tile for tile.
 *
 * Costs are integers (10 orthogonal, 14 diagonal — a 1.4 approximation of √2),
 * which keeps the whole search in exact integer arithmetic.
 */

import type { GameMap } from '../map.js';

const COST_STRAIGHT = 10;
const COST_DIAGONAL = 14;

/**
 * Give up after this many expansions.
 *
 * This bounds the cost of a *failed* search, which is the expensive case: a
 * reachable goal on a 128x128 map is found in a few hundred expansions thanks to
 * the octile heuristic, while an unreachable one explores until the budget runs
 * out. Callers must back off after a failure (see `pathCooldown`) rather than
 * retrying next tick.
 */
const MAX_EXPANSIONS = 2500;

/** Neighbour offsets: 4 orthogonal first, then 4 diagonal. Fixed order. */
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

export class AStar {
  private readonly width: number;
  private readonly height: number;

  private readonly gScore: Int32Array;
  private readonly cameFrom: Int32Array;
  /**
   * Generation stamp per tile instead of clearing the arrays between searches.
   * Clearing 16k entries per path request would dominate the cost; comparing a
   * stamp is free and equally deterministic.
   */
  private readonly stampAt: Int32Array;
  private stamp = 0;

  /**
   * Binary heap of open nodes, as parallel arrays.
   *
   * Sized by *pushes*, not by tile count. A node is re-pushed every time it is
   * relaxed to a cheaper score, so the heap can hold far more entries than there
   * are tiles — up to eight per expansion. Sizing this to the tile count would
   * overflow, and because writing past the end of a typed array fails silently
   * rather than throwing, the result is a quietly corrupted heap and wrong
   * paths, not a crash.
   */
  private readonly heapF: Int32Array;
  private readonly heapNode: Int32Array;
  private heapSize = 0;

  /** Scratch for the reconstructed path, reversed in place before returning. */
  private readonly scratch: number[] = [];

  /** Whether the current search is for a player on the rotated half. */
  private flip = false;
  private readonly tileCount: number;

  constructor(map: GameMap) {
    this.width = map.width;
    this.height = map.height;
    const n = map.width * map.height;
    this.tileCount = n;
    this.gScore = new Int32Array(n);
    this.cameFrom = new Int32Array(n);
    this.stampAt = new Int32Array(n);
    const maxPushes = (MAX_EXPANSIONS + 1) * NEIGHBOURS.length + 1;
    this.heapF = new Int32Array(maxPushes);
    this.heapNode = new Int32Array(maxPushes);
  }

  /**
   * Find a path from `startIndex` to `goalIndex`.
   *
   * Returns tile indices from the first step after the start through the goal,
   * or an empty array when no route exists. The start tile is omitted because
   * the unit is already standing on it.
   *
   * `flip` is `World.flipOf` for the unit the path is for: it decides the frame
   * every tie is broken in, so a mirrored request gets the mirrored route.
   */
  find(
    map: GameMap,
    startIndex: number,
    goalIndex: number,
    out: number[] = [],
    flip = false,
  ): number[] {
    out.length = 0;
    if (startIndex < 0 || goalIndex < 0) return out;
    if (startIndex === goalIndex) return out;

    this.flip = flip;
    this.stamp++;
    this.heapSize = 0;

    const gx = goalIndex % this.width;
    const gy = (goalIndex / this.width) | 0;

    this.gScore[startIndex] = 0;
    this.cameFrom[startIndex] = -1;
    this.stampAt[startIndex] = this.stamp;
    this.push(
      this.heuristic(startIndex % this.width, (startIndex / this.width) | 0, gx, gy),
      startIndex,
    );

    let expansions = 0;
    let found = false;

    while (this.heapSize > 0) {
      const current = this.pop();
      if (current === goalIndex) {
        found = true;
        break;
      }
      if (++expansions > MAX_EXPANSIONS) break;

      const cx = current % this.width;
      const cy = (current / this.width) | 0;
      const cg = this.gScore[current]!;

      for (let n = 0; n < NEIGHBOURS.length; n++) {
        const [ox, oy, cost] = NEIGHBOURS[n]!;
        // Enumerated in the canonical frame, so the first of two equal-cost
        // parents is the mirrored one for a mirrored search.
        const dx = flip ? -ox : ox;
        const dy = flip ? -oy : oy;
        const nx = cx + dx;
        const ny = cy + dy;
        if (nx < 0 || ny < 0 || nx >= this.width || ny >= this.height) continue;
        if (!map.isWalkable(nx, ny)) continue;

        // Forbid cutting corners diagonally between two blocked tiles, which
        // otherwise lets units slip through walls that visually touch.
        if (dx !== 0 && dy !== 0) {
          if (!map.isWalkable(cx + dx, cy)) continue;
          if (!map.isWalkable(cx, cy + dy)) continue;
        }

        const ni = ny * this.width + nx;
        const tentative = cg + cost;
        const seen = this.stampAt[ni] === this.stamp;
        if (seen && tentative >= this.gScore[ni]!) continue;

        this.gScore[ni] = tentative;
        this.cameFrom[ni] = current;
        this.stampAt[ni] = this.stamp;
        this.push(tentative + this.heuristic(nx, ny, gx, gy), ni);
      }
    }

    if (!found) return out;

    // Walk the parent chain back and reverse, dropping the start tile.
    this.scratch.length = 0;
    let node = goalIndex;
    while (node !== -1 && node !== startIndex) {
      this.scratch.push(node);
      node = this.cameFrom[node]!;
    }
    for (let i = this.scratch.length - 1; i >= 0; i--) out.push(this.scratch[i]!);
    return out;
  }

  /**
   * Octile distance, scaled to match the integer step costs. Admissible, so A*
   * still returns a shortest path.
   */
  private heuristic(ax: number, ay: number, bx: number, by: number): number {
    const dx = ax > bx ? ax - bx : bx - ax;
    const dy = ay > by ? ay - by : by - ay;
    const lo = dx < dy ? dx : dy;
    const hi = dx < dy ? dy : dx;
    return COST_STRAIGHT * (hi - lo) + COST_DIAGONAL * lo;
  }

  private push(f: number, node: number): void {
    let i = this.heapSize++;
    this.heapF[i] = f;
    this.heapNode[i] = node;
    // Sift up.
    while (i > 0) {
      const parent = (i - 1) >> 1;
      if (this.less(i, parent)) {
        this.swap(i, parent);
        i = parent;
      } else break;
    }
  }

  private pop(): number {
    const top = this.heapNode[0]!;
    this.heapSize--;
    if (this.heapSize > 0) {
      this.heapF[0] = this.heapF[this.heapSize]!;
      this.heapNode[0] = this.heapNode[this.heapSize]!;
      // Sift down.
      let i = 0;
      for (;;) {
        const l = 2 * i + 1;
        const r = l + 1;
        let best = i;
        if (l < this.heapSize && this.less(l, best)) best = l;
        if (r < this.heapSize && this.less(r, best)) best = r;
        if (best === i) break;
        this.swap(i, best);
        i = best;
      }
    }
    return top;
  }

  /**
   * Strict total order on heap entries: f-score, then the larger cost so far
   * (the node nearer the goal), then tile index in the canonical frame.
   *
   * The tile-index tiebreak is the load-bearing part. Without it, equal-f nodes
   * pop in an order that depends on heap history, and two peers exploring the
   * same grid produce different paths. Preferring the deeper node first is a
   * magnitude, so it is the same for a mirrored search, and it stops the
   * frontier fanning out across every equally good tile before it commits.
   */
  private less(a: number, b: number): boolean {
    const fa = this.heapF[a]!;
    const fb = this.heapF[b]!;
    if (fa !== fb) return fa < fb;
    const na = this.heapNode[a]!;
    const nb = this.heapNode[b]!;
    const ga = this.gScore[na]!;
    const gb = this.gScore[nb]!;
    if (ga !== gb) return ga > gb;
    return this.canonical(na) < this.canonical(nb);
  }

  private canonical(node: number): number {
    return this.flip ? this.tileCount - 1 - node : node;
  }

  private swap(a: number, b: number): void {
    const f = this.heapF[a]!;
    this.heapF[a] = this.heapF[b]!;
    this.heapF[b] = f;
    const n = this.heapNode[a]!;
    this.heapNode[a] = this.heapNode[b]!;
    this.heapNode[b] = n;
  }
}

/**
 * Nearest walkable tile to a target, by true distance.
 *
 * Used when a player right-clicks onto a cliff or inside a building footprint:
 * rather than refusing the order, units walk to the closest reachable tile.
 *
 * "Nearest" is the straight-line distance and ties go to the lowest tile index
 * in the requester's canonical frame (`flip`). It used to return the first
 * walkable tile met on the first ring that had one, scanning each ring from its
 * top-left corner — which is the tile nearest the map's origin, not the tile
 * nearest the target, and the *same* absolute corner for both halves of the
 * map. Every attack-move on a building was aimed at it, so both armies
 * converged on the same corner of either base, five tiles from where a true
 * mirror would have sent one of them. A whole ring is a Chebyshev shell, so
 * the search carries on until the next ring cannot beat the best so far.
 */
export function nearestWalkable(
  map: GameMap,
  tx: number,
  ty: number,
  maxRing = 12,
  flip = false,
): number {
  if (map.isWalkable(tx, ty)) return map.index(tx, ty);
  let best = -1;
  let bestDist = Number.POSITIVE_INFINITY;
  let bestKey = 0;
  for (let r = 1; r <= maxRing && r * r < bestDist; r++) {
    for (let dy = -r; dy <= r; dy++) {
      for (let dx = -r; dx <= r; dx++) {
        // Only the ring perimeter.
        if (dx > -r && dx < r && dy > -r && dy < r) continue;
        const nx = tx + dx;
        const ny = ty + dy;
        if (!map.isWalkable(nx, ny)) continue;
        const d = dx * dx + dy * dy;
        const tile = map.index(nx, ny);
        const key = map.canonicalIndex(tile, flip);
        if (d < bestDist || (d === bestDist && key < bestKey)) {
          bestDist = d;
          best = tile;
          bestKey = key;
        }
      }
    }
  }
  return best;
}
