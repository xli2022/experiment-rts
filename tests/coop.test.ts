/**
 * Co-op: two players a side.
 *
 * Nearly every rule in the simulation used to be able to ask "is this mine?"
 * and get the right answer, because everyone who was not you was your enemy.
 * That is no longer true, and the failures it produces are quiet ones — a unit
 * that chases a partner it can never damage, fog that stops at your own units,
 * a match that ends the moment one of two allies loses their last building. So
 * the tests here are mostly about *sides*: what changed is the question being
 * asked, not the machinery answering it.
 *
 * The 1v1 is checked too, in the same file and by the same assertions, because
 * it is now the degenerate case of the same code — a team per player, with team
 * ids equal to player ids — and the cheapest way to know that generalisation
 * did not cost anything is to state it as a property.
 */

import { describe, expect, it } from 'vitest';
import { HeadlessMatch } from '../src/ai/headless.js';
import { PATCHES_PER_BASE, defOf } from '../src/config/rules.js';
import { CommandType } from '../src/sim/commands.js';
import { fromInt } from '../src/sim/fixed.js';
import {
  coopMatch,
  duelMatch,
  hostOf,
  hostedBy,
  humanCount,
  isBotSlot,
  matchConfig,
  teamsFor,
} from '../src/sim/match.js';
import { GameMap, generateMap } from '../src/sim/map.js';
import { mirror } from '../src/sim/mapgen.js';
import { colourSlotFor } from '../src/render/models/procedural.js';
import { Simulation } from '../src/sim/tick.js';
import {
  BotKind,
  BuildState,
  EntityType,
  MapLayout,
  NEUTRAL,
  NO_ENTITY,
  Order,
  Tile,
  type PlayerId,
} from '../src/sim/types.js';
import { executeCommand } from '../src/sim/systems/orders.js';
import type { World } from '../src/sim/world.js';
import { scriptedAgents } from './helpers/agents.js';

const SEEDS = [0x51ce7a11, 0, 7, 99, 0xdecafbad | 0];

/** Spawn a unit for `owner` at a tile, and return its slot. */
function spawnAt(world: World, type: EntityType, owner: PlayerId, x: number, y: number): number {
  const id = world.pool.spawn(type, owner, fromInt(x), fromInt(y));
  expect(id).not.toBe(NO_ENTITY);
  return id & 0xffff;
}

/** Every live entity a player owns. */
function owned(world: World, player: PlayerId): number[] {
  const out: number[] = [];
  for (let i = 0; i < world.pool.count; i++) {
    if (world.pool.alive[i] === 1 && world.pool.owner[i] === player) out.push(i);
  }
  return out;
}

/**
 * Reachability between two tiles, with some regions optionally sealed.
 *
 * Expressed over `walkingDistance` rather than as a second flood fill: the two
 * ran the same block-stamping and the same queue walk, and the same test called
 * both on the same arguments — so a fix to one would have left "is connected"
 * and "how far" answering about different maps, with nothing failing.
 */
function connected(
  map: GameMap,
  a: { tileX: number; tileY: number },
  b: { tileX: number; tileY: number },
  blocks: { x: number; y: number; r: number }[],
): boolean {
  return walkingDistance(map, a, b, blocks) >= 0;
}

/** Four-directional walking distance between two tiles, or -1 if unreachable. */
function walkingDistance(
  map: GameMap,
  a: { tileX: number; tileY: number },
  b: { tileX: number; tileY: number },
  blocks: { x: number; y: number; r: number }[],
): number {
  const size = map.width;
  const blocked = Uint8Array.from(map.tiles);
  for (const block of blocks) {
    for (let y = block.y - block.r; y <= block.y + block.r; y++) {
      for (let x = block.x - block.r; x <= block.x + block.r; x++) {
        if (x < 0 || y < 0 || x >= size || y >= size) continue;
        const dx = x - block.x;
        const dy = y - block.y;
        if (dx * dx + dy * dy > block.r * block.r) continue;
        blocked[y * size + x] = Tile.Cliff;
      }
    }
  }

  const goal = map.index(b.tileX, b.tileY);
  const dist = new Int32Array(size * size).fill(-1);
  const queue = [map.index(a.tileX, a.tileY)];
  dist[queue[0]!] = 0;
  for (let head = 0; head < queue.length; head++) {
    const cur = queue[head]!;
    if (cur === goal) return dist[cur]!;
    const cx = cur % size;
    const cy = (cur / size) | 0;
    for (let d = 0; d < 4; d++) {
      const nx = cx + (d === 0 ? 1 : d === 1 ? -1 : 0);
      const ny = cy + (d === 2 ? 1 : d === 3 ? -1 : 0);
      if (nx < 0 || ny < 0 || nx >= size || ny >= size) continue;
      const ni = ny * size + nx;
      if (dist[ni] !== -1 || blocked[ni] === Tile.Cliff) continue;
      dist[ni] = dist[cur]! + 1;
      queue.push(ni);
    }
  }
  return -1;
}

describe('match configuration', () => {
  it('gives each duel player their own side, numbered like the players', () => {
    // The whole reason nothing about the 1v1 had to change. If team ids ever
    // stop matching player ids here, `world.winner` quietly starts meaning
    // something else in every existing match.
    expect(teamsFor(MapLayout.Lanes)).toEqual([0, 1]);
  });

  it('splits the four-corner map two a side, first half against second', () => {
    expect(teamsFor(MapLayout.Quarters)).toEqual([0, 0, 1, 1]);
  });

  it('normalises the bot roster rather than trusting the order given', () => {
    // Two peers that built the same roster in different orders must produce the
    // same config, or the checksum folds in a difference that is not one.
    const a = coopMatch(7, { botPlayers: [3, 2] });
    const b = coopMatch(7, { botPlayers: [2, 3, 2] });
    expect(a.bots).toEqual(b.bots);
    expect(a.bots.map((s) => s.player)).toEqual([2, 3]);
  });

  it('drops bot slots that are not in the match', () => {
    const config = duelMatch(7, { botPlayers: [1, 5, -1] });
    expect(config.bots.map((s) => s.player)).toEqual([1]);
  });

  it('leaves the humans on a contiguous prefix of the roster', () => {
    // The transports number their peers from zero and every bot is dealt to
    // one of those peers, so a bot in slot 0 with a human in slot 2 would
    // leave a slot nobody sends for and stall every peer forever.
    for (const config of [
      duelMatch(1, { botPlayers: [1] }),
      coopMatch(1),
      coopMatch(1, { botPlayers: [1, 2, 3] }),
    ]) {
      const humans = humanCount(config);
      for (const bot of config.bots) expect(bot.player).toBeGreaterThanOrEqual(humans);
    }
  });

  it('takes per-slot brains, and names win over the shorthand', () => {
    // A scripted partner beside neural opponents is the roster the lobby
    // builds for solo co-op; the two ways of naming a slot must agree.
    const mixed = coopMatch(7, {
      botSlots: [
        { player: 3, kind: BotKind.Neural },
        { player: 1, kind: BotKind.Scripted },
        { player: 2, kind: BotKind.Neural },
      ],
    });
    expect(mixed.bots).toEqual([
      { player: 1, kind: BotKind.Scripted },
      { player: 2, kind: BotKind.Neural },
      { player: 3, kind: BotKind.Neural },
    ]);
    const both = coopMatch(7, {
      botPlayers: [2, 3],
      kind: BotKind.Scripted,
      botSlots: [{ player: 3, kind: BotKind.Neural }],
    });
    expect(both.bots).toEqual([
      { player: 2, kind: BotKind.Scripted },
      { player: 3, kind: BotKind.Neural },
    ]);
    // Naming slots either way switches the co-op default off entirely.
    expect(coopMatch(7, { botSlots: [{ player: 3, kind: BotKind.Neural }] }).bots).toEqual([
      { player: 3, kind: BotKind.Neural },
    ]);
  });

  it('deals every bot to a human peer, round-robin in roster order', () => {
    // Derived from the agreed config and never negotiated, so every peer
    // computes the same dealing. Single-player hosts everything on peer 0;
    // two tabs take one bot each; a bot below the human prefix cannot occur.
    const duel = duelMatch(1, { botPlayers: [1], kind: BotKind.Neural });
    expect(isBotSlot(duel, 1)).toBe(true);
    expect(isBotSlot(duel, 0)).toBe(false);
    expect(hostOf(duel, 0)).toBe(0);
    expect(hostOf(duel, 1)).toBe(0);
    expect(hostedBy(duel, 0)).toEqual([1]);

    const coop = coopMatch(1, { botPlayers: [2, 3] });
    expect(hostOf(coop, 2)).toBe(0);
    expect(hostOf(coop, 3)).toBe(1);
    expect(hostedBy(coop, 0)).toEqual([2]);
    expect(hostedBy(coop, 1)).toEqual([3]);

    const solo = coopMatch(1, {
      botSlots: [
        { player: 1, kind: BotKind.Scripted },
        { player: 2, kind: BotKind.Neural },
        { player: 3, kind: BotKind.Neural },
      ],
    });
    expect(humanCount(solo)).toBe(1);
    expect(hostedBy(solo, 0)).toEqual([1, 2, 3]);

    const mixedPair = coopMatch(1, {
      botSlots: [
        { player: 2, kind: BotKind.Scripted },
        { player: 3, kind: BotKind.Neural },
      ],
    });
    expect(hostedBy(mixedPair, 0)).toEqual([2]);
    expect(hostedBy(mixedPair, 1)).toEqual([3]);

    // No humans at all: nothing is on a wire, and a headless driver owns it all.
    const allBots = coopMatch(1, { botPlayers: [0, 1, 2, 3] });
    expect(hostedBy(allBots, 0)).toEqual([0, 1, 2, 3]);
  });

  it('folds the roster into the checksum', () => {
    // A lobby mismatch is otherwise an inexplicable divergence some seconds in.
    // Folded in, it is a desync on the very first comparison — including which
    // bot plays a slot, which decides who hosts it.
    const scripted = new Simulation(coopMatch(5, { kind: BotKind.Scripted }));
    const neural = new Simulation(coopMatch(5, { kind: BotKind.Neural }));
    expect(scripted.checksum()).not.toBe(neural.checksum());
  });
});

describe('sides', () => {
  it('makes partners friendly and opponents hostile', () => {
    const world = new Simulation(coopMatch(SEEDS[0]!)).world;
    expect(world.areAllied(0, 1)).toBe(true);
    expect(world.areAllied(2, 3)).toBe(true);
    expect(world.areAllied(0, 2)).toBe(false);
    expect(world.areAllied(1, 3)).toBe(false);
    // A player is their own ally: every caller is really asking "may I shoot
    // this", and the answer for your own units is the same no.
    expect(world.areAllied(0, 0)).toBe(true);
  });

  it('never treats a neutral entity as an ally or an enemy of anyone', () => {
    const world = new Simulation(coopMatch(SEEDS[0]!)).world;
    expect(world.areAllied(NEUTRAL, 0)).toBe(false);
    let patch = -1;
    for (let i = 0; i < world.pool.count; i++) {
      if (world.pool.alive[i] === 1 && world.pool.type[i] === EntityType.MineralPatch) {
        patch = i;
        break;
      }
    }
    expect(patch).toBeGreaterThanOrEqual(0);
    for (let p = 0; p < world.players.length; p++) expect(world.isHostile(patch, p)).toBe(false);
  });

  it('keeps every duel player hostile to the other', () => {
    const world = new Simulation(duelMatch(SEEDS[0]!)).world;
    expect(world.areAllied(0, 1)).toBe(false);
  });
});

describe('the four-corner map', () => {
  it('is exactly symmetric under a 180 degree rotation', () => {
    for (const seed of SEEDS) {
      const map = generateMap(seed, 152, MapLayout.Quarters);
      let mismatched = 0;
      for (let y = 0; y < map.height; y++) {
        for (let x = 0; x < map.width; x++) {
          const here = map.index(x, y);
          const there = map.index(map.width - 1 - x, map.height - 1 - y);
          if (map.tiles[here] !== map.tiles[there]) mismatched++;
          if (map.elevation[here] !== map.elevation[there]) mismatched++;
        }
      }
      expect(`seed ${seed}: ${mismatched} mismatched`).toBe(`seed ${seed}: 0 mismatched`);
    }
  });

  it('stores starts and expansions in mirrored halves', () => {
    // The convention the whole fairness argument rests on: `i` and `i + n/2`
    // are rotations of each other, so the first half is one side of the map and
    // the second is the other.
    const map = new Simulation(coopMatch(SEEDS[0]!)).world.map;
    for (const sites of [map.starts, map.expansions]) {
      const half = sites.length >> 1;
      expect(half).toBeGreaterThan(0);
      for (let i = 0; i < half; i++) {
        const a = sites[i]!;
        const b = sites[i + half]!;
        expect(mirror({ x: a.tileX, y: a.tileY }, map.width)).toEqual({
          x: b.tileX,
          y: b.tileY,
        });
      }
    }
  });

  it('gives each side its own half of the map', () => {
    // A team to each side, not two diagonal pairs. Splitting the four corners
    // diagonally is the other arrangement that keeps the halves mirrored, and
    // it puts each player's partner across the map from them — a 1v1 played
    // twice rather than a co-op match.
    const world = new Simulation(coopMatch(SEEDS[0]!)).world;
    const middle = world.map.height / 2;
    for (let p = 0; p < world.players.length; p++) {
      const y = world.map.starts[p]!.tileY;
      if (world.teamOf(p) === 0) expect(y).toBeLessThan(middle);
      else expect(y).toBeGreaterThan(middle);
    }
  });

  it('puts each player furthest of all from their opposite number', () => {
    const world = new Simulation(coopMatch(SEEDS[0]!)).world;
    const starts = world.map.starts;
    const dist = (a: PlayerId, b: PlayerId): number => {
      const dx = starts[a]!.tileX - starts[b]!.tileX;
      const dy = starts[a]!.tileY - starts[b]!.tileY;
      return Math.sqrt(dx * dx + dy * dy);
    };
    // `p + n/2` is the slot whose opening is this one's 180-degree rotation,
    // and on four corners that is the far diagonal. The other opponent is the
    // one directly across the front line — which is what gives each player
    // someone to face and a partner to flank with.
    expect(dist(0, 2)).toBeGreaterThan(dist(0, 1));
    expect(dist(0, 2)).toBeGreaterThan(dist(0, 3));
  });

  it('gives all four openings the same thing', () => {
    for (const seed of SEEDS) {
      const sim = new Simulation(coopMatch(seed));
      const world = sim.world;
      expect(world.map.starts.length).toBe(4);

      for (let p = 0; p < 4; p++) {
        const mine = owned(world, p);
        const workers = mine.filter((i) => world.pool.type[i] === EntityType.Worker);
        const posts = mine.filter((i) => world.pool.type[i] === EntityType.CommandPost);
        expect(posts.length).toBe(1);
        expect(world.pool.buildState[posts[0]!]).toBe(BuildState.Complete);
        expect(workers.length).toBe(6);

        // A base one patch short is a losing opening no amount of skill
        // recovers, and on this map three of the four sites are laid out by
        // rules the duel map never exercised.
        let patches = 0;
        const start = world.map.starts[p]!;
        for (let i = 0; i < world.pool.count; i++) {
          if (world.pool.alive[i] !== 1) continue;
          if (world.pool.type[i] !== EntityType.MineralPatch) continue;
          const dx = world.pool.tileX[i]! - start.tileX;
          const dy = world.pool.tileY[i]! - start.tileY;
          if (dx * dx + dy * dy <= 12 * 12) patches++;
        }
        expect(`seed ${seed} player ${p}: ${patches}`).toBe(
          `seed ${seed} player ${p}: ${PATCHES_PER_BASE}`,
        );
      }
    }
  });

  it('puts each opening mineral line on the outside of its own base', () => {
    // The authored offsets run up and to the left, which tucks the line into
    // the corner for a base on the left of the map and shoves it toward the
    // middle for one on the right. Both are canonical sites here, so one rule
    // for all of them is not enough — the line is reflected for the right-hand
    // base, and this is what says so.
    const world = new Simulation(coopMatch(SEEDS[0]!)).world;
    const centre = world.map.width / 2;
    for (let p = 0; p < 4; p++) {
      const start = world.map.starts[p]!;
      let sum = 0;
      let n = 0;
      for (let i = 0; i < world.pool.count; i++) {
        if (world.pool.alive[i] !== 1) continue;
        if (world.pool.type[i] !== EntityType.MineralPatch) continue;
        const dx = world.pool.tileX[i]! - start.tileX;
        const dy = world.pool.tileY[i]! - start.tileY;
        if (dx * dx + dy * dy > 12 * 12) continue;
        sum += world.pool.tileX[i]!;
        n++;
      }
      const patchCentre = sum / n;
      // Further from the middle of the map than the base itself.
      expect(Math.abs(patchCentre - centre)).toBeGreaterThan(Math.abs(start.tileX - centre));
    }
  });

  it('connects every start to every other', () => {
    for (const seed of SEEDS) {
      const map = generateMap(seed, 152, MapLayout.Quarters);
      for (let a = 0; a < map.starts.length; a++) {
        for (let b = a + 1; b < map.starts.length; b++) {
          expect(connected(map, map.starts[a]!, map.starts[b]!, [])).toBe(true);
        }
      }
    }
  });

  it('gives a team a road home that does not pass through the middle', () => {
    // The reason allies share an edge at all. Sever the centre — the ground both
    // teams have to fight over — and a partner is still a short walk away along
    // the back lane. Sever the back lane instead and the only way to a partner
    // is the long way round the map, through the enemy's half: the route
    // survives, because the map is a ring, but it stops being a road home.
    for (const seed of SEEDS) {
      const map = generateMap(seed, 152, MapLayout.Quarters);
      const size = map.width;
      const middle = { x: (size >> 1) - 1, y: (size >> 1) - 1, r: 30 };
      const backLane = { x: size >> 1, y: Math.floor(size * 0.1), r: 26 };
      const a = map.starts[0]!;
      const b = map.starts[1]!;

      const direct = walkingDistance(map, a, b, [middle]);
      const detour = walkingDistance(map, a, b, [middle, backLane]);
      expect(connected(map, a, b, [middle])).toBe(true);
      // Roughly the width of the map, so the back lane really does run straight
      // between the two bases rather than looping out and back.
      expect(direct).toBeGreaterThan(0);
      expect(direct).toBeLessThan(size * 1.3);
      expect(detour).toBeGreaterThan(direct * 2);
    }
  });

  it('carves lanes, not a field and not a maze', () => {
    for (const seed of SEEDS) {
      const map = generateMap(seed, 152, MapLayout.Quarters);
      let open = 0;
      for (let i = 0; i < map.tiles.length; i++) {
        if (map.tiles[i] !== Tile.Cliff) open++;
      }
      const fraction = open / map.tiles.length;
      expect(fraction).toBeGreaterThan(0.25);
      expect(fraction).toBeLessThan(0.5);
    }
  });

  it('seals the border', () => {
    const map = generateMap(3, 152, MapLayout.Quarters);
    for (let i = 0; i < map.width; i++) {
      expect(map.isGroundWalkable(i, 0)).toBe(false);
      expect(map.isGroundWalkable(i, map.height - 1)).toBe(false);
      expect(map.isGroundWalkable(0, i)).toBe(false);
      expect(map.isGroundWalkable(map.width - 1, i)).toBe(false);
    }
  });

  it('leaves the duel map alone', () => {
    // Two layouts share one carver, and the one that already exists is tuned.
    const a = generateMap(SEEDS[0]!, 128, MapLayout.Lanes);
    const b = generateMap(SEEDS[0]!);
    expect(Array.from(a.tiles)).toEqual(Array.from(b.tiles));
    expect(a.starts).toEqual(b.starts);
  });
});

describe('player colours', () => {
  it('keeps a duel blue against red, and gives four players two hue families', () => {
    // The palette is laid out by side — entries 0 and 1 are one team's, 2 and 3
    // the other's — so a roster split down the middle reads straight off the
    // player id at four players and *not* at two. Indexed by raw id, a duel
    // painted the lone opponent teal: one shade from the local player's blue,
    // a hair from the mineral colour on the minimap, and contradicting the red
    // skin their own combat units were still wearing. Nothing in the suite
    // could see it, which is why it is a test rather than a comment.
    expect(colourSlotFor(0, 2)).toBe(0);
    expect(colourSlotFor(1, 2)).toBe(2);
    for (let p = 0; p < 4; p++) expect(colourSlotFor(p, 4)).toBe(p);

    // The property underneath both: partners share a hue family, opponents do
    // not. Checked against the real team split rather than against the indices.
    for (const config of [duelMatch(1), coopMatch(1)]) {
      const world = new Simulation(config).world;
      const n = world.players.length;
      for (let a = 0; a < n; a++) {
        for (let b = 0; b < n; b++) {
          const sameFamily = colourSlotFor(a, n) >> 1 === colourSlotFor(b, n) >> 1;
          expect(`${a},${b}:${sameFamily}`).toBe(`${a},${b}:${world.areAllied(a, b)}`);
        }
      }
    }
  });
});

describe('friendly fire', () => {
  /**
   * Stand two units next to each other well away from anyone's base and let
   * them look at each other for a few seconds.
   *
   * Away from a base deliberately: workers have a weapon and will shoot
   * anything that comes near their minerals, so a fight staged at a start
   * location measures the workers as much as the units under test.
   */
  function stage(allied: boolean): { world: World; a: number; b: number; shots: number } {
    const sim = new Simulation(coopMatch(SEEDS[0]!));
    const world = sim.world;
    const centre = world.map.width >> 1;
    const other: PlayerId = allied ? 1 : 2;
    const a = spawnAt(world, EntityType.Burstbot, 0, centre, centre);
    const b = spawnAt(world, EntityType.Burstbot, other, centre + 2, centre);

    let shots = 0;
    for (let t = 0; t < 60; t++) {
      sim.step([]);
      shots += world.events.shots.length / 2;
    }
    return { world, a, b, shots };
  }

  it('does not let a unit acquire a partner as a target', () => {
    const { world, a, b, shots } = stage(true);
    expect(shots).toBe(0);
    expect(world.pool.alive[a]).toBe(1);
    expect(world.pool.alive[b]).toBe(1);
    expect(world.pool.combatTarget[a]).toBe(NO_ENTITY);
  });

  it('still shoots an opponent standing in the same place', () => {
    // The control. Without it the test above passes just as well when target
    // acquisition is broken outright.
    const { shots } = stage(false);
    expect(shots).toBeGreaterThan(0);
  });

  it('refuses an explicit attack order aimed at a partner', () => {
    // Combat would refuse the damage anyway, so the cost of letting the order
    // through is not a misfire but an order that never ends: the unit walks to
    // its partner and stands there holding an Attack it can never resolve.
    const sim = new Simulation(coopMatch(SEEDS[0]!));
    const world = sim.world;
    const centre = world.map.width >> 1;
    const mine = spawnAt(world, EntityType.Burstbot, 0, centre, centre);
    const ally = spawnAt(world, EntityType.Burstbot, 1, centre + 6, centre);
    const enemy = spawnAt(world, EntityType.Burstbot, 2, centre - 6, centre);

    executeCommand(world, {
      type: CommandType.Attack,
      player: 0,
      units: [world.pool.idAt(mine)],
      target: world.pool.idAt(ally),
    });
    expect(world.pool.order[mine]).toBe(Order.None);

    executeCommand(world, {
      type: CommandType.Attack,
      player: 0,
      units: [world.pool.idAt(mine)],
      target: world.pool.idAt(enemy),
    });
    expect(world.pool.order[mine]).toBe(Order.Attack);
  });
});

describe('victory is decided per side', () => {
  /** Destroy every building a player owns, the way combat eventually would. */
  function razeBuildings(world: World, player: PlayerId): void {
    for (const i of owned(world, player)) {
      if (!defOf(world.pool.type[i]! as EntityType).isBuilding) continue;
      world.pool.destroy(world.pool.idAt(i));
    }
  }

  it('keeps the match running while a partner still stands', () => {
    const sim = new Simulation(coopMatch(SEEDS[0]!));
    const world = sim.world;
    razeBuildings(world, 0);
    sim.step([]);

    expect(world.player(0).defeated).toBe(true);
    expect(world.player(1).defeated).toBe(false);
    expect(world.matchOver).toBe(false);
  });

  it('ends it when the whole side is gone, and names the side that won', () => {
    const sim = new Simulation(coopMatch(SEEDS[0]!));
    const world = sim.world;
    razeBuildings(world, 0);
    razeBuildings(world, 1);
    sim.step([]);

    expect(world.matchOver).toBe(true);
    // A *side*, not a player. Both surviving opponents won it together.
    expect(world.winner).toBe(1);
    expect(world.teamOf(2)).toBe(1);
  });

  it('still reads as a player id in a duel', () => {
    const sim = new Simulation(duelMatch(SEEDS[0]!));
    const world = sim.world;
    razeBuildings(world, 1);
    sim.step([]);
    expect(world.matchOver).toBe(true);
    expect(world.winner).toBe(0);
  });
});

describe('surrender', () => {
  /** A live army for `player`, well away from anyone's base. */
  function armFor(world: World, player: PlayerId, count = 4): number[] {
    const centre = world.map.width >> 1;
    const out: number[] = [];
    for (let k = 0; k < count; k++) {
      out.push(spawnAt(world, EntityType.Burstbot, player, centre + k, centre + player * 3));
    }
    world.recomputeSupply();
    return out;
  }

  it('ends a duel on the tick it lands, and the other player wins', () => {
    const sim = new Simulation(duelMatch(SEEDS[0]!));
    const world = sim.world;
    executeCommand(world, { type: CommandType.Surrender, player: 1 });
    sim.step([]);

    expect(world.player(1).defeated).toBe(true);
    expect(world.matchOver).toBe(true);
    expect(world.winner).toBe(0);
  });

  it('takes a co-op player out while their side plays on', () => {
    const sim = new Simulation(coopMatch(SEEDS[0]!));
    const world = sim.world;
    executeCommand(world, { type: CommandType.Surrender, player: 0 });
    sim.step([]);

    expect(world.player(0).defeated).toBe(true);
    expect(world.player(1).defeated).toBe(false);
    expect(world.matchOver).toBe(false);
    // And the side they left can still win it for them.
    expect(world.teamOf(0)).toBe(world.teamOf(1));
  });

  it('takes everything the player still owned with it', () => {
    // The seam this closes: elimination is per player and the match is per
    // team, so a co-op player who conceded used to leave an army on the field
    // that fought on and could take no orders — `executeCommand` drops commands
    // from a defeated player. Out is out.
    const sim = new Simulation(coopMatch(SEEDS[0]!));
    const world = sim.world;
    const army = armFor(world, 0);
    expect(owned(world, 0).length).toBeGreaterThan(army.length);

    executeCommand(world, { type: CommandType.Surrender, player: 0 });
    sim.step([]);

    expect(owned(world, 0)).toEqual([]);
    // The partner is untouched — this is elimination, not a team wipe.
    expect(owned(world, 1).length).toBeGreaterThan(0);
  });

  it('frees the ground a departing player left behind', () => {
    // Destroyed buildings have to release their footprint or the map keeps a
    // hole nobody can build on for the rest of the match.
    const sim = new Simulation(coopMatch(SEEDS[0]!));
    const world = sim.world;
    const post = owned(world, 0).find((i) => world.pool.type[i] === EntityType.CommandPost)!;
    const tileX = world.pool.tileX[post]!;
    const tileY = world.pool.tileY[post]!;
    const footprint = defOf(EntityType.CommandPost).footprint;
    expect(world.map.canPlace(tileX, tileY, footprint)).toBe(false);

    executeCommand(world, { type: CommandType.Surrender, player: 0 });
    sim.step([]);

    expect(world.map.canPlace(tileX, tileY, footprint)).toBe(true);
  });

  it('reports the deaths, so they are seen rather than blinked away', () => {
    const sim = new Simulation(coopMatch(SEEDS[0]!));
    const world = sim.world;
    const before = owned(world, 0).length;
    executeCommand(world, { type: CommandType.Surrender, player: 0 });
    sim.step([]);
    // The presentation layer reads `events.deaths` after the step, so queueing
    // them is what makes a conceded base blow up instead of vanishing.
    //
    // Exactly, not at least: `>=` would also pass a sweep that queued the whole
    // *team*, or that queued every entity twice, which are the two ways this
    // could go wrong without anything else noticing.
    expect(world.events.deaths.length).toBe(before);
  });

  it('leaves the partner able to win it for both of them', () => {
    // The claim the co-op lobby makes on screen, and the reason conceding does
    // not simply end the match for your side.
    const sim = new Simulation(coopMatch(SEEDS[0]!));
    const world = sim.world;
    executeCommand(world, { type: CommandType.Surrender, player: 0 });
    sim.step([]);
    expect(world.matchOver).toBe(false);

    for (const p of [2, 3] as PlayerId[]) {
      for (const i of owned(world, p)) {
        if (defOf(world.pool.type[i]! as EntityType).isBuilding) {
          world.pool.destroy(world.pool.idAt(i));
        }
      }
    }
    sim.step([]);

    expect(world.matchOver).toBe(true);
    expect(world.winner).toBe(world.teamOf(1));
  });
});

describe('the stalemate rule razes a base too', () => {
  /**
   * A player reduced to buildings alone, with `minerals` banked.
   *
   * The other route into `victorySystem`'s sweep, and the one nobody concedes
   * their way into: a player who still owns structures but has nothing alive
   * and nothing affordable is out, and since elimination now destroys what a
   * player still owns, that verdict demolishes a standing base. It was harmless
   * while it only set a flag, so it is worth pinning both ways round.
   */
  function stranded(minerals: number): { sim: Simulation; world: World } {
    const sim = new Simulation(coopMatch(SEEDS[0]!));
    const world = sim.world;
    for (const i of owned(world, 0)) {
      if (defOf(world.pool.type[i]! as EntityType).isBuilding) continue;
      world.pool.destroy(world.pool.idAt(i));
    }
    world.player(0).minerals = minerals;
    world.recomputeSupply();
    return { sim, world };
  }

  it('takes the base of a player with nothing alive and nothing affordable', () => {
    const { sim, world } = stranded(0);
    expect(owned(world, 0).length).toBeGreaterThan(0);
    sim.step([]);

    expect(world.player(0).defeated).toBe(true);
    expect(owned(world, 0)).toEqual([]);
    // Elimination is still per player: the partner is untouched.
    expect(world.player(1).defeated).toBe(false);
    expect(owned(world, 1).length).toBeGreaterThan(0);
  });

  it('leaves a player who can still afford a worker alone', () => {
    const { sim, world } = stranded(defOf(EntityType.Worker).mineralCost);
    const before = owned(world, 0).length;
    sim.step([]);

    expect(world.player(0).defeated).toBe(false);
    expect(owned(world, 0).length).toBe(before);
  });

  it('leaves a player whose worker is already paid for and queued', () => {
    // "That position cannot change for the rest of time" has to be true, not
    // nearly true. Minerals spent on a unit already in a queue are gone from the
    // bank, so an affordability test alone declares the player stuck one tick
    // before the worker walks out — and now demolishes the Command Post it is
    // standing in.
    const { sim, world } = stranded(defOf(EntityType.Worker).mineralCost);
    const post = owned(world, 0).find((i) => world.pool.type[i] === EntityType.CommandPost)!;
    executeCommand(world, {
      type: CommandType.Train,
      player: 0,
      building: world.pool.idAt(post),
      unit: EntityType.Worker,
    });
    expect(world.player(0).minerals).toBe(0);
    const before = owned(world, 0).length;
    sim.step([]);

    expect(world.player(0).defeated).toBe(false);
    expect(owned(world, 0).length).toBe(before);
  });
});

describe('the bot on a side', () => {
  /** How close any combat unit stands to a Command Post it is hostile to. */
  function closestToEnemyPost(world: World): number {
    let closest = Number.POSITIVE_INFINITY;
    for (let i = 0; i < world.pool.count; i++) {
      if (world.pool.alive[i] !== 1) continue;
      const type = world.pool.type[i]! as EntityType;
      if (type !== EntityType.Burstbot && type !== EntityType.Slicebot) continue;
      const owner = world.pool.owner[i]!;
      for (let j = 0; j < world.pool.count; j++) {
        if (world.pool.alive[j] !== 1) continue;
        if (world.pool.type[j] !== EntityType.CommandPost) continue;
        if (!world.isHostile(j, owner)) continue;
        const dx = (world.pool.posX[i]! - world.pool.posX[j]!) / 65536;
        const dy = (world.pool.posY[i]! - world.pool.posY[j]!) / 65536;
        const d = Math.sqrt(dx * dx + dy * dy);
        if (d < closest) closest = d;
      }
    }
    return closest;
  }

  /** A four-bot co-op match, played out through the hosted path. */
  function playCoop(ticks: number) {
    const config = matchConfig(MapLayout.Quarters, SEEDS[0]!, { botPlayers: [0, 1, 2, 3] });
    const sim = new HeadlessMatch(config, scriptedAgents(config));
    const world = sim.world;
    let alliedShots = 0;
    let hostileShots = 0;
    let closestEver = Number.POSITIVE_INFINITY;
    for (let t = 0; t < ticks && !world.matchOver; t++) {
      sim.step();
      const shots = world.events.shots;
      for (let k = 0; k + 1 < shots.length; k += 2) {
        const attacker = world.pool.owner[shots[k]!]!;
        const victim = world.pool.owner[shots[k + 1]!]!;
        if (world.areAllied(attacker, victim)) alliedShots++;
        else hostileShots++;
      }
      closestEver = Math.min(closestEver, closestToEnemyPost(world));
    }
    return { sim, world, alliedShots, hostileShots, closestEver };
  }

  it('never fires on its partner, and does fire on the other side', () => {
    const { alliedShots, hostileShots } = playCoop(3600);
    expect(alliedShots).toBe(0);
    expect(hostileShots).toBeGreaterThan(20);
  });

  it('gets an army across the map and into an enemy base', () => {
    // The coverage assertion for everything else about the bot. A bot that
    // builds and never arrives keeps a co-op match running forever, and every
    // claim about its strategy would be untested.
    //
    // The closest approach over the whole match, not where the armies stand
    // on the last tick: an army that has been into a base and come home to
    // answer a raid has done exactly what this asks, and a snapshot of tick
    // 6000 reads it as never having left.
    const { closestEver } = playCoop(6000);
    expect(closestEver).toBeLessThan(20);
  });

  /**
   * Park a bot's army out in the middle of the map, then raid its base.
   *
   * Returns how far the army walked back toward home. The army is staged on
   * neutral ground rather than in an enemy base on purpose — a body keeps its
   * last position, so units that simply died there would read as units that
   * declined to come home.
   */
  function stageRaid(raided: boolean): number {
    const config = matchConfig(MapLayout.Quarters, SEEDS[0]!, { botPlayers: [0] });
    const sim = new HeadlessMatch(config, scriptedAgents(config));
    const world = sim.world;
    const home = world.map.starts[0]!;
    const middle = { tileX: world.map.width >> 1, tileY: world.map.height >> 1 };

    const defenders: number[] = [];
    for (let k = 0; k < 6; k++) {
      defenders.push(spawnAt(world, EntityType.Burstbot, 0, middle.tileX + k - 3, middle.tileY));
    }
    // A raiding party inside the bot's base, close enough to count as an
    // attack. Armed, because a lone scouting worker deliberately does not.
    if (raided) {
      for (let k = 0; k < 3; k++) {
        spawnAt(world, EntityType.Burstbot, 2, home.tileX + 7, home.tileY + k);
      }
    }
    world.recomputeSupply();

    const before = defenders.map((i) => distanceToTile(world, i, home));
    for (let t = 0; t < 600; t++) sim.step();

    let gained = 0;
    let alive = 0;
    for (let k = 0; k < defenders.length; k++) {
      const i = defenders[k]!;
      if (world.pool.alive[i] !== 1) continue;
      alive++;
      gained += before[k]! - distanceToTile(world, i, home);
    }
    expect(alive).toBeGreaterThan(0);
    return gained / alive;
  }

  it('walks its army home when its base is attacked', () => {
    // The thing the old bot never did: it would cross the map while its own
    // Command Post was being shot, which is the most expensive mistake a bot
    // can make and the one a human notices first.
    expect(stageRaid(true)).toBeGreaterThan(15);
  });

  it('leaves its army where it is when nothing attacks', () => {
    // The contrast that gives the assertion above its meaning: same staging,
    // and the only difference is whether there is a raid to answer.
    expect(stageRaid(false)).toBeLessThan(3);
  });
});

function distanceToTile(world: World, index: number, at: { tileX: number; tileY: number }): number {
  const dx = world.pool.posX[index]! / 65536 - at.tileX;
  const dy = world.pool.posY[index]! / 65536 - at.tileY;
  return Math.sqrt(dx * dx + dy * dy);
}
