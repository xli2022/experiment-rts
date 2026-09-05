/**
 * Visibility is computed once, in one place, and the fog draws it.
 *
 * The neural bot reads the same `Visibility` the fog renderer does, so a
 * discrepancy between the two would be a fog leak in one direction or a blind
 * bot in the other.
 */

import { describe, expect, it } from 'vitest';
import { FogRenderer } from '../src/render/fog.js';
import { fromInt } from '../src/sim/fixed.js';
import { GameMap } from '../src/sim/map.js';
import { coopMatch } from '../src/sim/match.js';
import { Simulation } from '../src/sim/tick.js';
import { EntityType, NO_ENTITY, type PlayerId } from '../src/sim/types.js';
import { EXPLORED, UNEXPLORED, VISIBLE, Visibility } from '../src/vision/visibility.js';

describe('visibility', () => {
  it('is what the fog renderer draws, by identity', () => {
    const fog = new FogRenderer(new GameMap(16));
    expect(fog.state).toBe(fog.visibility.state);
    const before = fog.version;
    fog.state[3] = VISIBLE;
    expect(fog.visibility.state[3]).toBe(VISIBLE);
    expect(fog.version).toBe(before);
  });

  it('sees through a partner and not through an opponent', () => {
    const sim = new Simulation(coopMatch(0x51ce7a11));
    const world = sim.world;
    const vis = new Visibility(world.map);

    const spots: { player: PlayerId; x: number; y: number }[] = [
      { player: 0, x: 40, y: 76 },
      { player: 1, x: 60, y: 76 },
      { player: 2, x: 80, y: 76 },
    ];
    const ids: number[] = [];
    for (const spot of spots) {
      const id = world.pool.spawn(
        EntityType.Burstbot,
        spot.player,
        fromInt(spot.x),
        fromInt(spot.y),
      );
      expect(id).not.toBe(NO_ENTITY);
      ids.push(id & 0xffff);
    }

    vis.update(world, 0);

    expect(vis.isVisibleAt(spots[0]!.x + 0.5, spots[0]!.y + 0.5)).toBe(true);
    expect(vis.isVisibleAt(spots[1]!.x + 0.5, spots[1]!.y + 0.5)).toBe(true);
    expect(vis.isVisibleAt(spots[2]!.x + 0.5, spots[2]!.y + 0.5)).toBe(false);
    expect(vis.canSee(world, ids[0]!, 0)).toBe(true);
    expect(vis.canSee(world, ids[1]!, 0)).toBe(true);
    expect(vis.canSee(world, ids[2]!, 0)).toBe(false);
    // The opponent's side sees the reverse.
    vis.update(world, 2);
    expect(vis.canSee(world, ids[2]!, 2)).toBe(true);
    expect(vis.canSee(world, ids[0]!, 2)).toBe(false);
  });

  it('decays what was seen to explored, and answers tile queries in bounds', () => {
    const sim = new Simulation(coopMatch(0x51ce7a11));
    const world = sim.world;
    const vis = new Visibility(world.map);
    const id = world.pool.spawn(EntityType.Burstbot, 0, fromInt(60), fromInt(76));
    const i = id & 0xffff;
    vis.update(world, 0);
    const tile = world.map.index(60, 76);
    expect(vis.isVisibleTile(tile)).toBe(true);
    expect(vis.isExploredTile(tile)).toBe(true);
    world.pool.posX[i] = fromInt(20);
    world.pool.posY[i] = fromInt(20);
    vis.update(world, 0);
    expect(vis.state[tile]).toBe(EXPLORED);
    expect(vis.isVisibleTile(tile)).toBe(false);
    expect(vis.isExploredTile(tile)).toBe(true);
    expect(vis.isVisibleTile(-1)).toBe(false);
    expect(vis.isExploredTile(vis.state.length)).toBe(false);
    expect(vis.state[world.map.index(100, 100)]).toBe(UNEXPLORED);
  });
});
