/**
 * Player commands — the only thing that ever crosses the network.
 *
 * In deterministic lockstep, peers never exchange entity state. They exchange
 * *intent*, and each peer applies the same intents to its own identical
 * simulation. That is why bandwidth is flat whether the battle has twenty units
 * or two thousand: a 200-unit attack-move is one command carrying a list of ids,
 * not 200 position updates every frame.
 *
 * Two consequences shape everything below:
 *
 * 1. Commands must be *small*, because they are resent redundantly for packet
 *    loss tolerance.
 * 2. Commands must be *validated inside the simulation*, identically on every
 *    peer. A command that is illegal (not enough minerals, unit already dead,
 *    building spot now occupied) must be rejected by the same rule everywhere —
 *    so validation lives in the sim, never in the UI. The UI may grey out a
 *    button, but the sim is what decides.
 */

import type { Fix } from './fixed.js';
import type { EntityId, EntityType, PlayerId } from './types.js';

export enum CommandType {
  Move = 0,
  AttackMove = 1,
  Attack = 2,
  Harvest = 3,
  Build = 4,
  Stop = 5,
  Hold = 6,
  Train = 7,
  CancelTrain = 8,
  Surrender = 9,
}

interface Base {
  /** Which player issued this. Set by the lockstep layer, never by the UI. */
  player: PlayerId;
}

export interface MoveCommand extends Base {
  type: CommandType.Move;
  units: EntityId[];
  x: Fix;
  y: Fix;
}

export interface AttackMoveCommand extends Base {
  type: CommandType.AttackMove;
  units: EntityId[];
  x: Fix;
  y: Fix;
}

export interface AttackCommand extends Base {
  type: CommandType.Attack;
  units: EntityId[];
  target: EntityId;
}

export interface HarvestCommand extends Base {
  type: CommandType.Harvest;
  units: EntityId[];
  target: EntityId;
}

export interface BuildCommand extends Base {
  type: CommandType.Build;
  /** The worker assigned to construct it. */
  worker: EntityId;
  building: EntityType;
  tileX: number;
  tileY: number;
}

export interface StopCommand extends Base {
  type: CommandType.Stop;
  units: EntityId[];
}

export interface HoldCommand extends Base {
  type: CommandType.Hold;
  units: EntityId[];
}

export interface TrainCommand extends Base {
  type: CommandType.Train;
  building: EntityId;
  unit: EntityType;
}

export interface CancelTrainCommand extends Base {
  type: CommandType.CancelTrain;
  building: EntityId;
  slot: number;
}

export interface SurrenderCommand extends Base {
  type: CommandType.Surrender;
}

export type Command =
  | MoveCommand
  | AttackMoveCommand
  | AttackCommand
  | HarvestCommand
  | BuildCommand
  | StopCommand
  | HoldCommand
  | TrainCommand
  | CancelTrainCommand
  | SurrenderCommand;

/**
 * All commands a single player issued for one turn.
 *
 * An empty `commands` array is still transmitted: it is the heartbeat that tells
 * other peers this player is alive and has nothing to say, letting the turn
 * advance. Without it the whole game would stall whenever someone stopped
 * clicking.
 */
export interface TurnCommands {
  turn: number;
  player: PlayerId;
  commands: Command[];
}

/**
 * Sort commands into a canonical order before applying them.
 *
 * Packets can arrive in any order, and two peers that apply the same commands in
 * different sequences will diverge — a move issued before an attack is not the
 * same as the reverse. Sorting by issuing player, then by command type, then by
 * the first unit id gives a total order that every peer computes identically
 * from the same set.
 */
export function sortCommands(commands: Command[]): Command[] {
  return commands.sort((a, b) => {
    if (a.player !== b.player) return a.player - b.player;
    if (a.type !== b.type) return a.type - b.type;
    return firstUnitId(a) - firstUnitId(b);
  });
}

function firstUnitId(c: Command): number {
  switch (c.type) {
    case CommandType.Move:
    case CommandType.AttackMove:
    case CommandType.Attack:
    case CommandType.Harvest:
    case CommandType.Stop:
    case CommandType.Hold:
      return c.units.length > 0 ? c.units[0]! : -1;
    case CommandType.Build:
      return c.worker;
    case CommandType.Train:
    case CommandType.CancelTrain:
      return c.building;
    case CommandType.Surrender:
      return -1;
  }
}
