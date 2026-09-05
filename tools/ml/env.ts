/**
 * The training environment: a headless match that a policy plays from Python.
 *
 * Every slot is a bot. A `policy` slot is played by whatever Python decides
 * each step; a `scripted` slot by the scripted bot, at full speed or thinking
 * less often; a `teacher` slot by the scripted bot at a human's cadence, whose
 * every command is also handed out as the label a student should have
 * produced from the same observation; an `idle` slot does nothing. All of
 * them run through `HeadlessMatch` — the same driver the browser uses, the
 * same input delay — so what a policy learns here is what it will meet in
 * play.
 *
 * One step is one decision: `DECISION_TICKS` ticks. A policy's command is
 * issued on the first of them, exactly when the browser's neural agent would
 * issue the reply to a decision it posted at the boundary.
 */

import type { Agent } from '../../src/ai/agent.js';
import { DECISION_TICKS, humanCadence } from '../../src/ai/cadence.js';
import { HeadlessMatch } from '../../src/ai/headless.js';
import {
  actionFromInts,
  allocAction,
  allocMasks,
  computeMasks,
  decode,
  encode,
  legalise,
  type Masks,
} from '../../src/ai/neural/actions.js';
import { allocFrame, type Frame } from '../../src/ai/neural/frame.js';
import { EntityMemory } from '../../src/ai/neural/memory.js';
import {
  allocObservation,
  encodeCritic,
  ObservationEncoder,
  type Observation,
  type RecentActions,
} from '../../src/ai/neural/observation.js';
import { ACTION_INTS, ActionType, CRITIC_LEN } from '../../src/ai/neural/spec.js';
import { ScriptedAgent } from '../../src/ai/scripted.js';
import { defOf } from '../../src/config/rules.js';
import { CommandType, type Command } from '../../src/sim/commands.js';
import { matchConfig } from '../../src/sim/match.js';
import {
  BotKind,
  MapLayout,
  NO_ENTITY,
  type EntityId,
  type EntityType,
  type PlayerId,
} from '../../src/sim/types.js';
import type { World } from '../../src/sim/world.js';
import { Visibility } from '../../src/vision/visibility.js';

export type SlotSpec =
  | { kind: 'policy' }
  | { kind: 'scripted'; thinkInterval?: number }
  | { kind: 'teacher'; thinkInterval?: number }
  | { kind: 'idle' };

export interface EnvConfig {
  seed: number;
  layout: MapLayout;
  /** One entry per roster slot. */
  slots: SlotSpec[];
  /** The match is truncated here. */
  maxTicks?: number;
  /** Weight of the potential-based shaping term. */
  shaping?: number;
  /** Discount the shaping is potential-based under. */
  gamma?: number;
  /** Cost of every decision, so a draw is never free. */
  timeCost?: number;
}

/** Everything one observed slot gets per step. Buffers are owned by the env and overwritten. */
export interface SlotObs {
  readonly player: PlayerId;
  readonly observation: Observation;
  readonly masks: Masks;
  readonly critic: Float32Array;
  readonly frame: Frame;
  /**
   * The teacher's decision, as `ACTION_INTS`: type Noop when it said nothing,
   * type -1 when it said something the student could not have. All -1 but the
   * Noop type for a policy slot.
   */
  readonly label: Int32Array;
}

export interface StepResult {
  /** One per observed slot, in `observed` order. */
  readonly rewards: Float32Array;
  readonly done: boolean;
  readonly truncated: boolean;
  readonly tick: number;
  readonly winner: number;
  /** Commands the observed slots issued this step, in `observed` order. */
  readonly issued: Int32Array;
}

/** One slot's eyes, kept in step every tick. */
class Eyes {
  readonly vis: Visibility;
  readonly mem: EntityMemory;
  readonly encoder: ObservationEncoder;
  readonly out: SlotObs;
  readonly recent: RecentActions & { lastUnits: Set<EntityId> } = {
    prevType: ActionType.Noop,
    sinceNonNoop: 0,
    recentCommands: 0,
    lastUnits: new Set(),
  };
  readonly commandTicks: number[] = [];

  constructor(world: World, player: PlayerId) {
    this.vis = new Visibility(world.map);
    this.mem = new EntityMemory(player);
    this.encoder = new ObservationEncoder(world, player);
    this.out = {
      player,
      observation: allocObservation(),
      masks: allocMasks(),
      critic: new Float32Array(CRITIC_LEN),
      frame: allocFrame(),
      label: new Int32Array(ACTION_INTS).fill(-1),
    };
  }

  look(world: World): void {
    this.vis.update(world, this.out.player);
    this.mem.update(world, this.vis);
  }

  noteCommand(world: World, command: Command | null, type: number): void {
    this.recent.prevType = type;
    this.recent.lastUnits.clear();
    if (command === null) {
      this.recent.sinceNonNoop++;
    } else {
      this.recent.sinceNonNoop = 0;
      if ('units' in command) for (const id of command.units) this.recent.lastUnits.add(id);
      if ('worker' in command) this.recent.lastUnits.add(command.worker);
      if (
        command.type === CommandType.Train ||
        command.type === CommandType.CancelTrain ||
        command.type === CommandType.SetRally
      ) {
        this.recent.lastUnits.add(command.building);
      }
      this.commandTicks.push(world.tick);
    }
    while (this.commandTicks.length > 0 && world.tick - this.commandTicks[0]! > 200)
      this.commandTicks.shift();
    this.recent.recentCommands = this.commandTicks.length;
  }

  observe(world: World): void {
    this.encoder.encode(this.vis, this.mem, this.recent, this.out.observation, this.out.frame);
    computeMasks(world, this.out.frame, this.vis, this.mem, this.out.masks);
    encodeCritic(world, this.out.player, this.out.critic);
  }
}

/** Plays whatever Python last decided, once, on the first tick after the decision. */
class PolicyAgent implements Agent {
  pending: Command | null = null;
  act(): Command[] {
    const command = this.pending;
    this.pending = null;
    return command === null ? [] : [command];
  }
}

/** The scripted bot at a human's cadence, remembering what it said at each boundary. */
class TeacherAgent implements Agent {
  private readonly inner: Agent;
  /** The command released at the last decision boundary, or null. */
  lastCommand: Command | null = null;
  constructor(thinkInterval?: number) {
    this.inner = humanCadence(
      new ScriptedAgent(thinkInterval === undefined ? {} : { thinkInterval }),
    );
  }
  act(world: World, player: PlayerId): Command[] {
    const commands = this.inner.act(world, player);
    if (world.tick % DECISION_TICKS === 0) this.lastCommand = commands[0] ?? null;
    return commands;
  }
}

/** Mineral value of everything alive a team owns, plus its banks. */
function teamValue(world: World, team: number): number {
  const pool = world.pool;
  let value = 0;
  for (let i = 0; i < pool.count; i++) {
    if (pool.alive[i] !== 1) continue;
    const owner = pool.owner[i]!;
    if (owner < 0 || world.teamOf(owner) !== team) continue;
    value += defOf(pool.type[i]! as EntityType).mineralCost;
  }
  for (let p = 0; p < world.players.length; p++) {
    if (world.teamOf(p) === team) value += world.player(p).minerals;
  }
  return value;
}

export class MatchEnv {
  readonly config: EnvConfig;
  private match!: HeadlessMatch;
  private eyes = new Map<PlayerId, Eyes>();
  private policies = new Map<PlayerId, PolicyAgent>();
  private teachers = new Map<PlayerId, TeacherAgent>();
  /** Slots an observation is produced for: policy and teacher slots, ascending. */
  readonly observed: PlayerId[] = [];
  private potential = new Float32Array(0);
  private readonly action = allocAction();
  private seed: number;
  private steps = 0;

  constructor(config: EnvConfig) {
    this.config = config;
    this.seed = config.seed;
    config.slots.forEach((slot, p) => {
      if (slot.kind === 'policy' || slot.kind === 'teacher') this.observed.push(p);
    });
    this.reset(config.seed);
  }

  get world(): World {
    return this.match.world;
  }

  get tick(): number {
    return this.match.world.tick;
  }

  get done(): boolean {
    return this.match.world.matchOver || this.tick >= (this.config.maxTicks ?? 24000);
  }

  /** Start over, on `seed` or the next seed after the last. */
  reset(seed?: number): void {
    this.seed = seed ?? (this.seed + 1) >>> 0;
    const slots = this.config.slots;
    const config = matchConfig(this.config.layout, this.seed, {
      botSlots: slots.map((slot, player) => ({
        player,
        kind: slot.kind === 'scripted' ? BotKind.Scripted : BotKind.Neural,
      })),
    });
    const agents: [PlayerId, Agent][] = [];
    this.policies.clear();
    this.teachers.clear();
    slots.forEach((slot, p) => {
      if (slot.kind === 'scripted') {
        agents.push([
          p,
          new ScriptedAgent(
            slot.thinkInterval === undefined ? {} : { thinkInterval: slot.thinkInterval },
          ),
        ]);
      } else if (slot.kind === 'policy') {
        const agent = new PolicyAgent();
        this.policies.set(p, agent);
        agents.push([p, agent]);
      } else if (slot.kind === 'teacher') {
        const agent = new TeacherAgent(slot.thinkInterval);
        this.teachers.set(p, agent);
        agents.push([p, agent]);
      }
    });
    this.match = new HeadlessMatch(config, agents);
    const world = this.match.world;
    this.eyes.clear();
    for (const p of this.observed) this.eyes.set(p, new Eyes(world, p));
    this.potential = new Float32Array(this.observed.length);
    this.steps = 0;
    // Tick zero: look once so the first observation is not blind.
    for (const eyes of this.eyes.values()) eyes.look(world);
    for (let k = 0; k < this.observed.length; k++)
      this.potential[k] = this.potentialOf(this.observed[k]!);
  }

  private potentialOf(player: PlayerId): number {
    const world = this.match.world;
    const team = world.teamOf(player);
    let enemy = 0;
    let teams = 0;
    for (let p = 0; p < world.players.length; p++) {
      const t = world.teamOf(p);
      if (t === team) continue;
      teams++;
      enemy += teamValue(world, t);
    }
    return (teamValue(world, team) - enemy / Math.max(1, teams)) / 1000;
  }

  /** The current observation for a slot. Valid until the next `step` or `reset`. */
  observe(player: PlayerId): SlotObs {
    const eyes = this.eyes.get(player);
    if (!eyes) throw new Error(`slot ${player} is not observed`);
    eyes.observe(this.match.world);
    const teacher = this.teachers.get(player);
    eyes.out.label.fill(-1);
    eyes.out.label[0] = ActionType.Noop;
    if (teacher) {
      // The label is the command the teacher released at this boundary,
      // expressed in the frame the student is looking at right now — and only
      // if the student could have said it: a label the masks refuse is marked
      // invalid (type -1) and is not a lesson, since the teacher saw more.
      const command = teacher.lastCommand;
      if (command !== null && !encode(command, eyes.out.frame, this.action)) {
        eyes.out.label[0] = -1;
      } else if (command !== null && !legalise(this.action, eyes.out.masks)) {
        eyes.out.label[0] = -1;
      } else if (command !== null) {
        eyes.out.label[0] = this.action.type;
        eyes.out.label[1] = this.action.entityType;
        eyes.out.label[2] = this.action.target;
        eyes.out.label[3] = this.action.cell;
        eyes.out.label[4] = this.action.sub;
        for (let k = 0; k < this.action.selection.length; k++)
          eyes.out.label[5 + k] = this.action.selection[k]!;
      }
    }
    return eyes.out;
  }

  /**
   * Apply one decision per policy slot — as `ACTION_INTS` integers, decoded
   * against that slot's last observation — and advance one decision's worth of
   * ticks. Teacher slots ignore what they are given.
   */
  step(actions: ReadonlyMap<PlayerId, ArrayLike<number>>): StepResult {
    const world = this.match.world;
    const issued = new Int32Array(this.observed.length);
    for (let k = 0; k < this.observed.length; k++) {
      const p = this.observed[k]!;
      const eyes = this.eyes.get(p)!;
      const policy = this.policies.get(p);
      if (policy) {
        const ints = actions.get(p);
        let command: Command | null = null;
        let type: number = ActionType.Noop;
        if (ints) {
          actionFromInts(ints, this.action);
          type = this.action.type;
          command = decode(this.action, world, eyes.out.frame);
        }
        policy.pending = command;
        eyes.noteCommand(world, command, type);
        if (command !== null) issued[k] = 1;
      }
    }

    for (let t = 0; t < DECISION_TICKS && !world.matchOver; t++) {
      this.match.step();
      for (const eyes of this.eyes.values()) eyes.look(world);
    }
    this.steps++;

    // Teachers issued through the driver; note what they said for the recent-
    // action features, so the student sees the same thing it will see in play.
    for (let k = 0; k < this.observed.length; k++) {
      const teacher = this.teachers.get(this.observed[k]!);
      if (!teacher) continue;
      const eyes = this.eyes.get(this.observed[k]!)!;
      eyes.noteCommand(
        world,
        teacher.lastCommand,
        teacher.lastCommand === null ? ActionType.Noop : this.typeOf(teacher.lastCommand),
      );
      if (teacher.lastCommand !== null) issued[k] = 1;
    }

    const done = this.done;
    const truncated = done && !world.matchOver;
    const rewards = new Float32Array(this.observed.length);
    const shaping = this.config.shaping ?? 1e-3;
    const gamma = this.config.gamma ?? 0.99;
    const timeCost = this.config.timeCost ?? 1e-4;
    for (let k = 0; k < this.observed.length; k++) {
      const p = this.observed[k]!;
      const next = this.potentialOf(p);
      let r = shaping * (gamma * next - this.potential[k]!) - timeCost;
      this.potential[k] = next;
      if (world.matchOver) {
        if (world.winner === NO_ENTITY) r += 0;
        else r += world.winner === world.teamOf(p) ? 1 : -1;
      }
      rewards[k] = r;
    }
    return { rewards, done, truncated, tick: world.tick, winner: world.winner, issued };
  }

  private typeOf(command: Command): number {
    switch (command.type) {
      case CommandType.Move:
        return ActionType.Move;
      case CommandType.AttackMove:
        return ActionType.AttackMove;
      case CommandType.Attack:
        return ActionType.Attack;
      case CommandType.Harvest:
        return ActionType.Harvest;
      case CommandType.Build:
        return ActionType.Build;
      case CommandType.Stop:
        return ActionType.Stop;
      case CommandType.Hold:
        return ActionType.Hold;
      case CommandType.Train:
        return ActionType.Train;
      case CommandType.CancelTrain:
        return ActionType.CancelTrain;
      case CommandType.SetRally:
        return ActionType.SetRally;
      default:
        return ActionType.Noop;
    }
  }

  /** Decisions taken since the last reset. */
  get decisions(): number {
    return this.steps;
  }

  dispose(): void {
    this.match.dispose();
  }
}

/** Parse a slot list like `policy,scripted@20,teacher,idle`. */
export function parseSlots(text: string): SlotSpec[] {
  return text.split(',').map((part) => {
    const [kind, arg] = part.trim().split('@');
    const thinkInterval = arg === undefined ? undefined : Number(arg);
    switch (kind) {
      case 'policy':
        return { kind: 'policy' };
      case 'scripted':
        return thinkInterval === undefined
          ? { kind: 'scripted' }
          : { kind: 'scripted', thinkInterval };
      case 'teacher':
        return thinkInterval === undefined
          ? { kind: 'teacher' }
          : { kind: 'teacher', thinkInterval };
      case 'idle':
        return { kind: 'idle' };
      default:
        throw new Error(`unknown slot kind ${kind}`);
    }
  });
}

export function parseLayout(text: string): MapLayout {
  if (text === 'lanes') return MapLayout.Lanes;
  if (text === 'quarters') return MapLayout.Quarters;
  throw new Error(`unknown layout ${text}`);
}
