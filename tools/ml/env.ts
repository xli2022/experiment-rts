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
import { ACTION_INTS, ACTION_TYPES, ActionType, CRITIC_LEN } from '../../src/ai/neural/spec.js';
import { ScriptedAgent } from '../../src/ai/scripted.js';
import { defOf } from '../../src/config/rules.js';
import { CommandType, type Command } from '../../src/sim/commands.js';
import { idIndex } from '../../src/sim/entities.js';
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

/** Training-label coverage for one teacher in the current match. */
export interface TeacherCoverage {
  decisions: number;
  valid: number;
  nonNoop: number;
  dropped: number;
  actions: Record<string, number>;
  droppedCommands: Record<string, number>;
  buildings: Record<string, number>;
  resumes: Record<string, number>;
  upgrades: Record<string, number>;
  units: Record<string, number>;
}

/** One slot's eyes, kept in step every tick. */
class Eyes {
  readonly vis: Visibility;
  readonly mem: EntityMemory;
  readonly encoder: ObservationEncoder;
  out: SlotObs;
  readonly recent: RecentActions & { lastUnits: Set<EntityId> } = {
    prevType: ActionType.Noop,
    sinceNonNoop: 0,
    recentCommands: 0,
    lastUnits: new Set(),
  };
  readonly commandTicks: number[] = [];
  /**
   * A teacher's observation and label captured together at its decision
   * boundary. Null until its first decision and for every other kind of slot.
   */
  held: SlotObs | null = null;

  constructor(world: World, player: PlayerId) {
    this.vis = new Visibility(world.map);
    this.mem = new EntityMemory(player);
    this.encoder = new ObservationEncoder(world, player);
    this.out = allocSlotObs(player);
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
        command.type === CommandType.SetRally ||
        command.type === CommandType.UpgradeBuilding ||
        command.type === CommandType.CancelUpgrade
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

  /**
   * Keep the labelled observation, and encode the next decision into the other
   * buffer so readers never see a partially overwritten decision.
   *
   * A swap rather than a copy. The two buffers alternate: the one just held was
   * serialised on the previous `observe` and is free to write again, and every
   * writer (`ObservationEncoder.encode`, `computeMasks`, `encodeCritic`)
   * overwrites its output whole rather than updating it in place, so there is
   * nothing to carry across. Copying instead meant naming all twenty fields by
   * hand, and a field added to `Observation`, `Masks` or `Frame` and forgotten
   * here would have paired labels with a stale copy of it — silently, with no
   * test failing and the training data quietly rotting.
   */
  hold(): void {
    const next = this.held ?? allocSlotObs(this.out.player);
    this.held = this.out;
    this.out = next;
  }
}

/** One slot's buffers. Two of these alternate for a teacher slot; see `hold`. */
function allocSlotObs(player: PlayerId): SlotObs {
  return {
    player,
    observation: allocObservation(),
    masks: allocMasks(),
    critic: new Float32Array(CRITIC_LEN),
    frame: allocFrame(),
    label: new Int32Array(ACTION_INTS).fill(-1),
  };
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

/** Decide at the boundary, then issue on the next tick like the neural policy. */
class TeacherAgent implements Agent {
  private readonly inner: Agent;
  private pending: Command | null = null;
  issuedThisStep = 0;
  constructor(
    private readonly decided: (world: World, player: PlayerId, command: Command | null) => void,
    thinkInterval?: number,
  ) {
    this.inner = humanCadence(
      new ScriptedAgent(thinkInterval === undefined ? {} : { thinkInterval }),
    );
  }
  act(world: World, player: PlayerId): Command[] {
    const release = this.pending;
    this.pending = null;
    const commands = this.inner.act(world, player);
    if (world.tick % DECISION_TICKS === 0) {
      this.pending = commands[0] ?? null;
      // Capture after the think, before issuing anything: Agent.act may read
      // but never mutate the world. The observation must describe this tick,
      // not the state at the beginning of the four-tick environment step.
      this.decided(world, player, this.pending);
    }
    if (release) this.issuedThisStep++;
    return release ? [release] : [];
  }
  dispose(): void {
    this.inner.dispose?.();
  }
}

/**
 * How much a side's own board presence counts toward its potential, against the
 * enemy presence it is trying to remove.
 *
 * Deliberately asymmetric. A symmetric differential prices a unit lost exactly
 * as dearly as a unit killed, so every attack is a local loss and the shaped
 * optimum is to sit still and keep what you have — which cannot win a match
 * decided by razing the other side's buildings. At a quarter weight an even
 * trade is a gain, which is the truth when the enemy's board is what has to end
 * up empty.
 */
const OWN_STANDING = 0.25;

/** Enemy structures, the thing that actually has to be destroyed, count treble. */
const ENEMY_BUILDINGS = 3;

export class MatchEnv {
  readonly config: EnvConfig;
  private match!: HeadlessMatch;
  private eyes = new Map<PlayerId, Eyes>();
  private policies = new Map<PlayerId, PolicyAgent>();
  private teachers = new Map<PlayerId, TeacherAgent>();
  private coverage = new Map<PlayerId, TeacherCoverage>();
  /** Slots an observation is produced for: policy and teacher slots, ascending. */
  readonly observed: PlayerId[] = [];
  private potential = new Float32Array(0);
  /** Per-team board presence, refreshed by `scanStanding` once a step. */
  private unitValue = new Float64Array(0);
  private buildingValue = new Float64Array(0);
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
    this.coverage.clear();
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
        this.coverage.set(p, {
          decisions: 0,
          valid: 0,
          nonNoop: 0,
          dropped: 0,
          actions: {},
          droppedCommands: {},
          buildings: {},
          resumes: {},
          upgrades: {},
          units: {},
        });
        const agent = new TeacherAgent((world, player, command) => {
          const eyes = this.eyes.get(player)!;
          // The driver runs before the environment's post-tick vision update.
          // Refresh here so newly visible entities belong to this same tick.
          eyes.look(world);
          eyes.observe(world);
          this.writeLabel(eyes, command);
          eyes.hold();
          eyes.noteCommand(
            world,
            command,
            command === null ? ActionType.Noop : this.typeOf(command),
          );
        }, slot.thinkInterval);
        this.teachers.set(p, agent);
        agents.push([p, agent]);
      }
    });
    this.match = new HeadlessMatch(config, agents);
    const world = this.match.world;
    this.eyes.clear();
    for (const p of this.observed) this.eyes.set(p, new Eyes(world, p));
    this.potential = new Float32Array(this.observed.length);
    let maxTeam = 0;
    for (let p = 0; p < world.players.length; p++) maxTeam = Math.max(maxTeam, world.teamOf(p));
    this.unitValue = new Float64Array(maxTeam + 1);
    this.buildingValue = new Float64Array(maxTeam + 1);
    this.steps = 0;
    // Tick zero: look once so the first observation is not blind.
    for (const eyes of this.eyes.values()) eyes.look(world);
    this.scanStanding();
    for (let k = 0; k < this.observed.length; k++)
      this.potential[k] = this.potentialOf(this.observed[k]!);
  }

  /**
   * Tally every team's board presence in one pass, splitting units from
   * structures so either weighting is then arithmetic.
   *
   * Called once per step rather than from `potentialOf`, which would otherwise
   * walk the whole entity pool once per enemy player per observed slot — twelve
   * scans a step on the four-slot Quarters layout, most of them recomputing the
   * same team's total.
   */
  private scanStanding(): void {
    const world = this.match.world;
    this.unitValue.fill(0);
    this.buildingValue.fill(0);
    const pool = world.pool;
    for (let i = 0; i < pool.count; i++) {
      if (pool.alive[i] !== 1) continue;
      const owner = pool.owner[i]!;
      if (owner < 0) continue;
      const team = world.teamOf(owner);
      const def = defOf(pool.type[i]! as EntityType);
      if (def.isBuilding) this.buildingValue[team] += def.mineralCost;
      else this.unitValue[team] += def.mineralCost;
    }
  }

  /** Reads the tally `scanStanding` left; call that first. */
  private potentialOf(player: PlayerId): number {
    const world = this.match.world;
    const team = world.teamOf(player);
    let enemy = 0;
    let seats = 0;
    for (let p = 0; p < world.players.length; p++) {
      const t = world.teamOf(p);
      if (t === team) continue;
      seats++;
      enemy += this.unitValue[t]! + ENEMY_BUILDINGS * this.buildingValue[t]!;
    }
    const own = this.unitValue[team]! + this.buildingValue[team]!;
    return (OWN_STANDING * own - enemy / Math.max(1, seats)) / 1000;
  }

  /**
   * The teacher's command written into `eyes.out.label`, encoded against the
   * frame `eyes.out` is currently holding — which must be the observation the
   * teacher chose it *from*. A command the student could not have expressed,
   * or one the masks refuse, is marked invalid (type -1) and is not a lesson.
   */
  private writeLabel(eyes: Eyes, command: Command | null): void {
    const counts = this.coverage.get(eyes.out.player)!;
    counts.decisions++;
    eyes.out.label.fill(-1);
    eyes.out.label[0] = ActionType.Noop;
    if (command === null) {
      counts.valid++;
      counts.actions.Noop = (counts.actions.Noop ?? 0) + 1;
      return;
    }
    let valid =
      encode(command, eyes.out.frame, this.action) && legalise(this.action, eyes.out.masks);
    if (valid && command.type === CommandType.Build) {
      const decoded = decode(this.action, this.world, eyes.out.frame);
      // A queued Build may name a site that is no longer workable. Coarse cell
      // legality alone can otherwise label a different, nearby new foundation.
      valid =
        decoded?.type === CommandType.Build &&
        decoded.tileX === command.tileX &&
        decoded.tileY === command.tileY;
    }
    if (!valid) {
      eyes.out.label[0] = -1;
      counts.dropped++;
      const kind = CommandType[command.type]!;
      counts.droppedCommands[kind] = (counts.droppedCommands[kind] ?? 0) + 1;
      return;
    }
    counts.valid++;
    counts.nonNoop++;
    const kind = ACTION_TYPES[this.action.type]!;
    counts.actions[kind] = (counts.actions[kind] ?? 0) + 1;
    if (command.type === CommandType.Build) {
      const name = defOf(command.building).name;
      counts.buildings[name] = (counts.buildings[name] ?? 0) + 1;
      const site = eyes.out.frame.constructionSites.get(
        command.tileY * eyes.out.frame.width + command.tileX,
      );
      if (site?.type === command.building) counts.resumes[name] = (counts.resumes[name] ?? 0) + 1;
    } else if (command.type === CommandType.Train) {
      const name = defOf(command.unit).name;
      counts.units[name] = (counts.units[name] ?? 0) + 1;
    } else if (command.type === CommandType.UpgradeBuilding) {
      const index = idIndex(command.building);
      const name = defOf(this.world.pool.type[index]! as EntityType).name;
      counts.upgrades[name] = (counts.upgrades[name] ?? 0) + 1;
    }
    eyes.out.label[0] = this.action.type;
    eyes.out.label[1] = this.action.entityType;
    eyes.out.label[2] = this.action.target;
    eyes.out.label[3] = this.action.cell;
    eyes.out.label[4] = this.action.sub;
    for (let k = 0; k < this.action.selection.length; k++)
      eyes.out.label[5 + k] = this.action.selection[k]!;
  }

  /**
   * The current observation for a slot. Valid until the next `step` or `reset`.
   *
   * A teacher slot reports its observation captured at the last decision
   * boundary, together with the command chosen there. The teacher issues that
   * command one tick later, just like a policy consuming this observation.
   * The first observation has an invalid label because no decision exists yet.
   */
  observe(player: PlayerId): SlotObs {
    const eyes = this.eyes.get(player);
    if (!eyes) throw new Error(`slot ${player} is not observed`);
    if (this.teachers.has(player) && eyes.held) return eyes.held;
    eyes.observe(this.match.world);
    if (!this.teachers.has(player)) {
      eyes.out.label.fill(-1);
      eyes.out.label[0] = ActionType.Noop;
      return eyes.out;
    }
    if (!eyes.held) {
      // Tick zero has no teacher decision. The next step captures the first
      // labelled observation at its decision boundary.
      eyes.out.label.fill(-1);
      eyes.out.label[0] = -1;
      return eyes.out;
    }
    return eyes.held;
  }

  /**
   * Apply one decision per policy slot — as `ACTION_INTS` integers, decoded
   * against that slot's last observation — and advance one decision's worth of
   * ticks. Teacher slots ignore what they are given.
   */
  step(actions: ReadonlyMap<PlayerId, ArrayLike<number>>): StepResult {
    const world = this.match.world;
    const issued = new Int32Array(this.observed.length);
    for (const teacher of this.teachers.values()) teacher.issuedThisStep = 0;
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

    // The teacher captured its own decision and observation at the boundary.
    // Count actual releases separately; the newest decision issues next tick.
    for (let k = 0; k < this.observed.length; k++) {
      const teacher = this.teachers.get(this.observed[k]!);
      if (!teacher) continue;
      issued[k] = teacher.issuedThisStep;
    }

    const done = this.done;
    const truncated = done && !world.matchOver;
    const rewards = new Float32Array(this.observed.length);
    const shaping = this.config.shaping ?? 1e-3;
    const gamma = this.config.gamma ?? 0.99;
    const timeCost = this.config.timeCost ?? 1e-4;
    this.scanStanding();
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
      case CommandType.UpgradeBuilding:
        return ActionType.UpgradeBuilding;
      case CommandType.CancelUpgrade:
        return ActionType.CancelUpgrade;
      default:
        return ActionType.Noop;
    }
  }

  /** Decisions taken since the last reset. */
  get decisions(): number {
    return this.steps;
  }

  /** A detached report; callers cannot mutate the next training observation. */
  teacherCoverage(player: PlayerId): TeacherCoverage {
    const counts = this.coverage.get(player);
    if (!counts) throw new Error(`slot ${player} is not a teacher`);
    return {
      ...counts,
      actions: { ...counts.actions },
      droppedCommands: { ...counts.droppedCommands },
      buildings: { ...counts.buildings },
      resumes: { ...counts.resumes },
      upgrades: { ...counts.upgrades },
      units: { ...counts.units },
    };
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
