# experiment-rts — contributor notes

A 3D real-time strategy game for the browser, with peer-to-peer multiplayer and
no server of our own.

## The one rule that matters

`src/sim/**` is a **sealed deterministic core**. Two players' machines run the
simulation independently and exchange only commands — never entity state — so if
the two simulations ever disagree by a single bit, the game silently becomes two
different games.

Inside `src/sim/**` (and `src/config/rules.ts`), you may not use:

| Banned                                                    | Use instead                                         |
| --------------------------------------------------------- | --------------------------------------------------- |
| `Math.random()`                                           | the seeded `Rng` on `World`                         |
| `Date.now()`, `performance.now()`, `new Date()`           | `world.tick`                                        |
| `Math.sin/cos/tan/atan2/pow/exp/log/hypot`                | see "no trigonometry" below                         |
| `for...in`, iteration over `Map`/`Set`                    | index loops over the entity pool                    |
| raw `float` positions                                     | Q16.16 fixed-point (`src/sim/fixed.ts`)             |
| importing `three`, DOM, `src/render`, `src/ui`, `src/net` | nothing — the sim imports only itself and `config/` |

`Math.sqrt` **is** allowed: IEEE-754 requires it to be correctly rounded, so it
is identical on every engine. The transcendentals are not, and genuinely differ
between V8, SpiderMonkey and JavaScriptCore.

`tests/sealed-sim.test.ts` enforces all of this mechanically. It is not a style
check — it is the thing standing between us and a desync bug that only
reproduces when a Chrome player faces a Firefox player.

It scans text, deliberately, so that `await import(...)` cannot slip past a
parser-based check. The cost is that prose matches too: a comment containing the
words `from "something"` is read as an import of `something`. Reword rather than
loosening the scan.

### No trigonometry

The simulation stores facing as a **normalised vector**, never an angle, and
turns via `vecRotateToward` (clamped lerp + renormalise). This is why the sim
needs no sin/cos at all, and therefore needs no lookup tables or CORDIC. Angles
are a rendering concern; convert at the boundary.

### Fixed-point gotchas

- `fmul` splits operands into 16-bit halves and uses `Math.imul`. The obvious
  `(a * b) >> 16` is **wrong** — `int32 * int32` reaches 2^62, past the 2^53
  where float64 stops representing integers exactly, and it silently disagrees
  with exact arithmetic on ~2.5% of operand pairs.
- Squared distances overflow int32 at map scale, so use `vecLenSqRaw` /
  `sqRange`, which stay in exact float64 integer range and compare safely.

## Map generation

`src/sim/mapgen.ts` carves lanes out of a solid block. The one rule: **every
write opens a tile and its 180-degree opposite**, inside a single `open()`. That
is the entire symmetry guarantee, and it must stay that way — lanes can then
wobble, overlap, and be carved in any order and the halves still come out
bit-identical. The previous approach wrote each shape twice and trusted the
copies to agree; they diverged on 926 tiles the moment the corridor brush gained
a random wobble, because each copy drew its own wobble from the same stream.

Two layouts share the carver: `Lanes` (two starts, 128) and `Quarters` (four
starts, 152). Adding a map is authoring polylines, not writing a generator.

### Start locations come in mirrored halves

`map.starts[i]` and `map.starts[i + n/2]` are exact 180-degree rotations of one
another, and `map.expansions` follows the same convention. Teams are then
*derived* from it — the first half against the second — which is what makes team
fairness structural instead of something to re-check. Two consequences worth
holding onto:

- **Player `p`'s opposite number is always `p + n/2`**, on any layout. Anything
  that wants "the enemy" without scanning the pool should ask for that rather
  than assuming `1 - p`.
- **Allies within a half are not mirrors of each other**, and cannot be: only
  one symmetry is guaranteed and it is the one between the sides. On `Quarters`
  the two players on a team hold different ground; their *opponents* hold the
  exact rotation of it.

Everything placed at setup is laid out from the canonical (first-half) site and
rotated for the other half — see `mirroredHalf`. One extra rule appears with
four corners: the authored mineral-line offsets run up and to the left, which
tucks the line into the corner for a base on the left of the map and shoves it
out toward the middle for one on the right. `Quarters` has a canonical base in
each, so the line is reflected about the base's own centre for the right-hand
one. About its *centre*, not its centre tile — the half-tile error is the same
one described under "Mirror fairness" below, and costs a whole tile of walking
on every trip for the rest of the match.

`map.elevation` is cosmetic and not checksummed — the simulation decides movement
from walkability alone — but it is still computed deterministically, because two
peers rendering visibly different terrain would look exactly like a desync.

## Mirror fairness is not the same as symmetry

The map is symmetric; that does not make the _match_ symmetric. Anything placed
at setup has to be laid out as an exact 180-degree rotation of player 0's, not by
applying the same offsets to a mirrored start. Three separate bugs came from
getting this subtly wrong, each worth a real advantage and none visible on
screen:

- A Command Post's footprint is 4, and `start - 2` cannot be symmetric about a
  tile — so player 1's whole base sat a tile nearer the middle of the map.
- Reflecting a mineral patch's _top-left corner_ about the base's centre tile
  rather than its centre is half a tile out, which rounds to a whole tile of
  extra walking on every trip.
- Units spawn facing +Y and trained units pop out on the +Y side of a building,
  so one player's reinforcements appeared four tiles nearer the front than the
  other's, every time.

**The simulation is deterministic but not rotation-equivariant.** Set both sides
up as exact mirrors, give them mirrored orders, and they still diverge. Two
independent causes, measured:

- **A\* breaks ties by tile index**, and row-major index is not invariant under a
  180-degree rotation. 27 of 30 mirrored start/goal pairs return a
  _differently shaped_ route — always the same length, never the same tiles.
  Units steer between waypoints, so a different-shaped path of equal tile cost
  is a different real distance. This is the big one: with no bots and exactly
  mirrored harvest orders, one side out-mined the other by a third.
- **`fmul` truncates toward negative infinity**, so `fmul(-a, b) !== -fmul(a, b)`
  for 99.99% of operand pairs — always by one ULP. This is the floor: even with
  direct steering and no pathfinding, two mirrored units drift apart within
  about ten ticks.

None of this is a desync risk — every peer computes the same numbers, which is
all lockstep needs — and none of it is perceptible in a human game. What it means
is that **an AI-vs-AI mirror match is not a fair fight**, and measuring balance
by running bot matches on a symmetric map will report whichever side the
tie-breaks happen to favour. Fixing it needs a rotation-invariant tie-break in
A\* and a change to `fmul`'s rounding, and `fmul` is the function everything
else's correctness rests on.

## Teams

A match is described by one agreed `MatchConfig` — seed, layout, a team per
player slot, and which slots the AI plays. It is checksummed, so a lobby that
let two peers disagree shows up as a desync on the first comparison rather than
as an inexplicable divergence some seconds in.

**A 1v1 is the degenerate case, and that is load-bearing.** Team ids equal player
ids there, so `world.winner` still reads as a player, `teamsFor(Lanes)` is
`[0, 1]`, and the duel map plus its opening are byte for byte what they were —
verified against the pre-teams build, not assumed.

`World.isHostile(index, player)` is the one place hostility is decided. It used
to live on the entity pool, where it could only compare owners — correct while
everyone was everyone else's enemy, and quietly wrong the moment two of them
share a side. The failures it prevents are all quiet ones: a unit that chases a
partner it can never damage, an attack order that can never resolve, fog that
stops at your own units, a match that ends when one of two allies loses their
last building.

**Elimination is per player, the match is per team, and being out takes
everything with it.** Those first two together left a seam: `executeCommand`
drops commands from a defeated player, so a co-op player who lost their last
building kept an army that fought on and could take no orders. The rule that
closes it is the genre's — an eliminated player's remaining units and buildings
are destroyed on the tick they go out — and it is what makes the surrender
button mean something in a team game rather than only in a 1v1.

`victorySystem` keys that sweep on *what a defeated player still owns*, not on
"newly defeated here". `Surrender` sets the flag itself, from a command that has
already executed by the time victory runs, so the obvious version emptied a
razed player's base and left a conceding player's standing — the one case the
button exists for. The deaths are queued into `world.events.deaths` and reaped
like any other, so a conceded base blows up instead of blinking away.

**Humans must occupy a contiguous prefix of the roster.** Lockstep indexes its
per-turn buffer by player id and only human slots ever appear on the wire, so a
bot in slot 0 with a human in slot 2 stalls every peer forever waiting for a turn
nobody sends. `coopMatch` puts the humans in 0 and 1 and the AI in 2 and 3 for
exactly this reason. A 2v2 therefore costs the same bandwidth as a 1v1: the two
bots are generated locally on both machines.

Vision is shared across a team, in the renderer. It cannot live in the
simulation — two peers would immediately hold different state — but it is
derived from checksummed data, so every peer could compute every side's fog if
it wanted to.

### The lobby handshake has to be waited for, not raced

`joinOnlineRoom` used to resolve the moment it saw the other peer join, and the
version check rode a message that arrived afterwards. That made the check
decorative: the match had already begun by the time a mismatch was discovered,
so all it could do was report a desync it existed to prevent. It now waits for
the peer's handshake, which also carries the *mode* — an opaque string covering
everything the lobby chose. The transport never interprets it and only compares
it for equality, so map, roster and difficulty are all covered by one check that
cannot drift out of step with what it is guarding.

## Rendering gotchas worth not rediscovering

- **`DataTexture` does not flip.** `CanvasTexture` is uploaded with three.js's
  default vertical flip and `DataTexture` is not, so a data-backed overlay on the
  same ground plane needs its rows written bottom-up. The fog got this wrong and
  lifted over the mirror image of where the player actually was — which on a
  rotationally symmetric map reads as "the fog is upside down". See
  `tests/fog.test.ts`.
- **Anything floating above a unit is UI, so place it in screen space.** Offset
  along the camera's own up vector, not world Y: world Y projects to a slanted
  screen direction away from the centre of the view, so health bars drifted
  sideways off their units. Size them per-depth too, or they merge into a
  hairline when zoomed out.
- **The fog plane is flat, and cliffs are not.** Terrain with height stands
  straight through the shroud, so cliffs are shaded to match the fog rather than
  covered by it (`TerrainRenderer.applyFog`).

## Authored unit models

The three combat units are skinned FBX rigs, converted to GLB and drawn as one
instanced mesh per type per team; the Worker deliberately remains procedural.
Every clip is sampled into a bone-matrix texture at load, and each instance
carries one number — the row it is posed on.
`src/render/models/animated.ts` bakes, `src/render/animatedUnits.ts` draws. Four
things about this were learned the hard way:

- **Bind the `AnimationMixer` to the scene, not the mesh.** Three.js resolves a
  track's target as either a skeleton bone or a descendant of the mixer's root,
  and an FBX rig's helper nodes are neither — a 3ds Max Biped puts `Bip001`
  above the bones. Rooted at the mesh, every clip silently lost its root motion,
  which for the five-bone aircraft was most of the animation.
- **Apply the asset's node transform yourself.** Three.js uses it as the model
  matrix when it draws a `SkinnedMesh`; a hand-rolled instanced draw has to fold
  it into the instance matrix. An FBX authored Z-up arrives as a 90-degree
  rotation, so skipping it draws every unit on its back.
- **Ground and scale are measured from different things.** A run cycle crouches
  below the bind pose, so grounding by the bind pose buries the feet; an attack
  clip swings a sword overhead, so scaling by the animated extent sizes the unit
  by its weapon. Ground from the animation minimum, scale from the bind pose.
- **Fit each model on the axis it reads by.** Walkers by height, aircraft by
  wingspan — fitting a wide ship on height sizes it by whatever fin sticks up.

### Blending between clips

Each instance names two rows of the bone texture and a weight between them. One
mechanism, two jobs: neighbouring frames of one clip smooth the 30Hz bake up to
display rate, and frames of two clips cross-fade so a unit eases into its swing.
The lerp is componentwise on the matrices, which shortens a bone when the poses
differ by a large rotation — measured on the real geometry, 0.0% between
neighbouring frames.

No rig ships an idle clip and none is synthesised: standing still holds frame
zero of the run.

### Pick the clip from events, not from state

Which clip a unit plays is decided by `poseFor` in `src/render/entities.ts`,
driven by `world.events.shots` — the shots the simulation actually fired. It used
to be inferred from the unit's _order_ and its speed, which merely correlate with
fighting, and both halves were wrong at once:

- A unit defending itself holds **no order at all**. It is idle, hitting whatever
  walked into range — the most common fight in the game — so the attack branch
  never ran.
- Units in contact are shoved apart by separation every tick, so a Slicebot in
  melee is never quite stationary, and a movement-first rule kept it running on
  the spot even when the order _was_ set.

The swing is timed from the shot, not from wall-clock, so the blow lands on the
frame the damage did; and it does not loop, so a cooldown longer than the clip
holds the follow-through instead of restarting the wind-up. `tests/pose.test.ts`.

Skins are KTX2/ETC1S, encoded by `npm run textures` from `assets/textures/`. A
compressed texture cannot be flipped as it uploads the way a PNG can, so the
vertical flip these UVs need is baked in by the encoder. Three.js resolves its
own transcoder WASM against `import.meta.url`, so leave `setTranscoderPath`
alone — pointing it at a hand-copied `public/` folder ships the same 580 KB
twice.

**Source art is not committed, only the encoded result.** `assets/` is
git-ignored: a unit's PNGs are around 1.3 MB each against 140 KB for the KTX2
the game loads, and the PNGs neither ship nor take part in a build. Adding a
unit means dropping its PNGs there, running `npm run textures`, and committing
the `.ktx2` under `public/models/`. The encoder is deterministic — re-encoding
the original art reproduces the committed files byte for byte — so nothing is
lost by keeping the sources outside the repository, but they are the only way
back to an editable image, so keep them wherever the originals live.

Athena2 imports are scripted by `scripts/import-athena2-models.mjs`. The 64 files
under Athena2's `Animations` directory are the source inventory and baked timing
authority; `scripts/athena2-models.mjs` maps those animation assets to public
names, source rigs, clips, and skins, and rejects drift between the two sets.
Multipart skins are joined against their common armature before export; rigid
parts are assigned proxy joints, leaving the runtime's one-mesh contract intact.
FBX 6.1 files are losslessly rewritten by Autodesk's FBX SDK through a temporary
Unity project, while malformed curve tables are repaired by Blender. A small
manifest allowlist of frame-for-frame parity-proven, single-renderer rigs uses a
compact skeletal fallback: an isolated Unity 2022.3.62f3 project samples each
controller-bound mesh, local transforms, and bind poses at every authored frame,
then the importer writes those values directly to standard glTF without another
FBX conversion. Multipart prefabs and controller outputs that differ from the
baked recorder are not eligible. A required move, attack, or death slot with
only one baked frame is treated as missing. The original `Skeleton` source
remains in the 64-source inventory under the unpublished `SkeletonOriginal`
manifest identity. The physical `ZombieSkeleton` source selected by the
`ZombieRespawned` animation asset is published under the canonical `Skeleton`
name instead. Incomplete units, explicitly unpublished units, and stale generated
assets are excluded; Mercenary is also explicitly unpublished, leaving 54 public
models today. The catalog requires a
Human, Robot, Monster, or Undead assignment for every published model and the
gallery groups cards in that authored faction order. The importer
stages PNGs under the ignored `assets/textures/` by default (or a disposable
`--texture-stage` path);
`npm run textures` remains the only KTX2 encoder, and the staging directory can
be removed after encoding.

## Verifying determinism

```sh
npm test                    # full suite, including the per-tick replay check
npm run determinism:cross   # same match under V8 and JavaScriptCore, diffed
```

`determinism:cross` is the strongest check we have and it is nearly free: Node
and Bun ship independent JavaScript engines, so agreement between them is real
evidence of portability rather than a self-consistency tautology.

**A determinism check is only as good as the systems the scripted match reaches,
and nothing tells you when it stops reaching one.** Both checks run
`tests/helpers/scripted.ts`, and for a long time that match could not afford a
Barracks — a single build attempt at tick 120 against a 150-mineral cost and a
50-mineral bank, rejected silently because validation lives in the simulation
and not in the helper. No barracks, no army, no fighting: **one shot in 6000
ticks, worker on worker.** Every claim about combat being deterministic was
untested, and the file's own comment said "long enough to reach combat".

Three things were needed to fix it and each was independently missing — retry
(the first attempt is unaffordable), staffing (the builder wanders back to
minerals and the shell sits unfinished), and one-site-at-a-time (or each retry
lays another foundation nobody finishes). Contact happens around tick 2800, so
both checks now run 4000 ticks.

`determinism.test.ts > covers combat, not just economy` asserts the coverage —
both players firing, more than 20 shots — because a comment claiming coverage is
worth nothing and this one was wrong for months.

**The probe runs a second leg driven by the bots, for the same reason.** One
fixed script cannot reach everything: it never sets a rally point, never packs
buildings tightly enough for a trained unit to land inside one, and never jams a
crowd against a cliff. Fixes to all three landed without moving a single
checksum, which is the same blindness in a new place. The bot builds, expands
and fights on its own, so it reaches states no script will, and it is a
simulation-side command source — running it under both engines costs nothing.

**And a third leg on the four-player map.** Both of the others are a 1v1 on the
duel map, so neither ever reaches four players, the larger grid, team hostility,
or the bot's team-level decisions — a whole map and half a roster's worth of
arithmetic that the strongest check in the project would otherwise never run
under a second engine.

## Architecture

```
src/sim/      sealed deterministic simulation (no rendering, no DOM, no clock)
src/net/      lockstep scheduler + transports (local, BroadcastChannel, WebRTC)
src/ai/       bot — a deterministic command source, identical on every peer
src/render/   three.js rendering, reads the sim, never writes it
src/input/    selection and camera control
src/ui/       DOM HUD overlay
src/config/   all balance data
```

The simulation advances at a fixed **20 ticks per second**. Rendering runs at
display rate and interpolates between the last two ticks; interpolation alpha is
clamped to [0,1] so a network stall freezes units rather than extrapolating them
through walls.

### Input delay is per-peer, and each peer must be told what to do with it

Commands execute `delayTurns` turns after they are issued. That number is _not_
shared: the wire carries the absolute turn each command belongs to, so a peer
files what it receives and never infers a schedule. Two peers can hold different
delays for the whole match without diverging by a bit.

Two things make it work, and both were arrived at the hard way:

- **The turns a peer has scheduled must stay a contiguous prefix.** Raising the
  delay opens a gap, and a turn nobody ever sends for blocks _everyone_ forever,
  so the gap is filled in the same packet. Lowering it cannot rewrite a turn
  already sent — a peer may have executed it — so the schedule simply pauses and
  lets the clock catch up.
- **A peer cannot adapt from its own stalls.** A stall says the _other_ peer is
  late, and raising your own delay changes only your own sends. Driven that way,
  the loop settles at one peer pinned to the ceiling and stalling permanently
  while the other sits at the floor and never learns it is the problem —
  measured at 120ms: 1133 stalled frames against 19 for the old fixed delay. So
  each packet carries `peerHeadroom`, how early the sender is receiving _your_
  packets, and each peer sizes its own delay from what it is told.

Measured against the fixed 200ms it replaced: LAN drops to 100ms with no stalls,
transpacific goes from 1893 stalled frames to 52, and a dire link runs 2.3x
closer to real time. `tests/inputDelay.test.ts`.

### Commands are the only thing on the wire

Player intent goes in, entity state never does. This is what keeps bandwidth
flat whether a battle has twenty units or two thousand. All validation —
affordability, ownership, placement legality — happens in
`src/sim/systems/orders.ts`, never in the UI. The UI may grey out a button, but
the simulation decides.

### The data channel is unordered, and packets must stay under 16 KB

A WebRTC data channel defaults to _ordered_ delivery — head-of-line blocking,
same as TCP. That is the wrong default here: each packet already repeats the
previous two turns' commands, so the packet held up behind a lost one is usually
the one carrying what the receiver is waiting for. Ordered delivery turns a loss
the protocol was built to absorb into a round-trip stall for **both** players.
So the channel is opened `{ordered: false}`, injected through Trystero's
`rtcPolyfill` hook since it takes no data-channel options of its own.

Reliability is deliberately kept. `maxRetransmits: 0` would lose the one-shot
version handshake that rides the same channel, and would push all-copies-lost
recovery onto the lockstep history resend, which is throttled to 120 ms — slower
than SCTP retransmitting in one round trip. The bandwidth saved would be nil.

**The constraint this creates:** Trystero splits payloads over ~16 KB into chunks
and reassembles them _by arrival order_, with no sequence number — so a
multi-chunk message is scrambled by the very reordering we asked for. Packets
stay far under that because `MAX_SELECTION` is 24 and bot commands never cross
the wire, but it is an invariant now, not a coincidence. `tests/wire.test.ts`
fails loudly if a change to the selection cap or the packet shape breaks it.

Reordering itself is harmless for a second, independent reason: the receive path
is structurally order-free (turns keyed absolutely, first write wins). Measured,
an "assume packets arrive in order" bug is _invisible_ under pure reordering —
the redundancy covers it — and only shows up, weakly, once 25% loss removes the
covering copies.

### The AI is not special

`src/ai/bot.ts` emits the same `Command` objects a human does, and because it is
deterministic it runs identically on every peer at zero bandwidth. Single-player
and multiplayer are therefore one code path, not two. That also means the only
producer of commands on the wire is local human input, which is what bounds
packet size.

**Two bots on a side are not two bots.** Run naively, an AI team is much weaker
than one bot with twice the economy: each half picks its own target, commits on
its own count, and gets beaten one at a time by an army that never had to fight
both at once. So *when to commit* and *what to hit* are decided over the team's
army rather than each bot's own — one centroid, not one each. Every bot on a side
derives it from the same world in the same ascending pass, so they agree with no
coordination channel and no shared state to keep in step.

Three other things the bot was simply missing, each worth more than any tuning:

- **It never came home.** An army crossing the map while its own Command Post is
  being shot is the most expensive mistake available, and the old one made it
  every match. Defence is checked before attack and, when it applies, is the only
  order issued.
- **It attacked the lowest-index enemy building**, which never changed. An army
  that had fought its way into a base would walk back out past a Barracks to keep
  pounding a Command Post it had already passed. The target is now the hostile
  structure nearest the team's army.
- **It trained workers from one Command Post.** An expansion's mineral line
  therefore sat empty for minutes after it finished. Every base trains now, and
  the worker target scales with how many there are to work.

Production queues two deep normally and to the cap once minerals pile up: at two
deep, every bot in a four-player match floated six to eight thousand minerals for
the last five minutes — an army it had paid for and never received.

Difficulty is behavioural only: no bonus income, no extra units, no cheating on
fog. `THINK_INTERVAL` is deliberately *not* a difficulty knob — every bot must
think on the same tick, or whoever thinks first gets a whole interval of head
start, which measurably decided a mirror matchup.

### Movement has three movers, and they are easy to miss

Direct steering, flow fields and A* path-following each advance units, and A* —
which carries most individual orders — walks its waypoints in a loop of its own
rather than through `stepToward`. Acceleration added to the shared helper alone
changed nothing for it, and nothing failed to say so. `tests/accel.test.ts`
pins the ramp for that reason.

Units accelerate by a fraction of their top speed per tick and brake by
`v = sqrt(2*a*d)` against the distance still to run — not to the next waypoint,
or they stutter at every corner of a path. `Math.sqrt` is the one non-trivial
function the sim may use; see the sealed-core rules above.

### Attack-move is a loop, and the missing half is the return

Attack-move has two transitions, not one. Stopping to fight was implemented;
starting again was not, and the omission is invisible in the code because
nothing looks wrong at the point where it should have happened.

Engaging calls `clearPath`, which wipes the route **and** `flowGoal`. That is
right — a unit holding its ground should not also be walking — but it deletes
the only record of where the unit was going. So an army sent across the map
stopped at the first thing it killed and stood there for the rest of the match,
still holding an `AttackMove` order it could never complete: sixteen tiles
short, order still set, looking for all the world like a pathfinding failure.

Three pieces fix it, and each is load-bearing:

- **`navGoal` holds the order's intent, not its route**, so it survives
  `clearPath`. Resuming has to restore the _same shared flow-field goal_ —
  rebuilding per unit is one Dijkstra sweep each, the collapse described under
  "Spread the arrival" above.
- **`resumeAdvance` runs before the movers**, so a unit that resumes this tick
  walks this tick, and it is what converts `AttackMove` to `None` on arrival.
  Nothing else does; without it the order is permanent.
- **`engageNearby` accepts `AttackMove`, not just `None`.** Idle units defend
  themselves and attack-movers go and take what they saw; a plain `Move` does
  neither, which is the entire reason attack-move exists as a separate order.

**The pursuit leash must be anchored where the chase began.** Measured from the
unit's current position the window slides along with a retreating enemy and the
chase ratchets indefinitely — a unit dragged sideways followed a fleeing
Burstbot 14.6 tiles off its route. Past the leash the anchor is _kept_, so the
unit turns back and can pick the fight up again on its way past; it resets only
when the target is dead or gone beyond acquisition. `tests/attackMove.test.ts`.

**Do not measure unit behaviour without checking the unit is alive.** A corpse
keeps its last `order` and its last position, so a dead unit looks exactly like
a unit that stopped and never resumed. Staging two identical Burstbots against
each other means the one under test usually dies, and the trap reports the bug
you were looking for whether or not it exists — it cost a first attempt at this
fix, "verified" against a body. Heal the unit under test, or give it something
weak to kill, and assert `alive` before reading anything else.

### Damage is one number, and the panel shows it

There was a counter triangle — Burstbot/Beamdrone/Slicebot, each dealing **double**
to one other, applied as a percentage inside `combatSystem`. It is gone. The
multiplier appeared nowhere on screen, so no honest damage figure could be shown
next to a unit; now `def.damage` is what a unit deals to everything it can
shoot, and the info panel prints it.

What decides matchups is what a player can read — range, speed, health — plus
one structural rule that is not a number at all: `canHitAir`, so a Slicebot
cannot touch a flyer. `tests/units.test.ts` stages a real fight for every armed
pair and asserts the blow equals the listed damage, which is the only way to
catch a multiplier creeping back in between the def and `applyDamage`.

**This makes the Burstbot the best buy at equal supply, and by a wide margin.**
Measured, 6 supply a side, both armies attack-moving into each other:

|           | dps/supply | hp/supply | range | result                                 |
| --------- | ---------- | --------- | ----- | -------------------------------------- |
| Burstbot  | 7.50       | 45        | 5.0   | beats Slicebots 5–0 and Beamdrones 6–0 |
| Slicebot  | 5.42       | 45        | 0.9   | —                                      |
| Beamdrone | 5.00       | 35        | 3.5   | —                                      |

The triangle was carrying the roster. Nothing is wrong with the _code_ — the
numbers in `config/rules.ts` were tuned around a 2x that no longer exists, and
retuning them is the outstanding job.

### Acquiring a target is also an instruction to walk to it

`engageNearby` steps an idle unit toward whatever combat picked out, so a target
a unit cannot actually shoot is not merely a wasted swing — it is a chase. That
is why `canHitAir` is enforced in `acquireTarget` rather than only at the moment
of firing: refusing the target and stopping the chase are the same fix.

Melee is ground-only, and that includes workers — a 0.6-tile reach is a swing by
any reading. Explicit attack orders are refused per unit rather than per order,
so a mixed selection still sends its Burstbots. `tests/air.test.ts`.

**Workers have a weapon.** Range 0.6, and they will shoot anything that comes
near their base. Any combat measurement staged near a start location is
measuring the workers as much as the units under test, which is how a melee unit
first appeared to be landing impossible blows on an aircraft.

### A building is a square, and units must approach its nearest face

Two separate mistakes made workers walk around a Command Post rather than
delivering where they stood, and each is worth avoiding again:

- **Reach was measured against the circle inscribed in the footprint.** That
  reaches half a tile past the middle of each face and falls short of every
  corner — on a 4-tile footprint the corners sit at 2.83 from centre against a
  reach of 2.67, so diagonal approaches could not deliver at all.
  `distanceSqTo` now measures to the box for anything with a footprint.
- **Units pathed at the building's centre.** That tile is not walkable, so A*
  substitutes the nearest walkable tile to it — the _same_ tile for every unit,
  whichever side it came from. `approachPoint` clamps the unit's position to the
  footprint so each one heads for its own near face.

Measured on the opening harvest: 314 deliveries in 4000 ticks before, 513 after,
and a loaded worker's detour fell from up to 2.6x the direct distance to about
1.0x from every side. `tests/dropoff.test.ts`.

### The click target is the ring, and rally lives above the units check

`pickAt` casts to the ground plane and tests the drawn selection ring, so what a
player can click is exactly what they can see. A screen-space tolerance was
tried and is worse: it does not shrink as the camera pulls back, so a distant
clump becomes a lottery. Buildings hit-test their footprint square, not a
circle. The ring size constant lives in `input/selection.ts` and the renderer
imports it — the two drifting apart is the whole failure mode.

Right-clicking the ground with a production building selected sets its rally
point, and that check has to run **before** `hasOwnUnits`. That guard
deliberately ignores buildings, since buildings take no movement orders — so a
selection holding nothing but a Barracks returned from `issueContextOrder`
before reaching the rally code, and the feature silently did nothing. It was
verified by calling `issueGroundOrder` directly, which walks straight past the
line that breaks it; only a real right-click finds this.

### Collision radius is the unit's size, everywhere

`radius` is the collision size, the weapon-reach margin (`attackRange + target
radius`) and what the selection ring is drawn from. Changing it therefore moves
combat balance as well as spacing — a bigger unit is a slightly easier target.
Flyers are the exception in one direction only: `collides` is false for them
because it also decides whether a thing occupies map tiles, so separation gives
them their own rule and matches air against air.

### Nothing may be written to a position a unit cannot occupy

`clampToMap` teleports a unit standing on a solid tile to the middle of the
nearest open one. That is a good backstop and a terrible routine event: if
whatever put the unit there does it again next tick, the unit snaps between two
positions for as long as the cause lasts, which is what players report as
"stuck". Three separate things were doing exactly that, and each needed fixing
at the source rather than in the backstop:

- **Production placed trained units at a point computed purely geometrically** —
  a fixed offset from the building's centre, checked against nothing. Measured
  over a bot match, 11 of 89 trained units appeared inside a building footprint
  and 3 on solid tiles. `spawnPointFor` now treats that point as a _preference_
  and falls back to the nearest standable tile; when there is no room at all it
  reports failure and production waits, exactly as it does when supply runs out.
- **A rally point was validated for being on the map, not for being ground.** A
  rally is a move order handed to every unit a building will ever train, so a
  rally on a cliff is the unreachable-destination bug repeated forever — and it
  never passed through `standableTarget`, because production writes the order
  straight onto the unit. Both ends are checked now: where the rally is set, and
  again at spawn, since a building can be raised on the spot afterwards.
- **Separation and direct steering wrote positions with no idea walls exist.** A
  crowd fighting against a cliff pushed its outer members into the rock every
  tick. This was 22 of the 25 relocations in a bot match. Both now go through
  `nudgeBy`, which tries the axes separately so a unit shoved into a wall slides
  along it instead of sticking.

After all three, the only relocations left in a bot match are the legitimate
one: a building raised on top of a unit that was standing there.

`tests/stuck.test.ts` pins each cause separately and then asserts the property
itself over whole matches — no unit standing on a solid tile, none ordered onto
one, and none holding a single move order longer than crossing the map could
take. **Check the invariant every tick, not at the end**: `clampToMap` repairs a
bad placement within a tick or two, so by the end of a run a unit that was
dropped inside a wall looks exactly like one that never was.

### A slot index is not an identity

The pool recycles slots and bumps a generation on every free, and `isAlive`
checks it, precisely so a stale reference can be spotted. Anything outside the
simulation that remembers an entity must remember the **handle**, not the index.

`Selection` remembered indices. So a dead unit's slot, refilled by the next unit
trained, quietly rejoined the selection — and a control group whose three
members had all died came back holding three Beamdrones that were never put in it.
Both the live selection and the stored groups had it, since both were index
lists; `prune` sees `alive === 1` on a reused slot and keeps it.

A group whose members are all dead is now forgotten rather than left empty:
empty is not a stable state, because those slots get reused. `tests/picking.test.ts`
covers the case the delete-on-empty alone does not — the player never presses
the key while the slots are empty, only later, when strangers are standing in
them.

### An order must name somewhere a unit can stand

Nothing stopped a right-click on a cliff being issued, and the resulting order
could never complete: A\* aims at the nearest walkable tile and stops there, but
arrival is measured against the ordered point. A lone unit stopped and kept its
order; a _group_ was worse, because a flow field cannot route to a solid tile at
all — twelve units pushed at the rock for the rest of the match, six tiles short,
and `settleArrivals` never applied because it only runs near the destination.

`standableTarget` snaps the commanded point to the nearest walkable tile before
anything else sees it, and refuses the order outright if there is nowhere within
`DESTINATION_SNAP_RINGS`. Fixing it at the point the order is created rather than
in the movers is what keeps it simple: every arrival rule downstream already
copes with a destination that exists.

### Every unit must be able to stop

Arrival is "within half a tile of the point you were given", and in a crowd most
units can never satisfy that: their spot is taken, and separation pushes them
out of it faster than they can walk back in. Nothing else ended the order, so
they pushed at the destination for the rest of the match, still holding a
`Move`. Measured on open ground, **20 of 24 units in one group move never came
to rest** — some 3.5 tiles from their point.

Three things fix it, and each covers a case the others do not:

- **The last stretch is steered at the unit's own formation slot**, not at the
  group's shared goal tile (`FORMATION_APPROACH`). The field aims everyone at
  one tile; near the destination that tile is full, so units funnel into a scrum
  and are shoved out of it forever. This is the half that makes a group _look_
  right, and it settles a 24-unit march at tick 183 instead of 261.
- **A unit that stops improving gives up** (`settleArrivals`). This is the
  guarantee, and it is the only thing that handles a destination with no room at
  all — ordered into a walled pocket, two units still never rested without it.
- **Formation slots are checked for walkability.** The old check was
  `tileOfPos(...) < 0`, which asks whether a point is _on the map_; solid rock is
  very much on the map.

The failure mode of a give-up rule is giving up too early, which reads as broken
pathfinding, so it is fenced three ways: it only runs within `SETTLE_RANGE` of
the destination (further out, a unit walking around an obstacle legitimately
fails to close the straight-line gap for a long stretch), progress must beat the
closest approach so far by a real margin rather than by separation jitter, and a
unit with a live combat target is fighting, not failing to arrive.
`tests/spread.test.ts` pins all three parts independently.

### Spread the arrival, never the pathfinding goal

A group move gives each unit its own destination so an army stops arriving in a
heap. The obvious way to do it is a trap: grouped moves navigate by flow field
and the field is cached **per goal tile**, so a per-unit goal turns one Dijkstra
sweep into one per unit. Not a crash — just a silent collapse, measured at 5
seconds to over 150 for the test match. The shared destination steers the group;
the spread point only decides where each unit finally stands.

The layout is a square spiral rather than a ring because a ring needs sine and
cosine. Separation rounds the corners off it anyway.

## Performance notes

Pathfinding dominates if left alone. Two measured lessons worth keeping:

- **Group moves use flow fields, individual errands use A\*.** One Dijkstra sweep
  from the destination serves a whole army; running per-unit A\* for a 30-unit
  attack-move cost ~60ms per tick. See `GROUP_PATH_THRESHOLD`.
- **Failed searches must back off.** A unit that cannot reach its target used to
  re-run a full-budget A\* every tick — 6.7 doomed searches per tick at 3ms each,
  which was 97% of total simulation time. See `pathCooldown`.

Both fixes together took the test match from 31s to 0.9s.

When touching pathfinding, size heap arrays by **pushes, not tiles**: Dijkstra
and A\* re-insert a node on every relaxation, and writing past the end of a typed
array fails _silently_, corrupting results rather than throwing.
