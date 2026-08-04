# experiment-rts — contributor notes

A 3D real-time strategy game for the browser, with peer-to-peer multiplayer and
no server of our own.

## The one rule that matters

`src/sim/**` is a **sealed deterministic core**. Two players' machines run the
simulation independently and exchange only commands — never entity state — so if
the two simulations ever disagree by a single bit, the game silently becomes two
different games.

Inside `src/sim/**` (and `src/config/rules.ts`), you may not use:

| Banned | Use instead |
|---|---|
| `Math.random()` | the seeded `Rng` on `World` |
| `Date.now()`, `performance.now()`, `new Date()` | `world.tick` |
| `Math.sin/cos/tan/atan2/pow/exp/log/hypot` | see "no trigonometry" below |
| `for...in`, iteration over `Map`/`Set` | index loops over the entity pool |
| raw `float` positions | Q16.16 fixed-point (`src/sim/fixed.ts`) |
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

`map.elevation` is cosmetic and not checksummed — the simulation decides movement
from walkability alone — but it is still computed deterministically, because two
peers rendering visibly different terrain would look exactly like a desync.

## Mirror fairness is not the same as symmetry

The map is symmetric; that does not make the *match* symmetric. Anything placed
at setup has to be laid out as an exact 180-degree rotation of player 0's, not by
applying the same offsets to a mirrored start. Three separate bugs came from
getting this subtly wrong, each worth a real advantage and none visible on
screen:

- A Command Post's footprint is 4, and `start - 2` cannot be symmetric about a
  tile — so player 1's whole base sat a tile nearer the middle of the map.
- Reflecting a mineral patch's *top-left corner* about the base's centre tile
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
  *differently shaped* route — always the same length, never the same tiles.
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
instanced mesh per type per team: every clip is sampled into a bone-matrix
texture at load, and each instance carries one number — the row it is posed on.
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
to be inferred from the unit's *order* and its speed, which merely correlate with
fighting, and both halves were wrong at once:

- A unit defending itself holds **no order at all**. It is idle, hitting whatever
  walked into range — the most common fight in the game — so the attack branch
  never ran.
- Units in contact are shoved apart by separation every tick, so a brawler in
  melee is never quite stationary, and a movement-first rule kept it running on
  the spot even when the order *was* set.

The swing is timed from the shot, not from wall-clock, so the blow lands on the
frame the damage did; and it does not loop, so a cooldown longer than the clip
holds the follow-through instead of restarting the wind-up. `tests/pose.test.ts`.

Skins are KTX2/ETC1S, encoded by `npm run textures` from `assets/textures/`
(source art, not served). A compressed texture cannot be flipped as it uploads
the way a PNG can, so the vertical flip these UVs need is baked in by the
encoder. Three.js resolves its own transcoder WASM against `import.meta.url`, so
leave `setTranscoderPath` alone — pointing it at a hand-copied `public/` folder
ships the same 580 KB twice.

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

Commands execute `delayTurns` turns after they are issued. That number is *not*
shared: the wire carries the absolute turn each command belongs to, so a peer
files what it receives and never infers a schedule. Two peers can hold different
delays for the whole match without diverging by a bit.

Two things make it work, and both were arrived at the hard way:

- **The turns a peer has scheduled must stay a contiguous prefix.** Raising the
  delay opens a gap, and a turn nobody ever sends for blocks *everyone* forever,
  so the gap is filled in the same packet. Lowering it cannot rewrite a turn
  already sent — a peer may have executed it — so the schedule simply pauses and
  lets the clock catch up.
- **A peer cannot adapt from its own stalls.** A stall says the *other* peer is
  late, and raising your own delay changes only your own sends. Driven that way,
  the loop settles at one peer pinned to the ceiling and stalling permanently
  while the other sits at the floor and never learns it is the problem —
  measured at 120ms: 1133 stalled frames against 19 for the old fixed delay. So
  each packet carries `peerHeadroom`, how early the sender is receiving *your*
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

A WebRTC data channel defaults to *ordered* delivery — head-of-line blocking,
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
and reassembles them *by arrival order*, with no sequence number — so a
multi-chunk message is scrambled by the very reordering we asked for. Packets
stay far under that because `MAX_SELECTION` is 24 and bot commands never cross
the wire, but it is an invariant now, not a coincidence. `tests/wire.test.ts`
fails loudly if a change to the selection cap or the packet shape breaks it.

Reordering itself is harmless for a second, independent reason: the receive path
is structurally order-free (turns keyed absolutely, first write wins). Measured,
an "assume packets arrive in order" bug is *invisible* under pure reordering —
the redundancy covers it — and only shows up, weakly, once 25% loss removes the
covering copies.

### The AI is not special

`src/ai/bot.ts` emits the same `Command` objects a human does, and because it is
deterministic it runs identically on every peer at zero bandwidth. Single-player
and multiplayer are therefore one code path, not two. That also means the only
producer of commands on the wire is local human input, which is what bounds
packet size.

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
  `clearPath`. Resuming has to restore the *same shared flow-field goal* —
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
rifleman 14.6 tiles off its route. Past the leash the anchor is *kept*, so the
unit turns back and can pick the fight up again on its way past; it resets only
when the target is dead or gone beyond acquisition. `tests/attackMove.test.ts`.

**Do not measure unit behaviour without checking the unit is alive.** A corpse
keeps its last `order` and its last position, so a dead unit looks exactly like
a unit that stopped and never resumed. Staging two identical riflemen against
each other means the one under test usually dies, and the trap reports the bug
you were looking for whether or not it exists — it cost a first attempt at this
fix, "verified" against a body. Heal the unit under test, or give it something
weak to kill, and assert `alive` before reading anything else.

### Damage is one number, and the panel shows it

There was a counter triangle — Rifleman/Gunship/Brawler, each dealing **double**
to one other, applied as a percentage inside `combatSystem`. It is gone. The
multiplier appeared nowhere on screen, so no honest damage figure could be shown
next to a unit; now `def.damage` is what a unit deals to everything it can
shoot, and the info panel prints it.

What decides matchups is what a player can read — range, speed, health — plus
one structural rule that is not a number at all: `canHitAir`, so a Brawler
cannot touch a flyer. `tests/units.test.ts` stages a real fight for every armed
pair and asserts the blow equals the listed damage, which is the only way to
catch a multiplier creeping back in between the def and `applyDamage`.

**This makes the Rifleman the best buy at equal supply, and by a wide margin.**
Measured, 6 supply a side, both armies attack-moving into each other:

| | dps/supply | hp/supply | range | result |
|---|---|---|---|---|
| Rifleman | 7.50 | 45 | 5.0 | beats Brawlers 5–0 and Gunships 6–0 |
| Brawler | 5.42 | 45 | 0.9 | — |
| Gunship | 5.00 | 35 | 3.5 | — |

The triangle was carrying the roster. Nothing is wrong with the *code* — the
numbers in `config/rules.ts` were tuned around a 2x that no longer exists, and
retuning them is the outstanding job.

### Acquiring a target is also an instruction to walk to it

`engageNearby` steps an idle unit toward whatever combat picked out, so a target
a unit cannot actually shoot is not merely a wasted swing — it is a chase. That
is why `canHitAir` is enforced in `acquireTarget` rather than only at the moment
of firing: refusing the target and stopping the chase are the same fix.

Melee is ground-only, and that includes workers — a 0.6-tile reach is a swing by
any reading. Explicit attack orders are refused per unit rather than per order,
so a mixed selection still sends its riflemen. `tests/air.test.ts`.

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
  substitutes the nearest walkable tile to it — the *same* tile for every unit,
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
  and are shoved out of it forever. This is the half that makes a group *look*
  right, and it settles a 24-unit march at tick 183 instead of 261.
- **A unit that stops improving gives up** (`settleArrivals`). This is the
  guarantee, and it is the only thing that handles a destination with no room at
  all — ordered into a walled pocket, two units still never rested without it.
- **Formation slots are checked for walkability.** The old check was
  `tileOfPos(...) < 0`, which asks whether a point is *on the map*; solid rock is
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
array fails *silently*, corrupting results rather than throwing.
