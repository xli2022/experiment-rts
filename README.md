# Experiment RTS

A 3D real-time strategy game that runs in the browser, with online multiplayer
and **no server to deploy**.

**▶ [Play it](https://xli2022.github.io/experiment-rts/)**

Gather minerals, build a base, train an army, and destroy every enemy structure.
Play against the AI, against someone in a second tab, or against a friend
anywhere — peer-to-peer over WebRTC. Or team up: **co-op** puts you and a
partner on one side of a four-corner map against two AI opponents.

```sh
npm install
npm run dev          # http://localhost:5173
```

Every push to `master` builds and publishes to GitHub Pages, gated on the test
suite — see [`.github/workflows/deploy.yml`](./.github/workflows/deploy.yml).

## Playing

|                  |                                                                    |
| ---------------- | ------------------------------------------------------------------ |
| Pan              | Arrow keys, screen edges, middle-drag, or the minimap              |
| Zoom             | mouse wheel                                                        |
| Select           | click, or drag a box                                               |
| Add to selection | `Shift` + click                                                    |
| Order            | right-click — attack an enemy, mine a patch, or move               |
| Attack-move      | `A` then click                                                     |
| Stop / Hold      | `S` / `H`                                                          |
| Control groups   | `Ctrl`+`1`–`9` to assign, `1`–`9` to recall                        |
| Build / train    | the letter shown on each command-card button                       |
| Repair           | right-click your own damaged or half-built structure with a worker |
| Surrender        | the flag button top-right — confirmed, and it cannot be undone     |
| Fullscreen       | `F`, or the button top-right                                       |
| Mute             | `M`                                                                |
| Model gallery    | `All units` button on the home screen                              |
| Cancel           | `Esc`                                                              |

Workers gather minerals, construct buildings, and repair them. The Command Post
trains workers, Supply Depots raise the supply cap, Barracks train the three
combat units, and Turrets defend. Build a second Command Post on an expansion to
mine two lines at once. You lose when your last structure falls — in co-op, when
the last structure on your _side_ falls.

### The three combat units

|                       | Damage        | Range | Health | Supply | Notes                          |
| --------------------- | ------------- | ----- | ------ | ------ | ------------------------------ |
| **Burstbot** (ranged) | 6 every 0.8s  | 5.0   | 45     | 1      | Cheapest, outranges everything |
| **Slicebot** (melee)  | 13 every 1.2s | 0.9   | 90     | 2      | Cannot touch air at all        |
| **Beamdrone** (air)   | 10 every 1.0s | 3.5   | 70     | 2      | Ignores terrain entirely       |

A unit deals its listed damage to everything it can shoot — there is no hidden
per-matchup multiplier, and the figure on the info panel is the figure you get.
What beats what is decided by the numbers above plus one hard rule: **a Slicebot
cannot reach a flyer**, so an army of them needs company.

Fog of war hides what you are not currently watching. It is a rendering feature
rather than a secret: peer-to-peer lockstep gives every client the whole game
state by construction, exactly as the genre's originals did. What it buys is
scouting and map control.

### Co-op

Two players share a side against two AI opponents, on a map with a base in each
corner and a team to each side of it. Allies:

- never damage each other, and cannot be ordered to try;
- **share vision** — whatever your partner scouts, you can see;
- win and lose together — a side is beaten only when every structure on it is
  gone, so a player whose own base is razed is on the winning side if their
  partner finishes the job. Being knocked out, or surrendering, does take
  everything you still own with it: you are out, and you watch.

Pick which AI plays in the lobby — every mode with an AI in it offers the same
two, and the choice follows you between them. **Scripted** plays one fixed,
tuned strategy and reads the whole map. **Neural** is a learned player that
sees only what you would — fog of war and its own memory — and runs its model
in your browser; the chip is live only in a build that ships a model (see
"Training the neural bot" below). It plays every mode with an AI in it, online
included — in online co-op each browser hosts one of the two bots, so both
players' browsers load the model before they connect. Neither is given
anything: no bonus income, no extra units.
And two scripted opponents play as one side: they count their armies together
before committing and push the same target, so they arrive together rather
than one at a time.

There is a solo option too, which fills your partner's seat with the scripted
AI. Playing co-op online works exactly like a normal online match — same room
code, both players pick the same mode.

### The maps

**Three Lanes** (1v1). Bases sit in opposite corners, joined by **three routes** — a middle lane through
the centre and two outer lanes that hug the map edge — plus a corner-to-corner
river crossing all three at the centre, and short connectors between mid and the
outer lanes. Everything else is cliff.

That means no single blockade closes the map, the centre is a four-way crossroads
worth holding, and a flank costs a real commitment rather than a free rotation.
The layout is generated, not authored: the whole map is carved with a brush that
opens every tile _and its 180-degree opposite_, so the two halves are identical
by construction rather than by inspection.

Two **expansions** sit in pockets off the lanes, one nearer each player, with a
mineral line already on them and room for a Command Post. Nobody owns one until
they build there, and the patches are smaller than a main — so taking one is a
decision about when your home line stops keeping up, not a free upgrade.

**Four Quarters** (2v2, co-op). A base in each corner and a team along each
edge, so your partner is next door and your opponents are across the map. Each
team gets a **back lane** of its own along its edge, which is what makes helping
a partner a short walk rather than a trip through the middle; two **flanks** run
down the left and right edges, and two **diagonals** cross at the centre. Six
expansions: a natural for each base, plus a contested pair halfway down each
flank. Same generator and the same mirrored brush, so the two sides are exact
180-degree rotations of each other.

## Multiplayer without a server

Three modes, one code path. The menu asks _who_ you are playing first and _how
you connect_ second, because those are two questions and the versus modes are
the same match over two different transports:

- **Skirmish vs AI** — the bot is a player hosted by your machine: it reads
  the game, issues the same commands you do, and they take the same path yours
  take.
- **Co-op vs AI** — the same thing with four slots instead of two: two humans,
  each hosting one of the two bots. A bot's commands ride the wire like a
  human's, a few hundred bytes a turn, and land after its host's input delay,
  so each bot reacts as fast as the connection of the player running it.
- **Versus another player** — 1v1, either over a shared room code or against a
  second tab on this computer.

The two ways to connect:

- **Two tabs** — a second tab on the same computer, over `BroadcastChannel`. No
  network at all.
- **Online** — share a room code. Peers find each other through the public
  [Nostr](https://nostr.com) relay network via
  [Trystero](https://github.com/dmotz/trystero), then talk **directly**,
  end-to-end encrypted. Nothing of ours sits in the middle.

### How it stays in sync

Peers never exchange game state. They exchange _commands_ — "these units, move
here" — and each machine runs an identical simulation. A 200-unit battle costs
the same bandwidth as a single click. This is what StarCraft does, and it is why
peer-to-peer works at all for a genre with thousands of moving objects.

The price is that both simulations must agree **exactly**, forever. The
contributor notes below hold the rules that guarantee it — fixed-point
arithmetic, a seeded RNG, no wall-clock time, and no reliance on JavaScript's
non-portable `Math` functions.

### The one honest caveat

Roughly one connection in ten cannot establish a direct peer link, because both
players sit behind symmetric NAT. Working around that needs a TURN relay, the
one piece that genuinely cannot be borrowed for free. When it happens the lobby
says so plainly instead of hanging; TURN credentials can be supplied to
`joinOnlineRoom` if you have them.

## Development

```sh
npm run dev                 # dev server
npm test                    # full suite
npm run typecheck           # tsc --noEmit
npm run build               # production build
npm run determinism:cross   # run the same match under V8 and JavaScriptCore
npm run mirror:probe        # where a mirrored match first stops mirroring
npm run ml:bench            # decisions per second of the neural bot's environment
npm run ml:arena -- --a scripted@10 --b scripted@20 --seeds 8   # bot against bot
```

`determinism:cross` is the strongest check here and it is close to free: Node
runs on V8 and Bun on JavaScriptCore, two independent JavaScript engines with
independently implemented maths, so agreement between them is real evidence of
portability. "Verifying determinism" in the contributor notes says what it runs
and why each leg exists.

The test suite also plays complete AI-vs-AI matches to a victory condition
across several seeds. That layer exists because a simulation can be perfectly
deterministic and still not be a game: it caught armies that grew into the
hundreds without ever fighting, an AI that hoarded 9,000 unspendable minerals,
and workers that stopped one step short of the building they were sent to make.

### Training the neural bot

The learned bot is trained in Python against matches served by the game's own
code under Bun — `tools/ml/serve.ts` steps headless matches one decision at a
time, `ml/` holds the model, imitation from the scripted bot, PPO against a
league, and the export to ONNX that the browser runs. It needs `bun`, a GPU
machine and a few hours:

```sh
cd ml && pip install -e '.[dev]' && pytest
rtsml-imitate --procs 16 --envs 8 --steps 5e6
rtsml-ppo --init runs/bc/best.pt --procs 16 --envs 8
rtsml-eval --ckpt runs/ppo/best.pt --seeds 64 --out eval.json
rtsml-export --ckpt runs/ppo/best.pt --evaluation eval.json --int8
```

See `ml/README.md` for what each step is and the gates between them. The
export writes `public/models/policy.onnx` and `policy.json`; a build that has
them offers the Neural chip, and

```sh
npm run build && npm run smoke:neural
```

plays the model in headless Chromium for half a minute and fails on any error,
skipped decision or silent slot.

### Layout

```
src/sim/      the deterministic simulation — no rendering, no DOM, no clock
src/net/      lockstep scheduler and the three transports
src/ai/       the bots — players hosted by a peer, scripted or neural
src/render/   Three.js rendering; reads the simulation, never writes it
src/input/    selection and camera
src/ui/       DOM overlay: HUD, minimap, lobby
src/config/   all balance data, in one table
```

Buildings and Workers use Three.js primitives at runtime. The three combat units
are authored models: skinned rigs with run, attack and death clips, baked at load
into a bone-matrix texture so a hundred of them mid-swing still cost one draw
call per type. Their skins are KTX2/ETC1S — a tenth the size of the source PNGs.

Only the encoded `.ktx2` textures are committed; the source texture art is not
in the repository. Put the PNGs in
`assets/textures/` (git-ignored) and re-encode with

```sh
npm run textures
```

Complete Athena2 rigs can be imported with the same pipeline. The importer
uses Athena2's 64-file `Assets/Art/Units/Animations` inventory as its source
authority. A required move, attack, or death slot with only one baked frame is
treated as missing. The original `Skeleton` source remains in that inventory but
is explicitly excluded; the retained `ZombieRespawned` rig from the physical
`ZombieSkeleton` folder now occupies the canonical public `Skeleton` slot. That
replacement is retained while Mercenary is explicitly unpublished, leaving 54
public models today. `all-units.json` assigns every published
model to Human, Robot, Monster, or Undead, and the gallery presents those four
factions in the authored order. For
each published complete unit the importer combines the clips into one GLB, joins
multipart meshes, and stages the exact prefab skins. Old FBX 6.1 rigs are
normalized with Unity's FBX SDK;
malformed curves and some multipart geometry use Blender 4.5. FireDragon and a
small explicit allowlist of single-renderer rigs use direct Unity sampling after
their complete run, attack, and death output has been checked frame-for-frame
against Athena2's baked geometry. The importer opens isolated copies of each
source FBX and controller in Unity 2022.3.62f3, samples the imported mesh, bind
poses, and local transforms at every authored frame, and writes those values
directly to a compact standard GLB. These files need no morph data, custom
extension, decoder, or model-specific runtime path. Multipart prefabs and rigs
whose controller output differs from the baked recorder remain on the normal FBX
path.

```sh
npm run import:athena2 -- --meshes <Imports folder> --textures <Textures folder> --animations <Animations folder>
npm run textures
```

The generated `public/units/all-units.json` catalogs only units with all
three required animations that are enabled for publication. `--animations` can
be omitted when that folder is the sibling of `Imports`; its baked frame counts
and rates define each exported clip's exact span. Incomplete, explicitly
excluded, and stale generated files are pruned. The normal import preserves the
three already-shipped combat GLBs; add `--include-existing` when rebuilding all
54 into an empty output directory.
`--unity <editor>` and `--blender <executable>` override tool discovery.
`assets/textures/` is only a transient encoder staging directory and can be
removed after `npm run textures` completes.
Importing art does not invent a new simulation unit or its balance data;
playable types are still registered explicitly in `unitModels.ts`.

## Contributor notes

Everything below is what the code does not say for itself: the rules that keep
two simulations identical, the measurements behind the decisions, and the
mistakes that were expensive enough to write down so nobody makes them twice.

### The one rule that matters

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

#### No trigonometry

The simulation stores facing as a **normalised vector**, never an angle, and
turns via `vecRotateToward` (clamped lerp + renormalise). This is why the sim
needs no sin/cos at all, and therefore needs no lookup tables or CORDIC. Angles
are a rendering concern; convert at the boundary.

#### Fixed-point gotchas

- `fmul` splits the operands' magnitudes into 16-bit halves whose partial
  products are exact, then applies the sign. The obvious `(a * b) >> 16` is
  **wrong** — `int32 * int32` reaches 2^62, past the 2^53 where float64 stops
  representing integers exactly, and it silently disagrees with exact
  arithmetic on ~2.5% of operand pairs. It truncates toward zero rather than
  flooring, so that `fmul(-a, b) === -fmul(a, b)`; see "The simulation is
  rotation-equivariant" for why that matters.
- Squared distances overflow int32 at map scale, so use `vecLenSqRaw` /
  `sqRange`, which stay in exact float64 integer range and compare safely.

### Map generation

`src/sim/mapgen.ts` carves lanes out of a solid block. The one rule: **every
write opens a tile and its 180-degree opposite**, inside a single `open()`. That
is the entire symmetry guarantee, and it must stay that way — lanes can then
wobble, overlap, and be carved in any order and the halves still come out
bit-identical. The previous approach wrote each shape twice and trusted the
copies to agree; they diverged on 926 tiles the moment the corridor brush gained
a random wobble, because each copy drew its own wobble from the same stream.

Two layouts share the carver: `Lanes` (two starts, 128) and `Quarters` (four
starts, 152). Adding a map is authoring polylines, not writing a generator.

#### Start locations come in mirrored halves

`map.starts[i]` and `map.starts[i + n/2]` are exact 180-degree rotations of one
another, and `map.expansions` follows the same convention. Teams are then
_derived_ from it — the first half against the second — which is what makes team
fairness structural instead of something to re-check. Two consequences worth
holding onto:

- **Player `p`'s opposite number is always `p + n/2`**, on any layout. Anything
  that wants "the enemy" without scanning the pool should ask for that rather
  than assuming `1 - p`.
- **Allies within a half are not mirrors of each other**, and cannot be: only
  one symmetry is guaranteed and it is the one between the sides. On `Quarters`
  the two players on a team hold different ground; their _opponents_ hold the
  exact rotation of it.

Everything placed at setup is laid out from the canonical (first-half) site and
rotated for the other half — see `mirroredHalf`. One extra rule appears with
four corners: the authored mineral-line offsets run up and to the left, which
tucks the line into the corner for a base on the left of the map and shoves it
out toward the middle for one on the right. `Quarters` has a canonical base in
each, so the line is reflected about the base's own centre for the right-hand
one. About its _centre_, not its centre tile — the half-tile error is the same
one described under "Mirror fairness" below, and costs a whole tile of walking
on every trip for the rest of the match.

`map.elevation` is cosmetic and not checksummed — the simulation decides movement
from walkability alone — but it is still computed deterministically, because two
peers rendering visibly different terrain would look exactly like a desync.

### The simulation is rotation-equivariant

The map is symmetric under a 180-degree rotation, and so is everything that
runs on it: a match whose second-half commands are the exact rotations of the
first half's stays an exact mirror to the last tick — every entity has a twin
at the rotated position with identical state, and both banks are equal, on
every tick. `tests/mirror.test.ts` enforces it mechanically, the way
`determinism.test.ts` enforces reproducibility: on a harvest-only match, on a
scripted match that builds, trains, rallies, paths across the map, fights and
expands, on a 20,000-tick scripted-bot mirror, on the same match with the
seats swapped, and on the four-corner map. `npm run mirror:probe` runs the same
scenarios and names the first tick, entity and field that broke the mirror,
which is the tool for finding out why a change broke it.

It was not always so, and the size of the effect is worth remembering. Before
the fix seat 1 won 15 of 16 bot mirror matches; Hard in seat 0 lost to Normal
in seat 1 five times in eight, so the seat was worth more than a difficulty
step; and with no bot at all, exactly mirrored harvest orders gave seat 1 12%
more minerals by tick 3000. Four families of cause, and each had to be closed
in full — one survivor re-breaks the mirror at the first tick it fires:

- **Rounding that is not symmetric under negation.** `fmul` floored, so
  `fmul(-a, b)` came out one below `-fmul(a, b)` for nearly every operand pair
  and every mover drifted one unit per axis per tick; it truncates toward zero
  now. `toInt` floors, and a position exactly on a tile boundary — every
  building centre, every approach point beside one — landed in the tile
  _inside_ a box on one half and _outside_ it on the other.
  `GameMap.tileOfPosFor(x, y, flip)` floors in the player's canonical frame
  instead, and every tile lookup made on an owned entity's behalf goes through
  it.
- **Ties broken by slot index or tile index.** Row-major tile index inverts
  under the rotation, so "lowest index wins" was "top-left wins" for one half
  and "bottom-right wins" for the other: A*'s heap and its first-parent rule,
  the flow-field descent, and `nearestWalkable`, which returned the first tile
  met on a ring scanned from its top-left corner and seeded every attack-move
  on a building. Slot indices are lower for seat 0 at setup and scrambled by
  the shared free list after the first death, so every ascending-index
  tie-break and every slot-keyed phase differed between twins. Entities now
  carry a `serial` — the owner's creation ordinal, which twins share — and
  ties break on `(ownerCanonical, serial)`; tiles break on
  `map.canonicalIndex(tile, flip)`.
- **Searches that walk absolute map directions.** The formation spiral, the
  bot's build spiral and its open-tile searches all walked the map's axes, so
  the second seat's base was a _translated_ copy of the first's rather than a
  rotated one: it put all of its first eight structures, both turrets among
  them, on its enemy-facing side, where the first seat put four and both
  turrets behind. Every one of them now walks the player's canonical frame,
  and the flow field seeds every walkable tile on the ring around an
  unwalkable goal rather than picking one.
- **Order-dependent passes across the halves.** Units are visited in slot
  order, and a pass that read another unit's position while writing its own
  saw a half-updated world in an order the other side did not share: a unit
  closing on an enemy stepped, and its opposite number then measured against
  the moved position and stopped one step short. Movement reads other units'
  positions from a tick-start snapshot; separation sums every pair's push
  before applying any; path requests are served round-robin per owner and in
  serial order within one; production spawns in creation order; two workers
  finishing on one patch in the same tick share it evenly.

Two rules keep it that way, and `tests/mirror.test.ts` is what says when one
has been broken:

- **Nothing decides by slot index or raw tile index.** Break ties on `serial`
  (after `World.ownerCanonical` when the owners can differ) or on
  `canonicalIndex`; phase timers on `serial`; convert a position to a tile
  with `tileOfPosFor(…, world.flipOf(owner))`; walk a spiral or a ring in the
  frame `flipOf` names.
- **A pass that reads other entities while writing its own reads a
  snapshot.** Accumulate, then apply.

One consequence to keep in mind when writing tests: two identical bots on the
mirrored map make mirrored moves for the whole match and can only draw, by
mutual elimination or by timeout. A test that needs a match to _resolve_ has
to make the two seats different — `tests/match.test.ts` plays the scripted bot
against a copy that thinks every 20 ticks instead of 10, and checks that
swapping the seats swaps the winner and changes nothing else. The half-speed
copy is not the weaker one, whatever it looks like: it still lands on every
beat the army logic fires on and re-issues its orders half as often, and
measured across eight seeds it wins 8–0 from either seat with identical tick
counts; every 30 ticks loses 8–0, every 40 wins 5–3. The interval is a way to
make a _different_ bot, not a strength dial, and nothing in the lobby sets it.

### Teams

A match is described by one agreed `MatchConfig` — seed, layout, a team per
player slot, and which slots the AI plays. It is checksummed, so a lobby that
let two peers disagree shows up as a desync on the first comparison rather than
as an inexplicable divergence some seconds in.

**A 1v1 is the degenerate case, and that is load-bearing.** Team ids equal player
ids there, so `world.winner` still reads as a player, `teamsFor(Lanes)` is
`[0, 1]`, and the duel map plus its opening are what they were — verified
against the pre-teams build when teams landed, not assumed; the only change
since is the starting workers standing on tile centres.

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

`victorySystem` keys that sweep on _what a defeated player still owns_, not on
"newly defeated here". `Surrender` sets the flag itself, from a command that has
already executed by the time victory runs, so the obvious version emptied a
razed player's base and left a conceding player's standing — the one case the
button exists for. The deaths are queued into `world.events.deaths` and reaped
like any other, so a conceded base blows up instead of blinking away.

**A peer that leaves ends the match for everyone.** Lockstep cannot advance
past a turn nobody will send, so there is no "play on without them" to offer —
`onPeerTimeout` exists to say so. It was never wired, which meant a departure
presented as a permanent "Waiting for your ally…" with no explanation; the
knocked-out dialog's Leave button then made that a one-click route. Conceding a
match nobody else is playing ends it outright rather than handing the player a
spectator seat at a bot fight.

**Humans must occupy a contiguous prefix of the roster.** The transports number
their peers from zero, and every bot slot is dealt to one of those peers by
`hostOf` — round-robin over the humans in roster order, derived from the
checksummed config and never negotiated, for the same reason slot assignment
is (`slotFromPeerIds`). A bot in slot 0 with a human in slot 2 would leave a
slot nobody sends for and stall every peer forever. `coopMatch` puts the humans
in 0 and 1 and the AI in 2 and 3, so two tabs host one bot each and a solo
co-op hosts all three on the one peer. A hosted bot's commands ride the wire
like a human's, so a 2v2 costs slightly more than a 1v1 — a few hundred bytes a
turn per bot, inside the budget `tests/wire.test.ts` sizes.

Vision is shared across a team, in the renderer. It cannot live in the
simulation — two peers would immediately hold different state — but it is
derived from checksummed data, so every peer could compute every side's fog if
it wanted to.

#### The lobby handshake has to be waited for, not raced

`joinOnlineRoom` used to resolve the moment it saw the other peer join, and the
version check rode a message that arrived afterwards. That made the check
decorative: the match had already begun by the time a mismatch was discovered,
so all it could do was report a desync it existed to prevent. It now waits for
the peer's handshake, which also carries the _mode_ — an opaque string covering
everything the lobby chose. The transport never interprets it and only compares
it for equality, so map, roster and bot kind are all covered by one check that
cannot drift out of step with what it is guarding.

#### The online transport is tested against a fake room

`joinOnlineRoom` needs a browser, WebRTC and a relay, so for a long time it was
the one piece of the multiplayer path no test ran: every hosted-bot proof went
over `LocalNetwork`, which models latency, jitter and loss but not the
handshake, peer discovery, a peer leaving, or a third peer in the room. It
takes a `RoomProvider` now — the slice of a Trystero room the transport uses,
the default wrapping Trystero at the one cast boundary — and
`tests/helpers/fakeRoom.ts` is a switchboard on the same virtual clock
`LocalNetwork` runs on: symmetric discovery with each report on its own
jittered path (the race `slotFromPeerIds` exists for), latency, jitter, loss,
targeted against broadcast sends, a member leaving, more than two peers in a
room, and Trystero's chunk-and-reassemble-by-arrival above
`TRANSPORT_CHUNK_BYTES`. `tests/online.test.ts` runs the handshake and each of
its refusals, and whole co-op matches with hosted bots of both kinds, over it.
Loss in the fake _drops_ where the real channel retransmits, so it is harsher
than the wire; what it cannot model is the network itself — ICE, NAT, TURN,
SCTP — which is why a real link still gets a manual soak before a release.

Two things it found before a real link could:

- **Packets are origin-checked, and sent to one peer.** A room can hold more
  than two (a spare tab, an unreaped reload, two pairs on one code), and the
  runner's own guard can only check that the `player` a packet _claims_ hosts
  the slots it fills — the claim itself is whatever the sender wrote. With
  bots hosted, a stranger could fill a bot slot the other peer hosts, and
  first-write-wins would apply it. So the transport pins the sender to the
  peer the handshake named and sends only to it; the two-tab transport does
  the same by the envelope's `from`.
- **A settled peer does not greet a newcomer.** It used to greet everyone it
  met, which is right before settling and wrong after: the newcomer settled
  against it, started a match, and waited forever for packets that only ever
  went to the pair. Left ungreeted, it times out with an honest message.

### Rendering gotchas worth not rediscovering

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

### Authored unit models

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

#### Blending between clips

Each instance names two rows of the bone texture and a weight between them. One
mechanism, two jobs: neighbouring frames of one clip smooth the 30Hz bake up to
display rate, and frames of two clips cross-fade so a unit eases into its swing.
The lerp is componentwise on the matrices, which shortens a bone when the poses
differ by a large rotation — measured on the real geometry, 0.0% between
neighbouring frames.

No rig ships an idle clip and none is synthesised: standing still holds frame
zero of the run.

#### Pick the clip from events, not from state

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
the `.ktx2` under `public/units/`. The encoder is deterministic — re-encoding
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

### Verifying determinism

```sh
npm test                    # full suite, including the per-tick replay check
npm run determinism:cross   # same match under V8 and JavaScriptCore, diffed
npm run mirror:probe        # where a mirrored match first stops mirroring
```

`determinism:cross` is the strongest check we have and it is nearly free: Node
and Bun ship independent JavaScript engines, so agreement between them is real
evidence of portability rather than a self-consistency tautology. It ends with
a leg that hashes the neural bot's observation stream under both — the encoder
is trained under Bun and run in the browser, so it gets the same treatment as
the simulation (see "The neural bot sees what a human sees").

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
and fights on its own, so it reaches states no script will, and it is a pure
function of the world — run through `HeadlessMatch`, the same driver and input
delay the browser uses, it costs nothing to check under both engines.

**And a fourth that plays through an elimination.** None of the other three ever
eliminates anybody — four seeds of four-bot co-op run twelve thousand ticks
without one — so the code that runs when a player goes out had never executed
under a second engine, and it is the most id-sensitive in the simulation:
`strip` decides the order slots return to the free list, and the free list
decides every entity id issued afterwards. The leg concedes on a fixed tick
rather than waiting for an elimination that does not come.

**And a third leg on the four-player map.** Both of the others are a 1v1 on the
duel map, so neither ever reaches four players, the larger grid, team hostility,
or the bot's team-level decisions — a whole map and half a roster's worth of
arithmetic that the strongest check in the project would otherwise never run
under a second engine.

### Architecture

The directory layout is under "Layout" above; this is how the pieces meet.

The simulation advances at a fixed **20 ticks per second**. Rendering runs at
display rate and interpolates between the last two ticks; interpolation alpha is
clamped to [0,1] so a network stall freezes units rather than extrapolating them
through walls.

#### Input delay is per-peer, and each peer must be told what to do with it

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

#### Commands are the only thing on the wire

Player intent goes in, entity state never does. This is what keeps bandwidth
flat whether a battle has twenty units or two thousand. All validation —
affordability, ownership, placement legality — happens in
`src/sim/systems/orders.ts`, never in the UI. The UI may grey out a button, but
the simulation decides.

#### The data channel is unordered, and packets must stay under 16 KB

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
stay far under that because `MAX_SELECTION` is 24, a hosted bot is capped at
`HOSTED_COMMANDS_PER_TURN`, and a networked transport hosts at most
`MAX_HOSTED_PER_PEER` bot per peer — `hostingProblem`, the one roster check the
runner and `Game` share — but it is an invariant now, not a coincidence.
`tests/wire.test.ts` fails loudly if a change to the selection cap or the
packet shape breaks it, and the fake room reproduces the scramble itself.

Reordering itself is harmless for a second, independent reason: the receive path
is structurally order-free (turns keyed absolutely, first write wins). Measured,
an "assume packets arrive in order" bug is _invisible_ under pure reordering —
the redundancy covers it — and only shows up, weakly, once 25% loss removes the
covering copies.

#### Bots are players

A bot is something that plays a slot: it reads the world through one interface
— `Agent.act(world, player)`, once a tick, returning the same `Command`s a
human's UI produces — and a host, `AgentDriver`, hands those to the lockstep
runner, where they cross the wire and execute one input delay later, exactly
like a human's. There are two kinds, `BotKind.Scripted` (`src/ai/bot.ts`
behind `ScriptedAgent`) and `BotKind.Neural` (the learned bot, `src/ai/neural`),
and they are interchangeable: the roster names a kind, `createAgent` turns it
into an `Agent`, and nothing downstream can tell which it got. The simulation
runs no bot at all — `Simulation.step` applies what it is given, `config.bots`
is nothing to it but a checksummed roster, and `tests/sealed-sim.test.ts`
refuses any import of `src/ai` from `src/sim`.

The scripted bot used to run _inside_ `Simulation.step`, on every peer, at zero
bandwidth. That only works for a bot that is a pure function of the world; the
neural bot samples its actions, so two peers running it would disagree on its
first decision. Rather than keep two kinds of bot on two paths, every bot is
hosted. What that costs: a bot's commands cross the wire, they land one input
delay after they are decided (the same latency a human has, which is fairer to
the human than the instant orders the old bot got), and in online co-op each
bot reacts at its host's adaptive delay — a known asymmetry between the two
bots on a side, and the price of hosting. Online hosting is live for both
kinds: the lobby loads the neural model _before_ joining a room, exactly as it
does before two tabs connect, so a peer that cannot run it never leaves the
other waiting on a match that will not start. `HOSTED_COMMANDS_PER_TURN` is the
budget a hosted slot gets on the wire; `AgentDriver` paces a bot inside it and
queues the rest, and the runner's own check on `issue` is the last line rather
than a normal path. `tests/wire.test.ts` sizes the worst packet from it: a
human's beyond-human burst plus one hosted bot at full budget, in every
repeated turn, still fits one transport chunk with room to spare.

**Tests and probes run bots through `HeadlessMatch`** — the lockstep loop with
the lockstep taken out: the same driver, and a command issued after tick t
executing at the start of turn `ceil(t / TICKS_PER_TURN) + INPUT_DELAY_TURNS`,
which is the runner's own rule. `tests/headless.test.ts` pins that a headless
match and a solo lockstep match agree tick for tick. The scripted bot is still
a pure function of the world, so a headless match is reproducible and
engine-independent, and the determinism and mirror probes stay exactly as
strong; that is now a policy rather than a structural necessity, so the
sealed-sim scan covers `src/ai/{agent,bot,cadence,driver,headless,scripted}.ts`
to keep it honest.

**One scripted bot.** Easy, Normal and Hard were three tunings of it; Hard is
the one that survives, with its constants as the one `TUNING`, and the lobby's
AI row offers Scripted or Neural. The merge is pinned by `tests/agent.test.ts`
against fixtures recorded from the pre-merge build — every command Hard decided
on every think of two whole matches, replayed and asked again.

**Two bots on a side are not two bots.** Run naively, an AI team is much weaker
than one bot with twice the economy: each half picks its own target, commits on
its own count, and gets beaten one at a time by an army that never had to fight
both at once. So _when to commit_ and _what to hit_ are decided over the team's
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

Nothing the scripted bot is given is special: no bonus income, no extra units,
no cheating on fog — it reads the whole map, which a human cannot, and that is
the one asymmetry. `THINK_INTERVAL` is the same for every bot — every bot must
think on the same tick, or whoever thinks first gets a whole interval of head
start, which measurably decided a mirror matchup — and it is not a strength
dial either; see "The simulation is rotation-equivariant" for the measurement.

#### The neural bot sees what a human sees

`src/ai/neural` is the codec behind `NeuralAgent`: what the bot observes, what
it can say, and the masks that keep the two honest. `SPEC` (`spec.ts`) is the
contract with the training code — `npm run ml:spec` dumps it to
`ml/rtsml/spec.json`, `tests/spec.test.ts` fails when the two differ, and a
model file names the `version` it was trained against so a stale one is refused
rather than run. Three decisions shape the codec, each taken once and then
leaned on:

- **Fog plus memory, nothing else.** `Visibility` (`src/vision/visibility.ts`,
  the renderer's fog lifted out so the two cannot drift; `FogRenderer` owns one
  and delegates) says what is in view now. `EntityMemory` keeps what has been
  seen — a building until its ground is seen empty, a unit for
  `UNIT_MEMORY_TICKS` after it was last in view — and never consults
  `pool.isAlive` on a handle it cannot see, which `tests/memory.test.ts` proves
  with a `Proxy`. The two HUD leaks (selecting an enemy through the shroud, the
  panel printing an enemy's hp and queue) are deliberately not inherited. Build
  is the one place fog bites the _masks_: placement legality depends on
  occupancy, so a building may only be placed in a cell the side can see, or
  the mask itself would say what stands in the fog.
- **The canonical frame.** Everything is encoded in the player's own frame
  through `flipOf` — positions, tiles, cells, row order — so the encoding for
  seat `p + n/2` equals seat `p`'s byte for byte on a mirrored match, on both
  maps (`tests/observation.test.ts`). The seat is invisible to the model, and
  that test is what says when a feature has let it back in. Rows are ordered by
  `serial` and canonical tile distance, never by slot index, for the reasons
  under "The simulation is rotation-equivariant"; the mismatches found on the
  way were all boundary cases of that section — a remembered position converted
  with `tileOfPos` instead of `tileOfPosFor`, a patch ordered by its rotated
  corner instead of its centre.
- **The human vocabulary, one command at a time.** A decision is
  `[type, entityType, target, cell, sub, selection × 24]`: every command the
  UI can produce except Surrender, with the UI's own limits — 24 units, one
  building per Train, cancel slot zero, a worker and a top-left tile per Build.
  `decode` turns one into a `Command`, `encode` turns a command back into the
  decision that would have produced it (the imitation labels), and
  `computeMasks` says what is legal _now_, from the simulation's own rules.
  `tests/actions.test.ts` round-trips every command the scripted bot emits and
  checks that ten thousand decisions drawn uniformly from the masks are all
  accepted by the simulation. Every store goes through `Math.fround`, so Bun
  and the browser produce the same bytes, and `scripts/cross-engine.sh` hashes
  the stream under both engines.

The masks were the cost. A legality scan over every tile for every footprint
ran 2.4 ms per decision under Bun, more than the four ticks it decided for;
asking the summed-area table only for the cells in view, through a
per-map-and-frame `GridIndex`, made it 0.23 ms and the whole decision about
0.5 ms. `npm run ml:bench` prints decisions per second and is the number to
watch when touching the encoder.

#### Training runs the game, not a copy of it

`tools/ml/env.ts` is a `HeadlessMatch` stepped one decision — `DECISION_TICKS`
ticks — at a time, with a `policy` slot played by whatever Python decides, a
`scripted@k` slot by the scripted bot thinking every k ticks, a `teacher` slot
by the scripted bot at the student's own cadence with each released command
handed out as a label, and `idle`. Same driver, same input delay as the
browser, so training latency is deployment latency by construction, and the
arena, the bench and the Python environment are one code path.
`tools/ml/serve.ts` puts many of them behind stdin/stdout; the frame layout is
in `tools/ml/README.md` and `ml/rtsml/protocol.py` is the same thing in Python.
`ml/README.md` covers the rest — imitation, PPO with a league, export.

Two things the pipeline depends on that are easy to break from either side:

- **Sampling is inside the model graph, and its noise is an input.** Each head
  takes `argmax(masked logits / T + Gumbel)`; the selection head keeps every
  legal row whose logit plus a Logistic draw is positive, at most 24 and never
  none. The exported ONNX carries no random number generator, the browser fills
  the noise from `crypto.getRandomValues`, and parity between torch and
  onnxruntime is an exact comparison of integers rather than a tolerance. The
  noise layout is `NOISE_SEGMENTS` in `spec.ts`; change a head and both sides
  change with it.
- **The teacher sees everything, so a label is checked before it is taught.**
  The scripted bot reads the whole map and decides before the student's frame
  is built; `encode` against the student's frame plus `legalise` against its
  masks drops what the student could not have said — a Train the bank no longer
  covers, a Build in fog — as type −1. What remains is what a human watching
  the same screen could have done.

**The think interval is not a strength dial.** `scripted@10` is the bot the
game ships; measured over eight seeds from both seats, `@20` beats it 8–0 and
`@40` beats it 5–3 while `@30` loses 8–0 — the interval changes _when_ the bot
commits, and some cadences suit its strategy. The rungs are distinct
reproducible opponents, the league weights each by how often the learner still
loses to it, and `HALF_SPEED_THINK_INTERVAL` exists only so the seat-fairness
tests have two different bots and therefore a winner. Any gate worth stating is
against `@10`.

#### The model runs in a worker, and the match never waits for it

`src/ai/neural/runtime.ts` is the browser side of `NeuralAgent`: a Web
Worker (`worker.ts`) running ONNX Runtime Web on its WebAssembly backend, one
decision at a time. The main thread encodes, posts the observation with its
buffers transferred, and carries on; the answer comes back by id and is read
at the next tick, or times out after two seconds and is counted. A decision
that comes due while one is in flight is _skipped_, never queued, so a slow
machine slows the bot's reactions and nothing else — and model latency feeds
nothing but that, never `adaptDelay`. Three failures in a row put a banner on
the HUD; the runner keeps sending empty turns for the slot either way, so the
match neither stalls nor desyncs.

Things about the packaging that were arrived at the hard way:

- **One thread, and the binary's URL is handed over explicitly.** GitHub Pages
  sends no cross-origin-isolation headers, so `SharedArrayBuffer` is absent
  and the threaded build cannot run; `numThreads` is 1. The `.wasm` is
  imported as `?url` in `browser.ts`, so Vite hashes it and honours
  `BASE_PATH`, and the worker is told where it is rather than guessing a path
  relative to a bundle whose shape it does not know. `onnxruntime-web` stays
  out of the dependency pre-bundle and the worker is an ES module
  (`vite.config.ts`). The binary is 14 MB (3.6 MB gzipped) and is fetched
  only when a neural slot is about to be played: the lobby loads the model
  _before_ connecting, so a peer that cannot run it never leaves the other
  waiting on a match that will not start.
- **A model names the codec it was trained for.** `public/models/policy.json`
  carries `specVersion`, and `WorkerRuntime.load` refuses any other than
  `SPEC.version`: a model trained against another feature layout would run
  happily and play nonsense. The Neural chip is live only when the manifest
  is there — the lobby asks for it on load — and no model is committed yet;
  `rtsml-export` writes one, and it goes in beside the unit skins.
- **`vite preview` serves at the build's base.** The config used to derive
  the base from `command === 'build'` alone, so a preview served the built
  `index.html` at `/` while its scripts pointed at `/experiment-rts/`, and the
  page loaded to a blank screen with one 404 in the console.

Measured with an untrained export of the shipped architecture (1.5 M
parameters, 1.7 MB as int8 — parity with torch exact for fp32 and 48 of 50 for
int8, which is what quantisation costs on a random-weight model's tiny
margins): warm-up 174 ms, then about 13 ms per decision inside the worker.
`npm run smoke:neural` plays `?skip=neural` in headless Chromium and fails on
any console error, failed decision, silent slot, or more than 1% of decisions
skipped. On a machine with no GPU the page renders in software and starves
the simulation — this container reached 192 ticks in 20 s, and the same 13 ms
of inference took 500 ms to come back — so when the page runs well below 20
ticks a second the script says so and gates on the model's own time instead.

#### Movement has three movers, and they are easy to miss

Direct steering, flow fields and A* path-following each advance units, and A* —
which carries most individual orders — walks its waypoints in a loop of its own
rather than through `stepToward`. Acceleration added to the shared helper alone
changed nothing for it, and nothing failed to say so. `tests/accel.test.ts`
pins the ramp for that reason.

Units accelerate by a fraction of their top speed per tick and brake by
`v = sqrt(2*a*d)` against the distance still to run — not to the next waypoint,
or they stutter at every corner of a path. `Math.sqrt` is the one non-trivial
function the sim may use; see the sealed-core rules above.

#### Attack-move is a loop, and the missing half is the return

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

#### Damage is one number, and the panel shows it

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

#### Acquiring a target is also an instruction to walk to it

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

#### A building is a square, and units must approach its nearest face

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

#### The click target is the ring, and rally lives above the units check

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

#### Collision radius is the unit's size, everywhere

`radius` is the collision size, the weapon-reach margin (`attackRange + target
radius`) and what the selection ring is drawn from. Changing it therefore moves
combat balance as well as spacing — a bigger unit is a slightly easier target.
Flyers are the exception in one direction only: `collides` is false for them
because it also decides whether a thing occupies map tiles, so separation gives
them their own rule and matches air against air.

#### Nothing may be written to a position a unit cannot occupy

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

#### A slot index is not an identity

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

#### An order must name somewhere a unit can stand

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

#### Every unit must be able to stop

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

#### Spread the arrival, never the pathfinding goal

A group move gives each unit its own destination so an army stops arriving in a
heap. The obvious way to do it is a trap: grouped moves navigate by flow field
and the field is cached **per goal tile**, so a per-unit goal turns one Dijkstra
sweep into one per unit. Not a crash — just a silent collapse, measured at 5
seconds to over 150 for the test match. The shared destination steers the group;
the spread point only decides where each unit finally stands.

The layout is a square spiral rather than a ring because a ring needs sine and
cosine. Separation rounds the corners off it anyway.

### Performance notes

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
