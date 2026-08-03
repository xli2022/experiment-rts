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

### Commands are the only thing on the wire

Player intent goes in, entity state never does. This is what keeps bandwidth
flat whether a battle has twenty units or two thousand. All validation —
affordability, ownership, placement legality — happens in
`src/sim/systems/orders.ts`, never in the UI. The UI may grey out a button, but
the simulation decides.

### The AI is not special

`src/ai/bot.ts` emits the same `Command` objects a human does, and because it is
deterministic it runs identically on every peer at zero bandwidth. Single-player
and multiplayer are therefore one code path, not two.

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
