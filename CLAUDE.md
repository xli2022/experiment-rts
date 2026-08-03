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
