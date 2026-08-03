# Experiment RTS

A 3D real-time strategy game that runs in the browser, with online multiplayer
and **no server to deploy**.

**▶ [Play it](https://xli2022.github.io/experiment-rts/)**

Gather minerals, build a base, train an army, and destroy every enemy structure.
Play against the AI, against someone in a second tab, or against a friend
anywhere — peer-to-peer over WebRTC.

```sh
npm install
npm run dev          # http://localhost:5173
```

Every push to `master` builds and publishes to GitHub Pages, gated on the test
suite — see [`.github/workflows/deploy.yml`](./.github/workflows/deploy.yml).

## Playing

| | |
|---|---|
| Pan | Arrow keys, screen edges, middle-drag, or the minimap |
| Zoom | mouse wheel |
| Select | click, or drag a box |
| Add to selection | `Shift` + click |
| Order | right-click — attack an enemy, mine a patch, or move |
| Attack-move | `A` then click |
| Stop / Hold | `S` / `H` |
| Control groups | `Ctrl`+`1`–`9` to assign, `1`–`9` to recall |
| Fullscreen | `F`, or the button top-right |
| Cancel | `Esc` |

Workers gather minerals and construct buildings. The Command Post trains
workers, Supply Depots raise the supply cap, Barracks train Riflemen and
Brawlers, and Turrets defend. You lose when your last structure falls.

## Multiplayer without a server

Three ways to play, all sharing one code path:

- **Skirmish vs AI** — the bot is a deterministic function of game state, so it
  runs inside the simulation on every machine. It costs no bandwidth and needs
  no host.
- **Two tabs** — a second tab on the same computer, over `BroadcastChannel`. No
  network at all.
- **Online** — share a room code. Peers find each other through the public
  [Nostr](https://nostr.com) relay network via
  [Trystero](https://github.com/dmotz/trystero), then talk **directly**,
  end-to-end encrypted. Nothing of ours sits in the middle.

### How it stays in sync

Peers never exchange game state. They exchange *commands* — "these units, move
here" — and each machine runs an identical simulation. A 200-unit battle costs
the same bandwidth as a single click. This is what StarCraft does, and it is why
peer-to-peer works at all for a genre with thousands of moving objects.

The price is that both simulations must agree **exactly**, forever. See
[CLAUDE.md](./CLAUDE.md) for the rules that guarantee it — fixed-point
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
```

`determinism:cross` is the strongest check here and it is close to free. Node
runs on V8 and Bun runs on JavaScriptCore — two independent JavaScript engines
with independently implemented maths. If they agree on every checkpoint of a
1,500-tick match, the simulation really is portable, and a Chrome player can
face a Safari player without drifting apart.

The test suite also plays complete AI-vs-AI matches to a victory condition
across several seeds. That layer exists because a simulation can be perfectly
deterministic and still not be a game: it caught armies that grew into the
hundreds without ever fighting, an AI that hoarded 9,000 unspendable minerals,
and workers that stopped one step short of the building they were sent to make.

### Layout

```
src/sim/      the deterministic simulation — no rendering, no DOM, no clock
src/net/      lockstep scheduler and the three transports
src/ai/       the bot, which is just another command source
src/render/   Three.js rendering; reads the simulation, never writes it
src/input/    selection and camera
src/ui/       DOM overlay: HUD, minimap, lobby
src/config/   all balance data, in one table
```

Units and buildings are built from Three.js primitives at runtime, behind a
`ModelProvider` interface — loaded assets can replace them later without
touching gameplay code.
