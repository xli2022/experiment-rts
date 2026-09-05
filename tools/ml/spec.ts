/**
 * Prints the codec spec as JSON.
 *
 *   npm run ml:spec > ml/rtsml/spec.json
 *
 * Python builds its model from that file, and `tests/spec.test.ts` fails when
 * the committed copy no longer matches `SPEC` — the one way the two sides can
 * drift is through a stale file, so the file is checked, not trusted.
 */

import { SPEC } from '../../src/ai/neural/spec.js';

console.log(JSON.stringify(SPEC, null, 2));
