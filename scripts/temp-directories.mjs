import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

/** Own directories from allocation until all work using them has completed. */
export async function withTemporaryDirectories(work) {
  const roots = [];
  const create = async (prefix) => {
    const root = await mkdtemp(join(tmpdir(), prefix));
    roots.push(root);
    return root;
  };
  try {
    return await work(create);
  } finally {
    await Promise.all(roots.map((root) => rm(root, { recursive: true, force: true })));
  }
}
