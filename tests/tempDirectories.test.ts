import { access, mkdir, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';
// @ts-expect-error Node build scripts do not have TypeScript declarations.
import { withTemporaryDirectories } from '../scripts/temp-directories.mjs';

type CreateDirectory = (prefix: string) => Promise<string>;

describe('import temporary directory ownership', () => {
  it('retains outputs for the complete import and cleans them after success', async () => {
    let root = '';
    const result = await withTemporaryDirectories(async (create: CreateDirectory) => {
      root = await create('rts-import-test-');
      await writeFile(join(root, 'model.fbx'), 'converted');
      await expect(access(join(root, 'model.fbx'))).resolves.toBeUndefined();
      return 'imported';
    });
    expect(result).toBe('imported');
    await expect(access(root)).rejects.toMatchObject({ code: 'ENOENT' });
  });

  it('cleans all allocated directories when a conversion fails before returning its paths', async () => {
    const roots: string[] = [];
    const failure = new Error('FBX conversion failed');
    await expect(
      withTemporaryDirectories(async (create: CreateDirectory) => {
        roots.push(await create('rts-normalize-test-'));
        await writeFile(join(roots[0]!, 'normalized.fbx'), 'converted');
        roots.push(await create('rts-join-test-'));
        await mkdir(join(roots[1]!, 'output'));
        await writeFile(join(roots[1]!, 'output', 'partial.fbx'), 'partial');
        throw failure;
      }),
    ).rejects.toBe(failure);
    expect(roots).toHaveLength(2);
    for (const root of roots) {
      await expect(access(root)).rejects.toMatchObject({ code: 'ENOENT' });
    }
  });
});
