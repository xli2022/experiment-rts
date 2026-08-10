"""Repair FBX animation curve tables that Three's loader cannot parse.

Arguments after `--` contain a JSON array of `{source, destination}` pairs.
Geometry from these files is subsequently joined by the separate geometry pass;
the authored take duration is reapplied by the JavaScript manifest.
"""

import json
import os
import sys

import bpy


def main():
    try:
        separator = sys.argv.index("--")
        manifest_path = sys.argv[separator + 1]
    except (ValueError, IndexError) as error:
        raise RuntimeError("expected a normalization manifest after --") from error

    with open(manifest_path, "r", encoding="utf-8") as manifest_file:
        jobs = json.load(manifest_file)

    for index, job in enumerate(jobs, start=1):
        source = os.path.abspath(job["source"])
        destination = os.path.abspath(job["destination"])
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        print(f"[FBX {index}/{len(jobs)}] {os.path.basename(source)}")

        bpy.ops.wm.read_factory_settings(use_empty=True)
        result = bpy.ops.wm.fbx_import(filepath=source)
        if "FINISHED" not in result:
            raise RuntimeError(f"Blender could not import {source}: {result}")
        result = bpy.ops.export_scene.fbx(
            filepath=destination,
            use_selection=False,
            add_leaf_bones=False,
            bake_anim=True,
            bake_anim_use_all_actions=True,
            bake_anim_use_nla_strips=False,
            bake_anim_simplify_factor=0,
            path_mode="STRIP",
        )
        if "FINISHED" not in result:
            raise RuntimeError(f"Blender could not export {destination}: {result}")


if __name__ == "__main__":
    main()
