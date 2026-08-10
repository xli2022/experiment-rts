"""Join all deforming pieces of an Athena2 source model into one skinned mesh.

Three's loader returns one SkinnedMesh per FBX skin cluster. Many Athena2 units
put their weapon, mount, or each body piece in a separate cluster, while the RTS
renderer intentionally consumes one geometry per unit type. Blender resolves
pieces per armature before joining, avoiding duplicate joints and preserving
every renderable part.

This script is invoked by `import-athena2-models.mjs`, not directly. The one
argument after `--` is a JSON array of `{source, destination}` pairs.
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

        deforming = []
        armatures = set()
        for obj in bpy.context.scene.objects:
            if obj.type != "MESH":
                continue
            modifiers = [modifier for modifier in obj.modifiers if modifier.type == "ARMATURE"]
            if not modifiers:
                continue
            deforming.append(obj)
            armatures.update(modifier.object for modifier in modifiers if modifier.object)

        if not deforming:
            raise RuntimeError(f"{source} contains no armature-deformed mesh")
        # Most units have one armature. A few old rigs carry a separately
        # skinned prop armature; join each armature's pieces independently and
        # let the JS pass unify the remaining skins by bind matrix.
        for armature in armatures:
            group = [
                obj
                for obj in deforming
                if any(
                    modifier.type == "ARMATURE" and modifier.object == armature
                    for modifier in obj.modifiers
                )
            ]
            bpy.ops.object.select_all(action="DESELECT")
            for obj in group:
                obj.select_set(True)
            active = max(group, key=lambda obj: len(obj.data.vertices))
            bpy.context.view_layer.objects.active = active
            if len(group) > 1:
                result = bpy.ops.object.join()
                if "FINISHED" not in result:
                    raise RuntimeError(f"Blender could not join {source}: {result}")

            # Team color comes from the external KTX2 skin. Flattening source
            # material slots avoids manufacturing extra glTF primitives.
            active.data.materials.clear()
            active.data.materials.append(bpy.data.materials.new("UnitMaterial"))
            for polygon in active.data.polygons:
                polygon.material_index = 0

        # Animation is deliberately omitted. The JS importer attaches the
        # untouched clips from Athena2, preserving their exact take spans.
        result = bpy.ops.export_scene.fbx(
            filepath=destination,
            use_selection=False,
            add_leaf_bones=False,
            bake_anim=False,
            path_mode="STRIP",
        )
        if "FINISHED" not in result:
            raise RuntimeError(f"Blender could not export {destination}: {result}")


if __name__ == "__main__":
    main()
