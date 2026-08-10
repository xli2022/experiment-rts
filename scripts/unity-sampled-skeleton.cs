// Invoked in a temporary Unity 2022 project by unity-sampled-skeleton.mjs.
// Unity is the authority for legacy FBX import, controller binding, and skin
// deformation. This script records the imported mesh, bind pose, hierarchy,
// and exact local transforms at every authored frame; Node then writes those
// values directly to glTF without passing animation data through FBX again.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

public static class RtsUnitySampledSkeleton
{
    [Serializable]
    private sealed class JobFile
    {
        public string prefabPath;
        public string controllerPath;
        public string outputPath;
        public List<ClipJob> clips = new List<ClipJob>();
    }

    [Serializable]
    private sealed class ClipJob
    {
        public string name;
        public string state;
        public int frames;
        public float frameRate;
    }

    [Serializable]
    private sealed class SampleFile
    {
        public string unityVersion;
        public List<NodeRecord> nodes = new List<NodeRecord>();
        public float[] restTransforms;
        public MeshRecord mesh;
        public List<ClipRecord> clips = new List<ClipRecord>();
    }

    [Serializable]
    private sealed class NodeRecord
    {
        public string name;
        public string path;
    }

    [Serializable]
    private sealed class ClipRecord
    {
        public string name;
        public int authoredFrames;
        public float frameRate;
        public float[] transforms;
    }

    [Serializable]
    private sealed class MeshRecord
    {
        public int rendererNode;
        public int[] boneNodes;
        public float[] vertices;
        public float[] normals;
        public float[] uv;
        public int[] triangles;
        public int[] boneIndices;
        public float[] boneWeights;
        public float[] bindPoses;
    }

    public static void Run()
    {
        try
        {
            var manifestPath = CommandLineValue("-rtsSampleManifest");
            if (string.IsNullOrEmpty(manifestPath) || !File.Exists(manifestPath))
                throw new FileNotFoundException("Unity sampled-skeleton manifest is absent", manifestPath);
            var job = JsonUtility.FromJson<JobFile>(File.ReadAllText(manifestPath));
            if (job == null || string.IsNullOrEmpty(job.prefabPath) ||
                string.IsNullOrEmpty(job.controllerPath) || string.IsNullOrEmpty(job.outputPath))
                throw new InvalidDataException("Unity sampled-skeleton manifest is malformed");
            if (job.clips == null || job.clips.Count == 0)
                throw new InvalidDataException("Unity sampled-skeleton manifest contains no clips");

            EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);
            var importer = AssetImporter.GetAtPath(job.prefabPath) as ModelImporter;
            if (importer == null) throw new FileNotFoundException("Geometry FBX importer is absent", job.prefabPath);
            if (!importer.isReadable)
            {
                importer.isReadable = true;
                importer.SaveAndReimport();
            }

            var prefab = AssetDatabase.LoadAssetAtPath<GameObject>(job.prefabPath);
            var controller = AssetDatabase.LoadAssetAtPath<RuntimeAnimatorController>(job.controllerPath);
            if (!prefab || !controller)
                throw new FileNotFoundException("Sampled-skeleton prefab or controller is absent");

            var root = UnityEngine.Object.Instantiate(prefab);
            root.name = "SampledSkeleton";
            root.transform.SetPositionAndRotation(Vector3.zero, Quaternion.identity);
            root.transform.localScale = Vector3.one;
            root.SetActive(true);
            // UnityEngine.Object has a fake-null state that C#'s ?? operator
            // does not observe. Some imported model prefabs expose that stale
            // Animator reference, so use Unity's overloaded truth test here.
            var animator = root.GetComponent<Animator>();
            if (!animator) animator = root.AddComponent<Animator>();
            animator.runtimeAnimatorController = controller;
            animator.applyRootMotion = false;
            animator.cullingMode = AnimatorCullingMode.AlwaysAnimate;
            animator.enabled = true;

            var transforms = root.GetComponentsInChildren<Transform>(true);
            var output = new SampleFile
            {
                unityVersion = Application.unityVersion,
                restTransforms = new float[transforms.Length * 10],
            };
            foreach (var transform in transforms)
            {
                output.nodes.Add(new NodeRecord
                {
                    name = transform.name,
                    path = AnimationUtility.CalculateTransformPath(transform, root.transform),
                });
            }
            for (var node = 0; node < transforms.Length; node++)
                WriteTransform(output.restTransforms, node * 10, transforms[node]);

            var renderers = root.GetComponentsInChildren<Renderer>(true)
                .Where(candidate => candidate is MeshRenderer || candidate is SkinnedMeshRenderer)
                .ToArray();
            if (renderers.Length != 1 || !(renderers[0] is SkinnedMeshRenderer renderer) || !renderer.sharedMesh)
                throw new InvalidDataException(
                    "Sampled-skeleton prefab must contain exactly one complete SkinnedMeshRenderer; found " +
                    string.Join(", ", renderers.Select(candidate =>
                        candidate.name + ":" + candidate.GetType().Name)));
            output.mesh = ReadMesh(renderer, transforms);

            foreach (var clip in job.clips)
            {
                if (string.IsNullOrEmpty(clip.name) || string.IsNullOrEmpty(clip.state) ||
                    clip.frames < 1 || !(clip.frameRate > 0))
                    throw new InvalidDataException("Sampled-skeleton clip timing is malformed");
                var record = new ClipRecord
                {
                    name = clip.name,
                    authoredFrames = clip.frames,
                    frameRate = clip.frameRate,
                    transforms = new float[(clip.frames + 1) * transforms.Length * 10],
                };
                animator.Rebind();
                animator.Play(clip.state, 0, 0f);
                animator.Update(0f);
                var state = animator.GetCurrentAnimatorStateInfo(0);
                if (state.shortNameHash != Animator.StringToHash(clip.state))
                    throw new InvalidDataException("Animator has no state named " + clip.state);
                for (var frame = 0; frame <= clip.frames; frame++)
                {
                    if (frame > 0) animator.Update(1f / clip.frameRate);
                    for (var node = 0; node < transforms.Length; node++)
                    {
                        var offset = (frame * transforms.Length + node) * 10;
                        WriteTransform(record.transforms, offset, transforms[node]);
                    }
                }
                output.clips.Add(record);
                Debug.LogFormat("SAMPLED {0}: {1} keys, {2} nodes", clip.name, clip.frames + 1, transforms.Length);
            }

            Directory.CreateDirectory(Path.GetDirectoryName(job.outputPath));
            File.WriteAllText(job.outputPath, JsonUtility.ToJson(output));
            Debug.Log("WROTE " + job.outputPath);
            EditorApplication.Exit(0);
        }
        catch (Exception exception)
        {
            Debug.LogException(exception);
            EditorApplication.Exit(1);
        }
    }

    private static MeshRecord ReadMesh(SkinnedMeshRenderer renderer, Transform[] transforms)
    {
        var source = renderer.sharedMesh;
        var vertices = source.vertices;
        var normals = source.normals;
        var uv = source.uv;
        var weights = source.boneWeights;
        var bindPoses = source.bindposes;
        if (normals.Length != vertices.Length || uv.Length != vertices.Length || weights.Length != vertices.Length)
            throw new InvalidDataException("Sampled-skeleton mesh attributes have inconsistent lengths");
        var record = new MeshRecord
        {
            rendererNode = Array.IndexOf(transforms, renderer.transform),
            boneNodes = renderer.bones.Select(bone => Array.IndexOf(transforms, bone)).ToArray(),
            vertices = new float[vertices.Length * 3],
            normals = new float[normals.Length * 3],
            uv = new float[uv.Length * 2],
            triangles = source.triangles,
            boneIndices = new int[weights.Length * 4],
            boneWeights = new float[weights.Length * 4],
            bindPoses = new float[bindPoses.Length * 16],
        };
        if (record.rendererNode < 0 || record.boneNodes.Any(node => node < 0) ||
            record.boneNodes.Length != bindPoses.Length)
            throw new InvalidDataException("Sampled-skeleton renderer has an invalid bone mapping");
        for (var index = 0; index < vertices.Length; index++)
        {
            record.vertices[index * 3 + 0] = vertices[index].x;
            record.vertices[index * 3 + 1] = vertices[index].y;
            record.vertices[index * 3 + 2] = vertices[index].z;
            record.normals[index * 3 + 0] = normals[index].x;
            record.normals[index * 3 + 1] = normals[index].y;
            record.normals[index * 3 + 2] = normals[index].z;
            record.uv[index * 2 + 0] = uv[index].x;
            record.uv[index * 2 + 1] = uv[index].y;
            record.boneIndices[index * 4 + 0] = weights[index].boneIndex0;
            record.boneIndices[index * 4 + 1] = weights[index].boneIndex1;
            record.boneIndices[index * 4 + 2] = weights[index].boneIndex2;
            record.boneIndices[index * 4 + 3] = weights[index].boneIndex3;
            record.boneWeights[index * 4 + 0] = weights[index].weight0;
            record.boneWeights[index * 4 + 1] = weights[index].weight1;
            record.boneWeights[index * 4 + 2] = weights[index].weight2;
            record.boneWeights[index * 4 + 3] = weights[index].weight3;
        }
        for (var matrix = 0; matrix < bindPoses.Length; matrix++)
            for (var column = 0; column < 4; column++)
                for (var row = 0; row < 4; row++)
                    record.bindPoses[matrix * 16 + column * 4 + row] = bindPoses[matrix][row, column];
        return record;
    }

    private static void WriteTransform(float[] values, int offset, Transform transform)
    {
        values[offset + 0] = transform.localPosition.x;
        values[offset + 1] = transform.localPosition.y;
        values[offset + 2] = transform.localPosition.z;
        values[offset + 3] = transform.localRotation.x;
        values[offset + 4] = transform.localRotation.y;
        values[offset + 5] = transform.localRotation.z;
        values[offset + 6] = transform.localRotation.w;
        values[offset + 7] = transform.localScale.x;
        values[offset + 8] = transform.localScale.y;
        values[offset + 9] = transform.localScale.z;
    }

    private static string CommandLineValue(string name)
    {
        var args = Environment.GetCommandLineArgs();
        for (var index = 0; index + 1 < args.Length; index++)
            if (args[index] == name) return args[index + 1];
        return null;
    }
}
