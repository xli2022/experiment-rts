// Invoked in a temporary Unity project by import-athena2-models.mjs.
//
// Athena2 contains a handful of FBX 6.1 files (and two newer files with curve
// tables Three cannot parse). Autodesk's FBX SDK can read those files and write
// their scene data back as FBX 7.4 without sampling or changing clip spans.

using System;
using System.IO;
using Autodesk.Fbx;
using UnityEditor;
using UnityEngine;

public static class RtsFbxNormalizer
{
    public static void Run()
    {
        try
        {
            var manifest = Argument("-rtsFbxManifest");
            var lines = File.ReadAllLines(manifest);
            for (var index = 0; index < lines.Length; index++)
            {
                var fields = lines[index].Split('\t');
                if (fields.Length != 2)
                    throw new InvalidDataException("Invalid FBX job: " + lines[index]);

                Debug.LogFormat("[FBX {0}/{1}] {2}", index + 1, lines.Length,
                    Path.GetFileName(fields[0]));
                RoundTrip(fields[0], fields[1]);
            }
            EditorApplication.Exit(0);
        }
        catch (Exception exception)
        {
            Debug.LogException(exception);
            EditorApplication.Exit(1);
        }
    }

    private static void RoundTrip(string source, string destination)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(destination));
        using (var manager = FbxManager.Create())
        {
            var settings = FbxIOSettings.Create(manager, Globals.IOSROOT);
            manager.SetIOSettings(settings);

            using (var scene = FbxScene.Create(manager, "Athena2"))
            using (var importer = FbxImporter.Create(manager, "Importer"))
            {
                if (!importer.Initialize(source, -1, settings) || !importer.Import(scene))
                    throw new InvalidOperationException(
                        "Could not read " + source + ": " + importer.GetStatus().GetErrorString());

                using (var exporter = FbxExporter.Create(manager, "Exporter"))
                {
                    if (!exporter.Initialize(destination, -1, settings))
                        throw new InvalidOperationException(
                            "Could not open " + destination + ": " + exporter.GetStatus().GetErrorString());
                    exporter.SetFileExportVersion("FBX201400");
                    if (!exporter.Export(scene))
                        throw new InvalidOperationException(
                            "Could not write " + destination + ": " + exporter.GetStatus().GetErrorString());
                }
            }
        }
    }

    private static string Argument(string name)
    {
        var args = Environment.GetCommandLineArgs();
        for (var index = 0; index + 1 < args.Length; index++)
            if (args[index] == name)
                return args[index + 1];
        throw new ArgumentException("Missing " + name);
    }
}
