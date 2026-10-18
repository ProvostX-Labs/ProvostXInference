using System.Buffers.Binary;
using System.Text;
using ProvostXInference.Decision.Abstractions;
using ProvostXInference.Logging;

namespace ProvostXInference.Decision.Engine;

/// <summary>
/// Wave-2 ONNX header validation: a REAL protobuf scan of the ModelProto wire —
/// ir_version (field 1, varint), the graph message (field 7) with its input
/// (field 11) and output (field 12) ValueInfoProto names, and opset_import
/// presence (field 8). Long initializers are SKIPPED by seek (never read into
/// memory), so a 500MB+ model validates in milliseconds. Any structural
/// violation throws — the load state machine maps that to fail-closed
/// model_invalid before any session init is attempted.
/// Wire-format reference: developers.onnxruntime.ai/onnx/proto (ModelProto,
/// GraphProto, ValueInfoProto field tables). Pure scanner — no state.
/// </summary>
public sealed class DecisionHeaderReader : IDecisionHeaderReader
{
    private const int ModelFieldIrVersion = 1;
    private const int ModelFieldGraph = 7;
    private const int ModelFieldOpsetImport = 8;
    private const int GraphFieldInput = 11;
    private const int GraphFieldOutput = 12;
    private const int ValueInfoFieldName = 1;
    private const int MaxNameLength = 4096;

    /// <summary>Validate that the file at <paramref name="path"/> is a loadable ONNX
    /// model (readable ModelProto with a graph exposing ≥1 named input and ≥1 named
    /// output). Throws <see cref="InvalidDataException"/> when the header is invalid.</summary>
    public void Validate(string path)
    {
        using var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read);
        try
        {
            ValidateStream(stream, stream.Length);
        }
        catch (InvalidDataException) { throw; }
        catch (Exception ex)
        {
            throw new InvalidDataException(
                $"ONNX model '{Path.GetFileName(path)}' could not be parsed: {ex.Message}");
        }
    }

    /// <summary>Scan the ModelProto fields. Pure per call (stream consumed sequentially).</summary>
    private static void ValidateStream(Stream stream, long length)
    {
        if (length < 2)
            throw new InvalidDataException("ONNX model is too small to be a model file");

        long irVersion = -1;
        var sawGraph = false;
        var opsetCount = 0;
        var inputCount = 0;
        var outputCount = 0;

        while (stream.Position < length)
        {
            var tag = ReadVarint(stream);
            var field = (int)(tag >> 3);
            var wire = (int)(tag & 0x7);

            if (field == ModelFieldIrVersion && wire == 0)
            {
                irVersion = (long)ReadVarint(stream);
                continue;
            }
            if (field == ModelFieldGraph && wire == 2)
            {
                sawGraph = true;
                var graphLength = (long)ReadVarint(stream);
                var graphEnd = stream.Position + graphLength;
                if (graphEnd > length)
                    throw new InvalidDataException("ONNX model graph length exceeds the file size");
                (inputCount, outputCount) = ScanGraph(stream, graphEnd);
                if (inputCount == 0)
                    throw new InvalidDataException("ONNX model graph declares no inputs");
                if (outputCount == 0)
                    throw new InvalidDataException("ONNX model graph declares no outputs");
                // Continue scanning the model fields after the graph (metadata etc.).
                stream.Seek(graphEnd, SeekOrigin.Begin);
                continue;
            }
            if (field == ModelFieldOpsetImport && wire == 2)
            {
                opsetCount++;
                SkipField(stream, wire, length);
                continue;
            }

            SkipField(stream, wire, length);
        }

        if (irVersion < 0)
            throw new InvalidDataException("ONNX model header is missing ir_version (field 1)");
        if (!sawGraph)
            throw new InvalidDataException("ONNX model header is missing the graph (field 7)");
        if (opsetCount == 0)
            throw new InvalidDataException("ONNX model header is missing opset_import (field 8)");
    }

    /// <summary>Scan GraphProto fields for input/output ValueInfoProto names. Pure per call.</summary>
    private static (int Inputs, int Outputs) ScanGraph(Stream stream, long end)
    {
        var inputs = 0;
        var outputs = 0;
        while (stream.Position < end)
        {
            var tag = ReadVarint(stream);
            var field = (int)(tag >> 3);
            var wire = (int)(tag & 0x7);

            if ((field == GraphFieldInput || field == GraphFieldOutput) && wire == 2)
            {
                var valueInfoLength = (long)ReadVarint(stream);
                var valueInfoEnd = stream.Position + valueInfoLength;
                if (valueInfoEnd > end)
                    throw new InvalidDataException("ONNX model value_info length exceeds the graph bounds");
                var name = ReadValueInfoName(stream, valueInfoEnd);
                if (string.IsNullOrEmpty(name))
                    throw new InvalidDataException(
                        field == GraphFieldInput
                            ? "ONNX model graph input is missing its name"
                            : "ONNX model graph output is missing its name");
                if (field == GraphFieldInput) inputs++;
                else outputs++;
                stream.Seek(valueInfoEnd, SeekOrigin.Begin);
                continue;
            }

            SkipField(stream, wire, end);
        }
        return (inputs, outputs);
    }

    /// <summary>Read ValueInfoProto field 1 (name, wire type 2); skips everything else. Pure per call.</summary>
    private static string ReadValueInfoName(Stream stream, long end)
    {
        var name = string.Empty;
        while (stream.Position < end)
        {
            var tag = ReadVarint(stream);
            var field = (int)(tag >> 3);
            var wire = (int)(tag & 0x7);
            if (field == ValueInfoFieldName && wire == 2)
            {
                var valueLength = (long)ReadVarint(stream);
                if (valueLength > MaxNameLength)
                    throw new InvalidDataException("ONNX model graph IO name is implausibly long");
                var buffer = new byte[valueLength];
                ReadExactly(stream, buffer);
                name = Encoding.UTF8.GetString(buffer);
                continue;
            }
            SkipField(stream, wire, end);
        }
        return name;
    }

    // ── protobuf wire primitives (pure per call) ────────────────────────

    private static ulong ReadVarint(Stream stream)
    {
        ulong result = 0;
        var shift = 0;
        for (var i = 0; i < 10; i++)
        {
            var b = stream.ReadByte();
            if (b < 0)
                throw new InvalidDataException("ONNX model ended inside a varint");
            result |= (ulong)(b & 0x7F) << shift;
            if ((b & 0x80) == 0) return result;
            shift += 7;
        }
        throw new InvalidDataException("ONNX model varint exceeds 10 bytes");
    }

    private static void SkipField(Stream stream, int wire, long end)
    {
        switch (wire)
        {
            case 0: // varint
                ReadVarint(stream);
                break;
            case 1: // 64-bit
                Require(stream, end, 8);
                stream.Seek(8, SeekOrigin.Current);
                break;
            case 2: // length-delimited
            {
                var length = (long)ReadVarint(stream);
                if (stream.Position + length > end)
                    throw new InvalidDataException("ONNX model length-delimited field exceeds its message bounds");
                stream.Seek(length, SeekOrigin.Current); // initializers/nodes skip by SEEK, never read
                break;
            }
            case 5: // 32-bit
                Require(stream, end, 4);
                stream.Seek(4, SeekOrigin.Current);
                break;
            default:
                throw new InvalidDataException($"ONNX model uses unsupported wire type {wire}");
        }
    }

    private static void Require(Stream stream, long end, long bytes)
    {
        if (stream.Position + bytes > end)
            throw new InvalidDataException("ONNX model fixed-width field exceeds its message bounds");
    }

    private static void ReadExactly(Stream stream, byte[] buffer)
    {
        var read = 0;
        while (read < buffer.Length)
        {
            var n = stream.Read(buffer, read, buffer.Length - read);
            if (n <= 0)
                throw new InvalidDataException("ONNX model ended inside a field value");
            read += n;
        }
    }
}