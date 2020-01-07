using System;
using System.IO;
using System.Linq;
using TensorShader;

namespace TensorShaderUtil.SnapshotSaver {
    /// <summary>標準バイナリファイル入出力</summary>
    public class DefaultBinaryShapshotSaver : SnapshotSaver {
        /// <summary>CRCエントリ</summary>
        protected static char[] crc_entry = "crc\0".ToCharArray();

        /// <summary>識別子</summary>
        protected static char[] signature = "TensorShaderShapshot\0".ToCharArray();

        /// <summary>ファイル読み込み</summary>
        public override Snapshot Load(Stream stream) {
            Snapshot snapshot = new Snapshot();

            byte[] data = null;

            using (var memory_stream = new MemoryStream()) {
                stream.CopyTo(memory_stream);

                data = memory_stream.ToArray();
            }

            if (data.Length < signature.Length + crc_entry.Length + 4) {
                throw new InvalidDataException("Invalid file length.");
            }

            UInt32 crc_expect = Crc32.ComputeHash(data, 0, data.LongLength - 4);
            UInt32 crc_actual = ((uint)data[data.Length - 1] << 24) | ((uint)data[data.Length - 2] << 16) | ((uint)data[data.Length - 3] << 8) | data[data.Length - 4];

            if (crc_expect != crc_actual) {
                throw new InvalidDataException($"Mismatch crc32 hash. expect:{crc_expect:x8}, actual:{crc_actual:x8}");
            }

            using (var memory_stream = new MemoryStream(data)) {
                using (var reader = new BinaryReader(memory_stream)) {
                    if (!reader.ReadChars(signature.Length).SequenceEqual(signature)) {
                        throw new InvalidDataException("Mismatch file signature.");
                    }

                    int count = reader.ReadInt32();
                    if (count < 0) {
                        throw new InvalidDataException("Invalid count.");
                    }

                    for (int i = 0; i < count; i++) {
                        string key = reader.ReadString();
                        Shape shape = ReadShape(reader);
                        float[] value = ReadValue(reader, shape.Length);

                        snapshot.Append(key, shape, value);
                    }

                    if (!reader.ReadChars(crc_entry.Length).SequenceEqual(crc_entry)) {
                        throw new InvalidDataException("Mismatch file signature.");
                    }
                }
            }

            return snapshot;
        }

        /// <summary>ファイル書き込み</summary>
        public override void Save(Stream stream, Snapshot snapshot) {
            byte[] data = null;

            using (var memory_stream = new MemoryStream()) {
                using (var writer = new BinaryWriter(memory_stream)) {
                    writer.Write(signature);
                    writer.Write(snapshot.Count);

                    foreach (var item in snapshot.Table) {
                        writer.Write(item.Key);
                        WriteShape(writer, item.Value.shape);
                        WriteValue(writer, item.Value.state);
                    }

                    writer.Write(crc_entry);
                }

                data = memory_stream.ToArray();
            }

            UInt32 crc = Crc32.ComputeHash(data, 0, data.Length);

            using (var writer = new BinaryWriter(stream)) {
                writer.Write(data);
                writer.Write(crc);
            }
        }

        /// <summary>形状読み込み</summary>
        protected static Shape ReadShape(BinaryReader reader) {
            if (reader.ReadString() != "shape") {
                throw new InvalidDataException("Not found shape header.");
            }

            ShapeType type = (ShapeType)Enum.ToObject(typeof(ShapeType), reader.ReadInt16());
            if (!Enum.IsDefined(typeof(ShapeType), type)) {
                throw new InvalidDataException("Invalid shape type.");
            }

            int ndim = reader.ReadInt32();
            if (ndim < 0) {
                throw new InvalidDataException("Invalid ndim.");
            }

            int[] s = new int[ndim];

            for (int i = 0; i < s.Length; i++) {
                s[i] = reader.ReadInt32();

                if (s[i] < 1) {
                    throw new InvalidDataException($"Invalid {i}th-axis.");
                }
            }

            return new Shape(type, s);
        }

        /// <summary>形状書き込み</summary>
        protected static void WriteShape(BinaryWriter writer, Shape shape) {
            writer.Write("shape");
            writer.Write((Int16)shape.Type);
            writer.Write(shape.Ndim);
            foreach (int length in (int[])shape) {
                writer.Write(length);
            }
        }

        /// <summary>float配列読み込み</summary>
        protected static float[] ReadValue(BinaryReader reader, long length) {
            if (reader.ReadString() != "value") {
                throw new InvalidDataException("Not found value header.");
            }

            if (reader.ReadInt64() != length) {
                throw new InvalidDataException("Invalid length.");
            }

            float[] val = new float[length];

            for (int i = 0; i < val.Length; i++) {
                val[i] = reader.ReadSingle();
            }

            return val;
        }

        /// <summary>float配列書き込み</summary>
        protected static void WriteValue(BinaryWriter writer, float[] value) {
            writer.Write("value");

            writer.Write(value.LongLength);

            foreach (float v in value) {
                writer.Write(v);
            }
        }
    }
}
