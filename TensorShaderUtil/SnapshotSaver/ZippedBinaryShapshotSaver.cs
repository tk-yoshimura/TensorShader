using System.IO.Compression;
using System.IO;
using TensorShader;

namespace TensorShaderUtil.SnapshotSaver {
    /// <summary>圧縮バイナリファイル入出力</summary>
    public class ZippedBinaryShapshotSaver : DefaultBinaryShapshotSaver {
        /// <summary>ファイル読み込み</summary>
        public override Snapshot Load(Stream stream) {
            Snapshot snapshot = null;

            using (DeflateStream deflate_stream = new DeflateStream(stream, CompressionMode.Decompress)) {
                snapshot = base.Load(deflate_stream);
            }

            return snapshot;
        }

        /// <summary>ファイル書き込み</summary>
        public override void Save(Stream stream, Snapshot snapshot) {
            using (DeflateStream deflate_stream = new DeflateStream(stream, CompressionMode.Compress)) {
                base.Save(deflate_stream, snapshot);
            }
        }
    }
}
