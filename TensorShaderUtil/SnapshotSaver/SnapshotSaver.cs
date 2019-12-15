using System.IO;
using TensorShader;

namespace TensorShaderUtil.SnapshotSaver {
    /// <summary>スナップショット 読み込み/書き込み</summary>
    public abstract class SnapshotSaver {
        /// <summary>ストリーム読み込み</summary>
        public abstract Snapshot Load(Stream stream);

        /// <summary>ストリーム書き込み</summary>
        public abstract void Save(Stream stream, Snapshot snapshot);

        /// <summary>ファイル読み込み</summary>
        public Snapshot Load(string filepath) {
            Snapshot snapshot = null;

            using (FileStream stream = new FileStream(filepath, FileMode.Open)) {
                snapshot = Load(stream);
            }

            return snapshot;
        }

        /// <summary>ファイル書き込み</summary>
        public void Save(string filepath, Snapshot snapshot) {
            using (FileStream stream = new FileStream(filepath, FileMode.Create)) {
                Save(stream, snapshot);
            }
        }
    }
}
