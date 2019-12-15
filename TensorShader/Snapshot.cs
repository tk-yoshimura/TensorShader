using System;
using System.Collections.Generic;

namespace TensorShader {
    /// <summary>スナップショット</summary>
    public class Snapshot {
        private readonly Dictionary<string, (Shape shape, float[] state)> table;

        /// <summary>状態テーブル</summary>
        public IReadOnlyDictionary<string, (Shape shape, float[] state)> Table => table;

        /// <summary>コンストラクタ</summary>
        public Snapshot() {
            this.table = new Dictionary<string, (Shape shape, float[] state)>();
        }

        /// <summary>スナップショットにテンソルの状態を追加</summary>
        public void Append(string key, Tensor tensor) {
            if (table.ContainsKey(key)) {
                throw new ArgumentException(nameof(key));
            }

            table.Add(key, (tensor.Shape, tensor.State));
        }

        /// <summary>スナップショットにfloat配列を追加</summary>
        public void Append(string key, Shape shape, float[] state) {
            if (table.ContainsKey(key)) {
                throw new ArgumentException(nameof(key));
            }

            table.Add(key, (shape, (float[])state.Clone()));
        }

        /// <summary>スナップショットにテンソルの状態を保存</summary>
        public void Save(string key, Tensor tensor) {
            if (table.ContainsKey(key)) {
                table[key] = (tensor.Shape, tensor.State);
            }
            else {
                table.Add(key, (tensor.Shape, tensor.State));
            }
        }

        /// <summary>スナップショットからテンソルを上書き</summary>
        public void Load(string key, Tensor tensor) {
            if (!table.ContainsKey(key)) {
                throw new KeyNotFoundException(nameof(key));
            }

            (Shape shape, float[] state) = table[key];

            if (tensor.Shape != shape) {
                throw new ArgumentException(ExceptionMessage.Shape(tensor.Shape, shape));
            }

            tensor.State = state;
        }

        /// <summary>キーを列挙</summary>
        public IEnumerable<string> Keys => table.Keys;

        /// <summary>キーの有無</summary>
        public bool ContainsKey(string key) => table.ContainsKey(key);

        /// <summary>要素数</summary>
        public int Count => table.Count;

        /// <summary>クリア</summary>
        public void Clear() {
            table.Clear();
        }
    }
}
