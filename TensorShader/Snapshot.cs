using System;
using System.Collections.Generic;
using System.Linq;

namespace TensorShader {
    /// <summary>スナップショット</summary>
    public class Snapshot {
        private readonly Dictionary<string, NdimArray<float>> table;

        /// <summary>状態テーブル</summary>
        public IReadOnlyDictionary<string, NdimArray<float>> Table => table;

        /// <summary>コンストラクタ</summary>
        public Snapshot() {
            this.table = new Dictionary<string, NdimArray<float>>();
        }

        /// <summary>スナップショットにテンソルの状態を追加</summary>
        public void Append(string key, Tensor tensor) {
            if (table.ContainsKey(key)) {
                throw new ArgumentException(nameof(key));
            }

            table.Add(key, tensor);
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
                table[key] = tensor;
            }
            else {
                table.Add(key, tensor);
            }
        }

        /// <summary>スナップショットからテンソルを上書き</summary>
        public void Load(string key, Tensor tensor) {
            if (!table.ContainsKey(key)) {
                throw new KeyNotFoundException(nameof(key));
            }

            NdimArray<float> arr = table[key];

            if (tensor.Shape != arr.Shape) {
                throw new ArgumentException(ExceptionMessage.Shape(tensor.Shape, arr.Shape));
            }

            tensor.State = arr;
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

        /// <summary>キー名リスト</summary>
        public override string ToString() {
            return string.Join(", ", table.Select((item)=>item.Key));
        }
    }
}
