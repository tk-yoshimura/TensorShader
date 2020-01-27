using System;

namespace TensorShader {

    /// <summary>多次元配列</summary>
    public class NdimArray<T> {

        /// <summary>値配列</summary>
        public T[] Value { private set; get; }

        /// <summary>形状</summary>
        public Shape Shape { private set; get; }

        /// <summary>長さ</summary>
        public int Length => Shape.Length;

        /// <summary>次元数</summary>
        public int Ndim => Shape.Ndim;

        /// <summary>タイプ</summary>
        public ShapeType Type => Shape.Type;

        /// <summary>コンストラクタ</summary>
        public NdimArray(T[] value, Shape shape, bool clone_value = false) {
            if (value == null) {
                throw new ArgumentNullException(nameof(value));
            }
            if (shape == null) {
                throw new ArgumentNullException(nameof(shape));
            }

            if (value.Length != shape.Length) {
                throw new ArgumentException(ExceptionMessage.Argument($"{value}.Length", value.Length, shape.Length));
            }

            this.Value = clone_value ? (T[])value.Clone() : value;
            this.Shape = shape;
        }

        /// <summary>コンストラクタ</summary>
        public NdimArray(Shape shape) {
            if (shape == null) {
                throw new ArgumentNullException(nameof(shape));
            }

            this.Value = new T[shape.Length];
            this.Shape = shape;
        }

        /// <summary>一次元配列へ変換</summary>
        public static explicit operator T[](NdimArray<T> array) {
            return array.Value;
        }

        /// <summary>インデクサ</summary>
        public T this[params int[] indexes] {
            get {
                return Value[Index(indexes)];
            }
            set {
                Value[Index(indexes)] = value;
            }
        }

        /// <summary>文字列化</summary>
        public override string ToString() {
            return Shape.ToString();
        }

        private long Index(params int[] indexes) {
            if (indexes.Length != Ndim) {
                throw new ArgumentException(ExceptionMessage.Argument($"{indexes}.Length", indexes.Length, Ndim));
            }

            long pos = 0;
            for (int dim = Ndim - 1; dim >= 0; dim--) {
                int index = indexes[dim];
                if (index < 0 || index >= Shape[dim]) {
                    throw new ArgumentOutOfRangeException(nameof(indexes));
                }
                pos *= Shape[dim];
                pos += index;
            }

            return pos;
        }
    }

    /// <summary>テンソルクラス</summary>
    public partial class Tensor {
        /// <summary>テンソルから変換</summary>
        public static implicit operator NdimArray<float>(Tensor tensor) {
            return new NdimArray<float>(tensor.State, tensor.Shape);
        }

        /// <summary>テンソルへ変換</summary>
        public static implicit operator Tensor(NdimArray<float> array) {
            return new Tensor(array.Shape, array.Value);
        }
    }
}
