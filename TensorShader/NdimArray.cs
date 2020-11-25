using System;

namespace TensorShader {

    /// <summary>多次元配列</summary>
    public partial class NdimArray<T> : ICloneable where T : struct, IComparable {

        /// <summary>値配列</summary>
        public T[] Value { private set; get; }

        /// <summary>形状</summary>
        public Shape Shape { private set; get; }

        /// <summary>形状タイプ</summary>
        public ShapeType Type => Shape.Type;

        /// <summary>要素数</summary>
        public int Length => Shape.Length;

        /// <summary>次元数</summary>
        public int Ndim => Shape.Ndim;

        /// <summary>幅</summary>
        public int Width => Shape.Width;

        /// <summary>高さ</summary>
        public int Height => Shape.Height;

        /// <summary>奥行き</summary>
        public int Depth => Shape.Depth;

        /// <summary>チャネル数</summary>
        public int Channels => Shape.Channels;

        /// <summary>入力チャネル数</summary>
        public int InChannels => Shape.InChannels;

        /// <summary>出力チャネル数</summary>
        public int OutChannels => Shape.OutChannels;

        /// <summary>バッチ数</summary>
        public int Batch => Shape.Batch;

        /// <summary>コンストラクタ</summary>
        public NdimArray(Shape shape, T[] value, bool clone_value = false) {
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

        /// <summary>形状変更</summary>
        public NdimArray<T> Reshape(Shape shape, bool clone_value = false) {
            if (shape.Length != Shape.Length) {
                throw new ArgumentException(nameof(shape));
            }

            return new NdimArray<T>(shape, Value, clone_value);
        }

        /// <summary>連結</summary>
        /// <remarks>指定軸方向の要素数が1である必要がある</remarks>
        public static NdimArray<T> Join(int axis, params NdimArray<T>[] arrays) {
            if (arrays == null || arrays.Length < 1) {
                throw new ArgumentException(nameof(arrays));
            }
            
            Shape shape = arrays[0].Shape;

            foreach (NdimArray<T> array in arrays) {
                if (array.Shape[axis] != 1 || array.Shape != shape) {
                    throw new ArgumentException(nameof(arrays));
                }
            }

            int stride = 1, size = shape.Length, n = arrays.Length;
            for (int i = 0; i < axis; i++) {
                stride *= shape[i];
            }

            int[] s = shape;
            s[axis] = n;

            NdimArray<T> array_stacked = new Shape(shape.Type, s);
            T[] dst = array_stacked.Value;

            for (int i = 0; i < arrays.Length; i++) {
                T[] src = arrays[i].Value;

                for (int j = 0, k = i * stride; j < size; j += stride, k += n * stride) {
                    Array.Copy(src, j, dst, k, stride);
                }
            }

            return array_stacked;
        }

        /// <summary>ディープコピー</summary>
        public object Clone() {
            return Copy();
        }

        /// <summary>ディープコピー</summary>
        public NdimArray<T> Copy() {
            return new NdimArray<T>(Shape, Value, clone_value: true);
        }
    }
}
