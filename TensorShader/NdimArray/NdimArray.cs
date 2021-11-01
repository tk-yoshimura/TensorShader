using System;
using System.Collections.Generic;
using System.Linq;

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
        public NdimArray(Shape shape, T[] value, bool clone_value = true) {
            if (value is null) {
                throw new ArgumentNullException(nameof(value));
            }
            if (shape is null) {
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
            if (shape is null) {
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

        private long Index(params int[] indexes) {
            if (indexes.Length != Ndim) {
                throw new ArgumentException(ExceptionMessage.Argument($"{nameof(indexes)}.Length", indexes.Length, Ndim));
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
        public NdimArray<T> Reshape(Shape shape, bool clone_value = true) {
            if (shape.Length != Shape.Length) {
                throw new ArgumentException(null, nameof(shape));
            }

            return new NdimArray<T>(shape, Value, clone_value);
        }

        /// <summary>連結</summary>
        /// <remarks>指定軸方向の要素数が1である必要がある</remarks>
        public static NdimArray<T> Join(int axis, params NdimArray<T>[] arrays) {
            if (arrays is null || arrays.Length < 1) {
                throw new ArgumentException(null, nameof(arrays));
            }

            Shape shape = arrays[0].Shape;

            foreach (NdimArray<T> array in arrays) {
                if (array.Shape[axis] != 1 || array.Shape != shape) {
                    throw new ArgumentException(null, nameof(arrays));
                }
            }

            int stride = shape.Stride(axis), size = shape.Length, n = arrays.Length;

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

        /// <summary>分割</summary>
        public NdimArray<T>[] Separate(int axis) {
            int stride = Shape.Stride(axis), n = Shape[axis];

            int[] s = Shape;
            s[axis] = 1;

            Shape shape = new(Shape.Type, s);

            int size = shape.Length;

            NdimArray<T>[] arrays_separated = (new NdimArray<T>[n]).Select((_) => (NdimArray<T>)shape).ToArray();

            T[] src = Value;

            for (int i = 0; i < arrays_separated.Length; i++) {
                T[] dst = arrays_separated[i].Value;

                for (int j = 0, k = i * stride; j < size; j += stride, k += n * stride) {
                    Array.Copy(src, k, dst, j, stride);
                }
            }

            return arrays_separated;
        }

        /// <summary>インデクサ</summary>
        /// <param name="index">バッチ次元のインデクス</param>
        public virtual NdimArray<T> this[int index] {
            set {
                if (index < 0 || index >= Batch) {
                    throw new ArgumentOutOfRangeException(nameof(index));
                }

                if (Shape.DataShape != value.Shape.DataShape || value.Shape.Batch != 1) {
                    throw new ArgumentException(null, nameof(value));
                }

                Array.Copy(value.Value, 0, Value, Shape.DataSize * index, Shape.DataSize);
            }
            get {
                if (index < 0 || index >= Batch) {
                    throw new ArgumentOutOfRangeException(nameof(index));
                }

                NdimArray<T> arr = new(Shape.DataShape);

                Array.Copy(Value, Shape.DataSize * index, arr.Value, 0, Shape.DataSize);

                return arr;
            }
        }

        /// <summary>データリスト</summary>
        public IEnumerable<(int index, NdimArray<T> data)> DataList {
            get {
                for (int i = 0; i < Batch; i++) {
                    yield return (i, this[i]);
                }
            }
        }

        /// <summary>バッチ方向に連結</summary>
        public static implicit operator NdimArray<T>(NdimArray<T>[] arrays) {
            if (arrays is null || arrays.Length == 0) {
                throw new ArgumentException(null, nameof(arrays));
            }

            Shape shape = arrays.First().Shape;

            if (shape.Type != ShapeType.Map) {
                throw new ArgumentException(ExceptionMessage.ShapeType(shape.Type, ShapeType.Map));
            }

            return Join(arrays.First().Shape.Ndim - 1, arrays);
        }

        /// <summary>バッチ方向に連結</summary>
        public static implicit operator NdimArray<T>(List<NdimArray<T>> arrays) {
            return arrays.ToArray();
        }

        /// <summary>文字列化</summary>
        public override string ToString() {
            return Shape.ToString();
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
