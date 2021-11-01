using System;
using System.Linq;

using TensorShaderCudaBackend;

namespace TensorShader {
    /// <summary>形状タイプ</summary>
    public enum ShapeType {
        /// <summary>未定義</summary>
        Undefined,
        /// <summary>スカラー</summary>
        Scalar,
        /// <summary>ベクター</summary>
        Vector,
        /// <summary>カーネル</summary>
        Kernel,
        /// <summary>マップ</summary>
        Map,
        /// <summary>行列</summary>
        Matrix,
        /// <summary>Column変換後行列</summary>
        Column,
    }

    /// <summary>形状クラス</summary>
    public class Shape {
        private readonly int[] shape;

        /// <summary>タイプ</summary>
        public ShapeType Type { private set; get; }

        /// <summary>次元数</summary>
        public int Ndim => shape.Length;

        /// <summary>幅</summary>
        public int Width {
            get {
                if (shape.Length > 2 && Type == ShapeType.Map) return shape[1];
                if (shape.Length > 2 && Type == ShapeType.Kernel) return shape[2];
                if (shape.Length > 3 && Type == ShapeType.Column) return shape[2];
                return 0;
            }
        }

        /// <summary>高さ</summary>
        public int Height {
            get {
                if (shape.Length > 3 && Type == ShapeType.Map) return shape[2];
                if (shape.Length > 3 && Type == ShapeType.Kernel) return shape[3];
                if (shape.Length > 4 && Type == ShapeType.Column) return shape[3];
                return 0;
            }
        }

        /// <summary>奥行き</summary>
        public int Depth {
            get {
                if (shape.Length > 4 && Type == ShapeType.Map) return shape[3];
                if (shape.Length > 4 && Type == ShapeType.Kernel) return shape[4];
                if (shape.Length > 5 && Type == ShapeType.Column) return shape[4];
                return 0;
            }
        }

        /// <summary>チャネル数</summary>
        public int Channels {
            get {
                if (shape.Length > 0 && (Type == ShapeType.Map || Type == ShapeType.Vector)) return shape[0];
                if (shape.Length > 1 && (Type == ShapeType.Kernel)) return shape[0] * shape[1];
                if (shape.Length > 1 && (Type == ShapeType.Column)) return shape[1];
                return 0;
            }
        }

        /// <summary>入力チャネル数</summary>
        public int InChannels {
            get {
                if (shape.Length > 0 && Type == ShapeType.Kernel) return shape[0];
                return 0;
            }
        }

        /// <summary>出力チャネル数</summary>
        public int OutChannels {
            get {
                if (shape.Length > 1 && Type == ShapeType.Kernel) return shape[1];
                return 0;
            }
        }

        /// <summary>バッチ数</summary>
        public int Batch { private set; get; }

        /// <summary>要素数</summary>
        public int Length { private set; get; }

        /// <summary>マップサイズ</summary>
        /// <remarks>要素数 / (チャネル数 x バッチ数)</remarks>
        public int MapSize { private set; get; }

        /// <summary>データサイズ</summary>
        /// <remarks>要素数 / バッチ数</remarks>
        public int DataSize { private set; get; }

        /// <summary>ストライド</summary>
        public int Stride(int axis) {
            int stride = 1;
            for (int i = 0; i < axis; i++) {
                stride *= shape[i];
            }

            return stride;
        }

        /// <summary>各軸の長さ</summary>
        public int this[int axis] => shape[axis];

        /// <summary>データ形状(次元にバッチを含むならば、バッチを1とする)</summary>
        public Shape DataShape {
            get {
                if (Type == ShapeType.Map || Type == ShapeType.Column) {
                    int[] ret_shape = (int[])this.shape.Clone();
                    ret_shape[ret_shape.Length - 1] = 1;

                    return new Shape(Type, ret_shape);
                }
                else {
                    return this;
                }
            }
        }

        /// <summary>コンストラクタ</summary>
        public Shape(ShapeType type, params int[] shape) {
            int length = 1;

            try {
                foreach (int s in shape) {
                    if (s < 1) {
                        throw new ArgumentException(ExceptionMessage.Shape(shape));
                    }

                    length *= s;
                }

                if ((uint)length > CudaArray<float>.MaxLength) {
                    throw new OverflowException();
                }
            }
            catch (OverflowException) {
                throw new ArgumentOutOfRangeException("The specified count of elements is too large.");
            }

            if (type == ShapeType.Scalar && shape.Length > 0) {
                throw new ArgumentException(null, nameof(type));
            }
            if (type == ShapeType.Vector && shape.Length > 1) {
                throw new ArgumentException(null, nameof(type));
            }

            if (shape.Length == 1) {
                type = ShapeType.Vector;
            }
            else if (shape.Length == 0) {
                type = ShapeType.Scalar;
            }

            this.shape = shape;
            this.Type = type;
            this.Length = length;

            this.Batch = (type == ShapeType.Map || type == ShapeType.Column) ? shape.Last() : 1;
            this.MapSize = (type == ShapeType.Map) ? length / (shape[0] * this.Batch)
                         : (type == ShapeType.Kernel) ? length / (shape[0] * shape[1])
                         : 1;
            this.DataSize = length / this.Batch;
        }

        /// <summary>スカラー</summary>
        public static Shape Scalar => new(ShapeType.Scalar);

        /// <summary>ベクター</summary>
        public static Shape Vector(int channels) {
            return new Shape(ShapeType.Vector, channels);
        }

        /// <summary>0次元マップ</summary>
        public static Shape Map0D(int channels, int batch = 1) {
            return new Shape(ShapeType.Map, channels, batch);
        }

        /// <summary>1次元マップ</summary>
        public static Shape Map1D(int channels, int width, int batch = 1) {
            return new Shape(ShapeType.Map, channels, width, batch);
        }

        /// <summary>2次元マップ</summary>
        public static Shape Map2D(int channels, int width, int height, int batch = 1) {
            return new Shape(ShapeType.Map, channels, width, height, batch);
        }

        /// <summary>3次元マップ</summary>
        public static Shape Map3D(int channels, int width, int height, int depth, int batch = 1) {
            return new Shape(ShapeType.Map, channels, width, height, depth, batch);
        }
        /// <summary>0次元フィルタ</summary>
        public static Shape Kernel0D(int inchannels, int outchannels) {
            return new Shape(ShapeType.Kernel, inchannels, outchannels);
        }

        /// <summary>1次元フィルタ</summary>
        public static Shape Kernel1D(int inchannels, int outchannels, int width) {
            if (width < 1 || width % 2 != 1) {
                throw new ArgumentException(null, nameof(width));
            }

            return new Shape(ShapeType.Kernel, inchannels, outchannels, width);
        }

        /// <summary>2次元フィルタ</summary>
        public static Shape Kernel2D(int inchannels, int outchannels, int width, int height) {
            if (width < 1 || width % 2 != 1 || height < 1 || height % 2 != 1) {
                throw new ArgumentException($"{nameof(width)}, {nameof(height)}");
            }

            return new Shape(ShapeType.Kernel, inchannels, outchannels, width, height);
        }

        /// <summary>3次元フィルタ</summary>
        public static Shape Kernel3D(int inchannels, int outchannels, int width, int height, int depth) {
            if (width < 1 || width % 2 != 1 || height < 1 || height % 2 != 1 || depth < 1 || depth % 2 != 1) {
                throw new ArgumentException($"{nameof(width)}, {nameof(height)}, {nameof(depth)}");
            }

            return new Shape(ShapeType.Kernel, inchannels, outchannels, width, height, depth);
        }

        /// <summary>形状クラス比較</summary>
        public static bool operator ==(Shape s1, Shape s2) {
            if (s1 is null && s2 is null) return true;
            if (s1 is null || s2 is null) return false;

            return s1.Equals(s2);
        }

        /// <summary>形状クラス比較</summary>
        public static bool operator !=(Shape s1, Shape s2) {
            return !(s1 == s2);
        }

        /// <summary>int配列へ変換</summary>
        public static implicit operator int[](Shape shape) {
            return (int[])shape.shape.Clone();
        }

        /// <summary>文字列化</summary>
        public override string ToString() {
            return $"{Type} ({string.Join(", ", shape)})";
        }

        /// <summary>等価か判定</summary>
        public override bool Equals(object obj) {
            if (obj is null || !(obj is Shape)) return false;

            Shape obj_shape = (Shape)obj;

            if (Type != obj_shape.Type) return false;

            return shape.SequenceEqual(obj_shape.shape);
        }

        /// <summary>ハッシュ値生成</summary>
        public override int GetHashCode() {
            return Type.GetHashCode() ^ Length.GetHashCode();
        }
    }
}
