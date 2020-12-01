using System;
using System.Linq;

using TensorShaderCudaBackend;

namespace TensorShader {
    /// <summary>テンソルクラス</summary>
    public partial class Tensor : ICloneable {
        /// <summary>形状</summary>
        public Shape Shape { protected set; get; }

        /// <summary>保有バッファ</summary>
        internal CudaArray<float> Buffer { private protected set; get; }

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
        /// <param name="shape">形状</param>
        /// <param name="value">初期値(任意指定)</param>
        internal Tensor(Shape shape, float[] value = null) {
            if (shape == null) {
                throw new ArgumentException(nameof(shape));
            }
            if (value != null && value.Length < shape.Length) {
                throw new ArgumentException(ExceptionMessage.Argument($"{value}.Length", value.Length, shape.Length));
            }

            this.Buffer = value ?? (new float[shape.Length]);
            this.Shape = shape;
        }

        /// <summary>コンストラクタ</summary>
        /// <param name="shape">形状</param>
        /// <param name="array">バッファ</param>
        protected internal Tensor(Shape shape, CudaArray<float> array) {
            if (shape == null) {
                throw new ArgumentException(nameof(shape));
            }
            if (array == null) {
                throw new ArgumentNullException(nameof(array));
            }
            if ((int)array.Length < shape.Length) {
                throw new ArgumentException(ExceptionMessage.Argument($"{array}.Length", (int)array.Length, shape.Length));
            }

            this.Buffer = array;
            this.Shape = shape;
        }

        /// <summary>状態</summary>
        public virtual NdimArray<float> State {
            get {
                float[] state = new float[Length];
                Buffer.Read(state, (ulong)Length);

                return new NdimArray<float>(Shape, state, clone_value: false);
            }
            set {
                if (value.Shape != Shape) {
                    throw new ArgumentException(ExceptionMessage.Shape(Shape, value.Shape));
                }

                Buffer.Write(value.Value, (ulong)Length);
            }
        }

        /// <summary>初期化</summary>
        public virtual void Clear(float val) {
            ArrayManipulation.Clear((uint)Length, val, Buffer);
        }

        /// <summary>初期化</summary>
        public void Zeroset() {
            Buffer.Zeroset((ulong)Length);
        }

        /// <summary>コピー</summary>
        /// <remarks>形状の一致不一致を問わず実行される</remarks>
        public virtual void CopyTo(Tensor tensor) {
            if (tensor.Length < Length) {
                throw new ArgumentException(ExceptionMessage.Argument("tensor.Length", tensor.Length, Length));
            }
            if (ReferenceEquals(Buffer, tensor.Buffer)) {
                return;
            }

            Buffer.CopyTo(tensor.Buffer, (ulong)Length);
        }

        /// <summary>部分コピー</summary>
        /// <remarks>
        /// 形状の一致不一致を問わず実行される
        /// コピー先に同じテンソルを指定時、領域が重なる場合の動作は不定
        /// </remarks>
        public void RegionCopyTo(Tensor tensor, uint src_index, uint dst_index, uint count) {
            if (src_index >= Length || src_index + count > Length) {
                throw new ArgumentOutOfRangeException();
            }
            if (dst_index >= tensor.Length || dst_index + count > tensor.Length) {
                throw new ArgumentOutOfRangeException();
            }

            Buffer.CopyTo(src_index, tensor.Buffer, dst_index, count);
        }

        /// <summary>形状変更</summary>
        public void Reshape(Shape shape) {
            if (Length != shape.Length) {
                throw new ArgumentException(nameof(shape));
            }

            this.Shape = shape;
        }

        /// <summary>クローン</summary>
        public object Clone() {
            return Copy();
        }

        /// <summary>コピー</summary>
        public virtual Tensor Copy() {
            if (this is OverflowCheckedTensor) {
                OverflowCheckedTensor tensor = new OverflowCheckedTensor(Shape);
                CopyTo(tensor);

                return tensor;
            }
            else {
                Tensor tensor = new Tensor(Shape);
                CopyTo(tensor);

                return tensor;
            }
        }

        /// <summary>定数</summary>
        public static Tensor Constant(Shape shape, float val) {
            Tensor tensor = new Tensor(shape);
            tensor.Clear(val);

            return tensor;
        }
    }

    /// <summary>バッファ共有テンソル</summary>
    internal class TemporaryTensor : Tensor {
        public TemporaryTensor(Shape shape, CudaArray<float> array)
            : base(shape, array) { }
    }

    /// <summary>領域外アクセスチェック有効テンソル</summary>
    /// <remarks>デバック用</remarks>
    internal class OverflowCheckedTensor : Tensor {
        private static readonly Random random = new Random();

        private const int canary_length = 32;
        private readonly float[] canary;

        /// <summary>コンストラクタ</summary>
        /// <param name="shape">形状</param>
        /// <param name="value">初期値(任意指定)</param>
        internal OverflowCheckedTensor(Shape shape, float[] value = null)
            : base(shape, new float[shape.Length + canary_length]) {
            if (shape == null) {
                throw new ArgumentException(nameof(shape));
            }
            if (value != null) {
                if (value.Length != shape.Length) {
                    throw new ArgumentException(ExceptionMessage.Argument($"{value}.Length", value.Length, shape.Length));
                }

                Buffer.Write(value, (ulong)shape.Length);
            }
            else {
                float[] r = new float[1] { (float)random.NextDouble() };

                Buffer.Write(r, 1);
            }

            this.canary = (new float[canary_length]).Select((_) => (float)random.NextDouble()).ToArray();

            CudaArray<float> canary = this.canary;

            canary.CopyTo(0, Buffer, (ulong)shape.Length, canary_length);

            this.Shape = shape;
        }

        /// <summary>状態</summary>
        /// <exception cref="AccessViolationException">領域外アクセスが検知されたとき</exception>
        public override NdimArray<float> State {
            get {
                CheckOverflow();

                float[] state = new float[Length];
                Buffer.Read(state, (ulong)Length);

                return new NdimArray<float>(Shape, state, clone_value: false);
            }

            set {
                if (value.Shape != Shape) {
                    throw new ArgumentException(ExceptionMessage.Shape(Shape, value.Shape));
                }

                Buffer.Write(value.Value, (ulong)Length);
            }
        }

        /// <summary>コピー</summary>
        public override void CopyTo(Tensor tensor) {
            Buffer.CopyTo(tensor.Buffer, (uint)Length);
        }

        /// <summary>領域外アクセスチェック</summary>
        public void CheckOverflow() {
            float[] value = Buffer.Value;

            for (int i = 0; i < canary_length; i++) {
                if (value[Shape.Length + i] != canary[i]) {
                    throw new AccessViolationException("Detected out of buffer access.");
                }
            }
        }
    }
}
