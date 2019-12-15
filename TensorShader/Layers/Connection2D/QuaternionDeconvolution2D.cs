using System;
using static TensorShader.Field;

namespace TensorShader.Layers {
    /// <summary>四元数2次元逆畳み込み</summary>
    public class QuaternionDeconvolution2D : Layer {
        /// <summary>重み</summary>
        public ParameterField W { private set; get; }

        /// <summary>バイアス</summary>
        public ParameterField Bias { private set; get; }

        /// <summary>ストライド</summary>
        public int Stride { private set; get; }

        /// <summary>パディングモード</summary>
        public PaddingMode PaddingMode { private set; get; }

        /// <summary>入力チャネル数</summary>
        public override int InChannels => W.Shape.OutChannels * 4;

        /// <summary>出力チャネル数</summary>
        public override int OutChannels => W.Shape.InChannels;

        /// <summary>カーネルサイズ</summary>
        public override int Width => W.Shape.Width;

        /// <summary>カーネルサイズ</summary>
        public override int Height => W.Shape.Height;

        /// <summary>コンストラクタ</summary>
        public QuaternionDeconvolution2D(int inchannels, int outchannels, int kwidth, int kheight, int stride, bool use_bias, PaddingMode pad_mode, string label)
            : base(label) {
            this.W = new ParameterField(
                new Tensor(Shape.Kernel2D(outchannels, inchannels / 4, kwidth, kheight)),
                Label + "/w",
                ParameterCategory.Kernel);

            this.Bias = use_bias
                ? new ParameterField(
                    new Tensor(Shape.Vector(outchannels)),
                    Label + "/bias",
                    ParameterCategory.Bias)
                : null;

            this.Stride = stride;
            this.PaddingMode = pad_mode;
        }

        /// <summary>適用</summary>
        public override Field Forward(params Field[] fields) {
            if (fields.Length != 1) {
                throw new ArgumentException(ExceptionMessage.ArgumentCount(nameof(fields), fields.Length, 1));
            }

            return Forward(fields[0]);
        }

        /// <summary>適用</summary>
        public Field Forward(Field x) {
            Field y = QuaternionDeconvolution2D(x, W, Stride);

            if (Bias != null) {
                y += Bias;
            }

            if (PaddingMode != PaddingMode.None) {
                int pad_x = W.Shape.Width / 2;
                int pad_y = W.Shape.Height / 2;

                if (PaddingMode == PaddingMode.Zero) {
                    y = ZeroPadding2D(y, pad_x, pad_x, pad_y, pad_y);
                }
                else if (PaddingMode == PaddingMode.Edge) {
                    y = EdgePadding2D(y, pad_x, pad_x, pad_y, pad_y);
                }
            }

            return y;
        }
    }
}
