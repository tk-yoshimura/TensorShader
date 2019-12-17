using System;
using static TensorShader.Field;

namespace TensorShader.Layers {
    /// <summary>3次元ベクトル1次元畳み込み</summary>
    public class TrivectorConvolution1D : Layer {
        /// <summary>重み</summary>
        public ParameterField W { private set; get; }

        /// <summary>バイアス</summary>
        public ParameterField Bias { private set; get; }

        /// <summary>パディングモード</summary>
        public PaddingMode PaddingMode { private set; get; }

        /// <summary>入力チャネル数</summary>
        public override int InChannels => W.Shape.InChannels / 4 * 3;

        /// <summary>出力チャネル数</summary>
        public override int OutChannels => W.Shape.OutChannels * 3;

        /// <summary>カーネルサイズ</summary>
        public override int Width => W.Shape.Width;

        /// <summary>コンストラクタ</summary>
        public TrivectorConvolution1D(int inchannels, int outchannels, int kwidth, bool use_bias, PaddingMode pad_mode, string label)
            : base(label) {
            this.W = new ParameterField(
                new Tensor(Shape.Kernel1D(inchannels / 3 * 4, outchannels / 3, kwidth)),
                Label + "/w",
                ParameterCategory.Kernel);

            this.Bias = use_bias
                ? new ParameterField(
                    new Tensor(Shape.Vector(outchannels)),
                    Label + "/bias",
                    ParameterCategory.Bias)
                : null;

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
            if (PaddingMode != PaddingMode.None) {
                int pad_x = W.Shape.Width / 2;

                if (PaddingMode == PaddingMode.Zero) {
                    x = ZeroPadding1D(x, pad_x, pad_x);
                }
                else if (PaddingMode == PaddingMode.Edge) {
                    x = EdgePadding1D(x, pad_x, pad_x);
                }
            }

            Field y = TrivectorConvolution1D(x, W);

            if (Bias != null) {
                y += Bias;
            }

            return y;
        }
    }
}
