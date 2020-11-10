using System;
using static TensorShader.Field;

namespace TensorShader.Layers {
    /// <summary>複素1次元逆畳み込み</summary>
    public class ComplexDeconvolution1D : Layer {
        /// <summary>重み</summary>
        public ParameterField W { private set; get; }

        /// <summary>バイアス</summary>
        public ParameterField Bias { private set; get; }

        /// <summary>パディングモード</summary>
        public PaddingMode PaddingMode { private set; get; }

        /// <summary>入力チャネル数</summary>
        public override int InChannels => W.Shape.OutChannels * 2;

        /// <summary>出力チャネル数</summary>
        public override int OutChannels => W.Shape.InChannels;

        /// <summary>カーネルサイズ</summary>
        public override int Width => W.Shape.Width;

        /// <summary>コンストラクタ</summary>
        public ComplexDeconvolution1D(int inchannels, int outchannels, int kwidth, bool use_bias, PaddingMode pad_mode, string label)
            : base(label) {
            this.W = new ParameterField(
                Shape.Kernel1D(outchannels, inchannels / 2, kwidth),
                Label + "/w",
                ParameterCategory.Kernel);

            this.Bias = use_bias
                ? new ParameterField(
                    Shape.Vector(outchannels),
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
            Field y = ComplexDeconvolution1D(x, W);

            if (Bias != null) {
                y += Bias;
            }

            if (PaddingMode != PaddingMode.None) {
                int pad_x = W.Shape.Width / 2;

                if (PaddingMode == PaddingMode.Zero) {
                    y = ZeroPadding1D(y, pad_x, pad_x);
                }
                else if (PaddingMode == PaddingMode.Edge) {
                    y = EdgePadding1D(y, pad_x, pad_x);
                }
            }

            return y;
        }
    }
}
