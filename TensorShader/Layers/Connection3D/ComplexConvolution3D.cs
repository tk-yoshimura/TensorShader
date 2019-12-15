using System;
using static TensorShader.Field;

namespace TensorShader.Layers {
    /// <summary>複素3次元畳み込み</summary>
    public class ComplexConvolution3D : Layer {
        /// <summary>重み</summary>
        public ParameterField W { private set; get; }

        /// <summary>バイアス</summary>
        public ParameterField Bias { private set; get; }

        /// <summary>ストライド</summary>
        public int Stride { private set; get; }

        /// <summary>パディングモード</summary>
        public PaddingMode PaddingMode { private set; get; }

        /// <summary>入力チャネル数</summary>
        public override int InChannels => W.Shape.InChannels;

        /// <summary>出力チャネル数</summary>
        public override int OutChannels => W.Shape.OutChannels * 2;

        /// <summary>カーネルサイズ</summary>
        public override int Width => W.Shape.Width;

        /// <summary>カーネルサイズ</summary>
        public override int Height => W.Shape.Height;

        /// <summary>カーネルサイズ</summary>
        public override int Depth => W.Shape.Depth;

        /// <summary>コンストラクタ</summary>
        public ComplexConvolution3D(int inchannels, int outchannels, int kwidth, int kheight, int kdepth, int stride, bool use_bias, PaddingMode pad_mode, string label)
            : base(label) {
            this.W = new ParameterField(
                new Tensor(Shape.Kernel3D(inchannels, outchannels / 2, kwidth, kheight, kdepth)),
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
            if (PaddingMode != PaddingMode.None) {
                int pad_x = W.Shape.Width / 2;
                int pad_y = W.Shape.Height / 2;
                int pad_z = W.Shape.Depth / 2;

                if (PaddingMode == PaddingMode.Zero) {
                    x = ZeroPadding3D(x, pad_x, pad_x, pad_y, pad_y, pad_z, pad_z);
                }
                else if (PaddingMode == PaddingMode.Edge) {
                    x = EdgePadding3D(x, pad_x, pad_x, pad_y, pad_y, pad_z, pad_z);
                }
            }

            Field y = ComplexConvolution3D(x, W, Stride);

            if (Bias != null) {
                y += Bias;
            }

            return y;
        }
    }
}
