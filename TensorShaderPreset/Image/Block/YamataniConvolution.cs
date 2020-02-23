using System;
using TensorShader;
using TensorShader.Layers;
using static TensorShader.Field;

namespace TensorShaderPreset.Image {
    /// <summary>Yamatani斉次応答性畳み込み</summary>
    /// <remarks>
    /// T.Yoshimura 2020
    /// https://www.techrxiv.org/articles/Yamatani_Activation_Edge_Homogeneous_Response_Super_Resolution_Neural_Network/11861187
    /// </remarks>
    public class YamataniConvolution : Layer {
        /// <summary>重み1</summary>
        public ParameterField W1 { private set; get; }

        /// <summary>重み2</summary>
        public ParameterField W2 { private set; get; }

        /// <summary>バイアス</summary>
        public ParameterField Bias { private set; get; }

        /// <summary>パディングモード</summary>
        public PaddingMode PaddingMode { private set; get; }

        /// <summary>入力チャネル数</summary>
        public override int InChannels => W1.Shape.InChannels;

        /// <summary>出力チャネル数</summary>
        public override int OutChannels => W1.Shape.OutChannels;

        /// <summary>カーネルサイズ</summary>
        public override int Width => W1.Shape.Width;

        /// <summary>カーネルサイズ</summary>
        public override int Height => W1.Shape.Height;

        /// <summary>Yamatani Slope</summary>
        public float Slope { private set; get; }

        /// <summary>コンストラクタ</summary>
        public YamataniConvolution(int inchannels, int outchannels, int ksize, float yamatani_slope, PaddingMode pad_mode, string label)
            : base(label) {
            this.W1 = new ParameterField(
                new Tensor(Shape.Kernel2D(inchannels, outchannels, ksize, ksize)),
                Label + "/w1",
                ParameterCategory.Kernel);

            this.W2 = new ParameterField(
                new Tensor(Shape.Kernel2D(inchannels, outchannels, ksize, ksize)),
                Label + "/w2",
                ParameterCategory.Kernel);

            this.Slope = yamatani_slope;
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
                int pad = W1.Shape.Width / 2;

                if (PaddingMode == PaddingMode.Zero) {
                    x = ZeroPadding2D(x, pad);
                }
                else if (PaddingMode == PaddingMode.Edge) {
                    x = EdgePadding2D(x, pad);
                }
            }

            Field h1 = Convolution2D(x, W1);
            Field h2 = Convolution2D(x, W2);

            Field y = Yamatani(h1, h2, Slope);

            return y;
        }
    }
}
