using System;
using static TensorShader.Field;

namespace TensorShader.Layers {
    /// <summary>3次元ベクトル全結合</summary>
    public class TrivectorDense : Layer {
        /// <summary>重み</summary>
        public ParameterField W { private set; get; }

        /// <summary>バイアス</summary>
        public ParameterField Bias { private set; get; }

        /// <summary>入力チャネル数</summary>
        public override int InChannels => W.Shape.InChannels / 4 * 3;

        /// <summary>出力チャネル数</summary>
        public override int OutChannels => W.Shape.OutChannels * 3;

        /// <summary>コンストラクタ</summary>
        public TrivectorDense(int inchannels, int outchannels, bool use_bias, string label)
            : base(label) {
            this.W = new ParameterField(
                Shape.Kernel0D(inchannels / 3 * 4, outchannels / 3),
                Label + "/w",
                ParameterCategory.Kernel);

            this.Bias = use_bias
                ? new ParameterField(
                    Shape.Vector(outchannels),
                    Label + "/bias",
                    ParameterCategory.Bias)
                : null;
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
            Field y = TrivectorDense(x, W);

            if (Bias is not null) {
                y += Bias;
            }

            return y;
        }
    }
}
