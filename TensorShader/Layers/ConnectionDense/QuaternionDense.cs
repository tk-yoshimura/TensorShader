using System;
using static TensorShader.Field;

namespace TensorShader.Layers {
    /// <summary>四元数全結合</summary>
    public class QuaternionDense : Layer {
        /// <summary>重み</summary>
        public ParameterField W { private set; get; }

        /// <summary>バイアス</summary>
        public ParameterField Bias { private set; get; }

        /// <summary>入力チャネル数</summary>
        public override int InChannels => W.Shape.InChannels;

        /// <summary>出力チャネル数</summary>
        public override int OutChannels => W.Shape.OutChannels * 4;

        /// <summary>コンストラクタ</summary>
        public QuaternionDense(int inchannels, int outchannels, bool use_bias, string label)
            : base(label) {
            this.W = new ParameterField(
                new Tensor(Shape.Kernel0D(inchannels, outchannels / 4)),
                Label + "/w",
                ParameterCategory.Kernel);

            this.Bias = use_bias
                ? new ParameterField(
                    new Tensor(Shape.Vector(outchannels)),
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
            Field y = QuaternionDense(x, W);

            if (Bias != null) {
                y += Bias;
            }

            return y;
        }
    }
}
