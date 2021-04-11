using static TensorShader.VariableNode;

namespace TensorShader {
    public partial class Field {
        /// <summary>チャネルごとの1次元畳み込み</summary>
        public static Field ChannelwiseConvolution1D(Field x, Field w) {
            Field y = new();
            Link link = new Links.Connection1D.ChannelwiseConvolution(x, w, y);

            link.Forward();

            return y;
        }
    }
}

namespace TensorShader.Links.Connection1D {
    /// <summary>チャネルごとの1次元畳み込み</summary>
    public class ChannelwiseConvolution : Link {

        /// <summary>入力項</summary>
        protected Field X => InFields[0];

        /// <summary>カーネル項</summary>
        protected Field W => InFields[1];

        /// <summary>出力項</summary>
        protected Field Y => OutField;

        /// <summary>コンストラクタ</summary>
        public ChannelwiseConvolution(Field infield, Field kernelfield, Field outfield)
            : base(new Field[] { infield, kernelfield }, outfield) { }

        /// <summary>順伝搬</summary>
        public override void Forward() {
            Y.AssignValue(ChannelwiseConvolution1D(X.Value, W.Value));
        }

        /// <summary>逆伝搬</summary>
        public override void Backward() {
            if (Y.Grad is null) {
                return;
            }

            if (X.EnableBackprop) {
                X.AddGrad(ChannelwiseDeconvolution1D(Y.Grad, W.Value));
            }

            if (W.EnableBackprop) {
                W.AddGrad(ChannelwiseKernelProduct1D(X.Value, Y.Grad, W.Shape.Width));
            }
        }
    }
}
