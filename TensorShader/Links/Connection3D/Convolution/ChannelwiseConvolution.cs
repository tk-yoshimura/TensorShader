using static TensorShader.VariableNode;

namespace TensorShader {
    public partial class Field {
        /// <summary>チャネルごとの3次元畳み込み</summary>
        public static Field ChannelwiseConvolution3D(Field x, Field w) {
            Field y = new Field();
            Link link = new Links.Connection3D.ChannelwiseConvolution(x, w, y);

            link.Forward();

            return y;
        }
    }
}

namespace TensorShader.Links.Connection3D {
    /// <summary>チャネルごとの3次元畳み込み</summary>
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
            Y.AssignValue(ChannelwiseConvolution3D(X.Value, W.Value));
        }

        /// <summary>逆伝搬</summary>
        public override void Backward() {
            if (Y.Grad == null) {
                return;
            }

            if (X.EnableBackprop) {
                X.AddGrad(ChannelwiseDeconvolution3D(Y.Grad, W.Value));
            }

            if (W.EnableBackprop) {
                W.AddGrad(ChannelwiseKernelProduct3D(X.Value, Y.Grad, W.Shape.Width, W.Shape.Height, W.Shape.Depth));
            }
        }
    }
}
