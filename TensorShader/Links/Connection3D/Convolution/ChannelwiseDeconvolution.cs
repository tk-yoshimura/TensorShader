using static TensorShader.VariableNode;

namespace TensorShader {
    public partial class Field {
        /// <summary>3次元逆畳み込み</summary>
        public static Field ChannelwiseDeconvolution3D(Field x, Field w) {
            Field y = new Field();
            Link link = new Links.Connection3D.ChannelwiseDeconvolution(x, w, y);

            link.Forward();

            return y;
        }
    }
}

namespace TensorShader.Links.Connection3D {
    /// <summary>3次元逆畳み込み</summary>
    public class ChannelwiseDeconvolution : Link {

        /// <summary>入力項</summary>
        protected Field X => InFields[0];

        /// <summary>カーネル項</summary>
        protected Field W => InFields[1];

        /// <summary>出力項</summary>
        protected Field Y => OutField;

        /// <summary>コンストラクタ</summary>
        public ChannelwiseDeconvolution(Field infield, Field kernelfield, Field outfield)
            : base(new Field[] { infield, kernelfield }, outfield) {}

        /// <summary>順伝搬</summary>
        public override void Forward() {
            Y.AssignValue(ChannelwiseDeconvolution3D(X.Value, W.Value));
        }

        /// <summary>逆伝搬</summary>
        public override void Backward() {
            if (Y.Grad == null) {
                return;
            }

            if (X.EnableBackprop) {
                X.AddGrad(ChannelwiseConvolution3D(Y.Grad, W.Value));
            }

            if (W.EnableBackprop) {
                W.AddGrad(ChannelwiseKernelProduct3D(Y.Grad, X.Value, W.Shape.Width, W.Shape.Height, W.Shape.Depth));
            }
        }
    }
}
