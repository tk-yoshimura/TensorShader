using static TensorShader.VariableNode;

namespace TensorShader {
    public partial class Field {
        /// <summary>1次元逆畳み込み</summary>
        public static Field ChannelwiseDeconvolution1D(Field x, Field w, int stride, Shape outshape = null) {
            Field y = new Field();
            Link link = new Links.Connection1D.ChannelwiseDeconvolution(x, w, y, stride, outshape);

            link.Forward();

            return y;
        }
    }
}

namespace TensorShader.Links.Connection1D {
    /// <summary>1次元逆畳み込み</summary>
    public class ChannelwiseDeconvolution : Link {
        /// <summary>ストライド</summary>
        public int Stride { private set; get; }

        /// <summary>出力形状</summary>
        public Shape OutShape { private set; get; }

        /// <summary>入力項</summary>
        protected Field X => InFields[0];

        /// <summary>カーネル項</summary>
        protected Field W => InFields[1];

        /// <summary>出力項</summary>
        protected Field Y => OutField;

        /// <summary>コンストラクタ</summary>
        public ChannelwiseDeconvolution(Field infield, Field kernelfield, Field outfield, int stride, Shape outshape)
            : base(new Field[] { infield, kernelfield }, outfield) {
            this.Stride = stride;
            this.OutShape = outshape;
        }

        /// <summary>順伝搬</summary>
        public override void Forward() {
            Y.AssignValue(ChannelwiseDeconvolution1D(X.Value, W.Value, Stride, OutShape));
        }

        /// <summary>逆伝搬</summary>
        public override void Backward() {
            if (Y.Grad == null) {
                return;
            }

            if (X.EnableBackprop) {
                X.AddGrad(ChannelwiseConvolution1D(Y.Grad, W.Value, Stride));
            }

            if (W.EnableBackprop) {
                W.AddGrad(ChannelwiseKernelProduct1D(Y.Grad, X.Value, W.Shape.Width, Stride));
            }
        }
    }
}
