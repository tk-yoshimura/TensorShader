using static TensorShader.VariableNode;

namespace TensorShader {
    public partial class Field {
        /// <summary>ポイントごとの1次元畳み込み</summary>
        public static Field PointwiseConvolution1D(Field x, Field w) {
            Field y = new Field();
            Link link = new Links.Connection1D.PointwiseConvolution(x, w, y);

            link.Forward();

            return y;
        }
    }
}

namespace TensorShader.Links.Connection1D {
    /// <summary>ポイントごとの1次元畳み込み</summary>
    public class PointwiseConvolution : Link {
        /// <summary>入力項</summary>
        protected Field X => InFields[0];

        /// <summary>カーネル項</summary>
        protected Field W => InFields[1];

        /// <summary>出力項</summary>
        protected Field Y => OutField;

        /// <summary>コンストラクタ</summary>
        public PointwiseConvolution(Field infield, Field kernelfield, Field outfield)
            : base(new Field[] { infield, kernelfield }, outfield) { }

        /// <summary>順伝搬</summary>
        public override void Forward() {
            Y.AssignValue(PointwiseConvolution1D(X.Value, W.Value));
        }

        /// <summary>逆伝搬</summary>
        public override void Backward() {
            if (Y.Grad == null) {
                return;
            }

            if (X.EnableBackprop) {
                X.AddGrad(PointwiseDeconvolution1D(Y.Grad, W.Value));
            }

            if (W.EnableBackprop) {
                W.AddGrad(PointwiseKernelProduct1D(X.Value, Y.Grad));
            }
        }
    }
}
