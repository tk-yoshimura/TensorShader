using static TensorShader.VariableNode;

namespace TensorShader {
    public partial class Field {
        /// <summary>ポイントごとの2次元逆畳み込み</summary>
        public static Field PointwiseDeconvolution2D(Field x, Field w) {
            Field y = new Field();
            Link link = new Links.Connection2D.PointwiseDeconvolution(x, w, y);

            link.Forward();

            return y;
        }
    }
}

namespace TensorShader.Links.Connection2D {
    /// <summary>ポイントごとの2次元逆畳み込み</summary>
    public class PointwiseDeconvolution : Link {
        /// <summary>入力項</summary>
        protected Field X => InFields[0];

        /// <summary>カーネル項</summary>
        protected Field W => InFields[1];

        /// <summary>出力項</summary>
        protected Field Y => OutField;

        /// <summary>コンストラクタ</summary>
        public PointwiseDeconvolution(Field infield, Field kernelfield, Field outfield)
            : base(new Field[] { infield, kernelfield }, outfield) { }

        /// <summary>順伝搬</summary>
        public override void Forward() {
            Y.AssignValue(PointwiseDeconvolution2D(X.Value, W.Value));
        }

        /// <summary>逆伝搬</summary>
        public override void Backward() {
            if (Y.Grad == null) {
                return;
            }

            if (X.EnableBackprop) {
                X.AddGrad(PointwiseConvolution2D(Y.Grad, W.Value));
            }

            if (W.EnableBackprop) {
                W.AddGrad(PointwiseKernelProduct2D(Y.Grad, X.Value));
            }
        }
    }
}
