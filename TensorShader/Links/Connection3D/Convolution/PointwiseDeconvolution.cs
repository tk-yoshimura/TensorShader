using static TensorShader.VariableNode;

namespace TensorShader {
    public partial class Field {
        /// <summary>ポイントごとの3次元逆畳み込み</summary>
        public static Field PointwiseDeconvolution3D(Field x, Field w) {
            Field y = new();
            Link link = new Links.Connection3D.PointwiseDeconvolution(x, w, y);

            link.Forward();

            return y;
        }
    }
}

namespace TensorShader.Links.Connection3D {
    /// <summary>ポイントごとの3次元逆畳み込み</summary>
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
            Y.AssignValue(PointwiseDeconvolution3D(X.Value, W.Value));
        }

        /// <summary>逆伝搬</summary>
        public override void Backward() {
            if (Y.Grad is null) {
                return;
            }

            if (X.EnableBackprop) {
                X.AddGrad(PointwiseConvolution3D(Y.Grad, W.Value));
            }

            if (W.EnableBackprop) {
                W.AddGrad(PointwiseKernelProduct3D(Y.Grad, X.Value));
            }
        }
    }
}
