using static TensorShader.VariableNode;

namespace TensorShader {
    public partial class Field {
        /// <summary>複素全結合</summary>
        public static Field ComplexDense(Field x, Field w) {
            Field y = new Field();
            Link link = new Links.ComplexConvolution.ComplexDense(x, w, y);

            link.Forward();

            return y;
        }
    }
}

namespace TensorShader.Links.ComplexConvolution {
    /// <summary>複素全結合</summary>
    public class ComplexDense : Link {
        /// <summary>入力項</summary>
        protected Field X => InFields[0];

        /// <summary>カーネル項</summary>
        protected Field W => InFields[1];

        /// <summary>出力項</summary>
        protected Field Y => OutField;

        /// <summary>コンストラクタ</summary>
        public ComplexDense(Field infield, Field kernelfield, Field outfield)
            : base(new Field[] { infield, kernelfield }, outfield) { }

        /// <summary>順伝搬</summary>
        public override void Forward() {
            Y.AssignValue(ComplexDense(X.Value, W.Value, gradmode: false));
        }

        /// <summary>逆伝搬</summary>
        public override void Backward() {
            if (Y.Grad == null) {
                return;
            }

            if (X.EnableBackprop) {
                X.AddGrad(ComplexTransposeDense(Y.Grad, W.Value, gradmode: true));
            }

            if (W.EnableBackprop) {
                W.AddGrad(ComplexKernelProductDense(X.Value, Y.Grad, transpose: false));
            }
        }
    }
}
