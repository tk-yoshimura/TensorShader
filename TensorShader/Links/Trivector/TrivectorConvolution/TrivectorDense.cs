using static TensorShader.VariableNode;

namespace TensorShader {
    public partial class Field {
        /// <summary>3次元ベクトル全結合</summary>
        public static Field TrivectorDense(Field x, Field w) {
            Field y = new Field();
            Link link = new Links.TrivectorConvolution.TrivectorDense(x, w, y);

            link.Forward();

            return y;
        }
    }
}

namespace TensorShader.Links.TrivectorConvolution {
    /// <summary>3次元ベクトル全結合</summary>
    public class TrivectorDense : Link {
        /// <summary>入力項</summary>
        protected Field X => InFields[0];

        /// <summary>カーネル項</summary>
        protected Field W => InFields[1];

        /// <summary>出力項</summary>
        protected Field Y => OutField;

        /// <summary>コンストラクタ</summary>
        public TrivectorDense(Field infield, Field kernelfield, Field outfield)
            : base(new Field[] { infield, kernelfield }, outfield) { }

        /// <summary>順伝搬</summary>
        public override void Forward() {
            Y.AssignValue(TrivectorDense(X.Value, W.Value, gradmode: false));
        }

        /// <summary>逆伝搬</summary>
        public override void Backward() {
            if (Y.Grad == null) {
                return;
            }

            if (X.EnableBackprop) {
                X.AddGrad(TrivectorTransposeDense(Y.Grad, W.Value, gradmode: true));
            }

            if (W.EnableBackprop) {
                W.AddGrad(TrivectorKernelProductDense(X.Value, Y.Grad, W.Value, transpose: false));
            }
        }
    }
}
