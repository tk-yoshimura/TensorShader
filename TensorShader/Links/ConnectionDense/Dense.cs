using static TensorShader.VariableNode;

namespace TensorShader {
    public partial class Field {
        /// <summary>全結合</summary>
        public static Field Dense(Field x, Field w) {
            Field y = new();
            Link link = new Links.ConnectionDense.Dense(x, w, y);

            link.Forward();

            return y;
        }
    }
}

namespace TensorShader.Links.ConnectionDense {
    /// <summary>全結合</summary>
    public class Dense : Link {
        /// <summary>入力項</summary>
        protected Field X => InFields[0];

        /// <summary>カーネル項</summary>
        protected Field W => InFields[1];

        /// <summary>出力項</summary>
        protected Field Y => OutField;

        /// <summary>コンストラクタ</summary>
        public Dense(Field infield, Field kernelfield, Field outfield)
            : base(new Field[] { infield, kernelfield }, outfield) { }

        /// <summary>順伝搬</summary>
        public override void Forward() {
            Y.AssignValue(Dense(X.Value, W.Value));
        }

        /// <summary>逆伝搬</summary>
        public override void Backward() {
            if (Y.Grad is null) {
                return;
            }

            if (X.EnableBackprop) {
                X.AddGrad(TransposeDense(Y.Grad, W.Value));
            }

            if (W.EnableBackprop) {
                W.AddGrad(KernelProductDense(X.Value, Y.Grad));
            }
        }
    }
}
