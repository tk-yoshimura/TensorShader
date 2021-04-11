using static TensorShader.VariableNode;

namespace TensorShader {
    public partial class Field {
        /// <summary>1次元線形補間</summary>
        /// <remarks>倍率2固定</remarks>
        public static Field LinearZoom1D(Field x) {
            Field y = new();
            Link link = new Links.Connection1D.LinearZoom(x, y);

            link.Forward();

            return y;
        }
    }
}

namespace TensorShader.Links.Connection1D {
    /// <summary>1次元線形補間</summary>
    /// <remarks>倍率2固定</remarks>
    public class LinearZoom : Link {
        /// <summary>入力項</summary>
        protected Field X => InFields[0];

        /// <summary>出力項</summary>
        protected Field Y => OutField;

        /// <summary>コンストラクタ</summary>
        public LinearZoom(Field infield, Field outfield)
            : base(new Field[] { infield }, outfield) {
        }

        /// <summary>順伝搬</summary>
        public override void Forward() {
            Y.AssignValue(LinearZoom1D(X.Value));
        }

        /// <summary>逆伝搬</summary>
        public override void Backward() {
            if (Y.Grad is null) {
                return;
            }

            if (X.EnableBackprop) {
                X.AddGrad(AveragePooling1D(Y.Grad, stride: 2));
            }
        }
    }
}
