using static TensorShader.VariableNode;

namespace TensorShader {
    public partial class Field {
        /// <summary>QuaternionRRelu</summary>
        /// <remarks>実部のみReluを適用</remarks>
        public static Field QuaternionRRelu(Field x) {
            Field y = new();
            Link link = new Links.QuaternionArithmetric.QuaternionRRelu(x, y);

            link.Forward();

            return y;
        }
    }
}

namespace TensorShader.Links.QuaternionArithmetric {
    /// <summary>QuaternionRRelu</summary>
    public class QuaternionRRelu : UnaryArithmetric.UnaryArithmetric {
        /// <summary>コンストラクタ</summary>
        public QuaternionRRelu(Field infield, Field outfield)
            : base(infield, outfield) { }

        /// <summary>順伝搬</summary>
        public override void Forward() {
            Y.AssignValue(QuaternionRRelu(X.Value));
        }

        /// <summary>逆伝搬</summary>
        public override void Backward() {
            if (Y.Grad is null) {
                return;
            }

            if (X.EnableBackprop) {
                X.AddGrad(QuaternionRReluGrad(Y.Grad, X.Value));
            }
        }
    }
}
