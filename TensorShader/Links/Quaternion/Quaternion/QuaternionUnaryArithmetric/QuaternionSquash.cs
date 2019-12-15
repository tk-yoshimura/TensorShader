using static TensorShader.VariableNode;

namespace TensorShader {
    public partial class Field {
        /// <summary>四元数に1/(1+sqrt(norm))を乗ずる</summary>
        public static Field QuaternionSquash(Field x) {
            Field y = new Field();
            Link link = new Links.QuaternionArithmetric.QuaternionSquash(x, y);

            link.Forward();

            return y;
        }
    }
}

namespace TensorShader.Links.QuaternionArithmetric {
    /// <summary>四元数に1/(1+sqrt(norm))を乗ずる</summary>
    public class QuaternionSquash : UnaryArithmetric.UnaryArithmetric {
        /// <summary>コンストラクタ</summary>
        public QuaternionSquash(Field infield, Field outfield)
            : base(infield, outfield) { }

        /// <summary>順伝搬</summary>
        public override void Forward() {
            Y.AssignValue(QuaternionSquash(X.Value));
        }

        /// <summary>逆伝搬</summary>
        public override void Backward() {
            if (Y.Grad == null) {
                return;
            }

            if (X.EnableBackprop) {
                X.AddGrad(NanAsZero(QuaternionSquashGrad(Y.Grad, X.Value)));
            }
        }
    }
}
