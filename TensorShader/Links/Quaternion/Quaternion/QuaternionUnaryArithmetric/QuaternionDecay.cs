using static TensorShader.VariableNode;

namespace TensorShader {
    public partial class Field {
        /// <summary>四元数減衰</summary>
        public static Field QuaternionDecay(Field x) {
            Field y = new Field();
            Link link = new Links.QuaternionArithmetric.QuaternionDecay(x, y);

            link.Forward();

            return y;
        }
    }
}

namespace TensorShader.Links.QuaternionArithmetric {
    /// <summary>四元数減衰</summary>
    public class QuaternionDecay : UnaryArithmetric.UnaryArithmetric {
        /// <summary>コンストラクタ</summary>
        public QuaternionDecay(Field infield, Field outfield)
            : base(infield, outfield) { }

        /// <summary>順伝搬</summary>
        public override void Forward() {
            Y.AssignValue(QuaternionDecay(X.Value));
        }

        /// <summary>逆伝搬</summary>
        public override void Backward() {
            if (Y.Grad == null) {
                return;
            }

            if (X.EnableBackprop) {
                X.AddGrad(QuaternionDecayGrad(Y.Grad, X.Value));
            }
        }
    }
}
