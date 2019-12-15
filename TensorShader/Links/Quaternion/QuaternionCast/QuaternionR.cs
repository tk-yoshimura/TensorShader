using static TensorShader.VariableNode;

namespace TensorShader {
    public partial class Field {
        /// <summary>四元数実部</summary>
        public static Field QuaternionR(Field x) {
            Field y = new Field();
            Link link = new Links.QuaternionArithmetric.QuaternionR(x, y);

            link.Forward();

            return y;
        }
    }
}

namespace TensorShader.Links.QuaternionArithmetric {
    /// <summary>四元数実部</summary>
    public class QuaternionR : UnaryArithmetric.UnaryArithmetric {
        /// <summary>コンストラクタ</summary>
        public QuaternionR(Field infield, Field outfield)
            : base(infield, outfield) { }

        /// <summary>順伝搬</summary>
        public override void Forward() {
            Y.AssignValue(QuaternionR(X.Value));
        }

        /// <summary>逆伝搬</summary>
        public override void Backward() {
            if (Y.Grad == null) {
                return;
            }

            if (X.EnableBackprop) {
                X.AddGrad(QuaternionPureR(Y.Grad));
            }
        }
    }
}
