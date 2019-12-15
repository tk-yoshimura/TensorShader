using static TensorShader.VariableNode;

namespace TensorShader {
    public partial class Field {
        /// <summary>四元数第3虚部</summary>
        public static Field QuaternionK(Field x) {
            Field y = new Field();
            Link link = new Links.QuaternionArithmetric.QuaternionK(x, y);

            link.Forward();

            return y;
        }
    }
}

namespace TensorShader.Links.QuaternionArithmetric {
    /// <summary>四元数第3虚部</summary>
    public class QuaternionK : UnaryArithmetric.UnaryArithmetric {
        /// <summary>コンストラクタ</summary>
        public QuaternionK(Field infield, Field outfield)
            : base(infield, outfield) { }

        /// <summary>順伝搬</summary>
        public override void Forward() {
            Y.AssignValue(QuaternionK(X.Value));
        }

        /// <summary>逆伝搬</summary>
        public override void Backward() {
            if (Y.Grad == null) {
                return;
            }

            if (X.EnableBackprop) {
                X.AddGrad(QuaternionPureK(Y.Grad));
            }
        }
    }
}
