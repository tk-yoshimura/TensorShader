using static TensorShader.VariableNode;

namespace TensorShader {
    public partial class Field {
        /// <summary>四元数正規化</summary>
        public static Field QuaternionNormalize(Field x) {
            Field y = new Field();
            Link link = new Links.QuaternionArithmetric.QuaternionNormalize(x, y);

            link.Forward();

            return y;
        }
    }
}

namespace TensorShader.Links.QuaternionArithmetric {
    /// <summary>四元数正規化</summary>
    public class QuaternionNormalize : UnaryArithmetric.UnaryArithmetric {
        /// <summary>コンストラクタ</summary>
        public QuaternionNormalize(Field infield, Field outfield)
            : base(infield, outfield) { }

        /// <summary>順伝搬</summary>
        public override void Forward() {
            Y.AssignValue(QuaternionNormalize(X.Value));
        }

        /// <summary>逆伝搬</summary>
        public override void Backward() {
            if (Y.Grad == null) {
                return;
            }

            if (X.EnableBackprop) {
                X.AddGrad(QuaternionNormalizeGrad(Y.Grad, X.Value));
            }
        }
    }
}
