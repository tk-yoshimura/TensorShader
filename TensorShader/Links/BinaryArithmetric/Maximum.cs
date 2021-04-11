using static TensorShader.VariableNode;

namespace TensorShader {
    public partial class Field {
        /// <summary>最大値</summary>
        public static Field Maximum(Field x1, Field x2) {
            Field y = new();
            Link link = new Links.BinaryArithmetric.Maximum(x1, x2, y);

            link.Forward();

            return y;
        }
    }
}

namespace TensorShader.Links.BinaryArithmetric {
    /// <summary>最大値</summary>
    internal class Maximum : BinaryArithmetric {
        /// <summary>コンストラクタ</summary>
        public Maximum(Field infield1, Field infield2, Field outfield)
            : base(infield1, infield2, outfield) { }

        /// <summary>順伝搬</summary>
        public override void Forward() {
            Y.AssignValue(Maximum(X1.Value, X2.Value));
        }

        /// <summary>逆伝搬</summary>
        public override void Backward() {
            if (Y.Grad is null) {
                return;
            }

            if (X1.EnableBackprop) {
                X1.AddGrad(AdjectShape(Y.Grad * GreaterThanOrEqual(X1.Value, X2.Value), X1.Shape));
            }

            if (X2.EnableBackprop) {
                X2.AddGrad(AdjectShape(Y.Grad * GreaterThanOrEqual(X2.Value, X1.Value), X2.Shape));
            }
        }
    }
}
