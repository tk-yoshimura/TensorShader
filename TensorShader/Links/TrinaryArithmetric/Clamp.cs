using static TensorShader.VariableNode;

namespace TensorShader {
    public partial class Field {
        /// <summary>Clamp</summary>
        public static Field Clamp(Field x, Field xmin, Field xmax) {
            Field y = new();
            Link link = new Links.TrinaryArithmetric.Clamp(x, xmin, xmax, y);

            link.Forward();

            return y;
        }
    }
}

namespace TensorShader.Links.TrinaryArithmetric {
    /// <summary>Clamp</summary>
    internal class Clamp : TrinaryArithmetric {
        /// <summary>コンストラクタ</summary>
        public Clamp(Field infield1, Field infield2, Field infield3, Field outfield)
            : base(infield1, infield2, infield3, outfield) { }

        /// <summary>順伝搬</summary>
        public override void Forward() {
            Y.AssignValue(Clamp(X1.Value, X2.Value, X3.Value));
        }

        /// <summary>逆伝搬</summary>
        public override void Backward() {
            if (Y.Grad is null) {
                return;
            }

            if (X1.EnableBackprop) {
                X1.AddGrad(Y.Grad * LessThanOrEqual(X2.Value, X1.Value) * LessThanOrEqual(X1.Value, X3.Value));
            }

            if (X2.EnableBackprop) {
                X2.AddGrad(Y.Grad * GreaterThan(X2.Value, X1.Value));
            }

            if (X3.EnableBackprop) {
                X3.AddGrad(Y.Grad * GreaterThan(X1.Value, X3.Value));
            }
        }
    }
}
