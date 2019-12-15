namespace TensorShader {
    public partial class Field {
        /// <summary>乗算</summary>
        public static Field Mul(Field x1, Field x2) {
            Field y = new Field();
            Link link = new Links.BinaryArithmetric.Mul(x1, x2, y);

            link.Forward();

            return y;
        }

        /// <summary>乗算</summary>
        public static Field operator *(Field x1, Field x2) {
            return Mul(x1, x2);
        }
    }
}

namespace TensorShader.Links.BinaryArithmetric {
    /// <summary>乗算</summary>
    internal class Mul : BinaryArithmetric {
        /// <summary>コンストラクタ</summary>
        public Mul(Field infield1, Field infield2, Field outfield)
            : base(infield1, infield2, outfield) { }

        /// <summary>順伝搬</summary>
        public override void Forward() {
            Y.AssignValue(X1.Value * X2.Value);
        }

        /// <summary>逆伝搬</summary>
        public override void Backward() {
            if (Y.Grad == null) {
                return;
            }

            if (X1.EnableBackprop) {
                X1.AddGrad(AdjectShape(Y.Grad * X2.Value, X1.Shape));
            }

            if (X2.EnableBackprop) {
                X2.AddGrad(AdjectShape(Y.Grad * X1.Value, X2.Shape));
            }
        }
    }
}
