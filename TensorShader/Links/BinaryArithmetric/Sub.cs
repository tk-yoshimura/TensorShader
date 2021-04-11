namespace TensorShader {
    public partial class Field {
        /// <summary>減算</summary>
        public static Field Sub(Field x1, Field x2) {
            Field y = new();
            Link link = new Links.BinaryArithmetric.Sub(x1, x2, y);

            link.Forward();

            return y;
        }

        /// <summary>減算</summary>
        public static Field operator -(Field x1, Field x2) {
            return Sub(x1, x2);
        }
    }
}

namespace TensorShader.Links.BinaryArithmetric {
    /// <summary>減算</summary>
    internal class Sub : BinaryArithmetric {
        /// <summary>コンストラクタ</summary>
        public Sub(Field infield1, Field infield2, Field outfield)
            : base(infield1, infield2, outfield) { }

        /// <summary>順伝搬</summary>
        public override void Forward() {
            Y.AssignValue(X1.Value - X2.Value);
        }

        /// <summary>逆伝搬</summary>
        public override void Backward() {
            if (Y.Grad is null) {
                return;
            }

            if (X1.EnableBackprop) {
                X1.AddGrad(AdjectShape(Y.Grad, X1.Shape));
            }

            if (X2.EnableBackprop) {
                X2.AddGrad(AdjectShape(-Y.Grad, X2.Shape));
            }
        }
    }
}
