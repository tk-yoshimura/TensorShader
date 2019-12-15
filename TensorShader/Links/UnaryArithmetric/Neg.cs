namespace TensorShader {
    public partial class Field {
        /// <summary>符号反転</summary>
        public static Field Neg(Field x) {
            Field y = new Field();
            Link link = new Links.UnaryArithmetric.Neg(x, y);

            link.Forward();

            return y;
        }

        /// <summary>符号反転</summary>
        public static Field operator -(Field x) {
            return Neg(x);
        }
    }
}

namespace TensorShader.Links.UnaryArithmetric {
    /// <summary>符号反転</summary>
    internal class Neg : UnaryArithmetric {
        /// <summary>コンストラクタ</summary>
        public Neg(Field infield, Field outfield)
            : base(infield, outfield) { }

        /// <summary>順伝搬</summary>
        public override void Forward() {
            Y.AssignValue(-X.Value);
        }

        /// <summary>逆伝搬</summary>
        public override void Backward() {
            if (Y.Grad == null) {
                return;
            }

            if (X.EnableBackprop) {
                X.AddGrad(-Y.Grad);
            }
        }
    }
}
