namespace TensorShader {
    public partial class Field {
        /// <summary>乗算</summary>
        public static Field Mul(Field x, float c) {
            Field y = new();
            Link link = new Links.UnaryArithmetric.Mul(x, y, c);

            link.Forward();

            return y;
        }

        /// <summary>乗算</summary>
        public static Field Mul(float c, Field x) {
            Field y = new();
            Link link = new Links.UnaryArithmetric.Mul(x, y, c);

            link.Forward();

            return y;
        }

        /// <summary>乗算</summary>
        public static Field operator *(Field x, float c) {
            return Mul(x, c);
        }

        /// <summary>乗算</summary>
        public static Field operator *(float c, Field x) {
            return Mul(c, x);
        }
    }
}

namespace TensorShader.Links.UnaryArithmetric {
    /// <summary>乗算</summary>
    internal class Mul : UnaryArithmetric {
        /// <summary>定数</summary>
        public float Constant { private set; get; }

        /// <summary>コンストラクタ</summary>
        public Mul(Field infield, Field outfield, float c)
            : base(infield, outfield) {

            this.Constant = c;
        }

        /// <summary>順伝搬</summary>
        public override void Forward() {
            Y.AssignValue(Constant * X.Value);
        }

        /// <summary>逆伝搬</summary>
        public override void Backward() {
            if (Y.Grad is null) {
                return;
            }

            if (X.EnableBackprop) {
                X.AddGrad(Constant * Y.Grad);
            }
        }
    }
}
