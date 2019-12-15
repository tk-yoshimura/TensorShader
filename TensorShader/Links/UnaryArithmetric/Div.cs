namespace TensorShader {
    public partial class Field {
        /// <summary>減算</summary>
        public static Field Div(Field x, float c) {
            Field y = new Field();
            Link link = new Links.UnaryArithmetric.LeftDiv(x, y, c);

            link.Forward();

            return y;
        }

        /// <summary>減算</summary>
        public static Field Div(float c, Field x) {
            Field y = new Field();
            Link link = new Links.UnaryArithmetric.RightDiv(x, y, c);

            link.Forward();

            return y;
        }

        /// <summary>減算</summary>
        public static Field operator /(Field x, float c) {
            return Div(x, c);
        }

        /// <summary>減算</summary>
        public static Field operator /(float c, Field x) {
            return Div(c, x);
        }
    }
}

namespace TensorShader.Links.UnaryArithmetric {
    /// <summary>減算</summary>
    internal class LeftDiv : UnaryArithmetric {
        /// <summary>定数</summary>
        public float Constant { private set; get; }

        /// <summary>コンストラクタ</summary>
        public LeftDiv(Field infield, Field outfield, float c)
            : base(infield, outfield) {
            this.Constant = c;
        }

        /// <summary>順伝搬</summary>
        public override void Forward() {
            Y.AssignValue(X.Value / Constant);
        }

        /// <summary>逆伝搬</summary>
        public override void Backward() {
            if (Y.Grad == null) {
                return;
            }

            if (X.EnableBackprop) {
                X.AddGrad(Y.Grad / Constant);
            }
        }
    }

    /// <summary>減算</summary>
    internal class RightDiv : UnaryArithmetric {
        /// <summary>定数</summary>
        public float Constant { private set; get; }

        /// <summary>コンストラクタ</summary>
        public RightDiv(Field infield, Field outfield, float c)
            : base(infield, outfield) {
            this.Constant = c;
        }

        /// <summary>順伝搬</summary>
        public override void Forward() {
            Y.AssignValue(Constant / X.Value);
        }

        /// <summary>逆伝搬</summary>
        public override void Backward() {
            if (Y.Grad == null) {
                return;
            }

            if (X.EnableBackprop) {
                X.AddGrad(-Y.Grad * Y.Value / X.Value);
            }
        }
    }
}
