namespace TensorShader {
    public partial class Field {
        /// <summary>減算</summary>
        public static Field Sub(Field x, float c) {
            Field y = new();
            Link link = new Links.UnaryArithmetric.LeftSub(x, y, c);

            link.Forward();

            return y;
        }

        /// <summary>減算</summary>
        public static Field Sub(float c, Field x) {
            Field y = new();
            Link link = new Links.UnaryArithmetric.RightSub(x, y, c);

            link.Forward();

            return y;
        }

        /// <summary>減算</summary>
        public static Field operator -(Field x, float c) {
            return Sub(x, c);
        }

        /// <summary>減算</summary>
        public static Field operator -(float c, Field x) {
            return Sub(c, x);
        }
    }
}

namespace TensorShader.Links.UnaryArithmetric {
    /// <summary>減算</summary>
    internal class LeftSub : UnaryArithmetric {
        /// <summary>定数</summary>
        public float Constant { private set; get; }

        /// <summary>コンストラクタ</summary>
        public LeftSub(Field infield, Field outfield, float c)
            : base(infield, outfield) {

            this.Constant = c;
        }

        /// <summary>順伝搬</summary>
        public override void Forward() {
            Y.AssignValue(X.Value - Constant);
        }

        /// <summary>逆伝搬</summary>
        public override void Backward() {
            if (Y.Grad is null) {
                return;
            }

            if (X.EnableBackprop) {
                X.AddGrad(Y.Grad);
            }
        }
    }

    /// <summary>減算</summary>
    internal class RightSub : UnaryArithmetric {
        /// <summary>定数</summary>
        public float Constant { private set; get; }

        /// <summary>コンストラクタ</summary>
        public RightSub(Field infield, Field outfield, float c)
            : base(infield, outfield) {

            this.Constant = c;
        }

        /// <summary>順伝搬</summary>
        public override void Forward() {
            Y.AssignValue(Constant - X.Value);
        }

        /// <summary>逆伝搬</summary>
        public override void Backward() {
            if (Y.Grad is null) {
                return;
            }

            if (X.EnableBackprop) {
                X.AddGrad(-Y.Grad);
            }
        }
    }
}
