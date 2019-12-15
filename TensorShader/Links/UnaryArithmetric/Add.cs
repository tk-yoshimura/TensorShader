namespace TensorShader {
    public partial class Field {
        /// <summary>加算</summary>
        public static Field Add(Field x, float c) {
            Field y = new Field();
            Link link = new Links.UnaryArithmetric.Add(x, y, c);

            link.Forward();

            return y;
        }

        /// <summary>加算</summary>
        public static Field Add(float c, Field x) {
            Field y = new Field();
            Link link = new Links.UnaryArithmetric.Add(x, y, c);

            link.Forward();

            return y;
        }

        /// <summary>加算</summary>
        public static Field operator +(Field x, float c) {
            return Add(x, c);
        }

        /// <summary>加算</summary>
        public static Field operator +(float c, Field x) {
            return Add(c, x);
        }
    }
}

namespace TensorShader.Links.UnaryArithmetric {
    /// <summary>加算</summary>
    internal class Add : UnaryArithmetric {
        /// <summary>定数</summary>
        public float Constant { private set; get; }

        /// <summary>コンストラクタ</summary>
        public Add(Field infield, Field outfield, float c)
            : base(infield, outfield) {
            this.Constant = c;
        }

        /// <summary>順伝搬</summary>
        public override void Forward() {
            Y.AssignValue(X.Value + Constant);
        }

        /// <summary>逆伝搬</summary>
        public override void Backward() {
            if (Y.Grad == null) {
                return;
            }

            if (X.EnableBackprop) {
                X.AddGrad(Y.Grad);
            }
        }
    }
}
