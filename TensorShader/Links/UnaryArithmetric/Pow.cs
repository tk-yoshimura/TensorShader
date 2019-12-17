using static TensorShader.VariableNode;

namespace TensorShader {
    public partial class Field {
        /// <summary>べき関数</summary>
        public static Field Pow(Field x, float alpha) {
            Field y = new Field();
            Link link = new Links.UnaryArithmetric.Pow(x, y, alpha);

            link.Forward();

            return y;
        }
    }
}

namespace TensorShader.Links.UnaryArithmetric {
    /// <summary>べき関数</summary>
    internal class Pow : UnaryArithmetric {
        /// <summary>指数</summary>
        public float Alpha { private set; get; }

        /// <summary>コンストラクタ</summary>
        public Pow(Field infield, Field outfield, float alpha)
            : base(infield, outfield) {

            this.Alpha = alpha;
        }

        /// <summary>順伝搬</summary>
        public override void Forward() {
            Y.AssignValue(Pow(X.Value, Alpha));
        }

        /// <summary>逆伝搬</summary>
        public override void Backward() {
            if (Y.Grad == null) {
                return;
            }

            if (X.EnableBackprop) {
                X.AddGrad(Y.Grad * Alpha * Pow(X.Value, Alpha - 1));
            }
        }
    }
}
