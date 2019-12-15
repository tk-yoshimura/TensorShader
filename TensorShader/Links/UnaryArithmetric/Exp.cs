using static TensorShader.VariableNode;

namespace TensorShader {
    public partial class Field {
        /// <summary>指数関数</summary>
        public static Field Exp(Field x) {
            Field y = new Field();
            Link link = new Links.UnaryArithmetric.Exp(x, y);

            link.Forward();

            return y;
        }
    }
}

namespace TensorShader.Links.UnaryArithmetric {
    /// <summary>指数関数</summary>
    internal class Exp : UnaryArithmetric {
        /// <summary>コンストラクタ</summary>
        public Exp(Field infield, Field outfield)
            : base(infield, outfield) { }

        /// <summary>順伝搬</summary>
        public override void Forward() {
            Y.AssignValue(Exp(X.Value));
        }

        /// <summary>逆伝搬</summary>
        public override void Backward() {
            if (Y.Grad == null) {
                return;
            }

            if (X.EnableBackprop) {
                X.AddGrad(Y.Value * Y.Grad);
            }
        }
    }
}
