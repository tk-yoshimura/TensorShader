using static TensorShader.VariableNode;

namespace TensorShader {
    public partial class Field {
        /// <summary>繰り上げ関数</summary>
        public static Field Ceil(Field x) {
            Field y = new Field();
            Link link = new Links.UnaryArithmetric.Ceil(x, y);

            link.Forward();

            return y;
        }
    }
}

namespace TensorShader.Links.UnaryArithmetric {
    /// <summary>繰り上げ関数</summary>
    internal class Ceil : UnaryArithmetric {
        /// <summary>コンストラクタ</summary>
        public Ceil(Field infield, Field outfield)
            : base(infield, outfield) { }

        /// <summary>順伝搬</summary>
        public override void Forward() {
            Y.AssignValue(Ceil(X.Value));
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
