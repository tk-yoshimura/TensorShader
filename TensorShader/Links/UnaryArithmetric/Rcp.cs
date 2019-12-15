using static TensorShader.VariableNode;

namespace TensorShader {
    public partial class Field {
        /// <summary>逆数</summary>
        public static Field Rcp(Field x) {
            Field y = new Field();
            Link link = new Links.UnaryArithmetric.Rcp(x, y);

            link.Forward();

            return y;
        }
    }
}

namespace TensorShader.Links.UnaryArithmetric {
    /// <summary>逆数</summary>
    internal class Rcp : UnaryArithmetric {
        /// <summary>コンストラクタ</summary>
        public Rcp(Field infield, Field outfield)
            : base(infield, outfield) { }

        /// <summary>順伝搬</summary>
        public override void Forward() {
            Y.AssignValue(Rcp(X.Value));
        }

        /// <summary>逆伝搬</summary>
        public override void Backward() {
            if (Y.Grad == null) {
                return;
            }

            if (X.EnableBackprop) {
                X.AddGrad(-Square(Y.Value) * Y.Grad);
            }
        }
    }
}
