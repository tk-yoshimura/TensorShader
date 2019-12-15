using static TensorShader.VariableNode;

namespace TensorShader {
    public partial class Field {
        /// <summary>余弦関数</summary>
        public static Field Cos(Field x) {
            Field y = new Field();
            Link link = new Links.UnaryArithmetric.Cos(x, y);

            link.Forward();

            return y;
        }
    }
}

namespace TensorShader.Links.UnaryArithmetric {
    /// <summary>余弦関数</summary>
    internal class Cos : UnaryArithmetric {
        /// <summary>コンストラクタ</summary>
        public Cos(Field infield, Field outfield)
            : base(infield, outfield) { }

        /// <summary>順伝搬</summary>
        public override void Forward() {
            Y.AssignValue(Cos(X.Value));
        }

        /// <summary>逆伝搬</summary>
        public override void Backward() {
            if (Y.Grad == null) {
                return;
            }

            if (X.EnableBackprop) {
                X.AddGrad(-Sin(X.Value) * Y.Grad);
            }
        }
    }
}
