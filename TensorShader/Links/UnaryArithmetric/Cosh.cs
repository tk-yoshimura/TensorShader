using static TensorShader.VariableNode;

namespace TensorShader {
    public partial class Field {
        /// <summary>双曲線余弦関数</summary>
        public static Field Cosh(Field x) {
            Field y = new Field();
            Link link = new Links.UnaryArithmetric.Cosh(x, y);

            link.Forward();

            return y;
        }
    }
}

namespace TensorShader.Links.UnaryArithmetric {
    /// <summary>双曲線余弦関数</summary>
    internal class Cosh : UnaryArithmetric {
        /// <summary>コンストラクタ</summary>
        public Cosh(Field infield, Field outfield)
            : base(infield, outfield) { }

        /// <summary>順伝搬</summary>
        public override void Forward() {
            Y.AssignValue(Cosh(X.Value));
        }

        /// <summary>逆伝搬</summary>
        public override void Backward() {
            if (Y.Grad == null) {
                return;
            }

            if (X.EnableBackprop) {
                X.AddGrad(Sinh(X.Value) * Y.Grad);
            }
        }
    }
}
