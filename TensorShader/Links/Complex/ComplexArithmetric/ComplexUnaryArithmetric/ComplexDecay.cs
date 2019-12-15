using static TensorShader.VariableNode;

namespace TensorShader {
    public partial class Field {
        /// <summary>複素数減衰</summary>
        public static Field ComplexDecay(Field x) {
            Field y = new Field();
            Link link = new Links.ComplexArithmetric.ComplexDecay(x, y);

            link.Forward();

            return y;
        }
    }
}

namespace TensorShader.Links.ComplexArithmetric {
    /// <summary>複素数減衰</summary>
    public class ComplexDecay : UnaryArithmetric.UnaryArithmetric {
        /// <summary>コンストラクタ</summary>
        public ComplexDecay(Field infield, Field outfield)
            : base(infield, outfield) { }

        /// <summary>順伝搬</summary>
        public override void Forward() {
            Y.AssignValue(ComplexDecay(X.Value));
        }

        /// <summary>逆伝搬</summary>
        public override void Backward() {
            if (Y.Grad == null) {
                return;
            }

            if (X.EnableBackprop) {
                X.AddGrad(ComplexDecayGrad(Y.Grad, X.Value));
            }
        }
    }
}
