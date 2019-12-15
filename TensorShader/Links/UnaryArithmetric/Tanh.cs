using static TensorShader.VariableNode;

namespace TensorShader {
    public partial class Field {
        /// <summary>双曲線正接関数</summary>
        public static Field Tanh(Field x) {
            Field y = new Field();
            Link link = new Links.UnaryArithmetric.Tanh(x, y);

            link.Forward();

            return y;
        }
    }
}

namespace TensorShader.Links.UnaryArithmetric {
    /// <summary>双曲線正接関数</summary>
    internal class Tanh : UnaryArithmetric {
        /// <summary>コンストラクタ</summary>
        public Tanh(Field infield, Field outfield)
            : base(infield, outfield) { }

        /// <summary>順伝搬</summary>
        public override void Forward() {
            Y.AssignValue(Tanh(X.Value));
        }

        /// <summary>逆伝搬</summary>
        public override void Backward() {
            if (Y.Grad == null) {
                return;
            }

            if (X.EnableBackprop) {
                X.AddGrad((1 - Square(Y.Value)) * Y.Grad);
            }
        }
    }
}
