using static TensorShader.VariableNode;

namespace TensorShader {
    public partial class Field {
        /// <summary>LeakyRelu</summary>
        public static Field LeakyRelu(Field x, float slope) {
            Field y = new Field();
            Link link = new Links.UnaryArithmetric.LeakyRelu(x, y, slope);

            link.Forward();

            return y;
        }
    }
}

namespace TensorShader.Links.UnaryArithmetric {
    /// <summary>LeakyRelu</summary>
    internal class LeakyRelu : UnaryArithmetric {
        /// <summary>負値乗数</summary>
        public float Slope { private set; get; }

        /// <summary>コンストラクタ</summary>
        public LeakyRelu(Field infield, Field outfield, float slope)
            : base(infield, outfield) {

            this.Slope = slope;
        }

        /// <summary>順伝搬</summary>
        public override void Forward() {
            Y.AssignValue(LeakyRelu(X.Value, Slope));
        }

        /// <summary>逆伝搬</summary>
        public override void Backward() {
            if (Y.Grad == null) {
                return;
            }

            if (X.EnableBackprop) {
                X.AddGrad(LeakyReluGrad(Slope, Y.Grad, X.Value));
            }
        }
    }
}
