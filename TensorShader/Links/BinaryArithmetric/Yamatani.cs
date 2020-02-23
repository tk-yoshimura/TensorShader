using static TensorShader.VariableNode;

namespace TensorShader {
    public partial class Field {
        /// <summary>Yamatani斉次応答性活性化関数</summary>
        /// <remarks>
        /// T.Yoshimura 2020
        /// https://www.techrxiv.org/articles/Yamatani_Activation_Edge_Homogeneous_Response_Super_Resolution_Neural_Network/11861187
        /// </remarks>
        public static Field Yamatani(Field x1, Field x2, float slope) {
            Field y = new Field();
            Link link = new Links.BinaryArithmetric.Yamatani(x1, x2, y, slope);

            link.Forward();

            return y;
        }
    }
}

namespace TensorShader.Links.BinaryArithmetric {
    /// <summary>Yamatani</summary>
    internal class Yamatani : BinaryArithmetric {
        /// <summary>Slope</summary>
        public float Slope { private set; get; }

        /// <summary>コンストラクタ</summary>
        public Yamatani(Field infield1, Field infield2, Field outfield, float slope)
            : base(infield1, infield2, outfield) {

            this.Slope = slope;
        }

        /// <summary>順伝搬</summary>
        public override void Forward() {
            Y.AssignValue(Yamatani(X1.Value, X2.Value, Slope));
        }

        /// <summary>逆伝搬</summary>
        public override void Backward() {
            if (Y.Grad == null) {
                return;
            }

            if (X1.EnableBackprop) {
                X1.AddGrad(Y.Grad * YamataniGrad(X1.Value, X2.Value, Slope));
            }

            if (X2.EnableBackprop) {
                X2.AddGrad(Y.Grad * YamataniGrad(X2.Value, X1.Value, Slope));
            }
        }
    }
}
