using static TensorShader.VariableNode;

namespace TensorShader {
    public partial class Field {
        /// <summary>3次元ベクトル減衰</summary>
        public static Field TrivectorDecay(Field x) {
            Field y = new Field();
            Link link = new Links.TrivectorArithmetric.TrivectorDecay(x, y);

            link.Forward();

            return y;
        }
    }
}

namespace TensorShader.Links.TrivectorArithmetric {
    /// <summary>3次元ベクトル減衰</summary>
    public class TrivectorDecay : UnaryArithmetric.UnaryArithmetric {
        /// <summary>コンストラクタ</summary>
        public TrivectorDecay(Field infield, Field outfield)
            : base(infield, outfield) { }

        /// <summary>順伝搬</summary>
        public override void Forward() {
            Y.AssignValue(TrivectorDecay(X.Value));
        }

        /// <summary>逆伝搬</summary>
        public override void Backward() {
            if (Y.Grad == null) {
                return;
            }

            if (X.EnableBackprop) {
                X.AddGrad(TrivectorDecayGrad(Y.Grad, X.Value));
            }
        }
    }
}
