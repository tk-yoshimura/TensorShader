using static TensorShader.VariableNode;

namespace TensorShader {
    public partial class Field {
        /// <summary>3次元ベクトル正規化</summary>
        public static Field TrivectorNormalize(Field x) {
            Field y = new Field();
            Link link = new Links.TrivectorArithmetric.TrivectorNormalize(x, y);

            link.Forward();

            return y;
        }
    }
}

namespace TensorShader.Links.TrivectorArithmetric {
    /// <summary>3次元ベクトル正規化</summary>
    public class TrivectorNormalize : UnaryArithmetric.UnaryArithmetric {
        /// <summary>コンストラクタ</summary>
        public TrivectorNormalize(Field infield, Field outfield)
            : base(infield, outfield) { }

        /// <summary>順伝搬</summary>
        public override void Forward() {
            Y.AssignValue(TrivectorNormalize(X.Value));
        }

        /// <summary>逆伝搬</summary>
        public override void Backward() {
            if (Y.Grad == null) {
                return;
            }

            if (X.EnableBackprop) {
                X.AddGrad(TrivectorNormalizeGrad(Y.Grad, X.Value));
            }
        }
    }
}
