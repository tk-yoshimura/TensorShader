using static TensorShader.VariableNode;

namespace TensorShader {
    public partial class Field {
        /// <summary>3次元ベクトルに1/(1+sqrt(norm))を乗ずる</summary>
        public static Field TrivectorSquash(Field x) {
            Field y = new Field();
            Link link = new Links.TrivectorArithmetric.TrivectorSquash(x, y);

            link.Forward();

            return y;
        }
    }
}

namespace TensorShader.Links.TrivectorArithmetric {
    /// <summary>3次元ベクトルに1/(1+sqrt(norm))を乗ずる</summary>
    public class TrivectorSquash : UnaryArithmetric.UnaryArithmetric {
        /// <summary>コンストラクタ</summary>
        public TrivectorSquash(Field infield, Field outfield)
            : base(infield, outfield) { }

        /// <summary>順伝搬</summary>
        public override void Forward() {
            Y.AssignValue(TrivectorSquash(X.Value));
        }

        /// <summary>逆伝搬</summary>
        public override void Backward() {
            if (Y.Grad == null) {
                return;
            }

            if (X.EnableBackprop) {
                X.AddGrad(NanAsZero(TrivectorSquashGrad(Y.Grad, X.Value)));
            }
        }
    }
}
