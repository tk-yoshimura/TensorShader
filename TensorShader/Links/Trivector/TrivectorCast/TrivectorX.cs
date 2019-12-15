using static TensorShader.VariableNode;

namespace TensorShader {
    public partial class Field {
        /// <summary>3次元ベクトルX成分</summary>
        public static Field TrivectorX(Field x) {
            Field y = new Field();
            Link link = new Links.TrivectorArithmetric.TrivectorX(x, y);

            link.Forward();

            return y;
        }
    }
}

namespace TensorShader.Links.TrivectorArithmetric {
    /// <summary>3次元ベクトルX成分</summary>
    public class TrivectorX : UnaryArithmetric.UnaryArithmetric {
        /// <summary>コンストラクタ</summary>
        public TrivectorX(Field infield, Field outfield)
            : base(infield, outfield) { }

        /// <summary>順伝搬</summary>
        public override void Forward() {
            Y.AssignValue(TrivectorX(X.Value));
        }

        /// <summary>逆伝搬</summary>
        public override void Backward() {
            if (Y.Grad == null) {
                return;
            }

            if (X.EnableBackprop) {
                X.AddGrad(TrivectorPureX(Y.Grad));
            }
        }
    }
}
