using static TensorShader.VariableNode;

namespace TensorShader {
    public partial class Field {
        /// <summary>3次元ベクトルY成分</summary>
        public static Field TrivectorY(Field x) {
            Field y = new();
            Link link = new Links.TrivectorArithmetric.TrivectorY(x, y);

            link.Forward();

            return y;
        }
    }
}

namespace TensorShader.Links.TrivectorArithmetric {
    /// <summary>3次元ベクトルY成分</summary>
    public class TrivectorY : UnaryArithmetric.UnaryArithmetric {
        /// <summary>コンストラクタ</summary>
        public TrivectorY(Field infield, Field outfield)
            : base(infield, outfield) { }

        /// <summary>順伝搬</summary>
        public override void Forward() {
            Y.AssignValue(TrivectorY(X.Value));
        }

        /// <summary>逆伝搬</summary>
        public override void Backward() {
            if (Y.Grad is null) {
                return;
            }

            if (X.EnableBackprop) {
                X.AddGrad(TrivectorPureY(Y.Grad));
            }
        }
    }
}
