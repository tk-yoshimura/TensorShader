using static TensorShader.VariableNode;

namespace TensorShader {
    public partial class Field {
        /// <summary>外積</summary>
        public static Field TrivectorCross(Field v, Field u) {
            Field y = new();
            Link link = new Links.QuaternionArithmetric.TrivectorCross(v, u, y);

            link.Forward();

            return y;
        }
    }
}

namespace TensorShader.Links.QuaternionArithmetric {
    /// <summary>外積</summary>
    public class TrivectorCross : BinaryArithmetric.BinaryArithmetric {
        /// <summary>コンストラクタ</summary>
        public TrivectorCross(Field infield1, Field infield2, Field outfield)
            : base(infield1, infield2, outfield) { }

        /// <summary>順伝搬</summary>
        public override void Forward() {
            Y.AssignValue(TrivectorCross(X1.Value, X2.Value));
        }

        /// <summary>逆伝搬</summary>
        public override void Backward() {
            if (Y.Grad is null) {
                return;
            }

            if (X1.EnableBackprop) {
                X1.AddGrad(TrivectorCross(X2.Value, Y.Grad));
            }

            if (X2.EnableBackprop) {
                X2.AddGrad(TrivectorCross(Y.Grad, X1.Value));
            }
        }
    }
}
