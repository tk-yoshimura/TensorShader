using static TensorShader.VariableNode;

namespace TensorShader {
    public partial class Field {
        /// <summary>絶対値制限</summary>
        public static Field LimitAbs(Field x, Field range) {
            Field y = new();
            Link link = new Links.BinaryArithmetric.LimitAbs(x, range, y);

            link.Forward();

            return y;
        }
    }
}

namespace TensorShader.Links.BinaryArithmetric {
    /// <summary>絶対値制限</summary>
    internal class LimitAbs : BinaryArithmetric {
        /// <summary>コンストラクタ</summary>
        public LimitAbs(Field infield1, Field infield2, Field outfield)
            : base(infield1, infield2, outfield) { }

        /// <summary>順伝搬</summary>
        public override void Forward() {
            Y.AssignValue(LimitAbs(X1.Value, X2.Value));
        }

        /// <summary>逆伝搬</summary>
        public override void Backward() {
            if (Y.Grad is null) {
                return;
            }

            if (X1.EnableBackprop) {
                X1.AddGrad(AdjectShape(Y.Grad * Equal(X1.Value, Y.Value), X1.Shape));
            }

            if (X2.EnableBackprop) {
                X2.AddGrad(AdjectShape(Y.Grad * Sign(X1.Value - X2.Value) * NotEqual(X1.Value, Y.Value), X2.Shape));
            }
        }
    }
}
