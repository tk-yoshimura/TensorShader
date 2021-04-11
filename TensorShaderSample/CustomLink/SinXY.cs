using TensorShader;
using TensorShader.Links.BinaryArithmetric;

using static TensorShader.VariableNode;

namespace CustomLink {

    public static class CustomBinaryArithmetric {
        /// <summary>カスタム2項演算</summary>
        public static Field SinXY(Field x1, Field x2) {
            Field y = new();
            Link link = new SinXY(x1, x2, y);

            link.Forward();

            return y;
        }
    }

    /// <summary>カスタム2項演算</summary>
    internal class SinXY : BinaryArithmetric {

        /// <summary>コンストラクタ</summary>
        public SinXY(Field infield1, Field infield2, Field outfield)
            : base(infield1, infield2, outfield) { }

        /// <summary>順伝搬</summary>
        public override void Forward() {
            Y.AssignValue(BinaryArithmetric(X1.Value, X2.Value, "sinxy_ew", "#y = sinf(#x1 * #x2);"));
        }

        /// <summary>逆伝搬</summary>
        public override void Backward() {
            if (Y.Grad is null) {
                return;
            }

            if (X1.EnableBackprop || X2.EnableBackprop) {
                VariableNode cos_xy = BinaryArithmetric(X1.Value, X2.Value, "cosxy_ew", "#y = cosf(#x1 * #x2);");

                if (X1.EnableBackprop) {
                    X1.AddGrad(Y.Grad * X2.Value * cos_xy);
                }

                if (X2.EnableBackprop) {
                    X2.AddGrad(Y.Grad * X1.Value * cos_xy);
                }
            }
        }
    }
}
