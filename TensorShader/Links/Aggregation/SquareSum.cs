using static TensorShader.VariableNode;

namespace TensorShader {
    public partial class Field {
        /// <summary>2乗和</summary>
        public static Field SquareSum(Field x, int[] axes = null, bool keepdims = false) {
            Field y = new Field();
            Link link = new Links.Aggregation.SquareSum(x, y, axes, keepdims);

            link.Forward();

            return y;
        }

        /// <summary>2乗和</summary>
        public static Field SquareSum(Field x, int axis, bool keepdims = false) {
            return SquareSum(x, new int[] { axis }, keepdims);
        }
    }
}

namespace TensorShader.Links.Aggregation {
    /// <summary>2乗和</summary>
    public class SquareSum : Aggregation {
        /// <summary>コンストラクタ</summary>
        public SquareSum(Field infield, Field outfield, int[] axes = null, bool keepdims = false)
            : base(infield, outfield, axes, keepdims) { }

        /// <summary>順伝搬</summary>
        public override void Forward() {
            Y.AssignValue(SquareSum(X.Value, Axes, KeepDims));
        }

        /// <summary>逆伝搬</summary>
        public override void Backward() {
            if (Y.Grad == null) {
                return;
            }

            if (X.EnableBackprop) {
                VariableNode g = AdjustShape(Y.Grad, X.Shape);

                X.AddGrad(2 * X.Value * g);
            }
        }
    }
}
