using static TensorShader.VariableNode;

namespace TensorShader {
    public partial class Field {
        /// <summary>2乗平均</summary>
        public static Field SquareAverage(Field x, int[] axes = null, bool keepdims = false) {
            Field y = new Field();
            Link link = new Links.Aggregation.SquareAverage(x, y, axes, keepdims);

            link.Forward();

            return y;
        }

        /// <summary>2乗平均</summary>
        public static Field SquareAverage(Field x, int axis, bool keepdims = false) {
            return SquareAverage(x, new int[] { axis }, keepdims);
        }
    }
}

namespace TensorShader.Links.Aggregation {
    /// <summary>2乗平均</summary>
    public class SquareAverage : Aggregation {
        /// <summary>コンストラクタ</summary>
        public SquareAverage(Field infield, Field outfield, int[] axes = null, bool keepdims = false)
            : base(infield, outfield, axes, keepdims) { }

        /// <summary>順伝搬</summary>
        public override void Forward() {
            Y.AssignValue(SquareAverage(X.Value, Axes, KeepDims));
        }

        /// <summary>逆伝搬</summary>
        public override void Backward() {
            if (Y.Grad == null) {
                return;
            }

            if (X.EnableBackprop) {
                VariableNode g = AdjustShape(Y.Grad, X.Shape);

                X.AddGrad(2 * X.Value * g / (X.Shape.Length / Y.Shape.Length));
            }
        }
    }
}
