using static TensorShader.VariableNode;

namespace TensorShader {
    public partial class Field {
        /// <summary>平均</summary>
        public static Field Average(Field x, int[] axes = null, bool keepdims = false) {
            Field y = new();
            Link link = new Links.Aggregation.Average(x, y, axes, keepdims);

            link.Forward();

            return y;
        }

        /// <summary>平均</summary>
        public static Field Average(Field x, int axis, bool keepdims = false) {
            return Average(x, new int[] { axis }, keepdims);
        }
    }
}

namespace TensorShader.Links.Aggregation {
    /// <summary>平均</summary>
    public class Average : Aggregation {
        /// <summary>コンストラクタ</summary>
        public Average(Field infield, Field outfield, int[] axes = null, bool keepdims = false)
            : base(infield, outfield, axes, keepdims) { }

        /// <summary>順伝搬</summary>
        public override void Forward() {
            Y.AssignValue(Average(X.Value, Axes, KeepDims));
        }

        /// <summary>逆伝搬</summary>
        public override void Backward() {
            if (Y.Grad is null) {
                return;
            }

            if (X.EnableBackprop) {
                VariableNode g = AdjustShape(Y.Grad, X.Shape);

                X.AddGrad(g / (X.Shape.Length / Y.Shape.Length));
            }
        }
    }
}
