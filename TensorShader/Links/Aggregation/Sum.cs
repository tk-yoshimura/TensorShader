using static TensorShader.VariableNode;

namespace TensorShader {
    public partial class Field {
        /// <summary>総和</summary>
        public static Field Sum(Field x, int[] axes = null, bool keepdims = false) {
            Field y = new Field();
            Link link = new Links.Aggregation.Sum(x, y, axes, keepdims);

            link.Forward();

            return y;
        }

        /// <summary>総和</summary>
        public static Field Sum(Field x, int axis, bool keepdims = false) {
            return Sum(x, new int[]{ axis }, keepdims);
        }
    }
}

namespace TensorShader.Links.Aggregation {
    /// <summary>総和</summary>
    public class Sum : Aggregation {
        /// <summary>コンストラクタ</summary>
        public Sum(Field infield, Field outfield, int[] axes = null, bool keepdims = false)
            : base(infield, outfield, axes, keepdims) { }

        /// <summary>順伝搬</summary>
        public override void Forward() {
            Y.AssignValue(Sum(X.Value, Axes, KeepDims));
        }

        /// <summary>逆伝搬</summary>
        public override void Backward() {
            if (Y.Grad == null) {
                return;
            }

            if (X.EnableBackprop) {
                VariableNode g = AdjustShape(Y.Grad, X.Shape);

                X.AddGrad(g);
            }
        }
    }
}
