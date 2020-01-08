using static TensorShader.VariableNode;

namespace TensorShader {
    public partial class Field {
        /// <summary>最大値</summary>
        public static Field Max(Field x, int[] axes = null, bool keepdims = false) {
            Field y = new Field();
            Link link = new Links.Aggregation.Max(x, y, axes, keepdims);

            link.Forward();

            return y;
        }

        /// <summary>最大値</summary>
        public static Field Max(Field x, int axis, bool keepdims = false) {
            return Max(x, new int[]{ axis }, keepdims);
        }
    }
}

namespace TensorShader.Links.Aggregation {
    /// <summary>最大値</summary>
    public class Max : Aggregation {
        /// <summary>コンストラクタ</summary>
        public Max(Field infield, Field outfield, int[] axes = null, bool keepdims = false)
            : base(infield, outfield, axes, keepdims) { }

        /// <summary>順伝搬</summary>
        public override void Forward() {
            Y.AssignValue(Max(X.Value, Axes, KeepDims));
        }
    }
}
