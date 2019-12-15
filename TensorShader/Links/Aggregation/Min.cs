using static TensorShader.VariableNode;

namespace TensorShader {
    public partial class Field {
        /// <summary>最小値</summary>
        public static Field Min(Field x, int[] axes = null, bool keepdims = false) {
            Field y = new Field();
            Link link = new Links.Aggregation.Min(x, y, axes, keepdims);

            link.Forward();

            return y;
        }
    }
}

namespace TensorShader.Links.Aggregation {
    /// <summary>最小値</summary>
    public class Min : Aggregation {
        /// <summary>コンストラクタ</summary>
        public Min(Field infield, Field outfield, int[] axes = null, bool keepdims = false)
            : base(infield, outfield, axes, keepdims) { }

        /// <summary>順伝搬</summary>
        public override void Forward() {
            Y.AssignValue(Min(X.Value, Axes, KeepDims));
        }
    }
}
