namespace TensorShader {
    public abstract partial class VariableNode {
        /// <summary>2乗平均</summary>
        public static VariableNode SquareAverage(VariableNode x, int[] axes = null, bool keepdims = false) {
            return Average(Square(x), axes, keepdims);
        }
    }

    public partial class Tensor {
        /// <summary>2乗平均</summary>
        public static Tensor SquareAverage(Tensor x, int[] axes = null, bool keepdims = false) {
            return Average(Square(x), axes, keepdims);
        }
    }
}
