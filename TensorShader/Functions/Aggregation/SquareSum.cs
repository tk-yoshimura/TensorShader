namespace TensorShader {
    public abstract partial class VariableNode {
        /// <summary>2乗和</summary>
        public static VariableNode SquareSum(VariableNode x, int[] axes = null, bool keepdims = false) {
            return Sum(Square(x), axes, keepdims);
        }
    }

    public partial class Tensor {
        /// <summary>2乗和</summary>
        public static Tensor SquareSum(Tensor x, int[] axes = null, bool keepdims = false) {
            return Sum(Square(x), axes, keepdims);
        }
    }
}
