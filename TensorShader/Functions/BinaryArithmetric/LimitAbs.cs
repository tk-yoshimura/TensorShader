namespace TensorShader {
    public abstract partial class VariableNode {
        /// <summary>絶対値制限</summary>
        public static VariableNode LimitAbs(VariableNode x, VariableNode xrange) {
            return Clamp(x, -xrange, xrange);
        }
    }

    public partial class Tensor {
        /// <summary>絶対値制限</summary>
        public static Tensor LimitAbs(Tensor x, Tensor xrange) {
            return Clamp(x, -xrange, xrange);
        }
    }
}
