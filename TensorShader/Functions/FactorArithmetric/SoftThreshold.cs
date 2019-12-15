namespace TensorShader {
    public abstract partial class VariableNode {
        /// <summary>軟判別しきい値関数</summary>
        internal static VariableNode SoftThreshold(VariableNode x, VariableNode alpha) {
            return Sign(x) * Maximum(Abs(x) - alpha, 0);
        }
    }

    public partial class Tensor {
        /// <summary>軟判別しきい値関数</summary>
        internal static Tensor SoftThreshold(Tensor x, Tensor alpha) {
            return Sign(x) * Maximum(Abs(x) - alpha, 0);
        }
    }
}
