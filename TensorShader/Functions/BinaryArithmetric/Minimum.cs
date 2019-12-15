namespace TensorShader {
    public abstract partial class VariableNode {
        /// <summary>最小要素</summary>
        public static VariableNode Minimum(VariableNode x1, VariableNode x2) {
            return BinaryArithmetric(x1, x2, new Operators.BinaryArithmetric.Minimum(x1.Shape));
        }
    }

    public partial class Tensor {
        /// <summary>最小要素</summary>
        public static Tensor Minimum(Tensor x1, Tensor x2) {
            return BinaryArithmetric(x1, x2, new Operators.BinaryArithmetric.Minimum(x1.Shape));
        }
    }
}
