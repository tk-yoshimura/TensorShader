namespace TensorShader {
    public abstract partial class VariableNode {
        /// <summary>最大要素</summary>
        public static VariableNode Maximum(VariableNode x1, VariableNode x2) {
            return BinaryArithmetric(x1, x2, new Operators.BinaryArithmetric.Maximum(x1.Shape));
        }
    }

    public partial class Tensor {
        /// <summary>最大要素</summary>
        public static Tensor Maximum(Tensor x1, Tensor x2) {
            return BinaryArithmetric(x1, x2, new Operators.BinaryArithmetric.Maximum(x1.Shape));
        }
    }
}
