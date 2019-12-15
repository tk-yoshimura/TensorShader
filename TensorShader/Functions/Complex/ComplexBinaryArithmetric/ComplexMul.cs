namespace TensorShader {
    public abstract partial class VariableNode {
        /// <summary>複素積</summary>
        public static VariableNode ComplexMul(VariableNode x1, VariableNode x2) {
            return ComplexBinaryArithmetric(x1, x2, new Operators.ComplexBinaryArithmetric.ComplexMul(x1.Shape));
        }
    }

    public partial class Tensor {
        /// <summary>複素積</summary>
        public static Tensor ComplexMul(Tensor x1, Tensor x2) {
            return ComplexBinaryArithmetric(x1, x2, new Operators.ComplexBinaryArithmetric.ComplexMul(x1.Shape));
        }
    }
}
