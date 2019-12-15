namespace TensorShader {
    public abstract partial class VariableNode {
        /// <summary>複素積勾配</summary>
        public static VariableNode ComplexMulGrad(VariableNode x1, VariableNode x2) {
            return ComplexBinaryArithmetric(x1, x2, new Operators.ComplexBinaryArithmetric.ComplexMulGrad(x1.Shape));
        }
    }

    public partial class Tensor {
        /// <summary>複素積勾配</summary>
        public static Tensor ComplexMulGrad(Tensor x1, Tensor x2) {
            return ComplexBinaryArithmetric(x1, x2, new Operators.ComplexBinaryArithmetric.ComplexMulGrad(x1.Shape));
        }
    }
}
