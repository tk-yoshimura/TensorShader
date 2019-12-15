namespace TensorShader {
    public abstract partial class VariableNode {
        /// <summary>複素2乗勾配</summary>
        public static VariableNode ComplexSquareGrad(VariableNode x1, VariableNode x2) {
            return ComplexBinaryArithmetric(x1, x2, new Operators.ComplexBinaryArithmetric.ComplexSquareGrad(x1.Shape));
        }
    }

    public partial class Tensor {
        /// <summary>複素2乗勾配</summary>
        public static Tensor ComplexSquareGrad(Tensor x1, Tensor x2) {
            return ComplexBinaryArithmetric(x1, x2, new Operators.ComplexBinaryArithmetric.ComplexSquareGrad(x1.Shape));
        }
    }
}
