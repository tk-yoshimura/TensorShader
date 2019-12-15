namespace TensorShader {
    public abstract partial class VariableNode {
        /// <summary>複素正規化勾配</summary>
        public static VariableNode ComplexNormalizeGrad(VariableNode x1, VariableNode x2) {
            return ComplexBinaryArithmetric(x1, x2, new Operators.ComplexBinaryArithmetric.ComplexNormalizeGrad(x1.Shape));
        }
    }

    public partial class Tensor {
        /// <summary>複素正規化勾配</summary>
        public static Tensor ComplexNormalizeGrad(Tensor x1, Tensor x2) {
            return ComplexBinaryArithmetric(x1, x2, new Operators.ComplexBinaryArithmetric.ComplexNormalizeGrad(x1.Shape));
        }
    }
}
