namespace TensorShader {
    public abstract partial class VariableNode {
        /// <summary>複素共役</summary>
        public static VariableNode ComplexConjugate(VariableNode x) {
            return ComplexUnaryArithmetric(x, new Operators.ComplexUnaryArithmetric.ComplexConjugate(x.Shape));
        }
    }

    public partial class Tensor {
        /// <summary>複素共役</summary>
        public static Tensor ComplexConjugate(Tensor x) {
            return ComplexUnaryArithmetric(x, new Operators.ComplexUnaryArithmetric.ComplexConjugate(x.Shape));
        }
    }
}
