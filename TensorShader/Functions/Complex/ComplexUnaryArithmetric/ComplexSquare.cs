namespace TensorShader {
    public abstract partial class VariableNode {
        /// <summary>複素2乗</summary>
        public static VariableNode ComplexSquare(VariableNode x) {
            return ComplexUnaryArithmetric(x, new Operators.ComplexUnaryArithmetric.ComplexSquare(x.Shape));
        }
    }

    public partial class Tensor {
        /// <summary>複素2乗</summary>
        public static Tensor ComplexSquare(Tensor x) {
            return ComplexUnaryArithmetric(x, new Operators.ComplexUnaryArithmetric.ComplexSquare(x.Shape));
        }
    }
}
