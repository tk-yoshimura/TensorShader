namespace TensorShader {
    public abstract partial class VariableNode {
        /// <summary>複素数正規化</summary>
        public static VariableNode ComplexNormalize(VariableNode v) {
            return ComplexUnaryArithmetric(v, new Operators.ComplexUnaryArithmetric.ComplexNormalize(v.Shape));
        }
    }

    public partial class Tensor {
        /// <summary>複素数正規化</summary>
        public static Tensor ComplexNormalize(Tensor v) {
            return ComplexUnaryArithmetric(v, new Operators.ComplexUnaryArithmetric.ComplexNormalize(v.Shape));
        }
    }
}
