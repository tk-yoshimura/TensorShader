namespace TensorShader {
    public abstract partial class VariableNode {
        /// <summary>複素数に1/(1+sqrt(norm))を乗ずる</summary>
        public static VariableNode ComplexSquash(VariableNode v) {
            return ComplexUnaryArithmetric(v, new Operators.ComplexUnaryArithmetric.ComplexSquash(v.Shape));
        }
    }

    public partial class Tensor {
        /// <summary>複素数に1/(1+sqrt(norm))を乗ずる</summary>
        public static Tensor ComplexSquash(Tensor v) {
            return ComplexUnaryArithmetric(v, new Operators.ComplexUnaryArithmetric.ComplexSquash(v.Shape));
        }
    }
}
