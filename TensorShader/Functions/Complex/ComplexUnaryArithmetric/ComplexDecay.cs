namespace TensorShader {
    public abstract partial class VariableNode {
        /// <summary>複素数減衰</summary>
        /// <remarks>(norm/(norm+1))を乗ずる</remarks>
        public static VariableNode ComplexDecay(VariableNode v) {
            return ComplexUnaryArithmetric(v, new Operators.ComplexUnaryArithmetric.ComplexDecay(v.Shape));
        }
    }

    public partial class Tensor {
        /// <summary>複素数減衰</summary>
        /// <remarks>(norm/(norm+1))を乗ずる</remarks>
        public static Tensor ComplexDecay(Tensor v) {
            return ComplexUnaryArithmetric(v, new Operators.ComplexUnaryArithmetric.ComplexDecay(v.Shape));
        }
    }
}
