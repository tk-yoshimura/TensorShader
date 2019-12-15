namespace TensorShader {
    public abstract partial class VariableNode {
        /// <summary>四元数減衰</summary>
        /// <remarks>(norm/(norm+1))を乗ずる</remarks>
        public static VariableNode QuaternionDecay(VariableNode v) {
            return QuaternionUnaryArithmetric(v, new Operators.QuaternionUnaryArithmetric.QuaternionDecay(v.Shape));
        }
    }

    public partial class Tensor {
        /// <summary>四元数減衰</summary>
        /// <remarks>(norm/(norm+1))を乗ずる</remarks>
        public static Tensor QuaternionDecay(Tensor v) {
            return QuaternionUnaryArithmetric(v, new Operators.QuaternionUnaryArithmetric.QuaternionDecay(v.Shape));
        }
    }
}
