namespace TensorShader {
    public abstract partial class VariableNode {
        /// <summary>四元数に1/(1+sqrt(norm))を乗ずる</summary>
        public static VariableNode QuaternionSquash(VariableNode v) {
            return QuaternionUnaryArithmetric(v, new Operators.QuaternionUnaryArithmetric.QuaternionSquash(v.Shape));
        }
    }

    public partial class Tensor {
        /// <summary>四元数に1/(1+sqrt(norm))を乗ずる</summary>
        public static Tensor QuaternionSquash(Tensor v) {
            return QuaternionUnaryArithmetric(v, new Operators.QuaternionUnaryArithmetric.QuaternionSquash(v.Shape));
        }
    }
}
