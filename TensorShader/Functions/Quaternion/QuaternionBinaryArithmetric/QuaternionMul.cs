namespace TensorShader {
    public abstract partial class VariableNode {
        /// <summary>四元数積</summary>
        public static VariableNode QuaternionMul(VariableNode x1, VariableNode x2) {
            return QuaternionBinaryArithmetric(x1, x2, new Operators.QuaternionBinaryArithmetric.QuaternionMul(x1.Shape));
        }
    }

    public partial class Tensor {
        /// <summary>四元数積</summary>
        public static Tensor QuaternionMul(Tensor x1, Tensor x2) {
            return QuaternionBinaryArithmetric(x1, x2, new Operators.QuaternionBinaryArithmetric.QuaternionMul(x1.Shape));
        }
    }
}
