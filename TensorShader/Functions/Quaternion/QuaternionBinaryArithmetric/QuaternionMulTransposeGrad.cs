namespace TensorShader {
    public abstract partial class VariableNode {
        /// <summary>四元数積転置勾配</summary>
        internal static VariableNode QuaternionMulTransposeGrad(VariableNode x1, VariableNode x2) {
            return QuaternionBinaryArithmetric(x1, x2, new Operators.QuaternionBinaryArithmetric.QuaternionMulTransposeGrad(x1.Shape));
        }
    }

    public partial class Tensor {
        /// <summary>四元数積転置勾配</summary>
        internal static Tensor QuaternionMulTransposeGrad(Tensor x1, Tensor x2) {
            return QuaternionBinaryArithmetric(x1, x2, new Operators.QuaternionBinaryArithmetric.QuaternionMulTransposeGrad(x1.Shape));
        }
    }
}
