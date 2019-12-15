namespace TensorShader {
    public abstract partial class VariableNode {
        /// <summary>四元数積勾配</summary>
        internal static VariableNode QuaternionMulGrad(VariableNode x1, VariableNode x2) {
            return QuaternionBinaryArithmetric(x1, x2, new Operators.QuaternionBinaryArithmetric.QuaternionMulGrad(x1.Shape));
        }
    }

    public partial class Tensor {
        /// <summary>四元数積勾配</summary>
        internal static Tensor QuaternionMulGrad(Tensor x1, Tensor x2) {
            return QuaternionBinaryArithmetric(x1, x2, new Operators.QuaternionBinaryArithmetric.QuaternionMulGrad(x1.Shape));
        }
    }
}
