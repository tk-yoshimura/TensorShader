namespace TensorShader {
    public abstract partial class VariableNode {
        /// <summary>四元数Squash勾配</summary>
        internal static VariableNode QuaternionSquashGrad(VariableNode x1, VariableNode x2) {
            return QuaternionBinaryArithmetric(x1, x2, new Operators.QuaternionBinaryArithmetric.QuaternionSquashGrad(x1.Shape));
        }
    }

    public partial class Tensor {
        /// <summary>四元数Squash勾配</summary>
        internal static Tensor QuaternionSquashGrad(Tensor x1, Tensor x2) {
            return QuaternionBinaryArithmetric(x1, x2, new Operators.QuaternionBinaryArithmetric.QuaternionSquashGrad(x1.Shape));
        }
    }
}
