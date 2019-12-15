namespace TensorShader {
    public abstract partial class VariableNode {
        /// <summary>四元数正規化勾配</summary>
        internal static VariableNode QuaternionNormalizeGrad(VariableNode x1, VariableNode x2) {
            return QuaternionBinaryArithmetric(x1, x2, new Operators.QuaternionBinaryArithmetric.QuaternionNormalizeGrad(x1.Shape));
        }
    }

    public partial class Tensor {
        /// <summary>四元数正規化勾配</summary>
        internal static Tensor QuaternionNormalizeGrad(Tensor x1, Tensor x2) {
            return QuaternionBinaryArithmetric(x1, x2, new Operators.QuaternionBinaryArithmetric.QuaternionNormalizeGrad(x1.Shape));
        }
    }
}
