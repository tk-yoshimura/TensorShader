namespace TensorShader {
    public abstract partial class VariableNode {
        /// <summary>四元数Decay勾配</summary>
        internal static VariableNode QuaternionDecayGrad(VariableNode x1, VariableNode x2) {
            return QuaternionBinaryArithmetric(x1, x2, new Operators.QuaternionBinaryArithmetric.QuaternionDecayGrad(x1.Shape));
        }
    }

    public partial class Tensor {
        /// <summary>四元数Decay勾配</summary>
        internal static Tensor QuaternionDecayGrad(Tensor x1, Tensor x2) {
            return QuaternionBinaryArithmetric(x1, x2, new Operators.QuaternionBinaryArithmetric.QuaternionDecayGrad(x1.Shape));
        }
    }
}
