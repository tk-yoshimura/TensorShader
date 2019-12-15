namespace TensorShader {
    public abstract partial class VariableNode {
        /// <summary>四元数RRelu勾配</summary>
        internal static VariableNode QuaternionRReluGrad(VariableNode x1, VariableNode x2) {
            return QuaternionBinaryArithmetric(x1, x2, new Operators.QuaternionBinaryArithmetric.QuaternionRReluGrad(x1.Shape));
        }
    }

    public partial class Tensor {
        /// <summary>四元数RRelu勾配</summary>
        internal static Tensor QuaternionRReluGrad(Tensor x1, Tensor x2) {
            return QuaternionBinaryArithmetric(x1, x2, new Operators.QuaternionBinaryArithmetric.QuaternionRReluGrad(x1.Shape));
        }
    }
}
