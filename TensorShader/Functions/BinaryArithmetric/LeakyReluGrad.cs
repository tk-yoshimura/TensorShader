namespace TensorShader {
    public abstract partial class VariableNode {
        /// <summary>LeakyRelu勾配</summary>
        public static VariableNode LeakyReluGrad(float slope, VariableNode x1, VariableNode x2) {
            return BinaryArithmetric(x1, x2, new Operators.BinaryArithmetric.LeakyReluGrad(slope, x1.Shape));
        }
    }

    public partial class Tensor {
        /// <summary>LeakyRelu勾配</summary>
        public static Tensor LeakyReluGrad(float slope, Tensor x1, Tensor x2) {
            return BinaryArithmetric(x1, x2, new Operators.BinaryArithmetric.LeakyReluGrad(slope, x1.Shape));
        }
    }
}
