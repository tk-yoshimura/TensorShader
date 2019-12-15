namespace TensorShader {
    public abstract partial class VariableNode {
        /// <summary>Elu勾配</summary>
        public static VariableNode EluGrad(float slope, VariableNode x1, VariableNode x2) {
            return BinaryArithmetric(x1, x2, new Operators.BinaryArithmetric.EluGrad(slope, x1.Shape));
        }
    }

    public partial class Tensor {
        /// <summary>Elu勾配</summary>
        public static Tensor EluGrad(float slope, Tensor x1, Tensor x2) {
            return BinaryArithmetric(x1, x2, new Operators.BinaryArithmetric.EluGrad(slope, x1.Shape));
        }
    }
}
