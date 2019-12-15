namespace TensorShader {
    public abstract partial class VariableNode {
        /// <summary>Relu勾配</summary>
        public static VariableNode ReluGrad(VariableNode x1, VariableNode x2) {
            return BinaryArithmetric(x1, x2, new Operators.BinaryArithmetric.ReluGrad(x1.Shape));
        }
    }

    public partial class Tensor {
        /// <summary>Relu勾配</summary>
        public static Tensor ReluGrad(Tensor x1, Tensor x2) {
            return BinaryArithmetric(x1, x2, new Operators.BinaryArithmetric.ReluGrad(x1.Shape));
        }
    }
}
