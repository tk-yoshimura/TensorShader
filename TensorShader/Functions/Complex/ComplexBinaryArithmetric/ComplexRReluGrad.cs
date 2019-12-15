namespace TensorShader {
    public abstract partial class VariableNode {
        /// <summary>複素RRelu勾配</summary>
        public static VariableNode ComplexRReluGrad(VariableNode x1, VariableNode x2) {
            return ComplexBinaryArithmetric(x1, x2, new Operators.ComplexBinaryArithmetric.ComplexRReluGrad(x1.Shape));
        }
    }

    public partial class Tensor {
        /// <summary>複素RRelu勾配</summary>
        public static Tensor ComplexRReluGrad(Tensor x1, Tensor x2) {
            return ComplexBinaryArithmetric(x1, x2, new Operators.ComplexBinaryArithmetric.ComplexRReluGrad(x1.Shape));
        }
    }
}
