namespace TensorShader {
    public abstract partial class VariableNode {
        /// <summary>複素実部Relu</summary>
        /// <remarks>実部のみReluを適用</remarks>
        public static VariableNode ComplexRRelu(VariableNode x) {
            return ComplexUnaryArithmetric(x, new Operators.ComplexUnaryArithmetric.ComplexRRelu(x.Shape));
        }
    }

    public partial class Tensor {
        /// <summary>複素実部Relu</summary>
        /// <remarks>実部のみReluを適用</remarks>
        public static Tensor ComplexRRelu(Tensor x) {
            return ComplexUnaryArithmetric(x, new Operators.ComplexUnaryArithmetric.ComplexRRelu(x.Shape));
        }
    }
}
