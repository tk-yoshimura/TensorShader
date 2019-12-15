namespace TensorShader {
    public abstract partial class VariableNode {
        /// <summary>四元数実部Relu</summary>
        /// <remarks>実部のみReluを適用</remarks>
        public static VariableNode QuaternionRRelu(VariableNode x) {
            return QuaternionUnaryArithmetric(x, new Operators.QuaternionUnaryArithmetric.QuaternionRRelu(x.Shape));
        }
    }

    public partial class Tensor {
        /// <summary>四元数実部Relu</summary>
        /// <remarks>実部のみReluを適用</remarks>
        public static Tensor QuaternionRRelu(Tensor x) {
            return QuaternionUnaryArithmetric(x, new Operators.QuaternionUnaryArithmetric.QuaternionRRelu(x.Shape));
        }
    }
}
