namespace TensorShader {
    public abstract partial class VariableNode {
        /// <summary>LeakyRelu</summary>
        public static VariableNode LeakyRelu(VariableNode x, float slope) {
            return UnaryArithmetric(x, new Operators.UnaryArithmetric.LeakyRelu(slope, x.Shape));
        }
    }

    public partial class Tensor {
        /// <summary>LeakyRelu</summary>
        public static Tensor LeakyRelu(Tensor x, float slope) {
            return UnaryArithmetric(x, new Operators.UnaryArithmetric.LeakyRelu(slope, x.Shape));
        }
    }
}
