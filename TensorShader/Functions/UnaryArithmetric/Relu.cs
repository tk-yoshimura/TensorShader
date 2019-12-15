namespace TensorShader {
    public abstract partial class VariableNode {
        /// <summary>Relu</summary>
        public static VariableNode Relu(VariableNode x) {
            return UnaryArithmetric(x, new Operators.UnaryArithmetric.Relu(x.Shape));
        }
    }

    public partial class Tensor {
        /// <summary>Relu</summary>
        public static Tensor Relu(Tensor x) {
            return UnaryArithmetric(x, new Operators.UnaryArithmetric.Relu(x.Shape));
        }
    }
}
