namespace TensorShader {
    public abstract partial class VariableNode {
        /// <summary>双曲線正接関数</summary>
        public static VariableNode Tanh(VariableNode x) {
            return UnaryArithmetric(x, new Operators.UnaryArithmetric.Tanh(x.Shape));
        }
    }

    public partial class Tensor {
        /// <summary>双曲線正接関数</summary>
        public static Tensor Tanh(Tensor x) {
            return UnaryArithmetric(x, new Operators.UnaryArithmetric.Tanh(x.Shape));
        }
    }
}
