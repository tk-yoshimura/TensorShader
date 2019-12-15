namespace TensorShader {
    public abstract partial class VariableNode {
        /// <summary>指数関数</summary>
        public static VariableNode Exp(VariableNode x) {
            return UnaryArithmetric(x, new Operators.UnaryArithmetric.Exp(x.Shape));
        }
    }

    public partial class Tensor {
        /// <summary>指数関数</summary>
        public static Tensor Exp(Tensor x) {
            return UnaryArithmetric(x, new Operators.UnaryArithmetric.Exp(x.Shape));
        }
    }
}
