namespace TensorShader {
    public abstract partial class VariableNode {
        /// <summary>平方根</summary>
        public static VariableNode Sqrt(VariableNode x) {
            return UnaryArithmetric(x, new Operators.UnaryArithmetric.Sqrt(x.Shape));
        }
    }

    public partial class Tensor {
        /// <summary>平方根</summary>
        public static Tensor Sqrt(Tensor x) {
            return UnaryArithmetric(x, new Operators.UnaryArithmetric.Sqrt(x.Shape));
        }
    }
}
