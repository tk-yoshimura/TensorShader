namespace TensorShader {
    public abstract partial class VariableNode {
        /// <summary>べき関数</summary>
        public static VariableNode Pow(VariableNode x, float alpha) {
            return UnaryArithmetric(x, new Operators.UnaryArithmetric.Pow(alpha, x.Shape));
        }
    }

    public partial class Tensor {
        /// <summary>べき関数</summary>
        public static Tensor Pow(Tensor x, float alpha) {
            return UnaryArithmetric(x, new Operators.UnaryArithmetric.Pow(alpha, x.Shape));
        }
    }
}
