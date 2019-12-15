namespace TensorShader {
    public abstract partial class VariableNode {
        /// <summary>Elu</summary>
        public static VariableNode Elu(VariableNode x, float alpha = 1) {
            return UnaryArithmetric(x, new Operators.UnaryArithmetric.Elu(alpha, x.Shape));
        }
    }

    public partial class Tensor {
        /// <summary>Elu</summary>
        public static Tensor Elu(Tensor x, float alpha = 1) {
            return UnaryArithmetric(x, new Operators.UnaryArithmetric.Elu(alpha, x.Shape));
        }
    }
}
