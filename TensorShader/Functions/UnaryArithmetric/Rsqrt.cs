namespace TensorShader {
    public abstract partial class VariableNode {
        /// <summary>逆平方根</summary>
        public static VariableNode Rsqrt(VariableNode x) {
            return UnaryArithmetric(x, new Operators.UnaryArithmetric.Rsqrt(x.Shape));
        }
    }

    public partial class Tensor {
        /// <summary>逆平方根</summary>
        public static Tensor Rsqrt(Tensor x) {
            return UnaryArithmetric(x, new Operators.UnaryArithmetric.Rsqrt(x.Shape));
        }
    }
}
