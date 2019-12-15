namespace TensorShader {
    public abstract partial class VariableNode {
        /// <summary>立方根</summary>
        public static VariableNode Cbrt(VariableNode x) {
            return UnaryArithmetric(x, new Operators.UnaryArithmetric.Cbrt(x.Shape));
        }
    }

    public partial class Tensor {
        /// <summary>立方根</summary>
        public static Tensor Cbrt(Tensor x) {
            return UnaryArithmetric(x, new Operators.UnaryArithmetric.Cbrt(x.Shape));
        }
    }
}
