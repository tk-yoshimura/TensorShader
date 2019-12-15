namespace TensorShader {
    public abstract partial class VariableNode {
        /// <summary>非数をゼロとして返す</summary>
        public static VariableNode NanAsZero(VariableNode x) {
            return UnaryArithmetric(x, new Operators.UnaryArithmetric.NanAsZero(x.Shape));
        }
    }

    public partial class Tensor {
        /// <summary>非数をゼロとして返す</summary>
        public static Tensor NanAsZero(Tensor x) {
            return UnaryArithmetric(x, new Operators.UnaryArithmetric.NanAsZero(x.Shape));
        }
    }
}
