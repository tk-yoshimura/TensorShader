namespace TensorShader {
    public abstract partial class VariableNode {
        /// <summary>符号</summary>
        public static VariableNode Sign(VariableNode x) {
            return UnaryArithmetric(x, new Operators.UnaryArithmetric.Sign(x.Shape));
        }
    }

    public partial class Tensor {
        /// <summary>符号</summary>
        public static Tensor Sign(Tensor x) {
            return UnaryArithmetric(x, new Operators.UnaryArithmetric.Sign(x.Shape));
        }
    }
}
