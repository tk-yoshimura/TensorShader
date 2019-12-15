namespace TensorShader {
    public abstract partial class VariableNode {
        /// <summary>絶対値</summary>
        public static VariableNode Abs(VariableNode x) {
            return UnaryArithmetric(x, new Operators.UnaryArithmetric.Abs(x.Shape));
        }
    }

    public partial class Tensor {
        /// <summary>絶対値</summary>
        public static Tensor Abs(Tensor x) {
            return UnaryArithmetric(x, new Operators.UnaryArithmetric.Abs(x.Shape));
        }
    }
}
