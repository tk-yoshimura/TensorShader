namespace TensorShader {
    public abstract partial class VariableNode {
        /// <summary>逆正弦関数</summary>
        public static VariableNode Arcsin(VariableNode x) {
            return UnaryArithmetric(x, new Operators.UnaryArithmetric.Arcsin(x.Shape));
        }
    }

    public partial class Tensor {
        /// <summary>逆正弦関数</summary>
        public static Tensor Arcsin(Tensor x) {
            return UnaryArithmetric(x, new Operators.UnaryArithmetric.Arcsin(x.Shape));
        }
    }
}
