namespace TensorShader {
    public abstract partial class VariableNode {
        /// <summary>逆余弦関数</summary>
        public static VariableNode Arccos(VariableNode x) {
            return UnaryArithmetric(x, new Operators.UnaryArithmetric.Arccos(x.Shape));
        }
    }

    public partial class Tensor {
        /// <summary>逆余弦関数</summary>
        public static Tensor Arccos(Tensor x) {
            return UnaryArithmetric(x, new Operators.UnaryArithmetric.Arccos(x.Shape));
        }
    }
}
