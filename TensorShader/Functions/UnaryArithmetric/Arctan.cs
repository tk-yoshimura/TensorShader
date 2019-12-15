namespace TensorShader {
    public abstract partial class VariableNode {
        /// <summary>逆正接関数</summary>
        public static VariableNode Arctan(VariableNode x) {
            return UnaryArithmetric(x, new Operators.UnaryArithmetric.Arctan(x.Shape));
        }
    }

    public partial class Tensor {
        /// <summary>逆正接関数</summary>
        public static Tensor Arctan(Tensor x) {
            return UnaryArithmetric(x, new Operators.UnaryArithmetric.Arctan(x.Shape));
        }
    }
}
