namespace TensorShader {
    public abstract partial class VariableNode {
        /// <summary>2進対数</summary>
        public static VariableNode Log2(VariableNode x) {
            return UnaryArithmetric(x, new Operators.UnaryArithmetric.Log2(x.Shape));
        }
    }

    public partial class Tensor {
        /// <summary>2進対数</summary>
        public static Tensor Log2(Tensor x) {
            return UnaryArithmetric(x, new Operators.UnaryArithmetric.Log2(x.Shape));
        }
    }
}
