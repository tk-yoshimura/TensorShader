namespace TensorShader {
    public abstract partial class VariableNode {
        /// <summary>常用対数</summary>
        public static VariableNode Log10(VariableNode x) {
            return UnaryArithmetric(x, new Operators.UnaryArithmetric.Log10(x.Shape));
        }
    }

    public partial class Tensor {
        /// <summary>常用対数</summary>
        public static Tensor Log10(Tensor x) {
            return UnaryArithmetric(x, new Operators.UnaryArithmetric.Log10(x.Shape));
        }
    }
}
