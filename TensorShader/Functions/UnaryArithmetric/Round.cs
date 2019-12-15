namespace TensorShader {
    public abstract partial class VariableNode {
        /// <summary>最近傍整数丸め関数</summary>
        public static VariableNode Round(VariableNode x) {
            return UnaryArithmetric(x, new Operators.UnaryArithmetric.Round(x.Shape));
        }
    }

    public partial class Tensor {
        /// <summary>最近傍整数丸め関数</summary>
        public static Tensor Round(Tensor x) {
            return UnaryArithmetric(x, new Operators.UnaryArithmetric.Round(x.Shape));
        }
    }
}
