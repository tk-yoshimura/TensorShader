namespace TensorShader {
    public abstract partial class VariableNode {
        /// <summary>最小要素</summary>
        public static VariableNode Minimum(VariableNode x, float c) {
            return UnaryArithmetric(x, new Operators.UnaryArithmetric.Minimum(c, x.Shape));
        }

        /// <summary>最小要素</summary>
        public static VariableNode Minimum(float c, VariableNode x) {
            return Minimum(x, c);
        }
    }

    public partial class Tensor {
        /// <summary>最小要素</summary>
        public static Tensor Minimum(Tensor x, float c) {
            return UnaryArithmetric(x, new Operators.UnaryArithmetric.Minimum(c, x.Shape));
        }

        /// <summary>最小要素</summary>
        public static Tensor Minimum(float c, Tensor x) {
            return Minimum(x, c);
        }
    }
}
