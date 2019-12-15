namespace TensorShader {
    public abstract partial class VariableNode {
        /// <summary>最大要素</summary>
        public static VariableNode Maximum(VariableNode x, float c) {
            return UnaryArithmetric(x, new Operators.UnaryArithmetric.Maximum(c, x.Shape));
        }

        /// <summary>最大要素</summary>
        public static VariableNode Maximum(float c, VariableNode x) {
            return Maximum(x, c);
        }
    }

    public partial class Tensor {
        /// <summary>最大要素</summary>
        public static Tensor Maximum(Tensor x, float c) {
            return UnaryArithmetric(x, new Operators.UnaryArithmetric.Maximum(c, x.Shape));
        }

        /// <summary>最大要素</summary>
        public static Tensor Maximum(float c, Tensor x) {
            return Maximum(x, c);
        }
    }
}
