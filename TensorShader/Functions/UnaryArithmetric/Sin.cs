namespace TensorShader {
    public abstract partial class VariableNode {
        /// <summary>正弦関数</summary>
        public static VariableNode Sin(VariableNode x) {
            return UnaryArithmetric(x, new Operators.UnaryArithmetric.Sin(x.Shape));
        }
    }

    public partial class Tensor {
        /// <summary>正弦関数</summary>
        public static Tensor Sin(Tensor x) {
            return UnaryArithmetric(x, new Operators.UnaryArithmetric.Sin(x.Shape));
        }
    }
}
