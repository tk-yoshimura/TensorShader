namespace TensorShader {
    public abstract partial class VariableNode {
        /// <summary>双曲線余弦関数</summary>
        public static VariableNode Cosh(VariableNode x) {
            return UnaryArithmetric(x, new Operators.UnaryArithmetric.Cosh(x.Shape));
        }
    }

    public partial class Tensor {
        /// <summary>双曲線余弦関数</summary>
        public static Tensor Cosh(Tensor x) {
            return UnaryArithmetric(x, new Operators.UnaryArithmetric.Cosh(x.Shape));
        }
    }
}
