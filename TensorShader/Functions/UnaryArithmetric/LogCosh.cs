namespace TensorShader {
    public abstract partial class VariableNode {
        /// <summary>対数双曲線余弦関数</summary>
        public static VariableNode LogCosh(VariableNode x) {
            return UnaryArithmetric(x, new Operators.UnaryArithmetric.LogCosh(x.Shape));
        }
    }

    public partial class Tensor {
        /// <summary>対数双曲線余弦関数</summary>
        public static Tensor LogCosh(Tensor x) {
            return UnaryArithmetric(x, new Operators.UnaryArithmetric.LogCosh(x.Shape));
        }
    }
}
