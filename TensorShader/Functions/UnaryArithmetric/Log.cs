namespace TensorShader {
    public abstract partial class VariableNode {
        /// <summary>対数関数</summary>
        public static VariableNode Log(VariableNode x) {
            return UnaryArithmetric(x, new Operators.UnaryArithmetric.Log(x.Shape));
        }
    }

    public partial class Tensor {
        /// <summary>対数関数</summary>
        public static Tensor Log(Tensor x) {
            return UnaryArithmetric(x, new Operators.UnaryArithmetric.Log(x.Shape));
        }
    }
}
