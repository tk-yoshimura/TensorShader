namespace TensorShader {
    public abstract partial class VariableNode {
        /// <summary>余弦関数</summary>
        public static VariableNode Cos(VariableNode x) {
            return UnaryArithmetric(x, new Operators.UnaryArithmetric.Cos(x.Shape));
        }
    }

    public partial class Tensor {
        /// <summary>余弦関数</summary>
        public static Tensor Cos(Tensor x) {
            return UnaryArithmetric(x, new Operators.UnaryArithmetric.Cos(x.Shape));
        }
    }
}
