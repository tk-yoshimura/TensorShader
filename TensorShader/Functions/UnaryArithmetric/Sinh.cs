namespace TensorShader {
    public abstract partial class VariableNode {
        /// <summary>双曲線正弦関数</summary>
        public static VariableNode Sinh(VariableNode x) {
            return UnaryArithmetric(x, new Operators.UnaryArithmetric.Sinh(x.Shape));
        }
    }

    public partial class Tensor {
        /// <summary>双曲線正弦関数</summary>
        public static Tensor Sinh(Tensor x) {
            return UnaryArithmetric(x, new Operators.UnaryArithmetric.Sinh(x.Shape));
        }
    }
}
