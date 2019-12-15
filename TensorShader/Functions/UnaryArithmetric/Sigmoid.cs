namespace TensorShader {
    public abstract partial class VariableNode {
        /// <summary>シグモイド関数</summary>
        public static VariableNode Sigmoid(VariableNode x) {
            return UnaryArithmetric(x, new Operators.UnaryArithmetric.Sigmoid(x.Shape));
        }
    }

    public partial class Tensor {
        /// <summary>シグモイド関数</summary>
        public static Tensor Sigmoid(Tensor x) {
            return UnaryArithmetric(x, new Operators.UnaryArithmetric.Sigmoid(x.Shape));
        }
    }
}
