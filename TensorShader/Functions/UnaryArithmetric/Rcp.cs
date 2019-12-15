namespace TensorShader {
    public abstract partial class VariableNode {
        /// <summary>逆数</summary>
        public static VariableNode Rcp(VariableNode x) {
            return UnaryArithmetric(x, new Operators.UnaryArithmetric.Rcp(x.Shape));
        }
    }

    public partial class Tensor {
        /// <summary>逆数</summary>
        public static Tensor Rcp(Tensor x) {
            return UnaryArithmetric(x, new Operators.UnaryArithmetric.Rcp(x.Shape));
        }
    }
}
