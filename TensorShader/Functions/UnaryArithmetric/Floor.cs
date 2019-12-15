namespace TensorShader {
    public abstract partial class VariableNode {
        /// <summary>切り捨て関数</summary>
        public static VariableNode Floor(VariableNode x) {
            return UnaryArithmetric(x, new Operators.UnaryArithmetric.Floor(x.Shape));
        }
    }

    public partial class Tensor {
        /// <summary>切り捨て関数</summary>
        public static Tensor Floor(Tensor x) {
            return UnaryArithmetric(x, new Operators.UnaryArithmetric.Floor(x.Shape));
        }
    }
}
