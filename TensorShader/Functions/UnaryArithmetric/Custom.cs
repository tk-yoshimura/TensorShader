namespace TensorShader {
    public abstract partial class VariableNode {
        /// <summary>カスタム1項演算</summary>
        public static VariableNode UnaryArithmetric(VariableNode x, string funcname, string funccode) {
            return UnaryArithmetric(x, new Operators.UnaryArithmetric.Custom(x.Shape, funcname, funccode));
        }
    }

    public partial class Tensor {
        /// <summary>カスタム1項演算</summary>
        public static Tensor UnaryArithmetric(Tensor x, string funcname, string funccode) {
            return UnaryArithmetric(x, new Operators.UnaryArithmetric.Custom(x.Shape, funcname, funccode));
        }
    }
}
