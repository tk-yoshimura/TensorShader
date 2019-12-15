namespace TensorShader {
    public abstract partial class VariableNode {
        /// <summary>符号反転</summary>
        public static VariableNode Neg(VariableNode x) {
            return UnaryArithmetric(x, new Operators.UnaryArithmetric.Neg(x.Shape));
        }

        /// <summary>符号反転</summary>
        public static VariableNode operator -(VariableNode x) {
            return Neg(x);
        }
    }

    public partial class Tensor {
        /// <summary>符号反転</summary>
        public static Tensor Neg(Tensor x) {
            return UnaryArithmetric(x, new Operators.UnaryArithmetric.Neg(x.Shape));
        }

        /// <summary>符号反転</summary>
        public static Tensor operator -(Tensor x) {
            return Neg(x);
        }
    }
}
