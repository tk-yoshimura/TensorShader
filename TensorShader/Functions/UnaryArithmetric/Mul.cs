namespace TensorShader {
    public abstract partial class VariableNode {
        /// <summary>乗算</summary>
        public static VariableNode Mul(VariableNode x, float c) {
            return UnaryArithmetric(x, new Operators.UnaryArithmetric.Mul(c, x.Shape));
        }

        /// <summary>乗算</summary>
        public static VariableNode Mul(float c, VariableNode x) {
            return Mul(x, c);
        }

        /// <summary>乗算</summary>
        public static VariableNode operator *(VariableNode x, float c) {
            return Mul(x, c);
        }

        /// <summary>乗算</summary>
        public static VariableNode operator *(float c, VariableNode x) {
            return Mul(c, x);
        }
    }

    public partial class Tensor {
        /// <summary>乗算</summary>
        public static Tensor Mul(Tensor x, float c) {
            return UnaryArithmetric(x, new Operators.UnaryArithmetric.Mul(c, x.Shape));
        }

        /// <summary>乗算</summary>
        public static Tensor Mul(float c, Tensor x) {
            return Mul(x, c);
        }

        /// <summary>乗算</summary>
        public static Tensor operator *(Tensor x, float c) {
            return Mul(x, c);
        }

        /// <summary>乗算</summary>
        public static Tensor operator *(float c, Tensor x) {
            return Mul(c, x);
        }
    }
}
