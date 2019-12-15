namespace TensorShader {
    public abstract partial class VariableNode {
        /// <summary>除算</summary>
        public static VariableNode Div(VariableNode x, float c) {
            return Mul(x, 1 / c);
        }

        /// <summary>除算</summary>
        public static VariableNode Div(float c, VariableNode x) {
            if (c == 1) {
                return Rcp(x);
            }

            return UnaryArithmetric(x, new Operators.UnaryArithmetric.Div(c, x.Shape));
        }

        /// <summary>除算</summary>
        public static VariableNode operator /(VariableNode x, float c) {
            return Div(x, c);
        }

        /// <summary>除算</summary>
        public static VariableNode operator /(float c, VariableNode x) {
            return Div(c, x);
        }
    }

    public partial class Tensor {
        /// <summary>除算</summary>
        public static Tensor Div(Tensor x, float c) {
            return Mul(x, 1 / c);
        }

        /// <summary>除算</summary>
        public static Tensor Div(float c, Tensor x) {
            if (c == 1) {
                return Rcp(x);
            }

            return UnaryArithmetric(x, new Operators.UnaryArithmetric.Div(c, x.Shape));
        }

        /// <summary>除算</summary>
        public static Tensor operator /(Tensor x, float c) {
            return Div(x, c);
        }

        /// <summary>除算</summary>
        public static Tensor operator /(float c, Tensor x) {
            return Div(c, x);
        }
    }
}
