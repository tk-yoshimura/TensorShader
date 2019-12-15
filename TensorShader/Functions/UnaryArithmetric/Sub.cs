namespace TensorShader {
    public abstract partial class VariableNode {
        /// <summary>減算</summary>
        public static VariableNode Sub(VariableNode x, float c) {
            return UnaryArithmetric(x, new Operators.UnaryArithmetric.Add(-c, x.Shape));
        }

        /// <summary>減算</summary>
        public static VariableNode Sub(float c, VariableNode x) {
            return UnaryArithmetric(x, new Operators.UnaryArithmetric.Sub(c, x.Shape));
        }

        /// <summary>減算</summary>
        public static VariableNode operator -(VariableNode x, float c) {
            return Sub(x, c);
        }

        /// <summary>減算</summary>
        public static VariableNode operator -(float c, VariableNode x) {
            return Sub(c, x);
        }
    }

    public partial class Tensor {
        /// <summary>減算</summary>
        public static Tensor Sub(Tensor x, float c) {
            return UnaryArithmetric(x, new Operators.UnaryArithmetric.Add(-c, x.Shape));
        }

        /// <summary>減算</summary>
        public static Tensor Sub(float c, Tensor x) {
            return UnaryArithmetric(x, new Operators.UnaryArithmetric.Sub(c, x.Shape));
        }

        /// <summary>減算</summary>
        public static Tensor operator -(Tensor x, float c) {
            return Sub(x, c);
        }

        /// <summary>減算</summary>
        public static Tensor operator -(float c, Tensor x) {
            return Sub(c, x);
        }
    }
}
