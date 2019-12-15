namespace TensorShader {
    public abstract partial class VariableNode {
        /// <summary>加算</summary>
        public static VariableNode Add(VariableNode x, float c) {
            return UnaryArithmetric(x, new Operators.UnaryArithmetric.Add(c, x.Shape));
        }

        /// <summary>加算</summary>
        public static VariableNode Add(float c, VariableNode x) {
            return Add(x, c);
        }

        /// <summary>加算</summary>
        public static VariableNode operator +(VariableNode x, float c) {
            return Add(x, c);
        }

        /// <summary>加算</summary>
        public static VariableNode operator +(float c, VariableNode x) {
            return Add(c, x);
        }
    }

    public partial class Tensor {
        /// <summary>加算</summary>
        public static Tensor Add(Tensor x, float c) {
            return UnaryArithmetric(x, new Operators.UnaryArithmetric.Add(c, x.Shape));
        }

        /// <summary>加算</summary>
        public static Tensor Add(float c, Tensor x) {
            return Add(x, c);
        }

        /// <summary>加算</summary>
        public static Tensor operator +(Tensor x, float c) {
            return Add(x, c);
        }

        /// <summary>加算</summary>
        public static Tensor operator +(float c, Tensor x) {
            return Add(c, x);
        }
    }
}
