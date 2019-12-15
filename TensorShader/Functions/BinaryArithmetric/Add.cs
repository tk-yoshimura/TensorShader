namespace TensorShader {
    public abstract partial class VariableNode {
        /// <summary>加算</summary>
        public static VariableNode Add(VariableNode x1, VariableNode x2) {
            if (x1.Shape == x2.Shape) {
                return BinaryArithmetric(x1, x2, new Operators.BinaryArithmetric.Add(x1.Shape));
            }
            if (x1.Shape.Ndim < x2.Shape.Ndim) {
                return BinaryLeftVectorArithmetric(x1, x2, new Operators.BinaryArithmetric.AddLeftVector(x1.Shape, x2.Shape));
            }
            else {
                return BinaryRightVectorArithmetric(x1, x2, new Operators.BinaryArithmetric.AddRightVector(x2.Shape, x1.Shape));
            }
        }

        /// <summary>加算</summary>
        public static VariableNode operator +(VariableNode x1, VariableNode x2) {
            return Add(x1, x2);
        }
    }

    public partial class Tensor {
        /// <summary>加算</summary>
        public static Tensor Add(Tensor x1, Tensor x2) {
            if (x1.Shape == x2.Shape) {
                return BinaryArithmetric(x1, x2, new Operators.BinaryArithmetric.Add(x1.Shape));
            }
            if (x1.Shape.Ndim < x2.Shape.Ndim) {
                return BinaryLeftVectorArithmetric(x1, x2, new Operators.BinaryArithmetric.AddLeftVector(x1.Shape, x2.Shape));
            }
            else {
                return BinaryRightVectorArithmetric(x1, x2, new Operators.BinaryArithmetric.AddRightVector(x2.Shape, x1.Shape));
            }
        }

        /// <summary>加算</summary>
        public static Tensor operator +(Tensor x1, Tensor x2) {
            return Add(x1, x2);
        }
    }
}
