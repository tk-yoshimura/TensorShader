namespace TensorShader {
    public abstract partial class VariableNode {
        /// <summary>除算</summary>
        public static VariableNode Div(VariableNode x1, VariableNode x2) {
            if (x1.Shape == x2.Shape) {
                return BinaryArithmetric(x1, x2, new Operators.BinaryArithmetric.Div(x1.Shape));
            }
            if (x1.Shape.Ndim < x2.Shape.Ndim) {
                return BinaryLeftVectorArithmetric(x1, x2, new Operators.BinaryArithmetric.DivLeftVector(x1.Shape, x2.Shape));
            }
            else {
                return BinaryRightVectorArithmetric(x1, x2, new Operators.BinaryArithmetric.DivRightVector(x2.Shape, x1.Shape));
            }
        }

        /// <summary>除算</summary>
        public static VariableNode operator /(VariableNode x1, VariableNode x2) {
            return Div(x1, x2);
        }
    }

    public partial class Tensor {
        /// <summary>除算</summary>
        public static Tensor Div(Tensor x1, Tensor x2) {
            if (x1.Shape == x2.Shape) {
                return BinaryArithmetric(x1, x2, new Operators.BinaryArithmetric.Div(x1.Shape));
            }
            if (x1.Shape.Ndim < x2.Shape.Ndim) {
                return BinaryLeftVectorArithmetric(x1, x2, new Operators.BinaryArithmetric.DivLeftVector(x1.Shape, x2.Shape));
            }
            else {
                return BinaryRightVectorArithmetric(x1, x2, new Operators.BinaryArithmetric.DivRightVector(x2.Shape, x1.Shape));
            }
        }

        /// <summary>除算</summary>
        public static Tensor operator /(Tensor x1, Tensor x2) {
            return Div(x1, x2);
        }
    }
}
