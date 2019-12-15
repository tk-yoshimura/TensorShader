namespace TensorShader {
    public abstract partial class VariableNode {
        /// <summary>イコール</summary>
        public static VariableNode Equal(VariableNode x1, VariableNode x2) {
            return BinaryArithmetric(x1, x2, new Operators.LogicalArithmetric.Equal(x1.Shape));
        }

        /// <summary>イコール</summary>
        public static VariableNode Equal(VariableNode x, float c) {
            return BinaryLeftConstantArithmetric(x, new Operators.LogicalArithmetric.EqualConstant(c, x.Shape));
        }

        /// <summary>イコール</summary>
        public static VariableNode Equal(float c, VariableNode x) {
            return BinaryLeftConstantArithmetric(x, new Operators.LogicalArithmetric.EqualConstant(c, x.Shape));
        }
    }

    public partial class Tensor {
        /// <summary>イコール</summary>
        public static Tensor Equal(Tensor x1, Tensor x2) {
            return BinaryArithmetric(x1, x2, new Operators.LogicalArithmetric.Equal(x1.Shape));
        }

        /// <summary>イコール</summary>
        public static Tensor Equal(Tensor x, float c) {
            return BinaryLeftConstantArithmetric(x, new Operators.LogicalArithmetric.EqualConstant(c, x.Shape));
        }

        /// <summary>イコール</summary>
        public static Tensor Equal(float c, Tensor x) {
            return BinaryLeftConstantArithmetric(x, new Operators.LogicalArithmetric.EqualConstant(c, x.Shape));
        }
    }
}
