namespace TensorShader {
    public abstract partial class VariableNode {
        /// <summary>ノットイコール</summary>
        public static VariableNode NotEqual(VariableNode x1, VariableNode x2) {
            return BinaryArithmetric(x1, x2, new Operators.LogicalArithmetric.NotEqual(x1.Shape));
        }

        /// <summary>ノットイコール</summary>
        public static VariableNode NotEqual(VariableNode x, float c) {
            return BinaryLeftConstantArithmetric(x, new Operators.LogicalArithmetric.NotEqualConstant(c, x.Shape));
        }

        /// <summary>ノットイコール</summary>
        public static VariableNode NotEqual(float c, VariableNode x) {
            return BinaryLeftConstantArithmetric(x, new Operators.LogicalArithmetric.NotEqualConstant(c, x.Shape));
        }
    }

    public partial class Tensor {
        /// <summary>ノットイコール</summary>
        public static Tensor NotEqual(Tensor x1, Tensor x2) {
            return BinaryArithmetric(x1, x2, new Operators.LogicalArithmetric.NotEqual(x1.Shape));
        }

        /// <summary>ノットイコール</summary>
        public static Tensor NotEqual(Tensor x, float c) {
            return BinaryLeftConstantArithmetric(x, new Operators.LogicalArithmetric.NotEqualConstant(c, x.Shape));
        }

        /// <summary>ノットイコール</summary>
        public static Tensor NotEqual(float c, Tensor x) {
            return BinaryLeftConstantArithmetric(x, new Operators.LogicalArithmetric.NotEqualConstant(c, x.Shape));
        }
    }
}
