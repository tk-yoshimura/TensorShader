namespace TensorShader {
    public abstract partial class VariableNode {
        /// <summary>小なりイコール</summary>
        public static VariableNode LessThanOrEqual(VariableNode x1, VariableNode x2) {
            return BinaryArithmetric(x1, x2, new Operators.LogicalArithmetric.LessThanOrEqual(x1.Shape));
        }

        /// <summary>小なりイコール</summary>
        public static VariableNode LessThanOrEqual(VariableNode x, float c) {
            return BinaryLeftConstantArithmetric(x, new Operators.LogicalArithmetric.GreaterThanOrEqualLeftConstant(c, x.Shape));
        }

        /// <summary>小なりイコール</summary>
        public static VariableNode LessThanOrEqual(float c, VariableNode x) {
            return BinaryLeftConstantArithmetric(x, new Operators.LogicalArithmetric.LessThanOrEqualLeftConstant(c, x.Shape));
        }
    }

    public partial class Tensor {
        /// <summary>小なりイコール</summary>
        public static Tensor LessThanOrEqual(Tensor x1, Tensor x2) {
            return BinaryArithmetric(x1, x2, new Operators.LogicalArithmetric.LessThanOrEqual(x1.Shape));
        }

        /// <summary>小なりイコール</summary>
        public static Tensor LessThanOrEqual(Tensor x, float c) {
            return BinaryLeftConstantArithmetric(x, new Operators.LogicalArithmetric.GreaterThanOrEqualLeftConstant(c, x.Shape));
        }

        /// <summary>小なりイコール</summary>
        public static Tensor LessThanOrEqual(float c, Tensor x) {
            return BinaryLeftConstantArithmetric(x, new Operators.LogicalArithmetric.LessThanOrEqualLeftConstant(c, x.Shape));
        }
    }
}
