namespace TensorShader {
    public abstract partial class VariableNode {
        /// <summary>大なりイコール</summary>
        public static VariableNode GreaterThanOrEqual(VariableNode x1, VariableNode x2) {
            return BinaryArithmetric(x1, x2, new Operators.LogicalArithmetric.GreaterThanOrEqual(x1.Shape));
        }

        /// <summary>大なりイコール</summary>
        public static VariableNode GreaterThanOrEqual(VariableNode x, float c) {
            return BinaryLeftConstantArithmetric(x, new Operators.LogicalArithmetric.LessThanOrEqualLeftConstant(c, x.Shape));
        }

        /// <summary>大なりイコール</summary>
        public static VariableNode GreaterThanOrEqual(float c, VariableNode x) {
            return BinaryLeftConstantArithmetric(x, new Operators.LogicalArithmetric.GreaterThanOrEqualLeftConstant(c, x.Shape));
        }
    }

    public partial class Tensor {
        /// <summary>大なりイコール</summary>
        public static Tensor GreaterThanOrEqual(Tensor x1, Tensor x2) {
            return BinaryArithmetric(x1, x2, new Operators.LogicalArithmetric.GreaterThanOrEqual(x1.Shape));
        }

        /// <summary>大なりイコール</summary>
        public static Tensor GreaterThanOrEqual(Tensor x, float c) {
            return BinaryLeftConstantArithmetric(x, new Operators.LogicalArithmetric.LessThanOrEqualLeftConstant(c, x.Shape));
        }

        /// <summary>大なりイコール</summary>
        public static Tensor GreaterThanOrEqual(float c, Tensor x) {
            return BinaryLeftConstantArithmetric(x, new Operators.LogicalArithmetric.GreaterThanOrEqualLeftConstant(c, x.Shape));
        }
    }
}
