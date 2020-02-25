namespace TensorShader {
    public abstract partial class VariableNode {
        /// <summary>3項演算</summary>
        internal static VariableNode TrinaryBiConstantArithmetric(VariableNode x, Operators.TrinaryArithmetric.TrinaryBiConstantArithmetric trinary_operator) {
            Function function = new Functions.TrinaryArithmetric.TrinaryBiConstantArithmetric(trinary_operator);

            return Apply(function, x)[0];
        }
    }

    public partial class Tensor {
        /// <summary>3項演算</summary>
        internal static Tensor TrinaryBiConstantArithmetric(Tensor x, Operators.TrinaryArithmetric.TrinaryBiConstantArithmetric trinary_operator) {
            Function function = new Functions.TrinaryArithmetric.TrinaryBiConstantArithmetric(trinary_operator);

            Shape y_shape = function.OutputShapes(x.Shape)[0];

            Tensor y = new Tensor(y_shape);

            function.Execute(new Tensor[] { x }, new Tensor[] { y });

            return y;
        }
    }
}

namespace TensorShader.Functions.TrinaryArithmetric {
    /// <summary>3項演算</summary>
    internal class TrinaryBiConstantArithmetric : Function {
        private readonly Operator trinary_operator;

        /// <summary>コンストラクタ</summary>
        public TrinaryBiConstantArithmetric(Operators.TrinaryArithmetric.TrinaryBiConstantArithmetric trinary_operator)
            : base(inputs: 1, outputs: 1, allow_resubstitution: true) {

            this.trinary_operator = trinary_operator;
        }

        public override string Name => trinary_operator.Name;

        /// <summary>出力テンソル形状を返す</summary>
        public override Shape[] OutputShapes(params Shape[] inshapes) {
            CheckInputShapes(inshapes);

            Shape shape = inshapes[0];

            return new Shape[] { shape };
        }

        public override void CheckInputShapes(params Shape[] inshapes) {
            base.CheckInputShapes(inshapes);
        }

        /// <summary>操作クラスを生成する</summary>
        internal override (Tensor[] tensors, Operator ope) GenerateOperator(Tensor[] intensors, Tensor[] outtensors) {
            CheckArgumentsCount(intensors, outtensors);

            Tensor intensor = intensors[0], outtensor = outtensors[0];

            return (new Tensor[] { intensor, outtensor }, trinary_operator);
        }
    }
}
