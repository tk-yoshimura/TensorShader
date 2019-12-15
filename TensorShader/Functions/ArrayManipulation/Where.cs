using System;

namespace TensorShader {
    public abstract partial class VariableNode {
        /// <summary>条件選択</summary>
        public static VariableNode Where(VariableNode condition, VariableNode x1, VariableNode x2, Shape shape) {
            Function function = new Functions.ArrayManipulation.Where(shape);

            return Apply(function, condition, x1, x2)[0];
        }
    }

    public partial class Tensor {
        /// <summary>条件選択</summary>
        public static Tensor Where(Tensor condition, Tensor x1, Tensor x2, Shape shape) {
            Function function = new Functions.ArrayManipulation.Where(shape);

            Tensor y = new Tensor(shape);

            function.Execute(new Tensor[] { condition, x1, x2 }, new Tensor[] { y });

            return y;
        }
    }
}

namespace TensorShader.Functions.ArrayManipulation {
    /// <summary>条件選択</summary>
    internal class Where : Function {
        /// <summary>出力形状</summary>
        public Shape Shape { private set; get; }

        /// <summary>コンストラクタ</summary>
        public Where(Shape shape)
            : base(inputs: 3, outputs: 1, allow_resubstitution: false) {
            this.Shape = shape;
        }

        /// <summary>出力テンソル形状を返す</summary>
        public override Shape[] OutputShapes(params Shape[] inshapes) {
            CheckInputShapes(inshapes);

            return new Shape[] { Shape };
        }

        public override void CheckInputShapes(params Shape[] inshapes) {
            base.CheckInputShapes(inshapes);

            for (int i = 1; i < inshapes.Length; i++) {
                if (inshapes[i] != inshapes[0]) {
                    throw new ArgumentException(ExceptionMessage.ShapeWithIndex(index: i, inshapes[i], inshapes[0]));
                }
            }
        }

        /// <summary>操作クラスを生成する</summary>
        internal override (Tensor[] tensors, Operator ope) GenerateOperator(Tensor[] intensors, Tensor[] outtensors) {
            CheckArgumentsCount(intensors, outtensors);

            Tensor incondition = intensors[0], intensor1 = intensors[1], intensor2 = intensors[2], outtensor = outtensors[0];

            return (
                new Tensor[] { incondition, intensor1, intensor2, outtensor },
                new Operators.ArrayManipulation.Where(incondition.Shape)
                );
        }
    }
}
