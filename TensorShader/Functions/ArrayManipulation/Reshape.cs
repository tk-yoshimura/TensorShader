using System;

namespace TensorShader {
    public abstract partial class VariableNode {
        /// <summary>形状変更</summary>
        public static VariableNode Reshape(VariableNode x, Shape shape) {
            Function function = new Functions.ArrayManipulation.Reshape(shape);

            return Apply(function, x)[0];
        }
    }

    public partial class Tensor {
        /// <summary>形状変更</summary>
        public static Tensor Reshape(Tensor x, Shape shape) {
            Function function = new Functions.ArrayManipulation.Reshape(shape);

            Tensor y = new Tensor(shape);

            function.Execute(new Tensor[] { x }, new Tensor[] { y });

            return y;
        }
    }
}

namespace TensorShader.Functions.ArrayManipulation {
    /// <summary>形状変更</summary>
    internal class Reshape : Function {
        /// <summary>出力形状</summary>
        public Shape OutShape { private set; get; }

        /// <summary>コンストラクタ</summary>
        public Reshape(Shape outshape)
            : base(inputs: 1, outputs: 1, allow_resubstitution: true) {
            this.OutShape = outshape;
        }

        /// <summary>出力テンソル形状を返す</summary>
        public override Shape[] OutputShapes(params Shape[] inshapes) {
            CheckInputShapes(inshapes);

            return new Shape[] { OutShape };
        }

        public override void CheckInputShapes(params Shape[] inshapes) {
            base.CheckInputShapes(inshapes);

            Shape inshape = inshapes[0];

            if (inshape.Length != OutShape.Length) {
                throw new ArgumentException(ExceptionMessage.TensorLength(inshape, OutShape));
            }
        }

        /// <summary>操作クラスを生成する</summary>
        internal override (Tensor[] tensors, Operator ope) GenerateOperator(Tensor[] intensors, Tensor[] outtensors) {
            CheckArgumentsCount(intensors, outtensors);

            Tensor intensor = intensors[0], outtensor = outtensors[0];

            return (new Tensor[] { intensor, outtensor }, new Operators.ArrayManipulation.Reshape(intensor.Shape, outtensor.Shape));
        }
    }
}
