using System;

namespace TensorShader {
    public abstract partial class VariableNode {
        /// <summary>チャネル拡張</summary>
        public static VariableNode ExpandChannel(VariableNode x, int expands) {
            Function function = new Functions.ArrayManipulation.ExpandChannel(x.Shape, expands);

            return Apply(function, x)[0];
        }
    }

    public partial class Tensor {
        /// <summary>チャネル拡張</summary>
        public static Tensor ExpandChannel(Tensor x, int expands) {
            Function function = new Functions.ArrayManipulation.ExpandChannel(x.Shape, expands);

            Tensor y = new(function.OutputShapes(x.Shape)[0]);

            function.Execute(new Tensor[] { x }, new Tensor[] { y });

            return y;
        }
    }
}

namespace TensorShader.Functions.ArrayManipulation {
    /// <summary>チャネル拡張</summary>
    internal class ExpandChannel : Function {
        /// <summary>出力形状</summary>
        public Shape Shape { private set; get; }

        /// <summary>拡張数</summary>
        public int Expands { private set; get; }

        /// <summary>コンストラクタ</summary>
        public ExpandChannel(Shape shape, int expands)
            : base(inputs: 1, outputs: 1, allow_resubstitution: false) {

            this.Shape = shape;
            this.Expands = expands;
        }

        /// <summary>出力テンソル形状を返す</summary>
        public override Shape[] OutputShapes(params Shape[] inshapes) {
            CheckInputShapes(inshapes);

            Shape inshape = inshapes[0];

            int[] s = inshape;
            s[0] *= Expands;

            return new Shape[] { new Shape(inshape.Type, s) };
        }

        public override void CheckInputShapes(params Shape[] inshapes) {
            base.CheckInputShapes(inshapes);

            if (inshapes[0].Ndim <= 0) {
                throw new ArgumentException(ExceptionMessage.Shape("Ndim", inshapes[0]));
            }
        }

        /// <summary>操作クラスを生成する</summary>
        internal override (Tensor[] tensors, Operator ope) GenerateOperator(Tensor[] intensors, Tensor[] outtensors) {
            CheckArgumentsCount(intensors, outtensors);

            Tensor intensor = intensors[0], outtensor = outtensors[0];

            return (
                new Tensor[] { intensor, outtensor },
                new Operators.ArrayManipulation.ExpandChannel(intensor.Shape, Expands)
                );
        }
    }
}
