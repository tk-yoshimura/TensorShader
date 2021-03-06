using System;

namespace TensorShader {
    public abstract partial class VariableNode {
        /// <summary>ImageToColumn変換</summary>
        public static VariableNode ImageToColumn1D(VariableNode x, int kwidth) {
            Function function =
                new Functions.Connection1D.ImageToColumn(kwidth);

            VariableNode y = Apply(function, x)[0];

            return y;
        }
    }

    public partial class Tensor {
        /// <summary>ImageToColumn変換</summary>
        public static Tensor ImageToColumn1D(Tensor x, int kwidth) {
            Function function =
                new Functions.Connection1D.ImageToColumn(kwidth);

            Tensor y = new(function.OutputShapes(x.Shape)[0]);

            function.Execute(new Tensor[] { x }, new Tensor[] { y });

            return y;
        }
    }
}

namespace TensorShader.Functions.Connection1D {
    /// <summary>ImageToColumn変換</summary>
    internal class ImageToColumn : Function {

        /// <summary>フィルタサイズ</summary>
        public int KernelWidth { private set; get; }

        /// <summary>コンストラクタ</summary>
        public ImageToColumn(int kwidth)
            : base(inputs: 1, outputs: 1, allow_resubstitution: false) {

            this.KernelWidth = kwidth;
        }

        /// <summary>出力テンソル形状を返す</summary>
        public override Shape[] OutputShapes(params Shape[] inshapes) {
            CheckInputShapes(inshapes);

            Shape inshape = inshapes[0];

            int outwidth = inshape.Width - KernelWidth + 1;

            Shape outshape = new(
                ShapeType.Column,
                KernelWidth,
                inshape.Channels,
                outwidth,
                inshape.Batch);

            return new Shape[] { outshape };
        }

        public override void CheckInputShapes(params Shape[] inshapes) {
            base.CheckInputShapes(inshapes);

            if (inshapes[0].Type != ShapeType.Map || inshapes[0].Ndim != 3) {
                throw new ArgumentException(ExceptionMessage.ShapeElements(inshapes[0], ("Ndim", 3), ("Type", ShapeType.Map)));
            }
        }

        /// <summary>操作クラスを生成する</summary>
        internal override (Tensor[] tensors, Operator ope) GenerateOperator(Tensor[] intensors, Tensor[] outtensors) {
            CheckArgumentsCount(intensors, outtensors);

            Shape shape = intensors[0].Shape;

            return (new Tensor[] { intensors[0], outtensors[0] },
                    new Operators.Connection1D.ImageToColumn(
                        shape.Width, shape.Channels,
                        KernelWidth, shape.Batch));
        }
    }
}
