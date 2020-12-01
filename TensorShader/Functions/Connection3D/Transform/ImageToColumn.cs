using System;

namespace TensorShader {
    public abstract partial class VariableNode {
        /// <summary>ImageToColumn変換</summary>
        public static VariableNode ImageToColumn3D(VariableNode x, int kwidth, int kheight, int kdepth) {
            Function function =
                new Functions.Connection3D.ImageToColumn(kwidth, kheight, kdepth);

            VariableNode y = Apply(function, x)[0];

            return y;
        }
    }

    public partial class Tensor {
        /// <summary>ImageToColumn変換</summary>
        public static Tensor ImageToColumn3D(Tensor x, int kwidth, int kheight, int kdepth) {
            Function function =
                new Functions.Connection3D.ImageToColumn(kwidth, kheight, kdepth);

            Tensor y = new Tensor(function.OutputShapes(x.Shape)[0]);

            function.Execute(new Tensor[] { x }, new Tensor[] { y });

            return y;
        }
    }
}

namespace TensorShader.Functions.Connection3D {
    /// <summary>ImageToColumn変換</summary>
    internal class ImageToColumn : Function {

        /// <summary>フィルタサイズ</summary>
        public int KernelWidth { private set; get; }

        /// <summary>フィルタサイズ</summary>
        public int KernelHeight { private set; get; }

        /// <summary>フィルタサイズ</summary>
        public int KernelDepth { private set; get; }

        /// <summary>コンストラクタ</summary>
        public ImageToColumn(int kwidth, int kheight, int kdepth)
            : base(inputs: 1, outputs: 1, allow_resubstitution: false) {

            this.KernelWidth = kwidth;
            this.KernelHeight = kheight;
            this.KernelDepth = kdepth;
        }

        /// <summary>出力テンソル形状を返す</summary>
        public override Shape[] OutputShapes(params Shape[] inshapes) {
            CheckInputShapes(inshapes);

            Shape inshape = inshapes[0];

            int outwidth = inshape.Width - KernelWidth + 1;
            int outheight = inshape.Height - KernelHeight + 1;
            int outdepth = inshape.Depth - KernelDepth + 1;

            Shape outshape = new Shape(
                ShapeType.Column,
                KernelWidth * KernelHeight * KernelDepth,
                inshape.Channels,
                outwidth,
                outheight,
                outdepth,
                inshape.Batch);

            return new Shape[] { outshape };
        }

        public override void CheckInputShapes(params Shape[] inshapes) {
            base.CheckInputShapes(inshapes);

            if (inshapes[0].Type != ShapeType.Map || inshapes[0].Ndim != 5) {
                throw new ArgumentException(ExceptionMessage.ShapeElements(inshapes[0], ("Ndim", 5), ("Type", ShapeType.Map)));
            }
        }

        /// <summary>操作クラスを生成する</summary>
        internal override (Tensor[] tensors, Operator ope) GenerateOperator(Tensor[] intensors, Tensor[] outtensors) {
            CheckArgumentsCount(intensors, outtensors);

            Shape shape = intensors[0].Shape;

            return (new Tensor[] { intensors[0], outtensors[0] },
                    new Operators.Connection3D.ImageToColumn(
                        shape.Width, shape.Height, shape.Depth, shape.Channels,
                        KernelWidth, KernelHeight, KernelDepth, shape.Batch));
        }
    }
}
