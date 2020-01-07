using System;

namespace TensorShader {
    public abstract partial class VariableNode {
        /// <summary>ColumnToImage変換</summary>
        public static VariableNode ColumnToImage2D(VariableNode x, int kwidth, int kheight) {
            Function function =
                new Functions.Connection2D.ColumnToImage(kwidth, kheight);

            VariableNode y = Apply(function, x)[0];

            return y;
        }
    }

    public partial class Tensor {
        /// <summary>ColumnToImage変換</summary>
        public static Tensor ColumnToImage2D(Tensor x, int kwidth, int kheight) {
            Function function =
                new Functions.Connection2D.ColumnToImage(kwidth, kheight);

            Tensor y = new Tensor(function.OutputShapes(x.Shape)[0]);

            function.Execute(new Tensor[] { x }, new Tensor[] { y });

            return y;
        }
    }
}

namespace TensorShader.Functions.Connection2D {
    /// <summary>ColumnToImage変換</summary>
    internal class ColumnToImage : Function {

        /// <summary>フィルタサイズ</summary>
        public int KernelWidth { private set; get; }

        /// <summary>フィルタサイズ</summary>
        public int KernelHeight { private set; get; }

        /// <summary>コンストラクタ</summary>
        public ColumnToImage(int kwidth, int kheight)
            : base(inputs: 1, outputs: 1, allow_resubstitution: false) {

            this.KernelWidth = kwidth;
            this.KernelHeight = kheight;
        }

        /// <summary>出力テンソル形状を返す</summary>
        public override Shape[] OutputShapes(params Shape[] inshapes) {
            CheckInputShapes(inshapes);

            Shape inshape = inshapes[0];

            int outwidth = inshape.Width + KernelWidth - 1;
            int outheight = inshape.Height + KernelHeight - 1;

            Shape outshape = new Shape(
                ShapeType.Map,
                inshape.Channels,
                outwidth,
                outheight,
                inshape.Batch);

            return new Shape[] { outshape };
        }

        public override void CheckInputShapes(params Shape[] inshapes) {
            base.CheckInputShapes(inshapes);

            if (inshapes[0].Type != ShapeType.Column || inshapes[0].Ndim != 5) {
                throw new ArgumentException(ExceptionMessage.TensorElements(inshapes[0], ("Ndim", 5), ("Type", ShapeType.Column)));
            }
        }

        /// <summary>操作クラスを生成する</summary>
        internal override (Tensor[] tensors, Operator ope) GenerateOperator(Tensor[] intensors, Tensor[] outtensors) {
            CheckArgumentsCount(intensors, outtensors);

            Shape shape = intensors[0].Shape;

            return (new Tensor[] { intensors[0], outtensors[0] },
                    new Operators.Connection2D.ColumnToImage(
                        shape.Width, shape.Height, shape.Channels,
                        KernelWidth, KernelHeight, shape.Batch));
        }
    }
}
