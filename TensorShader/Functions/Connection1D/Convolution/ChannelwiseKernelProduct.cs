using System;

namespace TensorShader {
    public abstract partial class VariableNode {
        /// <summary>チャネルごとの1次元カーネル積</summary>
        public static VariableNode ChannelwiseKernelProduct1D(VariableNode x, VariableNode y, int kwidth) {
            Function function =
                new Functions.Connection1D.ChannelwiseKernelProduct(x.Shape, y.Shape, kwidth);

            VariableNode w = Apply(function, x, y)[0];

            return w;
        }
    }

    public partial class Tensor {
        /// <summary>チャネルごとの1次元カーネル積</summary>
        public static Tensor ChannelwiseKernelProduct1D(Tensor x, Tensor y, int kwidth) {
            Functions.Connection1D.ChannelwiseKernelProduct function =
                new Functions.Connection1D.ChannelwiseKernelProduct(x.Shape, y.Shape, kwidth);

            Tensor w = new Tensor(function.OutShape);

            function.Execute(new Tensor[] { x, y }, new Tensor[] { w });

            return w;
        }
    }
}

namespace TensorShader.Functions.Connection1D {
    /// <summary>チャネルごとの1次元カーネル積</summary>
    internal class ChannelwiseKernelProduct : Function {
        /// <summary>入力特徴マップ形状</summary>
        public Shape InShape { private set; get; }

        /// <summary>出力特徴マップ形状</summary>
        public Shape OutShape { private set; get; }

        /// <summary>カーネル形状</summary>
        public Shape KernelShape { private set; get; }

        /// <summary>コンストラクタ</summary>
        public ChannelwiseKernelProduct(Shape inshape, Shape outshape, int kwidth)
            : base(inputs: 2, outputs: 1, allow_resubstitution: false) {

            if (inshape.Type != ShapeType.Map || inshape.Ndim != 3) {
                throw new ArgumentException(ExceptionMessage.TensorElements(inshape, ("Ndim", 3), ("Type", ShapeType.Map)));
            }

            if (outshape.Type != ShapeType.Map || outshape.Ndim != 3) {
                throw new ArgumentException(ExceptionMessage.TensorElements(outshape, ("Ndim", 3), ("Type", ShapeType.Map)));
            }

            this.InShape = inshape;
            this.OutShape = outshape;
            this.KernelShape = Shape.Kernel1D(inshape.Channels, 1, kwidth);
        }

        /// <summary>出力テンソル形状を返す</summary>
        public override Shape[] OutputShapes(params Shape[] inshapes) {
            CheckInputShapes(inshapes);

            return new Shape[] { KernelShape };
        }

        public override void CheckInputShapes(params Shape[] inshapes) {
            base.CheckInputShapes(inshapes);

            if (inshapes[0] != InShape) {
                throw new ArgumentException(ExceptionMessage.ShapeWithIndex(index: 0, inshapes[0], InShape));
            }

            if (inshapes[1] != OutShape) {
                throw new ArgumentException(ExceptionMessage.ShapeWithIndex(index: 1, inshapes[1], OutShape));
            }
        }

        /// <summary>操作クラスを生成する</summary>
        internal override (Tensor[] tensors, Operator ope) GenerateOperator(Tensor[] intensors, Tensor[] outtensors) {
            CheckArgumentsCount(intensors, outtensors);

            return (new Tensor[] { intensors[0], intensors[1], outtensors[0] },
                    new Operators.Connection1D.ChannelwiseKernelProduct(
                        InShape.Width,
                        InShape.Channels,
                        KernelShape.Width,
                        InShape.Batch));
        }
    }
}
