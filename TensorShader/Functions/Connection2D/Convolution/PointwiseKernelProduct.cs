using System;

namespace TensorShader {
    public abstract partial class VariableNode {
        /// <summary>ポイントごとの2次元カーネル積</summary>
        public static VariableNode PointwiseKernelProduct2D(VariableNode x, VariableNode y) {
            Function function =
                new Functions.Connection2D.PointwiseKernelProduct(x.Shape, y.Shape);

            VariableNode w = Apply(function, x, y)[0];

            return w;
        }
    }

    public partial class Tensor {
        /// <summary>ポイントごとの2次元カーネル積</summary>
        public static Tensor PointwiseKernelProduct2D(Tensor x, Tensor y) {
            Functions.Connection2D.PointwiseKernelProduct function =
                new Functions.Connection2D.PointwiseKernelProduct(x.Shape, y.Shape);

            Tensor w = new Tensor(function.OutShape);

            function.Execute(new Tensor[] { x, y }, new Tensor[] { w });

            return w;
        }
    }
}

namespace TensorShader.Functions.Connection2D {
    /// <summary>ポイントごとの2次元カーネル積</summary>
    internal class PointwiseKernelProduct : Function {
        /// <summary>入力特徴マップ形状</summary>
        public Shape InShape { private set; get; }

        /// <summary>出力特徴マップ形状</summary>
        public Shape OutShape { private set; get; }

        /// <summary>カーネル形状</summary>
        public Shape KernelShape { private set; get; }

        /// <summary>コンストラクタ</summary>
        public PointwiseKernelProduct(Shape inshape, Shape outshape)
            : base(inputs: 2, outputs: 1, allow_resubstitution: false) {

            if (inshape.Type != ShapeType.Map || inshape.Ndim != 4) {
                throw new ArgumentException(ExceptionMessage.TensorElements(inshape, ("Ndim", 4), ("Type", ShapeType.Map)));
            }

            if (outshape.Type != ShapeType.Map || outshape.Ndim != 4) {
                throw new ArgumentException(ExceptionMessage.TensorElements(outshape, ("Ndim", 4), ("Type", ShapeType.Map)));
            }

            this.InShape = inshape;
            this.OutShape = outshape;
            this.KernelShape = Shape.Kernel0D(inshape.Channels, outshape.Channels);
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
                    new Operators.Connection2D.PointwiseKernelProduct(
                        InShape.Width, InShape.Height,
                        InShape.Channels, OutShape.Channels,
                        InShape.Batch));
        }
    }
}
