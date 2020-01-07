using System;

namespace TensorShader {
    public abstract partial class VariableNode {
        /// <summary>ポイントごとの3次元逆畳み込み</summary>
        public static VariableNode PointwiseDeconvolution3D(VariableNode x, VariableNode w) {
            Function function =
                new Functions.Connection3D.PointwiseDeconvolution(x.Shape, w.Shape);

            VariableNode y = Apply(function, x, w)[0];

            return y;
        }
    }

    public partial class Tensor {
        /// <summary>ポイントごとの3次元逆畳み込み</summary>
        public static Tensor PointwiseDeconvolution3D(Tensor x, Tensor w) {
            Functions.Connection3D.PointwiseDeconvolution function =
                new Functions.Connection3D.PointwiseDeconvolution(x.Shape, w.Shape);

            Tensor y = new Tensor(function.OutShape);

            function.Execute(new Tensor[] { x, w }, new Tensor[] { y });

            return y;
        }
    }
}

namespace TensorShader.Functions.Connection3D {
    /// <summary>ポイントごとの3次元逆畳み込み</summary>
    internal class PointwiseDeconvolution : Function {
        /// <summary>入力特徴マップ形状</summary>
        public Shape InShape { private set; get; }

        /// <summary>出力特徴マップ形状</summary>
        public Shape OutShape { private set; get; }

        /// <summary>カーネル形状</summary>
        public Shape KernelShape { private set; get; }

        /// <summary>コンストラクタ</summary>
        public PointwiseDeconvolution(Shape inshape, Shape kernelshape)
            : base(inputs: 2, outputs: 1, allow_resubstitution: false) {

            if (inshape.Type != ShapeType.Map || inshape.Ndim != 5) {
                throw new ArgumentException(ExceptionMessage.TensorElements(inshape, ("Ndim", 5), ("Type", ShapeType.Map)));
            }

            if (kernelshape.Type != ShapeType.Kernel || kernelshape.Ndim != 2) {
                throw new ArgumentException(ExceptionMessage.TensorElements(kernelshape, ("Ndim", 2), ("Type", ShapeType.Kernel)));
            }

            if (inshape.Channels != kernelshape.OutChannels) {
                throw new ArgumentException(ExceptionMessage.TensorElements(kernelshape, ("OutChannels", inshape.Channels)));
            }

            this.InShape = inshape;
            this.OutShape = Shape.Map3D(kernelshape.InChannels, inshape.Width, inshape.Height, inshape.Depth, inshape.Batch);
            this.KernelShape = kernelshape;
        }

        /// <summary>出力テンソル形状を返す</summary>
        public override Shape[] OutputShapes(params Shape[] inshapes) {
            CheckInputShapes(inshapes);

            return new Shape[] { OutShape };
        }

        public override void CheckInputShapes(params Shape[] inshapes) {
            base.CheckInputShapes(inshapes);

            if (inshapes[0] != InShape) {
                throw new ArgumentException(ExceptionMessage.ShapeWithIndex(index: 0, inshapes[0], InShape));
            }

            if (inshapes[1] != KernelShape) {
                throw new ArgumentException(ExceptionMessage.ShapeWithIndex(index: 1, inshapes[1], KernelShape));
            }
        }

        /// <summary>操作クラスを生成する</summary>
        internal override (Tensor[] tensors, Operator ope) GenerateOperator(Tensor[] intensors, Tensor[] outtensors) {
            CheckArgumentsCount(intensors, outtensors);

            return (new Tensor[] { intensors[0], intensors[1], outtensors[0] },
                    new Operators.Connection3D.PointwiseDeconvolution(
                        OutShape.Width, OutShape.Height, OutShape.Depth,
                        InShape.Channels, OutShape.Channels,
                        InShape.Batch));
        }
    }
}
