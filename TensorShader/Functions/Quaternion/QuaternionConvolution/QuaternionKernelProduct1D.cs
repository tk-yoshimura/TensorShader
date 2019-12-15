using System;

namespace TensorShader {
    public abstract partial class VariableNode {
        /// <summary>四元数1次元カーネル積</summary>
        public static VariableNode QuaternionKernelProduct1D(VariableNode x, VariableNode y, int kwidth, int stride, bool transpose = false) {
            Function function =
                new Functions.QuaternionConvolution.QuaternionKernelProduct1D(x.Shape, y.Shape, kwidth, stride, transpose);

            VariableNode w = Apply(function, x, y)[0];

            return w;
        }
    }

    public partial class Tensor {
        /// <summary>四元数1次元カーネル積</summary>
        public static Tensor QuaternionKernelProduct1D(Tensor x, Tensor y, int kwidth, int stride, bool transpose = false) {
            Functions.QuaternionConvolution.QuaternionKernelProduct1D function =
                new Functions.QuaternionConvolution.QuaternionKernelProduct1D(x.Shape, y.Shape, kwidth, stride, transpose);

            Tensor w = new Tensor(function.OutShape);

            function.Execute(new Tensor[] { x, y }, new Tensor[] { w });

            return w;
        }
    }
}

namespace TensorShader.Functions.QuaternionConvolution {
    /// <summary>四元数1次元カーネル積</summary>
    internal class QuaternionKernelProduct1D : Function {
        /// <summary>入力特徴マップ形状</summary>
        public Shape InShape { private set; get; }

        /// <summary>出力特徴マップ形状</summary>
        public Shape OutShape { private set; get; }

        /// <summary>カーネル形状</summary>
        public Shape KernelShape { private set; get; }

        /// <summary>ストライド</summary>
        public int Stride { private set; get; }

        /// <summary>転置</summary>
        public bool Transpose { private set; get; }

        /// <summary>コンストラクタ</summary>
        public QuaternionKernelProduct1D(Shape inshape, Shape outshape, int kwidth, int stride, bool transpose) :
            base(inputs: 2, outputs: 1, allow_resubstitution: false) {
            if (inshape.Type != ShapeType.Map || inshape.Ndim != 3) {
                throw new ArgumentException(ExceptionMessage.TensorElements(inshape, ("Ndim", 3), ("Type", ShapeType.Map)));
            }

            if (outshape.Type != ShapeType.Map || outshape.Ndim != 3) {
                throw new ArgumentException(ExceptionMessage.TensorElements(outshape, ("Ndim", 3), ("Type", ShapeType.Map)));
            }

            if (inshape.Channels % 4 != 0) {
                throw new AggregateException(ExceptionMessage.TensorLengthMultiple("Channels", inshape, inshape.Channels, 4));
            }

            if (outshape.InChannels % 4 != 0) {
                throw new AggregateException(ExceptionMessage.TensorLengthMultiple("Channels", outshape, outshape.Channels, 4));
            }

            if (stride < 1) {
                throw new ArgumentException(nameof(stride));
            }

            this.InShape = inshape;
            this.OutShape = outshape;
            this.KernelShape = Shape.Kernel1D(inshape.Channels, outshape.Channels / 4, kwidth);
            this.Stride = stride;
            this.Transpose = transpose;
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
                    new Operators.QuaternionConvolution.QuaternionKernelProduct1D(
                        InShape.Width,
                        InShape.Channels, OutShape.Channels,
                        KernelShape.Width,
                        Stride, Transpose, InShape.Batch));
        }
    }
}
