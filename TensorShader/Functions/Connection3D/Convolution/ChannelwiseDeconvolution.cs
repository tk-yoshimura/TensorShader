using System;

namespace TensorShader {
    public abstract partial class VariableNode {
        /// <summary>チャネルごとの3次元逆畳み込み</summary>
        public static VariableNode ChannelwiseDeconvolution3D(VariableNode x, VariableNode w, int stride, Shape outshape = null) {
            if (outshape == null) {
                int outwidth = (x.Shape.Width - 1) * stride + w.Shape.Width;
                int outheight = (x.Shape.Height - 1) * stride + w.Shape.Height;
                int outdepth = (x.Shape.Depth - 1) * stride + w.Shape.Depth;

                outshape = Shape.Map3D(w.Shape.InChannels, outwidth, outheight, outdepth, x.Shape.Batch);
            }

            Function function =
                new Functions.Connection3D.ChannelwiseDeconvolution(outshape, w.Shape, stride);

            VariableNode y = Apply(function, x, w)[0];

            return y;
        }
    }

    public partial class Tensor {
        /// <summary>チャネルごとの3次元逆畳み込み</summary>
        public static Tensor ChannelwiseDeconvolution3D(Tensor x, Tensor w, int stride, Shape outshape = null) {
            if (outshape == null) {
                int outwidth = (x.Shape.Width - 1) * stride + w.Shape.Width;
                int outheight = (x.Shape.Height - 1) * stride + w.Shape.Height;
                int outdepth = (x.Shape.Depth - 1) * stride + w.Shape.Depth;

                outshape = Shape.Map3D(w.Shape.InChannels, outwidth, outheight, outdepth, x.Shape.Batch);
            }

            Functions.Connection3D.ChannelwiseDeconvolution function =
                new Functions.Connection3D.ChannelwiseDeconvolution(outshape, w.Shape, stride);

            Tensor y = new Tensor(function.OutShape);

            function.Execute(new Tensor[] { x, w }, new Tensor[] { y });

            return y;
        }
    }
}

namespace TensorShader.Functions.Connection3D {
    /// <summary>チャネルごとの3次元逆畳み込み</summary>
    internal class ChannelwiseDeconvolution : Function {
        /// <summary>入力特徴マップ形状</summary>
        public Shape InShape { private set; get; }

        /// <summary>出力特徴マップ形状</summary>
        public Shape OutShape { private set; get; }

        /// <summary>カーネル形状</summary>
        public Shape KernelShape { private set; get; }

        /// <summary>ストライド</summary>
        public int Stride { private set; get; }

        /// <summary>コンストラクタ</summary>
        public ChannelwiseDeconvolution(Shape outshape, Shape kernelshape, int stride) :
            base(inputs: 2, outputs: 1, allow_resubstitution: false) {
            if (outshape.Type != ShapeType.Map || outshape.Ndim != 5) {
                throw new ArgumentException(ExceptionMessage.TensorElements(outshape, ("Ndim", 5), ("Type", ShapeType.Map)));
            }

            if (kernelshape.Type != ShapeType.Kernel || kernelshape.Ndim != 5) {
                throw new ArgumentException(ExceptionMessage.TensorElements(kernelshape, ("Ndim", 5), ("Type", ShapeType.Kernel)));
            }

            if (outshape.Channels != kernelshape.InChannels) {
                throw new ArgumentException(ExceptionMessage.TensorElements(kernelshape, ("InChannels", outshape.Channels)));
            }

            if (stride < 1) {
                throw new ArgumentException(nameof(stride));
            }

            int inwidth = (outshape.Width - kernelshape.Width) / stride + 1;
            int inheight = (outshape.Height - kernelshape.Height) / stride + 1;
            int indepth = (outshape.Depth - kernelshape.Depth) / stride + 1;

            this.InShape = Shape.Map3D(outshape.Channels, inwidth, inheight, indepth, outshape.Batch);
            this.OutShape = outshape;
            this.KernelShape = kernelshape;
            this.Stride = stride;
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
                    new Operators.Connection3D.ChannelwiseDeconvolution(
                        OutShape.Width, OutShape.Height, OutShape.Depth,
                        InShape.Channels,
                        KernelShape.Width, KernelShape.Height, KernelShape.Depth,
                        Stride, InShape.Batch));
        }
    }
}
