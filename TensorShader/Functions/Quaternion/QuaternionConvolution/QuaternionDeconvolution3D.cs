using System;

namespace TensorShader {
    public abstract partial class VariableNode {
        /// <summary>四元数3次元逆畳み込み</summary>
        public static VariableNode QuaternionDeconvolution3D(VariableNode x, VariableNode w, int stride, bool gradmode = false, Shape outshape = null) {
            if (outshape == null) {
                int outwidth = (x.Shape.Width - 1) * stride + w.Shape.Width;
                int outheight = (x.Shape.Height - 1) * stride + w.Shape.Height;
                int outdepth = (x.Shape.Depth - 1) * stride + w.Shape.Depth;

                outshape = Shape.Map3D(w.Shape.InChannels, outwidth, outheight, outdepth, x.Shape.Batch);
            }

            Function function =
                new Functions.QuaternionConvolution.QuaternionDeconvolution3D(outshape, w.Shape, stride, gradmode);

            VariableNode y = Apply(function, x, w)[0];

            return y;
        }
    }

    public partial class Tensor {
        /// <summary>四元数3次元逆畳み込み</summary>
        public static Tensor QuaternionDeconvolution3D(Tensor x, Tensor w, int stride, bool gradmode = false, Shape outshape = null) {
            if (outshape == null) {
                int outwidth = (x.Shape.Width - 1) * stride + w.Shape.Width;
                int outheight = (x.Shape.Height - 1) * stride + w.Shape.Height;
                int outdepth = (x.Shape.Depth - 1) * stride + w.Shape.Depth;

                outshape = Shape.Map3D(w.Shape.InChannels, outwidth, outheight, outdepth, x.Shape.Batch);
            }

            Functions.QuaternionConvolution.QuaternionDeconvolution3D function =
                new Functions.QuaternionConvolution.QuaternionDeconvolution3D(outshape, w.Shape, stride, gradmode);

            Tensor y = new Tensor(function.OutShape);

            function.Execute(new Tensor[] { x, w }, new Tensor[] { y });

            return y;
        }
    }
}

namespace TensorShader.Functions.QuaternionConvolution {
    /// <summary>四元数3次元逆畳み込み</summary>
    internal class QuaternionDeconvolution3D : Function {
        /// <summary>入力特徴マップ形状</summary>
        public Shape InShape { private set; get; }

        /// <summary>出力特徴マップ形状</summary>
        public Shape OutShape { private set; get; }

        /// <summary>カーネル形状</summary>
        public Shape KernelShape { private set; get; }

        /// <summary>ストライド</summary>
        public int Stride { private set; get; }

        /// <summary>勾配</summary>
        public bool GradMode { private set; get; }

        /// <summary>コンストラクタ</summary>
        public QuaternionDeconvolution3D(Shape outshape, Shape kernelshape, int stride, bool gradmode) :
            base(inputs: 2, outputs: 1, allow_resubstitution: false) {
            if (outshape.Type != ShapeType.Map || outshape.Ndim != 5) {
                throw new ArgumentException(ExceptionMessage.TensorElements(outshape, ("Ndim", 5), ("Type", ShapeType.Map)));
            }

            if (kernelshape.Type != ShapeType.Kernel || kernelshape.Ndim != 5) {
                throw new ArgumentException(ExceptionMessage.TensorElements(kernelshape, ("Ndim", 5), ("Type", ShapeType.Kernel)));
            }

            if (outshape.Channels % 4 != 0) {
                throw new AggregateException(ExceptionMessage.TensorLengthMultiple("Channels", outshape, outshape.Channels, 4));
            }

            if (kernelshape.InChannels % 4 != 0) {
                throw new AggregateException(ExceptionMessage.TensorLengthMultiple("InChannels", kernelshape, kernelshape.Channels, 4));
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

            this.InShape = Shape.Map3D(kernelshape.OutChannels * 4, inwidth, inheight, indepth, outshape.Batch);
            this.OutShape = outshape;
            this.KernelShape = kernelshape;
            this.Stride = stride;
            this.GradMode = gradmode;
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
                    new Operators.QuaternionConvolution.QuaternionDeconvolution3D(
                        OutShape.Width, OutShape.Height, OutShape.Depth,
                        InShape.Channels, OutShape.Channels,
                        KernelShape.Width, KernelShape.Height, KernelShape.Depth,
                        Stride, GradMode, InShape.Batch));
        }
    }
}
