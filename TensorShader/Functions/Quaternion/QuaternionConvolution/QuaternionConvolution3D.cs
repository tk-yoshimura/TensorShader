using System;

namespace TensorShader {
    public abstract partial class VariableNode {
        /// <summary>四元数3次元畳み込み</summary>
        public static VariableNode QuaternionConvolution3D(VariableNode x, VariableNode w, bool gradmode = false) {
            Function function =
                new Functions.QuaternionConvolution.QuaternionConvolution3D(x.Shape, w.Shape, gradmode);

            VariableNode y = Apply(function, x, w)[0];

            return y;
        }
    }

    public partial class Tensor {
        /// <summary>四元数3次元畳み込み</summary>
        public static Tensor QuaternionConvolution3D(Tensor x, Tensor w, bool gradmode = false) {
            Functions.QuaternionConvolution.QuaternionConvolution3D function =
                new Functions.QuaternionConvolution.QuaternionConvolution3D(x.Shape, w.Shape, gradmode);

            Tensor y = new Tensor(function.OutShape);

            function.Execute(new Tensor[] { x, w }, new Tensor[] { y });

            return y;
        }
    }
}

namespace TensorShader.Functions.QuaternionConvolution {
    /// <summary>四元数3次元畳み込み</summary>
    internal class QuaternionConvolution3D : Function {
        /// <summary>入力特徴マップ形状</summary>
        public Shape InShape { private set; get; }

        /// <summary>出力特徴マップ形状</summary>
        public Shape OutShape { private set; get; }

        /// <summary>カーネル形状</summary>
        public Shape KernelShape { private set; get; }

        /// <summary>勾配</summary>
        public bool GradMode { private set; get; }

        /// <summary>コンストラクタ</summary>
        public QuaternionConvolution3D(Shape inshape, Shape kernelshape, bool gradmode)
            : base(inputs: 2, outputs: 1, allow_resubstitution: false) {

            if (inshape.Type != ShapeType.Map || inshape.Ndim != 5) {
                throw new ArgumentException(ExceptionMessage.TensorElements(inshape, ("Ndim", 5), ("Type", ShapeType.Map)));
            }

            if (kernelshape.Type != ShapeType.Kernel || kernelshape.Ndim != 5) {
                throw new ArgumentException(ExceptionMessage.TensorElements(kernelshape, ("Ndim", 5), ("Type", ShapeType.Kernel)));
            }

            if (inshape.Channels % 4 != 0) {
                throw new AggregateException(ExceptionMessage.TensorLengthMultiple("Channels", inshape, inshape.Channels, 4));
            }

            if (kernelshape.InChannels % 4 != 0) {
                throw new AggregateException(ExceptionMessage.TensorLengthMultiple("InChannels", kernelshape, kernelshape.Channels, 4));
            }

            if (inshape.Channels != kernelshape.InChannels) {
                throw new ArgumentException(ExceptionMessage.TensorElements(kernelshape, ("InChannels", inshape.Channels)));
            }

            int outwidth = inshape.Width - kernelshape.Width + 1;
            int outheight = inshape.Height - kernelshape.Height + 1;
            int outdepth = inshape.Depth - kernelshape.Depth + 1;

            this.InShape = inshape;
            this.OutShape = Shape.Map3D(kernelshape.OutChannels * 4, outwidth, outheight, outdepth, inshape.Batch);
            this.KernelShape = kernelshape;
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
                    new Operators.QuaternionConvolution.QuaternionConvolution3D(
                        InShape.Width, InShape.Height, InShape.Depth,
                        InShape.Channels, OutShape.Channels,
                        KernelShape.Width, KernelShape.Height, KernelShape.Depth,
                        GradMode, InShape.Batch));
        }
    }
}
