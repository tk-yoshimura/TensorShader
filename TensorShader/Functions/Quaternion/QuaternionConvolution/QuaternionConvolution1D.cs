using System;

namespace TensorShader {
    public abstract partial class VariableNode {
        /// <summary>四元数1次元畳み込み</summary>
        public static VariableNode QuaternionConvolution1D(VariableNode x, VariableNode w, int stride, bool gradmode = false) {
            Function function =
                new Functions.QuaternionConvolution.QuaternionConvolution1D(x.Shape, w.Shape, stride, gradmode);

            VariableNode y = Apply(function, x, w)[0];

            return y;
        }
    }

    public partial class Tensor {
        /// <summary>四元数1次元畳み込み</summary>
        public static Tensor QuaternionConvolution1D(Tensor x, Tensor w, int stride, bool gradmode = false) {
            Functions.QuaternionConvolution.QuaternionConvolution1D function =
                new Functions.QuaternionConvolution.QuaternionConvolution1D(x.Shape, w.Shape, stride, gradmode);

            Tensor y = new Tensor(function.OutShape);

            function.Execute(new Tensor[] { x, w }, new Tensor[] { y });

            return y;
        }
    }
}

namespace TensorShader.Functions.QuaternionConvolution {
    /// <summary>四元数1次元畳み込み</summary>
    internal class QuaternionConvolution1D : Function {
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
        public QuaternionConvolution1D(Shape inshape, Shape kernelshape, int stride, bool gradmode) :
            base(inputs: 2, outputs: 1, allow_resubstitution: false) {
            if (inshape.Type != ShapeType.Map || inshape.Ndim != 3) {
                throw new ArgumentException(ExceptionMessage.TensorElements(inshape, ("Ndim", 3), ("Type", ShapeType.Map)));
            }

            if (kernelshape.Type != ShapeType.Kernel || kernelshape.Ndim != 3) {
                throw new ArgumentException(ExceptionMessage.TensorElements(kernelshape, ("Ndim", 3), ("Type", ShapeType.Kernel)));
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

            if (stride < 1) {
                throw new ArgumentException(nameof(stride));
            }

            int outwidth = (inshape.Width - kernelshape.Width) / stride + 1;

            this.InShape = inshape;
            this.OutShape = Shape.Map1D(kernelshape.OutChannels * 4, outwidth, inshape.Batch);
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
                    new Operators.QuaternionConvolution.QuaternionConvolution1D(
                        InShape.Width,
                        InShape.Channels, OutShape.Channels,
                        KernelShape.Width,
                        Stride, GradMode, InShape.Batch));
        }
    }
}
