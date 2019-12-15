using System;

namespace TensorShader {
    public abstract partial class VariableNode {
        /// <summary>複素2次元畳み込み</summary>
        public static VariableNode ComplexConvolution2D(VariableNode x, VariableNode w, int stride, bool gradmode = false) {
            Function function =
                new Functions.ComplexConvolution.ComplexConvolution2D(x.Shape, w.Shape, stride, gradmode);

            VariableNode y = Apply(function, x, w)[0];

            return y;
        }
    }

    public partial class Tensor {
        /// <summary>複素2次元畳み込み</summary>
        public static Tensor ComplexConvolution2D(Tensor x, Tensor w, int stride, bool gradmode = false) {
            Functions.ComplexConvolution.ComplexConvolution2D function =
                new Functions.ComplexConvolution.ComplexConvolution2D(x.Shape, w.Shape, stride, gradmode);

            Tensor y = new Tensor(function.OutShape);

            function.Execute(new Tensor[] { x, w }, new Tensor[] { y });

            return y;
        }
    }
}

namespace TensorShader.Functions.ComplexConvolution {
    /// <summary>複素2次元畳み込み</summary>
    internal class ComplexConvolution2D : Function {
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
        public ComplexConvolution2D(Shape inshape, Shape kernelshape, int stride, bool gradmode) :
            base(inputs: 2, outputs: 1, allow_resubstitution: false) {
            if (inshape.Type != ShapeType.Map || inshape.Ndim != 4) {
                throw new ArgumentException(ExceptionMessage.TensorElements(inshape, ("Ndim", 4), ("Type", ShapeType.Map)));
            }

            if (kernelshape.Type != ShapeType.Kernel || kernelshape.Ndim != 4) {
                throw new ArgumentException(ExceptionMessage.TensorElements(kernelshape, ("Ndim", 4), ("Type", ShapeType.Kernel)));
            }

            if (inshape.Channels % 2 != 0) {
                throw new AggregateException(ExceptionMessage.TensorLengthMultiple("Channels", inshape, inshape.Channels, 2));
            }

            if (kernelshape.InChannels % 2 != 0) {
                throw new AggregateException(ExceptionMessage.TensorLengthMultiple("InChannels", kernelshape, kernelshape.Channels, 2));
            }

            if (inshape.Channels != kernelshape.InChannels) {
                throw new ArgumentException(ExceptionMessage.TensorElements(kernelshape, ("InChannels", inshape.Channels)));
            }

            if (stride < 1) {
                throw new ArgumentException(nameof(stride));
            }

            int outwidth = (inshape.Width - kernelshape.Width) / stride + 1;
            int outheight = (inshape.Height - kernelshape.Height) / stride + 1;

            this.InShape = inshape;
            this.OutShape = Shape.Map2D(kernelshape.OutChannels * 2, outwidth, outheight, inshape.Batch);
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
                    new Operators.ComplexConvolution.ComplexConvolution2D(
                        InShape.Width, InShape.Height,
                        InShape.Channels, OutShape.Channels,
                        KernelShape.Width, KernelShape.Height,
                        Stride, GradMode, InShape.Batch));
        }
    }
}
