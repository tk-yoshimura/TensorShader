using System;

namespace TensorShader {
    public abstract partial class VariableNode {
        /// <summary>複素全結合</summary>
        public static VariableNode ComplexDense(VariableNode x, VariableNode w, bool gradmode = false) {
            Function function =
                new Functions.ComplexConvolution.ComplexDense(x.Shape, w.Shape, gradmode);

            VariableNode y = Apply(function, x, w)[0];

            return y;
        }
    }

    public partial class Tensor {
        /// <summary>複素全結合</summary>
        public static Tensor ComplexDense(Tensor x, Tensor w, bool gradmode = false) {
            Functions.ComplexConvolution.ComplexDense function =
                new Functions.ComplexConvolution.ComplexDense(x.Shape, w.Shape, gradmode);

            Tensor y = new Tensor(function.OutShape);

            function.Execute(new Tensor[] { x, w }, new Tensor[] { y });

            return y;
        }
    }
}

namespace TensorShader.Functions.ComplexConvolution {
    /// <summary>複素全結合</summary>
    internal class ComplexDense : Function {
        /// <summary>入力特徴マップ形状</summary>
        public Shape InShape { private set; get; }

        /// <summary>出力特徴マップ形状</summary>
        public Shape OutShape { private set; get; }

        /// <summary>カーネル形状</summary>
        public Shape KernelShape { private set; get; }

        /// <summary>勾配</summary>
        public bool GradMode { private set; get; }

        /// <summary>コンストラクタ</summary>
        public ComplexDense(Shape inshape, Shape kernelshape, bool gradmode)
            : base(inputs: 2, outputs: 1, allow_resubstitution: false) {

            if (inshape.Type != ShapeType.Map || inshape.Ndim != 2) {
                throw new ArgumentException(ExceptionMessage.ShapeElements(inshape, ("Ndim", 2), ("Type", ShapeType.Map)));
            }

            if (kernelshape.Type != ShapeType.Kernel || kernelshape.Ndim != 2) {
                throw new ArgumentException(ExceptionMessage.ShapeElements(kernelshape, ("Ndim", 2), ("Type", ShapeType.Kernel)));
            }

            if (inshape.Channels % 2 != 0) {
                throw new AggregateException(ExceptionMessage.LengthMultiple("Channels", inshape, inshape.Channels, 2));
            }

            if (kernelshape.InChannels % 2 != 0) {
                throw new AggregateException(ExceptionMessage.LengthMultiple("InChannels", kernelshape, kernelshape.Channels, 2));
            }

            if (inshape.Channels != kernelshape.InChannels) {
                throw new ArgumentException(ExceptionMessage.ShapeElements(kernelshape, ("InChannels", inshape.Channels)));
            }

            this.InShape = inshape;
            this.OutShape = Shape.Map0D(kernelshape.OutChannels * 2, inshape.Batch);
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
                    new Operators.ComplexConvolution.ComplexDense(
                        InShape.Channels, OutShape.Channels,
                        GradMode, InShape.Batch));
        }
    }
}
