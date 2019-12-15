using System;

namespace TensorShader {
    public abstract partial class VariableNode {
        /// <summary>複素全結合カーネル積</summary>
        public static VariableNode ComplexKernelProductDense(VariableNode x, VariableNode y, bool transpose = false) {
            Function function =
                new Functions.ComplexConvolution.ComplexKernelProductDense(x.Shape, y.Shape, transpose);

            VariableNode w = Apply(function, x, y)[0];

            return w;
        }
    }

    public partial class Tensor {
        /// <summary>複素全結合カーネル積</summary>
        public static Tensor ComplexKernelProductDense(Tensor x, Tensor y, bool transpose = false) {
            Functions.ComplexConvolution.ComplexKernelProductDense function =
                new Functions.ComplexConvolution.ComplexKernelProductDense(x.Shape, y.Shape, transpose);

            Tensor w = new Tensor(function.OutShape);

            function.Execute(new Tensor[] { x, y }, new Tensor[] { w });

            return w;
        }
    }
}

namespace TensorShader.Functions.ComplexConvolution {
    /// <summary>複素全結合カーネル積</summary>
    internal class ComplexKernelProductDense : Function {
        /// <summary>入力特徴マップ形状</summary>
        public Shape InShape { private set; get; }

        /// <summary>出力特徴マップ形状</summary>
        public Shape OutShape { private set; get; }

        /// <summary>カーネル形状</summary>
        public Shape KernelShape { private set; get; }

        /// <summary>転置</summary>
        public bool Transpose { private set; get; }

        /// <summary>コンストラクタ</summary>
        public ComplexKernelProductDense(Shape inshape, Shape outshape, bool transpose) :
            base(inputs: 2, outputs: 1, allow_resubstitution: false) {
            if (inshape.Type != ShapeType.Map || inshape.Ndim != 2) {
                throw new ArgumentException(ExceptionMessage.TensorElements(inshape, ("Ndim", 2), ("Type", ShapeType.Map)));
            }

            if (outshape.Type != ShapeType.Map || outshape.Ndim != 2) {
                throw new ArgumentException(ExceptionMessage.TensorElements(outshape, ("Ndim", 2), ("Type", ShapeType.Map)));
            }

            if (inshape.Channels % 2 != 0) {
                throw new AggregateException(ExceptionMessage.TensorLengthMultiple("Channels", inshape, inshape.Channels, 2));
            }

            if (outshape.Channels % 2 != 0) {
                throw new AggregateException(ExceptionMessage.TensorLengthMultiple("Channels", outshape, outshape.Channels, 2));
            }

            this.InShape = inshape;
            this.OutShape = outshape;
            this.KernelShape = Shape.Kernel0D(inshape.Channels, outshape.Channels / 2);
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
                    new Operators.ComplexConvolution.ComplexKernelProductDense(
                        InShape.Channels, OutShape.Channels,
                        Transpose, InShape.Batch));
        }
    }
}
