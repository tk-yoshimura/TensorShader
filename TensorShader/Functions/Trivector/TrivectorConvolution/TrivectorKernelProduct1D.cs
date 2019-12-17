using System;

namespace TensorShader {
    public abstract partial class VariableNode {
        /// <summary>3次元ベクトル1次元カーネル積</summary>
        public static VariableNode TrivectorKernelProduct1D(VariableNode x, VariableNode y, VariableNode q, int kwidth, bool transpose = false) {
            Function function =
                new Functions.TrivectorConvolution.TrivectorKernelProduct1D(x.Shape, y.Shape, kwidth, transpose);

            VariableNode w = Apply(function, x, y, q)[0];

            return w;
        }
    }

    public partial class Tensor {
        /// <summary>3次元ベクトル1次元カーネル積</summary>
        public static Tensor TrivectorKernelProduct1D(Tensor x, Tensor y, Tensor q, int kwidth, bool transpose = false) {
            Functions.TrivectorConvolution.TrivectorKernelProduct1D function =
                new Functions.TrivectorConvolution.TrivectorKernelProduct1D(x.Shape, y.Shape, kwidth, transpose);

            Tensor w = new Tensor(function.OutShape);

            function.Execute(new Tensor[] { x, y, q }, new Tensor[] { w });

            return w;
        }
    }
}

namespace TensorShader.Functions.TrivectorConvolution {
    /// <summary>3次元ベクトル1次元カーネル積</summary>
    internal class TrivectorKernelProduct1D : Function {
        /// <summary>入力特徴マップ形状</summary>
        public Shape InShape { private set; get; }

        /// <summary>出力特徴マップ形状</summary>
        public Shape OutShape { private set; get; }

        /// <summary>カーネル形状</summary>
        public Shape KernelShape { private set; get; }

        /// <summary>転置</summary>
        public bool Transpose { private set; get; }

        /// <summary>コンストラクタ</summary>
        public TrivectorKernelProduct1D(Shape inshape, Shape outshape, int kwidth, bool transpose)
            : base(inputs: 3, outputs: 1, allow_resubstitution: false) {
            
            if (inshape.Type != ShapeType.Map || inshape.Ndim != 3) {
                throw new ArgumentException(ExceptionMessage.TensorElements(inshape, ("Ndim", 3), ("Type", ShapeType.Map)));
            }

            if (outshape.Type != ShapeType.Map || outshape.Ndim != 3) {
                throw new ArgumentException(ExceptionMessage.TensorElements(outshape, ("Ndim", 3), ("Type", ShapeType.Map)));
            }

            if (inshape.Channels % 3 != 0) {
                throw new AggregateException(ExceptionMessage.TensorLengthMultiple("Channels", inshape, inshape.Channels, 3));
            }

            if (outshape.Channels % 3 != 0) {
                throw new AggregateException(ExceptionMessage.TensorLengthMultiple("Channels", outshape, outshape.Channels, 3));
            }

            this.InShape = inshape;
            this.OutShape = outshape;
            this.KernelShape = Shape.Kernel1D(inshape.Channels / 3 * 4, outshape.Channels / 3, kwidth);
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

            return (new Tensor[] { intensors[0], intensors[1], intensors[2], outtensors[0] },
                    new Operators.TrivectorConvolution.TrivectorKernelProduct1D(
                        InShape.Width,
                        InShape.Channels, OutShape.Channels,
                        KernelShape.Width,
                        Transpose, InShape.Batch));
        }
    }
}
