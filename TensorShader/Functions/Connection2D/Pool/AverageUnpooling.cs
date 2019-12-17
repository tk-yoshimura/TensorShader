using System;

namespace TensorShader {
    public abstract partial class VariableNode {
        /// <summary>2次元平均値逆プーリング</summary>
        public static VariableNode AverageUnpooling2D(VariableNode x, int stride, Shape outshape = null) {
            if (outshape == null) {
                int outwidth = x.Shape.Width * stride;
                int outheight = x.Shape.Height * stride;

                outshape = Shape.Map2D(x.Shape.Channels, outwidth, outheight, x.Shape.Batch);
            }

            Function function =
                new Functions.Connection2D.AverageUnpooling(outshape, stride);

            VariableNode y = Apply(function, x)[0];

            return y;
        }
    }

    public partial class Tensor {
        /// <summary>2次元平均値逆プーリング</summary>
        public static Tensor AverageUnpooling2D(Tensor x, int stride, Shape outshape = null) {
            if (outshape == null) {
                int outwidth = x.Shape.Width * stride;
                int outheight = x.Shape.Height * stride;

                outshape = Shape.Map2D(x.Shape.Channels, outwidth, outheight, x.Shape.Batch);
            }

            Function function =
                new Functions.Connection2D.AverageUnpooling(outshape, stride);

            Tensor y = new Tensor(function.OutputShapes(x.Shape)[0]);

            function.Execute(new Tensor[] { x }, new Tensor[] { y });

            return y;
        }
    }
}

namespace TensorShader.Functions.Connection2D {
    /// <summary>2次元平均値逆プーリング</summary>
    internal class AverageUnpooling : Function {
        /// <summary>ストライド</summary>
        public int Stride { private set; get; }

        /// <summary>入力特徴マップ形状</summary>
        public Shape InShape { private set; get; }

        /// <summary>出力特徴マップ形状</summary>
        public Shape OutShape { private set; get; }

        /// <summary>コンストラクタ</summary>
        public AverageUnpooling(Shape outshape, int stride)
            : base(inputs: 1, outputs: 1, allow_resubstitution: false) {

            if (outshape.Type != ShapeType.Map || outshape.Ndim != 4) {
                throw new ArgumentException(ExceptionMessage.TensorElements(outshape, ("Ndim", 4), ("Type", ShapeType.Map)));
            }

            if (stride < 2) {
                throw new ArgumentException(nameof(stride));
            }

            this.Stride = stride;
            this.InShape = Shape.Map2D(outshape.Channels, outshape.Width / stride, outshape.Height / stride, outshape.Batch);
            this.OutShape = outshape;
        }

        /// <summary>出力テンソル形状を返す</summary>
        public override Shape[] OutputShapes(params Shape[] inshapes) {
            CheckInputShapes(inshapes);

            return new Shape[] { OutShape };
        }

        public override void CheckInputShapes(params Shape[] inshapes) {
            base.CheckInputShapes(inshapes);

            if (inshapes[0] != InShape) {
                throw new ArgumentException(ExceptionMessage.Shape(inshapes[0], InShape));
            }
        }

        /// <summary>操作クラスを生成する</summary>
        internal override (Tensor[] tensors, Operator ope) GenerateOperator(Tensor[] intensors, Tensor[] outtensors) {
            CheckArgumentsCount(intensors, outtensors);

            Shape shape = outtensors[0].Shape;

            return (new Tensor[] { intensors[0], outtensors[0] },
                    new Operators.Connection2D.AverageUnpooling(
                        shape.Width, shape.Height, shape.Channels,
                        Stride, shape.Batch));
        }
    }
}
