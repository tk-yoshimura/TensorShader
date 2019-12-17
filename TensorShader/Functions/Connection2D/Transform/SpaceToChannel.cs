using System;

namespace TensorShader {
    public abstract partial class VariableNode {
        /// <summary>空間次元をチャネル次元に展開</summary>
        public static VariableNode SpaceToChannel2D(VariableNode x, int scale) {
            Function function =
                new Functions.Connection2D.SpaceToChannel(scale);

            VariableNode y = Apply(function, x)[0];

            return y;
        }
    }

    public partial class Tensor {
        /// <summary>空間次元をチャネル次元に展開</summary>
        public static Tensor SpaceToChannel2D(Tensor x, int scale) {
            Function function =
                new Functions.Connection2D.SpaceToChannel(scale);

            Tensor y = new Tensor(function.OutputShapes(x.Shape)[0]);

            function.Execute(new Tensor[] { x }, new Tensor[] { y });

            return y;
        }
    }
}

namespace TensorShader.Functions.Connection2D {
    /// <summary>空間次元をチャネル次元に展開</summary>
    internal class SpaceToChannel : Function {
        /// <summary>倍率</summary>
        public int Scale { private set; get; }

        /// <summary>コンストラクタ</summary>
        public SpaceToChannel(int scale)
            : base(inputs: 1, outputs: 1, allow_resubstitution: false) {

            this.Scale = scale;
        }

        /// <summary>出力テンソル形状を返す</summary>
        public override Shape[] OutputShapes(params Shape[] inshapes) {
            CheckInputShapes(inshapes);

            Shape inshape = inshapes[0];

            Shape outshape = Shape.Map2D(
                inshape.Channels * Scale * Scale,
                inshape.Width / Scale,
                inshape.Height / Scale,
                inshape.Batch);

            return new Shape[] { outshape };
        }

        public override void CheckInputShapes(params Shape[] inshapes) {
            base.CheckInputShapes(inshapes);

            if (inshapes[0].Type != ShapeType.Map || inshapes[0].Ndim != 4) {
                throw new ArgumentException(ExceptionMessage.TensorElements(inshapes[0], ("Ndim", 4), ("Type", ShapeType.Map)));
            }

            if (inshapes[0].Width % Scale != 0) {
                throw new ArgumentException(ExceptionMessage.TensorLengthMultiple("Width", inshapes[0], inshapes[0].Width, Scale));
            }

            if (inshapes[0].Height % Scale != 0) {
                throw new ArgumentException(ExceptionMessage.TensorLengthMultiple("Height", inshapes[0], inshapes[0].Height, Scale));
            }
        }

        /// <summary>操作クラスを生成する</summary>
        internal override (Tensor[] tensors, Operator ope) GenerateOperator(Tensor[] intensors, Tensor[] outtensors) {
            CheckArgumentsCount(intensors, outtensors);

            Shape shape = intensors[0].Shape;

            return (new Tensor[] { intensors[0], outtensors[0] },
                    new Operators.Connection2D.SpaceToChannel(
                        shape.Width, shape.Height, shape.Channels,
                        Scale, shape.Batch));
        }
    }
}
