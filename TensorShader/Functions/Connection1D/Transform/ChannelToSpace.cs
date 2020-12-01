using System;

namespace TensorShader {
    public abstract partial class VariableNode {
        /// <summary>チャネル次元を空間方向に展開</summary>
        public static VariableNode ChannelToSpace1D(VariableNode x, int scale) {
            Function function =
                new Functions.Connection1D.ChannelToSpace(scale);

            VariableNode y = Apply(function, x)[0];

            return y;
        }
    }

    public partial class Tensor {
        /// <summary>チャネル次元を空間方向に展開</summary>
        public static Tensor ChannelToSpace1D(Tensor x, int scale) {
            Function function =
                new Functions.Connection1D.ChannelToSpace(scale);

            Tensor y = new Tensor(function.OutputShapes(x.Shape)[0]);

            function.Execute(new Tensor[] { x }, new Tensor[] { y });

            return y;
        }
    }
}

namespace TensorShader.Functions.Connection1D {
    /// <summary>チャネル次元を空間方向に展開</summary>
    internal class ChannelToSpace : Function {
        /// <summary>倍率</summary>
        public int Scale { private set; get; }

        /// <summary>コンストラクタ</summary>
        public ChannelToSpace(int scale)
            : base(inputs: 1, outputs: 1, allow_resubstitution: false) {

            this.Scale = scale;
        }

        /// <summary>出力テンソル形状を返す</summary>
        public override Shape[] OutputShapes(params Shape[] inshapes) {
            CheckInputShapes(inshapes);

            Shape inshape = inshapes[0];

            Shape outshape = Shape.Map1D(
                inshape.Channels / Scale,
                inshape.Width * Scale,
                inshape.Batch);

            return new Shape[] { outshape };
        }

        public override void CheckInputShapes(params Shape[] inshapes) {
            base.CheckInputShapes(inshapes);

            if (inshapes[0].Type != ShapeType.Map || inshapes[0].Ndim != 3) {
                throw new ArgumentException(ExceptionMessage.ShapeElements(inshapes[0], ("Ndim", 3), ("Type", ShapeType.Map)));
            }

            if (inshapes[0].Channels % Scale != 0) {
                throw new ArgumentException(ExceptionMessage.LengthMultiple("Channels", inshapes[0], inshapes[0].Channels, Scale));
            }
        }

        /// <summary>操作クラスを生成する</summary>
        internal override (Tensor[] tensors, Operator ope) GenerateOperator(Tensor[] intensors, Tensor[] outtensors) {
            CheckArgumentsCount(intensors, outtensors);

            Shape shape = intensors[0].Shape;

            return (new Tensor[] { intensors[0], outtensors[0] },
                    new Operators.Connection1D.ChannelToSpace(
                        shape.Width, shape.Channels,
                        Scale, shape.Batch));
        }
    }
}
