using System;

namespace TensorShader {
    public abstract partial class VariableNode {
        /// <summary>2次元最近傍補間</summary>
        /// <remarks>倍率2固定</remarks>
        public static VariableNode NeighborZoom2D(VariableNode x) {
            Function function =
                new Functions.Connection2D.NeighborZoom();

            VariableNode y = Apply(function, x)[0];

            return y;
        }
    }

    public partial class Tensor {
        /// <summary>2次元最近傍補間</summary>
        /// <remarks>倍率2固定</remarks>
        public static Tensor NeighborZoom2D(Tensor x) {
            Function function =
                new Functions.Connection2D.NeighborZoom();

            Tensor y = new Tensor(function.OutputShapes(x.Shape)[0]);

            function.Execute(new Tensor[] { x }, new Tensor[] { y });

            return y;
        }
    }
}

namespace TensorShader.Functions.Connection2D {
    /// <summary>2次元最近傍補間</summary>
    /// <remarks>倍率2固定</remarks>
    internal class NeighborZoom : Function {
        /// <summary>コンストラクタ</summary>
        public NeighborZoom()
            : base(inputs: 1, outputs: 1, allow_resubstitution: false) { }

        /// <summary>出力テンソル形状を返す</summary>
        public override Shape[] OutputShapes(params Shape[] inshapes) {
            CheckInputShapes(inshapes);

            Shape inshape = inshapes[0];

            Shape outshape = Shape.Map2D(
                inshape.Channels,
                inshape.Width * 2,
                inshape.Height * 2,
                inshape.Batch);

            return new Shape[] { outshape };
        }

        public override void CheckInputShapes(params Shape[] inshapes) {
            base.CheckInputShapes(inshapes);

            if (inshapes[0].Type != ShapeType.Map || inshapes[0].Ndim != 4) {
                throw new ArgumentException(ExceptionMessage.TensorElements(inshapes[0], ("Ndim", 4), ("Type", ShapeType.Map)));
            }
        }

        /// <summary>操作クラスを生成する</summary>
        internal override (Tensor[] tensors, Operator ope) GenerateOperator(Tensor[] intensors, Tensor[] outtensors) {
            CheckArgumentsCount(intensors, outtensors);

            Shape shape = intensors[0].Shape;

            return (new Tensor[] { intensors[0], outtensors[0] },
                    new Operators.Connection2D.NeighborZoom(
                        shape.Width, shape.Height, shape.Channels,
                        shape.Batch));
        }
    }
}
