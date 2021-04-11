using System;

namespace TensorShader {
    public abstract partial class VariableNode {
        /// <summary>1次元線形補間</summary>
        /// <remarks>倍率2固定</remarks>
        public static VariableNode LinearZoom1D(VariableNode x) {
            Function function =
                new Functions.Connection1D.LinearZoom();

            VariableNode y = Apply(function, x)[0];

            return y;
        }
    }

    public partial class Tensor {
        /// <summary>1次元線形補間</summary>
        /// <remarks>倍率2固定</remarks>
        public static Tensor LinearZoom1D(Tensor x) {
            Function function =
                new Functions.Connection1D.LinearZoom();

            Tensor y = new(function.OutputShapes(x.Shape)[0]);

            function.Execute(new Tensor[] { x }, new Tensor[] { y });

            return y;
        }
    }
}

namespace TensorShader.Functions.Connection1D {
    /// <summary>1次元線形補間</summary>
    /// <remarks>倍率2固定</remarks>
    internal class LinearZoom : Function {
        /// <summary>コンストラクタ</summary>
        public LinearZoom()
            : base(inputs: 1, outputs: 1, allow_resubstitution: false) { }

        /// <summary>出力テンソル形状を返す</summary>
        public override Shape[] OutputShapes(params Shape[] inshapes) {
            CheckInputShapes(inshapes);

            Shape inshape = inshapes[0];

            Shape outshape = Shape.Map1D(
                inshape.Channels,
                inshape.Width * 2,
                inshape.Batch);

            return new Shape[] { outshape };
        }

        public override void CheckInputShapes(params Shape[] inshapes) {
            base.CheckInputShapes(inshapes);

            if (inshapes[0].Type != ShapeType.Map || inshapes[0].Ndim != 3) {
                throw new ArgumentException(ExceptionMessage.ShapeElements(inshapes[0], ("Ndim", 3), ("Type", ShapeType.Map)));
            }
        }

        /// <summary>操作クラスを生成する</summary>
        internal override (Tensor[] tensors, Operator ope) GenerateOperator(Tensor[] intensors, Tensor[] outtensors) {
            CheckArgumentsCount(intensors, outtensors);

            Shape shape = intensors[0].Shape;

            return (new Tensor[] { intensors[0], outtensors[0] },
                    new Operators.Connection1D.LinearZoom(
                        shape.Width, shape.Channels,
                        shape.Batch));
        }
    }
}
