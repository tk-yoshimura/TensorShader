using System;

namespace TensorShader {
    public abstract partial class VariableNode {
        /// <summary>2次元最大値プーリング</summary>
        public static VariableNode MaxPooling2D(VariableNode x, int stride) {
            Function function =
                new Functions.Connection2D.MaxPooling(stride);

            VariableNode y = Apply(function, x)[0];

            return y;
        }
    }

    public partial class Tensor {
        /// <summary>2次元最大値プーリング</summary>
        public static Tensor MaxPooling2D(Tensor x, int stride) {
            Function function =
                new Functions.Connection2D.MaxPooling(stride);

            Tensor y = new Tensor(function.OutputShapes(x.Shape)[0]);

            function.Execute(new Tensor[] { x }, new Tensor[] { y });

            return y;
        }
    }
}

namespace TensorShader.Functions.Connection2D {
    /// <summary>2次元最大値プーリング</summary>
    internal class MaxPooling : Function {
        /// <summary>ストライド</summary>
        public int Stride { private set; get; }

        /// <summary>コンストラクタ</summary>
        public MaxPooling(int stride)
            : base(inputs: 1, outputs: 1, allow_resubstitution: false) {

            this.Stride = stride;
        }

        /// <summary>出力テンソル形状を返す</summary>
        public override Shape[] OutputShapes(params Shape[] inshapes) {
            CheckInputShapes(inshapes);

            Shape inshape = inshapes[0];

            Shape outshape = Shape.Map2D(inshape.Channels,
                                         inshape.Width / Stride,
                                         inshape.Height / Stride,
                                         inshape.Batch);

            return new Shape[] { outshape };
        }

        public override void CheckInputShapes(params Shape[] inshapes) {
            base.CheckInputShapes(inshapes);

            if (inshapes[0].Type != ShapeType.Map || inshapes[0].Ndim != 4) {
                throw new ArgumentException(ExceptionMessage.ShapeElements(inshapes[0], ("Ndim", 4), ("Type", ShapeType.Map)));
            }
        }

        /// <summary>操作クラスを生成する</summary>
        internal override (Tensor[] tensors, Operator ope) GenerateOperator(Tensor[] intensors, Tensor[] outtensors) {
            CheckArgumentsCount(intensors, outtensors);

            Shape shape = intensors[0].Shape;

            return (new Tensor[] { intensors[0], outtensors[0] },
                    new Operators.Connection2D.MaxPooling(
                        shape.Width, shape.Height, shape.Channels,
                        Stride, shape.Batch));
        }
    }
}
