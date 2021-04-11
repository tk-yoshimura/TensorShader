using System;

namespace TensorShader {
    public abstract partial class VariableNode {
        /// <summary>トリミング</summary>
        public static VariableNode Trimming2D(VariableNode x, int trim) {
            Function function =
                new Functions.Connection2D.Trimming(trim);

            VariableNode y = Apply(function, x)[0];

            return y;
        }

        /// <summary>トリミング</summary>
        public static VariableNode Trimming2D(VariableNode x, int trim_left, int trim_right, int trim_top, int trim_bottom) {
            Function function =
                new Functions.Connection2D.Trimming(trim_left, trim_right, trim_top, trim_bottom);

            VariableNode y = Apply(function, x)[0];

            return y;
        }
    }

    public partial class Tensor {
        /// <summary>トリミング</summary>
        public static Tensor Trimming2D(Tensor x, int trim) {
            Function function =
                new Functions.Connection2D.Trimming(trim);

            Tensor y = new(function.OutputShapes(x.Shape)[0]);

            function.Execute(new Tensor[] { x }, new Tensor[] { y });

            return y;
        }

        /// <summary>トリミング</summary>
        public static Tensor Trimming2D(Tensor x, int trim_left, int trim_right, int trim_top, int trim_bottom) {
            Function function =
                new Functions.Connection2D.Trimming(trim_left, trim_right, trim_top, trim_bottom);

            Tensor y = new(function.OutputShapes(x.Shape)[0]);

            function.Execute(new Tensor[] { x }, new Tensor[] { y });

            return y;
        }
    }
}

namespace TensorShader.Functions.Connection2D {
    /// <summary>トリミング</summary>
    internal class Trimming : Function {
        /// <summary>トリミング左幅</summary>
        public int TrimLeft { private set; get; }

        /// <summary>トリミング右幅</summary>
        public int TrimRight { private set; get; }

        /// <summary>トリミング上幅</summary>
        public int TrimTop { private set; get; }

        /// <summary>トリミング下幅</summary>
        public int TrimBottom { private set; get; }

        /// <summary>コンストラクタ</summary>
        public Trimming(int trim)
            : this(trim, trim, trim, trim) { }

        /// <summary>コンストラクタ</summary>
        public Trimming(int trim_left, int trim_right, int trim_top, int trim_bottom)
            : base(inputs: 1, outputs: 1, allow_resubstitution: false) {

            this.TrimLeft = trim_left;
            this.TrimRight = trim_right;
            this.TrimTop = trim_top;
            this.TrimBottom = trim_bottom;
        }

        /// <summary>出力テンソル形状を返す</summary>
        public override Shape[] OutputShapes(params Shape[] inshapes) {
            CheckInputShapes(inshapes);

            Shape inshape = inshapes[0];

            Shape outshape = Shape.Map2D(inshape.Channels,
                                         inshape.Width - TrimLeft - TrimRight,
                                         inshape.Height - TrimTop - TrimBottom,
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
                    new Operators.Connection2D.Trimming(
                        shape.Width, shape.Height, shape.Channels,
                        TrimLeft, TrimRight, TrimTop, TrimBottom,
                        shape.Batch));
        }
    }
}
