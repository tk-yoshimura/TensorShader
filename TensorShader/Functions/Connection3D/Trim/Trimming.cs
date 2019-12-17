using System;

namespace TensorShader {
    public abstract partial class VariableNode {
        /// <summary>トリミング</summary>
        public static VariableNode Trimming3D(VariableNode x, int trim) {
            Function function =
                new Functions.Connection3D.Trimming(trim);

            VariableNode y = Apply(function, x)[0];

            return y;
        }

        /// <summary>トリミング</summary>
        public static VariableNode Trimming3D(VariableNode x, int trim_left, int trim_right, int trim_top, int trim_bottom, int trim_front, int trim_rear) {
            Function function =
                new Functions.Connection3D.Trimming(trim_left, trim_right, trim_top, trim_bottom, trim_front, trim_rear);

            VariableNode y = Apply(function, x)[0];

            return y;
        }
    }

    public partial class Tensor {
        /// <summary>トリミング</summary>
        public static Tensor Trimming3D(Tensor x, int trim) {
            Function function =
                new Functions.Connection3D.Trimming(trim);

            Tensor y = new Tensor(function.OutputShapes(x.Shape)[0]);

            function.Execute(new Tensor[] { x }, new Tensor[] { y });

            return y;
        }

        /// <summary>トリミング</summary>
        public static Tensor Trimming3D(Tensor x, int trim_left, int trim_right, int trim_top, int trim_bottom, int trim_front, int trim_rear) {
            Function function =
                new Functions.Connection3D.Trimming(trim_left, trim_right, trim_top, trim_bottom, trim_front, trim_rear);

            Tensor y = new Tensor(function.OutputShapes(x.Shape)[0]);

            function.Execute(new Tensor[] { x }, new Tensor[] { y });

            return y;
        }
    }
}

namespace TensorShader.Functions.Connection3D {
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

        /// <summary>トリミング前幅</summary>
        public int TrimFront { private set; get; }

        /// <summary>トリミング後幅</summary>
        public int TrimRear { private set; get; }

        /// <summary>コンストラクタ</summary>
        public Trimming(int trim)
            : this(trim, trim, trim, trim, trim, trim) { }

        /// <summary>コンストラクタ</summary>
        public Trimming(int trim_left, int trim_right, int trim_top, int trim_bottom, int trim_front, int trim_rear)
            : base(inputs: 1, outputs: 1, allow_resubstitution: false) {

            this.TrimLeft = trim_left;
            this.TrimRight = trim_right;
            this.TrimTop = trim_top;
            this.TrimBottom = trim_bottom;
            this.TrimFront = trim_front;
            this.TrimRear = trim_rear;
        }

        /// <summary>出力テンソル形状を返す</summary>
        public override Shape[] OutputShapes(params Shape[] inshapes) {
            CheckInputShapes(inshapes);

            Shape inshape = inshapes[0];

            Shape outshape = Shape.Map3D(inshape.Channels,
                                         inshape.Width - TrimLeft - TrimRight,
                                         inshape.Height - TrimTop - TrimBottom,
                                         inshape.Depth - TrimFront - TrimRear,
                                         inshape.Batch);

            return new Shape[] { outshape };
        }

        public override void CheckInputShapes(params Shape[] inshapes) {
            base.CheckInputShapes(inshapes);

            if (inshapes[0].Type != ShapeType.Map || inshapes[0].Ndim != 5) {
                throw new ArgumentException(ExceptionMessage.TensorElements(inshapes[0], ("Ndim", 5), ("Type", ShapeType.Map)));
            }
        }

        /// <summary>操作クラスを生成する</summary>
        internal override (Tensor[] tensors, Operator ope) GenerateOperator(Tensor[] intensors, Tensor[] outtensors) {
            CheckArgumentsCount(intensors, outtensors);

            Shape shape = intensors[0].Shape;

            return (new Tensor[] { intensors[0], outtensors[0] },
                    new Operators.Connection3D.Trimming(
                        shape.Width, shape.Height, shape.Depth,
                        shape.Channels,
                        TrimLeft, TrimRight, TrimTop, TrimBottom, TrimFront, TrimRear,
                        shape.Batch));
        }
    }
}
