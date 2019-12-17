using System;

namespace TensorShader {
    public abstract partial class VariableNode {
        /// <summary>トリミング</summary>
        public static VariableNode Trimming1D(VariableNode x, int trim) {
            Function function =
                new Functions.Connection1D.Trimming(trim);

            VariableNode y = Apply(function, x)[0];

            return y;
        }

        /// <summary>トリミング</summary>
        public static VariableNode Trimming1D(VariableNode x, int trim_left, int trim_right) {
            Function function =
                new Functions.Connection1D.Trimming(trim_left, trim_right);

            VariableNode y = Apply(function, x)[0];

            return y;
        }
    }

    public partial class Tensor {
        /// <summary>トリミング</summary>
        public static Tensor Trimming1D(Tensor x, int trim) {
            Function function =
                new Functions.Connection1D.Trimming(trim);

            Tensor y = new Tensor(function.OutputShapes(x.Shape)[0]);

            function.Execute(new Tensor[] { x }, new Tensor[] { y });

            return y;
        }

        /// <summary>トリミング</summary>
        public static Tensor Trimming1D(Tensor x, int trim_left, int trim_right) {
            Function function =
                new Functions.Connection1D.Trimming(trim_left, trim_right);

            Tensor y = new Tensor(function.OutputShapes(x.Shape)[0]);

            function.Execute(new Tensor[] { x }, new Tensor[] { y });

            return y;
        }
    }
}

namespace TensorShader.Functions.Connection1D {
    /// <summary>トリミング</summary>
    internal class Trimming : Function {
        /// <summary>トリミング左幅</summary>
        public int TrimLeft { private set; get; }

        /// <summary>トリミング右幅</summary>
        public int TrimRight { private set; get; }

        /// <summary>コンストラクタ</summary>
        public Trimming(int trim)
            : this(trim, trim) { }

        /// <summary>コンストラクタ</summary>
        public Trimming(int trim_left, int trim_right)
            : base(inputs: 1, outputs: 1, allow_resubstitution: false) {

            this.TrimLeft = trim_left;
            this.TrimRight = trim_right;
        }

        /// <summary>出力テンソル形状を返す</summary>
        public override Shape[] OutputShapes(params Shape[] inshapes) {
            CheckInputShapes(inshapes);

            Shape inshape = inshapes[0];

            Shape outshape = Shape.Map1D(inshape.Channels,
                                         inshape.Width - TrimLeft - TrimRight,
                                         inshape.Batch);

            return new Shape[] { outshape };
        }

        public override void CheckInputShapes(params Shape[] inshapes) {
            base.CheckInputShapes(inshapes);

            if (inshapes[0].Type != ShapeType.Map || inshapes[0].Ndim != 3) {
                throw new ArgumentException(ExceptionMessage.TensorElements(inshapes[0], ("Ndim", 3), ("Type", ShapeType.Map)));
            }
        }

        /// <summary>操作クラスを生成する</summary>
        internal override (Tensor[] tensors, Operator ope) GenerateOperator(Tensor[] intensors, Tensor[] outtensors) {
            CheckArgumentsCount(intensors, outtensors);

            Shape shape = intensors[0].Shape;

            return (new Tensor[] { intensors[0], outtensors[0] },
                    new Operators.Connection1D.Trimming(
                        shape.Width, shape.Channels,
                        TrimLeft, TrimRight,
                        shape.Batch));
        }
    }
}
