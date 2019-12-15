namespace TensorShader {
    public abstract partial class VariableNode {
        /// <summary>2次元エッジパディング</summary>
        public static VariableNode EdgePadding2D(VariableNode x, int pad) {
            Function function =
                new Functions.Connection2D.EdgePadding(pad);

            VariableNode y = Apply(function, x)[0];

            return y;
        }

        /// <summary>2次元エッジパディング</summary>
        public static VariableNode EdgePadding2D(VariableNode x, int pad_left, int pad_right, int pad_top, int pad_bottom) {
            Function function =
                new Functions.Connection2D.EdgePadding(pad_left, pad_right, pad_top, pad_bottom);

            VariableNode y = Apply(function, x)[0];

            return y;
        }
    }

    public partial class Tensor {
        /// <summary>2次元エッジパディング</summary>
        public static Tensor EdgePadding2D(Tensor x, int pad) {
            Function function =
                new Functions.Connection2D.EdgePadding(pad);

            Tensor y = new Tensor(function.OutputShapes(x.Shape)[0]);

            function.Execute(new Tensor[] { x }, new Tensor[] { y });

            return y;
        }

        /// <summary>2次元エッジパディング</summary>
        public static Tensor EdgePadding2D(Tensor x, int pad_left, int pad_right, int pad_top, int pad_bottom) {
            Function function =
                new Functions.Connection2D.EdgePadding(pad_left, pad_right, pad_top, pad_bottom);

            Tensor y = new Tensor(function.OutputShapes(x.Shape)[0]);

            function.Execute(new Tensor[] { x }, new Tensor[] { y });

            return y;
        }
    }
}

namespace TensorShader.Functions.Connection2D {
    /// <summary>2次元エッジパディング</summary>
    internal class EdgePadding : Padding {
        /// <summary>コンストラクタ</summary>
        public EdgePadding(int pad)
            : base(pad) { }

        /// <summary>コンストラクタ</summary>
        public EdgePadding(int pad_left, int pad_right, int pad_top, int pad_bottom)
            : base(pad_left, pad_right, pad_top, pad_bottom) { }

        /// <summary>操作クラスを生成する</summary>
        internal override (Tensor[] tensors, Operator ope) GenerateOperator(Tensor[] intensors, Tensor[] outtensors) {
            CheckArgumentsCount(intensors, outtensors);

            Shape shape = intensors[0].Shape;

            return (new Tensor[] { intensors[0], outtensors[0] },
                    new Operators.Connection2D.EdgePadding(
                        shape.Width, shape.Height, shape.Channels,
                        PadLeft, PadRight, PadTop, PadBottom,
                        shape.Batch));
        }
    }
}
