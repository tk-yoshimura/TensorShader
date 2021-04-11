namespace TensorShader {
    public abstract partial class VariableNode {
        /// <summary>2次元ゼロパディング</summary>
        public static VariableNode ZeroPadding2D(VariableNode x, int pad) {
            Function function =
                new Functions.Connection2D.ZeroPadding(pad);

            VariableNode y = Apply(function, x)[0];

            return y;
        }

        /// <summary>2次元ゼロパディング</summary>
        public static VariableNode ZeroPadding2D(VariableNode x, int pad_left, int pad_right, int pad_top, int pad_bottom) {
            Function function =
                new Functions.Connection2D.ZeroPadding(pad_left, pad_right, pad_top, pad_bottom);

            VariableNode y = Apply(function, x)[0];

            return y;
        }
    }

    public partial class Tensor {
        /// <summary>2次元ゼロパディング</summary>
        public static Tensor ZeroPadding2D(Tensor x, int pad) {
            Function function =
                new Functions.Connection2D.ZeroPadding(pad);

            Tensor y = new(function.OutputShapes(x.Shape)[0]);

            function.Execute(new Tensor[] { x }, new Tensor[] { y });

            return y;
        }

        /// <summary>2次元ゼロパディング</summary>
        public static Tensor ZeroPadding2D(Tensor x, int pad_left, int pad_right, int pad_top, int pad_bottom) {
            Function function =
                new Functions.Connection2D.ZeroPadding(pad_left, pad_right, pad_top, pad_bottom);

            Tensor y = new(function.OutputShapes(x.Shape)[0]);

            function.Execute(new Tensor[] { x }, new Tensor[] { y });

            return y;
        }
    }
}

namespace TensorShader.Functions.Connection2D {
    /// <summary>2次元ゼロパディング</summary>
    internal class ZeroPadding : Padding {
        /// <summary>コンストラクタ</summary>
        public ZeroPadding(int pad)
            : base(pad) { }

        /// <summary>コンストラクタ</summary>
        public ZeroPadding(int pad_left, int pad_right, int pad_top, int pad_bottom)
            : base(pad_left, pad_right, pad_top, pad_bottom) { }

        /// <summary>操作クラスを生成する</summary>
        internal override (Tensor[] tensors, Operator ope) GenerateOperator(Tensor[] intensors, Tensor[] outtensors) {
            CheckArgumentsCount(intensors, outtensors);

            Shape shape = intensors[0].Shape;

            return (new Tensor[] { intensors[0], outtensors[0] },
                    new Operators.Connection2D.ZeroPadding(
                        shape.Width, shape.Height, shape.Channels,
                        PadLeft, PadRight, PadTop, PadBottom,
                        shape.Batch));
        }
    }
}
