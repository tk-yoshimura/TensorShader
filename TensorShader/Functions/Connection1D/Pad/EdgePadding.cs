namespace TensorShader {
    public abstract partial class VariableNode {
        /// <summary>1次元エッジパディング</summary>
        public static VariableNode EdgePadding1D(VariableNode x, int pad) {
            Function function =
                new Functions.Connection1D.EdgePadding(pad);

            VariableNode y = Apply(function, x)[0];

            return y;
        }

        /// <summary>1次元エッジパディング</summary>
        public static VariableNode EdgePadding1D(VariableNode x, int pad_left, int pad_right) {
            Function function =
                new Functions.Connection1D.EdgePadding(pad_left, pad_right);

            VariableNode y = Apply(function, x)[0];

            return y;
        }
    }

    public partial class Tensor {
        /// <summary>1次元エッジパディング</summary>
        public static Tensor EdgePadding1D(Tensor x, int pad) {
            Function function =
                new Functions.Connection1D.EdgePadding(pad);

            Tensor y = new(function.OutputShapes(x.Shape)[0]);

            function.Execute(new Tensor[] { x }, new Tensor[] { y });

            return y;
        }

        /// <summary>1次元エッジパディング</summary>
        public static Tensor EdgePadding1D(Tensor x, int pad_left, int pad_right) {
            Function function =
                new Functions.Connection1D.EdgePadding(pad_left, pad_right);

            Tensor y = new(function.OutputShapes(x.Shape)[0]);

            function.Execute(new Tensor[] { x }, new Tensor[] { y });

            return y;
        }
    }
}

namespace TensorShader.Functions.Connection1D {
    /// <summary>1次元エッジパディング</summary>
    internal class EdgePadding : Padding {
        /// <summary>コンストラクタ</summary>
        public EdgePadding(int pad)
            : base(pad) { }

        /// <summary>コンストラクタ</summary>
        public EdgePadding(int pad_left, int pad_right)
            : base(pad_left, pad_right) { }

        /// <summary>操作クラスを生成する</summary>
        internal override (Tensor[] tensors, Operator ope) GenerateOperator(Tensor[] intensors, Tensor[] outtensors) {
            CheckArgumentsCount(intensors, outtensors);

            Shape shape = intensors[0].Shape;

            return (new Tensor[] { intensors[0], outtensors[0] },
                    new Operators.Connection1D.EdgePadding(
                        shape.Width, shape.Channels,
                        PadLeft, PadRight,
                        shape.Batch));
        }
    }
}
