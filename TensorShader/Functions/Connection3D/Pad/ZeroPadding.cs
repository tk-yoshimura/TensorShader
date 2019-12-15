namespace TensorShader {
    public abstract partial class VariableNode {
        /// <summary>3次元ゼロパディング</summary>
        public static VariableNode ZeroPadding3D(VariableNode x, int pad) {
            Function function =
                new Functions.Connection3D.ZeroPadding(pad);

            VariableNode y = Apply(function, x)[0];

            return y;
        }

        /// <summary>3次元ゼロパディング</summary>
        public static VariableNode ZeroPadding3D(VariableNode x, int pad_left, int pad_right, int pad_top, int pad_bottom, int pad_front, int pad_rear) {
            Function function =
                new Functions.Connection3D.ZeroPadding(pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_rear);

            VariableNode y = Apply(function, x)[0];

            return y;
        }
    }

    public partial class Tensor {
        /// <summary>3次元ゼロパディング</summary>
        public static Tensor ZeroPadding3D(Tensor x, int pad) {
            Function function =
                new Functions.Connection3D.ZeroPadding(pad);

            Tensor y = new Tensor(function.OutputShapes(x.Shape)[0]);

            function.Execute(new Tensor[] { x }, new Tensor[] { y });

            return y;
        }

        /// <summary>3次元ゼロパディング</summary>
        public static Tensor ZeroPadding3D(Tensor x, int pad_left, int pad_right, int pad_top, int pad_bottom, int pad_front, int pad_rear) {
            Function function =
                new Functions.Connection3D.ZeroPadding(pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_rear);

            Tensor y = new Tensor(function.OutputShapes(x.Shape)[0]);

            function.Execute(new Tensor[] { x }, new Tensor[] { y });

            return y;
        }
    }
}

namespace TensorShader.Functions.Connection3D {
    /// <summary>3次元ゼロパディング</summary>
    internal class ZeroPadding : Padding {
        /// <summary>コンストラクタ</summary>
        public ZeroPadding(int pad)
            : base(pad) { }

        /// <summary>コンストラクタ</summary>
        public ZeroPadding(int pad_left, int pad_right, int pad_top, int pad_bottom, int pad_front, int pad_rear)
            : base(pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_rear) { }

        /// <summary>操作クラスを生成する</summary>
        internal override (Tensor[] tensors, Operator ope) GenerateOperator(Tensor[] intensors, Tensor[] outtensors) {
            CheckArgumentsCount(intensors, outtensors);

            Shape shape = intensors[0].Shape;

            return (new Tensor[] { intensors[0], outtensors[0] },
                    new Operators.Connection3D.ZeroPadding(
                        shape.Width, shape.Height, shape.Depth,
                        shape.Channels,
                        PadLeft, PadRight, PadTop, PadBottom, PadFront, PadRear,
                        shape.Batch));
        }
    }
}
