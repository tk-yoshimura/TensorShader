using System;

namespace TensorShader {
    public abstract partial class VariableNode {
        /// <summary>1次元最大値逆プーリング</summary>
        public static VariableNode MaxUnpooling1D(VariableNode gx, VariableNode y, VariableNode x, int stride) {
            Function function =
                new Functions.Connection1D.MaxUnpooling(stride);

            VariableNode gy = Apply(function, gx, y, x)[0];

            return gy;
        }
    }

    public partial class Tensor {
        /// <summary>1次元最大値逆プーリング</summary>
        public static Tensor MaxUnpooling1D(Tensor gx, Tensor y, Tensor x, int stride) {
            Function function =
                new Functions.Connection1D.MaxUnpooling(stride);

            Tensor gy = new Tensor(function.OutputShapes(x.Shape)[0]);

            function.Execute(new Tensor[] { gx, y, x }, new Tensor[] { gy });

            return y;
        }
    }
}

namespace TensorShader.Functions.Connection1D {
    /// <summary>1次元最大値逆プーリング</summary>
    internal class MaxUnpooling : Function {
        /// <summary>ストライド</summary>
        public int Stride { private set; get; }

        /// <summary>コンストラクタ</summary>
        public MaxUnpooling(int stride)
            : base(inputs: 3, outputs: 1, allow_resubstitution: false) {

            if (stride < 2) {
                throw new ArgumentException(nameof(stride));
            }

            this.Stride = stride;
        }

        /// <summary>出力テンソル形状を返す</summary>
        public override Shape[] OutputShapes(params Shape[] inshapes) {
            CheckInputShapes(inshapes);

            return new Shape[] { inshapes[1] };
        }

        /// <summary>操作クラスを生成する</summary>
        internal override (Tensor[] tensors, Operator ope) GenerateOperator(Tensor[] intensors, Tensor[] outtensors) {
            CheckArgumentsCount(intensors, outtensors);

            Shape shape = outtensors[0].Shape;

            return (new Tensor[] { intensors[0], intensors[1], intensors[2], outtensors[0] },
                    new Operators.Connection1D.MaxUnpooling(
                        shape.Width, shape.Channels,
                        Stride, shape.Batch));
        }
    }
}
