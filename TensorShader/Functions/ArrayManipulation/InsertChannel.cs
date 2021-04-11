using System;

namespace TensorShader {
    public abstract partial class VariableNode {
        /// <summary>チャネル方向に挿入</summary>
        internal static VariableNode InsertChannel(VariableNode x, int index, int channels) {
            Function function = new Functions.ArrayManipulation.InsertChannel(index, channels);

            return Apply(function, x)[0];
        }
    }

    public partial class Tensor {
        /// <summary>チャネル方向に挿入</summary>
        internal static Tensor InsertChannel(Tensor x, int index, int channels) {
            Function function = new Functions.ArrayManipulation.InsertChannel(index, channels);

            Tensor y = new(function.OutputShapes(x.Shape)[0]);

            function.Execute(new Tensor[] { x }, new Tensor[] { y });

            return y;
        }
    }
}

namespace TensorShader.Functions.ArrayManipulation {
    /// <summary>チャネル方向に挿入</summary>
    internal class InsertChannel : Function {
        /// <summary>インデクス</summary>
        public int Index { private set; get; }

        /// <summary>出力チャネル数</summary>
        public int Channels { private set; get; }

        /// <summary>コンストラクタ</summary>
        public InsertChannel(int index, int channels)
            : base(inputs: 1, outputs: 1, allow_resubstitution: false) {

            this.Index = index;
            this.Channels = channels;
        }

        /// <summary>出力テンソル形状を返す</summary>
        public override Shape[] OutputShapes(params Shape[] inshapes) {
            CheckInputShapes(inshapes);

            Shape inshape = inshapes[0];

            int[] s = inshape;
            s[0] = Channels;

            return new Shape[] { new Shape(inshape.Type, s) };
        }

        public override void CheckInputShapes(params Shape[] inshapes) {
            base.CheckInputShapes(inshapes);

            if (inshapes[0].Ndim <= 0) {
                throw new ArgumentException(ExceptionMessage.Shape("Ndim", inshapes[0]));
            }
        }

        /// <summary>操作クラスを生成する</summary>
        internal override (Tensor[] tensors, Operator ope) GenerateOperator(Tensor[] intensors, Tensor[] outtensors) {
            CheckArgumentsCount(intensors, outtensors);

            Tensor intensor = intensors[0], outtensor = outtensors[0];

            return (
                new Tensor[] { intensor, outtensor },
                new Operators.ArrayManipulation.InsertChannel(intensor.Shape, Index, outtensor.Shape)
                );
        }
    }
}
