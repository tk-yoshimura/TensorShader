using System;
using System.Linq;

namespace TensorShader {
    public abstract partial class VariableNode {
        /// <summary>インデクスをOneHot特徴量に変換</summary>
        public static VariableNode OneHotVector(VariableNode x, int channels) {
            Function function = new Functions.Indexer.OneHotVector(channels);

            VariableNode y = Apply(function, x)[0];

            return y;
        }
    }

    public partial class Tensor {
        /// <summary>インデクスをOneHot特徴量に変換</summary>
        public static Tensor OneHotVector(Tensor x, int channels) {
            Function function = new Functions.Indexer.OneHotVector(channels);

            Tensor y = new Tensor(Shape.Map0D(channels, x.Channels));

            function.Execute(new Tensor[] { x }, new Tensor[] { y });

            return y;
        }
    }
}

namespace TensorShader.Functions.Indexer {
    /// <summary>インデクスをOneHot特徴量に変換</summary>
    internal class OneHotVector : Function {
        /// <summary>チャネル数</summary>
        public int Channels { private set; get; }

        /// <summary>コンストラクタ</summary>
        public OneHotVector(int channels)
            : base(inputs: 1, outputs: 1, allow_resubstitution: false) {
            this.Channels = channels;
        }

        /// <summary>出力テンソル形状を返す</summary>
        public override Shape[] OutputShapes(params Shape[] inshapes) {
            CheckInputShapes(inshapes);

            Shape inshape = inshapes[0];

            return new Shape[] { new Shape(ShapeType.Map, (new int[] { Channels }).Concat((int[])inshape).ToArray()) };
        }

        public override void CheckInputShapes(params Shape[] inshapes) {
            base.CheckInputShapes(inshapes);

            if (inshapes[0].Ndim < 1) {
                throw new ArgumentException(ExceptionMessage.Shape("Ndim", inshapes[0]));
            }
        }

        /// <summary>操作クラスを生成する</summary>
        internal override (Tensor[] tensors, Operator ope) GenerateOperator(Tensor[] intensors, Tensor[] outtensors) {
            CheckArgumentsCount(intensors, outtensors);

            Tensor intensor = intensors[0], outtensor = outtensors[0];

            return (new Tensor[] { intensor, outtensor },
                new Operators.Indexer.OneHotVector(Channels, intensor.Shape));
        }
    }
}
