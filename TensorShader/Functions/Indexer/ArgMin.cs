using System;

namespace TensorShader {
    public abstract partial class VariableNode {
        /// <summary>次元ごとに最小インデクスを抽出</summary>
        public static VariableNode ArgMin(VariableNode x) {
            Function function = new Functions.Indexer.ArgMin();

            VariableNode y = Apply(function, x)[0];

            return y;
        }
    }

    public partial class Tensor {
        /// <summary>次元ごとに最小インデクスを抽出</summary>
        public static Tensor ArgMin(Tensor x) {
            Function function = new Functions.Indexer.ArgMin();

            Tensor y = new Tensor(Shape.Vector(x.Batch));

            function.Execute(new Tensor[] { x }, new Tensor[] { y });

            return y;
        }
    }
}

namespace TensorShader.Functions.Indexer {
    /// <summary>次元ごとに最小インデクスを抽出</summary>
    internal class ArgMin : Function {
        /// <summary>コンストラクタ</summary>
        public ArgMin()
            : base(inputs: 1, outputs: 1, allow_resubstitution: false) { }

        /// <summary>出力テンソル形状を返す</summary>
        public override Shape[] OutputShapes(params Shape[] inshapes) {
            CheckInputShapes(inshapes);

            Shape inshape = inshapes[0];

            return new Shape[] { Shape.Vector(inshape.Batch) };
        }

        public override void CheckInputShapes(params Shape[] inshapes) {
            base.CheckInputShapes(inshapes);

            if (inshapes[0].Type != ShapeType.Map || inshapes[0].Ndim != 2) {
                throw new ArgumentException(ExceptionMessage.TensorElements(inshapes[0], ("Ndim", 2), ("Type", ShapeType.Map)));
            }
        }

        /// <summary>操作クラスを生成する</summary>
        internal override (Tensor[] tensors, Operator ope) GenerateOperator(Tensor[] intensors, Tensor[] outtensors) {
            CheckArgumentsCount(intensors, outtensors);

            Tensor intensor = intensors[0], outtensor = outtensors[0];

            return (new Tensor[] { intensor, outtensor },
                new Operators.Indexer.ArgMin(intensor.Channels, intensor.Batch));
        }
    }
}
