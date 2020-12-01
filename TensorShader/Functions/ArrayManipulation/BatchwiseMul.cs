using System;

namespace TensorShader {
    public abstract partial class VariableNode {
        /// <summary>バッチ方向に乗算</summary>
        public static VariableNode BatchwiseMul(VariableNode x, VariableNode v) {
            Function function = new Functions.ArrayManipulation.BatchwiseMul();

            VariableNode y = Apply(function, x, v)[0];

            return y;
        }
    }
}

namespace TensorShader.Functions.ArrayManipulation {
    /// <summary>バッチ方向に乗算</summary>
    internal class BatchwiseMul : Function {
        /// <summary>コンストラクタ</summary>
        public BatchwiseMul()
            : base(inputs: 2, outputs: 1, allow_resubstitution: false) { }

        /// <summary>出力テンソル形状を返す</summary>
        public override Shape[] OutputShapes(params Shape[] inshapes) {
            CheckInputShapes(inshapes);

            return new Shape[] { inshapes[0] };
        }

        public override void CheckInputShapes(params Shape[] inshapes) {
            base.CheckInputShapes(inshapes);

            if (inshapes[1].Type != ShapeType.Vector || inshapes[0].Batch != inshapes[1].Channels) {
                throw new ArgumentException(ExceptionMessage.ShapeElements(inshapes[1], ("Type", ShapeType.Vector), ("Channels", inshapes[0].Batch)));
            }
        }

        /// <summary>操作クラスを生成する</summary>
        internal override (Tensor[] tensors, Operator ope) GenerateOperator(Tensor[] intensors, Tensor[] outtensors) {
            CheckArgumentsCount(intensors, outtensors);

            Tensor intensor = intensors[0], invector = intensors[1], outtensor = outtensors[0];

            return (new Tensor[] { intensor, invector, outtensor },
                new Operators.ArrayManipulation.BatchwiseMul(intensor.Shape));
        }
    }
}
