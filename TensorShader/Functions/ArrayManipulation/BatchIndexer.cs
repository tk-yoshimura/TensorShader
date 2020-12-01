using System;

namespace TensorShader {
    public abstract partial class VariableNode {
        /// <summary>インデクサ</summary>
        public VariableNode this[int index] {
            get {
                Function function = new Functions.ArrayManipulation.BatchIndexer(index);

                VariableNode y = Apply(function, this)[0];

                return y;
            }
        }
    }
}

namespace TensorShader.Functions.ArrayManipulation {
    /// <summary>インデクサ</summary>
    internal class BatchIndexer : Function {
        /// <summary>インデクサ</summary>
        public int Index { private set; get; }

        /// <summary>コンストラクタ</summary>
        public BatchIndexer(int index)
            : base(inputs: 1, outputs: 1, allow_resubstitution: false) {

            this.Index = index;
        }

        /// <summary>出力テンソル形状を返す</summary>
        public override Shape[] OutputShapes(params Shape[] inshapes) {
            CheckInputShapes(inshapes);

            Shape inshape = inshapes[0];

            int[] s = inshape;
            s[s.Length - 1] = 1;

            return new Shape[] { new Shape(ShapeType.Map, s) };
        }

        public override void CheckInputShapes(params Shape[] inshapes) {
            base.CheckInputShapes(inshapes);

            Shape inshape = inshapes[0];

            if (inshape.Ndim < 2 || inshape.Type != ShapeType.Map) {
                throw new ArgumentException(ExceptionMessage.ShapeElements(inshape, ("Ndim>", 2), ("Type", ShapeType.Map)));
            }
        }

        /// <summary>操作クラスを生成する</summary>
        internal override (Tensor[] tensors, Operator ope) GenerateOperator(Tensor[] intensors, Tensor[] outtensors) {
            CheckArgumentsCount(intensors, outtensors);

            Tensor intensor = intensors[0], outtensor = outtensors[0];

            return (new Tensor[] { intensor, outtensor },
                new Operators.ArrayManipulation.BatchIndexer(intensor.Shape, Index));
        }
    }
}
