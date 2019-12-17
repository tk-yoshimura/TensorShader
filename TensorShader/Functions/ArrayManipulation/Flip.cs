using System;

namespace TensorShader {
    public abstract partial class VariableNode {
        /// <summary>軸反転</summary>
        public static VariableNode Flip(VariableNode x, int axis) {
            Function function = new Functions.ArrayManipulation.Flip(axis);

            return Apply(function, x)[0];
        }
    }

    public partial class Tensor {
        /// <summary>軸反転</summary>
        public static Tensor Flip(Tensor x, int axis) {
            Function function = new Functions.ArrayManipulation.Flip(axis);

            Tensor y = new Tensor(x.Shape);

            function.Execute(new Tensor[] { x }, new Tensor[] { y });

            return y;
        }
    }
}

namespace TensorShader.Functions.ArrayManipulation {
    /// <summary>軸反転</summary>
    internal class Flip : Function {
        /// <summary>軸</summary>
        public int Axis { private set; get; }

        /// <summary>コンストラクタ</summary>
        public Flip(int axis)
            : base(inputs: 1, outputs: 1, allow_resubstitution: false) {

            this.Axis = axis;
        }

        /// <summary>出力テンソル形状を返す</summary>
        public override Shape[] OutputShapes(params Shape[] inshapes) {
            CheckInputShapes(inshapes);

            return inshapes;
        }

        public override void CheckInputShapes(params Shape[] inshapes) {
            base.CheckInputShapes(inshapes);

            if (Axis >= inshapes[0].Ndim) {
                throw new ArgumentOutOfRangeException(nameof(Axis));
            }
        }

        /// <summary>操作クラスを生成する</summary>
        internal override (Tensor[] tensors, Operator ope) GenerateOperator(Tensor[] intensors, Tensor[] outtensors) {
            CheckArgumentsCount(intensors, outtensors);

            Tensor intensor = intensors[0], outtensor = outtensors[0];

            return (
                new Tensor[] { intensor, outtensor },
                new Operators.ArrayManipulation.Flip(intensor.Shape, Axis)
                );
        }
    }
}
