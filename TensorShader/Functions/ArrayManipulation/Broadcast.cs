using System;
using System.Linq;

namespace TensorShader {
    public abstract partial class VariableNode {
        /// <summary>ブロードキャスト</summary>
        public static VariableNode Broadcast(VariableNode x, Shape shape) {
            while (!((int[])x.Shape).SequenceEqual((int[])shape)) {
                Function function = new Functions.ArrayManipulation.Broadcast(x.Shape, shape);

                x = Apply(function, x)[0];
            }

            return x;
        }
    }

    public partial class Tensor {
        /// <summary>ブロードキャスト</summary>
        public static Tensor Broadcast(Tensor x, Shape shape) {
            while (!((int[])x.Shape).SequenceEqual((int[])shape)) {
                Function function = new Functions.ArrayManipulation.Broadcast(x.Shape, shape);

                Tensor y = new Tensor(shape);

                function.Execute(new Tensor[] { x }, new Tensor[] { y });

                x = y;
            }

            return x;
        }
    }
}

namespace TensorShader.Functions.ArrayManipulation {
    /// <summary>ブロードキャスト</summary>
    internal class Broadcast : Function {
        /// <summary>入力形状</summary>
        public Shape InShape { private set; get; }

        /// <summary>出力形状</summary>
        public Shape OutShape { private set; get; }

        /// <summary>ブロードキャスト後の目的形状</summary>
        public Shape TargetShape { private set; get; }

        /// <summary>コンストラクタ</summary>
        public Broadcast(Shape inshape, Shape targetshape)
            : base(inputs: 1, outputs: 1, allow_resubstitution: false) {

            if (inshape.Ndim > targetshape.Ndim) {
                throw new ArgumentException(ExceptionMessage.Broadcast(inshape, targetshape));
            }

            for (int i = 0; i < inshape.Ndim; i++) {
                if (inshape[i] != 1 && inshape[i] != targetshape[i]) {
                    throw new ArgumentException(ExceptionMessage.Broadcast(inshape, targetshape));
                }
            }

            this.InShape = inshape;
            this.OutShape = Operators.ArrayManipulation.Broadcast.BroadcastShape(inshape, targetshape);
            this.TargetShape = targetshape;
        }

        /// <summary>出力テンソル形状を返す</summary>
        public override Shape[] OutputShapes(params Shape[] inshapes) {
            CheckInputShapes(inshapes);

            return new Shape[] { OutShape };
        }

        public override void CheckInputShapes(params Shape[] inshapes) {
            base.CheckInputShapes(inshapes);
        }

        /// <summary>操作クラスを生成する</summary>
        internal override (Tensor[] tensors, Operator ope) GenerateOperator(Tensor[] intensors, Tensor[] outtensors) {
            CheckArgumentsCount(intensors, outtensors);

            Tensor intensor = intensors[0], outtensor = outtensors[0];

            return (new Tensor[] { intensor, outtensor }, new Operators.ArrayManipulation.Broadcast(InShape, OutShape));
        }
    }
}
