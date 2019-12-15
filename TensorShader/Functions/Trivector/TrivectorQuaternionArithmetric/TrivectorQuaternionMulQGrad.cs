using System;

namespace TensorShader {
    public abstract partial class VariableNode {
        /// <summary>3次元ベクトル四元数回転積四元数勾配</summary>
        internal static VariableNode TrivectorQuaternionMulQGrad(VariableNode v, VariableNode u, VariableNode q) {
            Function function = new Functions.TrivectorArithmetric.TrivectorQuaternionMulQGrad();

            return Apply(function, v, u, q)[0];
        }
    }

    public partial class Tensor {
        /// <summary>3次元ベクトル四元数回転積四元数勾配</summary>
        internal static Tensor TrivectorQuaternionMulQGrad(Tensor v, Tensor u, Tensor q) {
            Function function = new Functions.TrivectorArithmetric.TrivectorQuaternionMulQGrad();

            Tensor p = new Tensor(q.Shape);

            function.Execute(new Tensor[] { v, u, q }, new Tensor[] { p });

            return p;
        }
    }
}

namespace TensorShader.Functions.TrivectorArithmetric {
    /// <summary>3次元ベクトル四元数回転積四元数勾配</summary>
    internal class TrivectorQuaternionMulQGrad : Function {
        /// <summary>コンストラクタ</summary>
        public TrivectorQuaternionMulQGrad()
            : base(inputs: 3, outputs: 1, allow_resubstitution: false) { }

        /// <summary>出力テンソル形状を返す</summary>
        public override Shape[] OutputShapes(params Shape[] inshapes) {
            CheckInputShapes(inshapes);

            return new Shape[] { inshapes[2] };
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

            Tensor intensor1 = intensors[0], intensor2 = intensors[1], intensor3 = intensors[2], outtensor = outtensors[0];

            return (new Tensor[] { intensor1, intensor2, intensor3, outtensor },
                    new Operators.TrivectorQuaternionArithmetric.TrivectorQuaternionMulQGrad(intensor1.Shape));
        }
    }
}
