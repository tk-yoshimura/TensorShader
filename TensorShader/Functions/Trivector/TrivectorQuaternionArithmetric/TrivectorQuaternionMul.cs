using System;

namespace TensorShader {
    public abstract partial class VariableNode {
        /// <summary>3次元ベクトル四元数回転積</summary>
        public static VariableNode TrivectorQuaternionMul(VariableNode v, VariableNode q) {
            Function function = new Functions.TrivectorArithmetric.TrivectorQuaternionMul();

            return Apply(function, v, q)[0];
        }
    }

    public partial class Tensor {
        /// <summary>3次元ベクトル四元数回転積</summary>
        public static Tensor TrivectorQuaternionMul(Tensor v, Tensor q) {
            Function function = new Functions.TrivectorArithmetric.TrivectorQuaternionMul();

            Tensor u = new Tensor(v.Shape);

            function.Execute(new Tensor[] { v, q }, new Tensor[] { u });

            return u;
        }
    }
}

namespace TensorShader.Functions.TrivectorArithmetric {
    /// <summary>3次元ベクトル四元数回転積</summary>
    internal class TrivectorQuaternionMul : Function {
        /// <summary>コンストラクタ</summary>
        public TrivectorQuaternionMul()
            : base(inputs: 2, outputs: 1, allow_resubstitution: false) { }

        /// <summary>出力テンソル形状を返す</summary>
        public override Shape[] OutputShapes(params Shape[] inshapes) {
            CheckInputShapes(inshapes);

            return new Shape[] { inshapes[0] };
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

            Tensor intensor1 = intensors[0], intensor2 = intensors[1], outtensor = outtensors[0];

            return (new Tensor[] { intensor1, intensor2, outtensor },
                new Operators.TrivectorQuaternionArithmetric.TrivectorQuaternionMul(intensor1.Shape));
        }
    }
}
