using System;

namespace TensorShader {
    public abstract partial class VariableNode {
        /// <summary>四元数純第3成分</summary>
        public static VariableNode QuaternionPureJ(VariableNode X) {
            Function function = new Functions.Quaternion.QuaternionPureJ();

            return Apply(function, X)[0];
        }
    }

    public partial class Tensor {
        /// <summary>四元数純第3成分</summary>
        public static Tensor QuaternionPureJ(Tensor X) {
            Function function = new Functions.Quaternion.QuaternionPureJ();

            Shape y_shape = function.OutputShapes(X.Shape)[0];

            Tensor y = new Tensor(y_shape);

            function.Execute(new Tensor[] { X }, new Tensor[] { y });

            return y;
        }
    }
}

namespace TensorShader.Functions.Quaternion {
    /// <summary>四元数純第3成分</summary>
    internal class QuaternionPureJ : Function {
        /// <summary>コンストラクタ</summary>
        public QuaternionPureJ()
            : base(inputs: 1, outputs: 1, allow_resubstitution: false) { }

        /// <summary>出力テンソル形状を返す</summary>
        public override Shape[] OutputShapes(params Shape[] inshapes) {
            CheckInputShapes(inshapes);

            Shape inshape = inshapes[0];

            int[] s = inshape;
            s[0] *= 4;

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

            return (new Tensor[] { intensor, outtensor },
                    new Operators.QuaternionCast.QuaternionPureJ(intensor.Shape));
        }
    }
}
