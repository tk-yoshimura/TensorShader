using System;

namespace TensorShader {
    public abstract partial class VariableNode {
        /// <summary>Hardmax</summary>
        public static VariableNode Hardmax(VariableNode x) {
            Function function = new Functions.Channelwise.Hardmax();

            VariableNode y = Apply(function, x)[0];

            return y;
        }
    }

    public partial class Tensor {
        /// <summary>Hardmax</summary>
        public static Tensor Hardmax(Tensor x) {
            Function function = new Functions.Channelwise.Hardmax();

            Tensor y = x.Shape;

            function.Execute(new Tensor[] { x }, new Tensor[] { y });

            return y;
        }
    }
}

namespace TensorShader.Functions.Channelwise {
    /// <summary>Hardmax</summary>
    internal class Hardmax : Function {
        /// <summary>コンストラクタ</summary>
        public Hardmax()
            : base(inputs: 1, outputs: 1, allow_resubstitution: false) { }

        /// <summary>出力テンソル形状を返す</summary>
        public override Shape[] OutputShapes(params Shape[] inshapes) {
            CheckInputShapes(inshapes);

            return inshapes;
        }

        public override void CheckInputShapes(params Shape[] inshapes) {
            base.CheckInputShapes(inshapes);

            if (inshapes[0].Type != ShapeType.Map) {
                throw new ArgumentException(ExceptionMessage.TensorType(inshapes[0].Type, ShapeType.Map));
            }
        }

        /// <summary>操作クラスを生成する</summary>
        internal override (Tensor[] tensors, Operator ope) GenerateOperator(Tensor[] intensors, Tensor[] outtensors) {
            CheckArgumentsCount(intensors, outtensors);

            Tensor intensor = intensors[0], outtensor = outtensors[0];

            return (new Tensor[] { intensor, outtensor },
                new Operators.Channelwise.Hardmax(intensor.Channels, intensor.Shape));
        }
    }
}
