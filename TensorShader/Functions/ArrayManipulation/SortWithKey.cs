using System;
using System.Linq;

namespace TensorShader {
    public abstract partial class VariableNode {
        /// <summary>ソート</summary>
        public static (VariableNode key, VariableNode value) SortWithKey(VariableNode k, VariableNode v, int axis) {
            Function function = new Functions.ArrayManipulation.SortWithKey(axis);

            VariableNode[] kv = Apply(function, k, v);

            return (kv[0], kv[1]);
        }
    }

    public partial class Tensor {
        /// <summary>ソート</summary>
        public static (Tensor key, Tensor value) SortWithKey(Tensor k, Tensor v, int axis) {
            Function function = new Functions.ArrayManipulation.SortWithKey(axis);

            Tensor key = new(k.Shape), value = new(v.Shape);

            function.Execute(new Tensor[] { k, v }, new Tensor[] { key, value });

            return (key, value);
        }
    }
}

namespace TensorShader.Functions.ArrayManipulation {
    /// <summary>ソート</summary>
    internal class SortWithKey : Function {
        /// <summary>軸</summary>
        public int Axis { private set; get; }

        /// <summary>コンストラクタ</summary>
        public SortWithKey(int axis)
            : base(inputs: 2, outputs: 2, allow_resubstitution: false) {

            this.Axis = axis;
        }

        /// <summary>出力テンソル形状を返す</summary>
        public override Shape[] OutputShapes(params Shape[] inshapes) {
            CheckInputShapes(inshapes);

            return inshapes;
        }

        public override void CheckInputShapes(params Shape[] inshapes) {
            base.CheckInputShapes(inshapes);

            if (inshapes[0] != inshapes[1]) {
                throw new ArgumentException(ExceptionMessage.Shape(inshapes[1], inshapes[0]));
            }
        }

        /// <summary>操作クラスを生成する</summary>
        internal override (Tensor[] tensors, Operator ope) GenerateOperator(Tensor[] intensors, Tensor[] outtensors) {
            CheckArgumentsCount(intensors, outtensors);

            Tensor intensor_key = intensors[0], intensor_value = intensors[1];
            Tensor outtensor_key = outtensors[0], outtensor_value = outtensors[1];

            if (intensors.Concat(outtensors).Distinct().Count() != 4) {
                throw new NotImplementedException();
            }

            return (
                new Tensor[] { intensor_key, intensor_value, outtensor_key, outtensor_value },
                new Operators.ArrayManipulation.SortWithKey(intensor_key.Shape, Axis)
                );
        }
    }
}
