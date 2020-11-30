using System.Collections.Generic;

namespace TensorShader.Operators.ArrayManipulation {
    /// <summary>ソート</summary>
    /// <remarks>4要素スライドソート</remarks>
    internal class SortWithKey : Operator {
        /// <summary>形状</summary>
        public Shape Shape { private set; get; }

        /// <summary>ストライド</summary>
        public int Stride { private set; get; }

        /// <summary>軸長さ</summary>
        public int AxisLength { private set; get; }

        /// <summary>コンストラクタ</summary>
        public SortWithKey(Shape shape, int axis) {
            this.arguments = new List<(ArgumentType type, Shape shape)>{
                (ArgumentType.In, shape),
                (ArgumentType.In, shape),
                (ArgumentType.Out, shape),
                (ArgumentType.Out, shape),
            };

            this.Shape = shape;
            this.Stride = shape.Stride(axis);
            this.AxisLength = shape[axis];
        }

        /// <summary>操作を実行</summary>
        public override void Execute(params Tensor[] tensors) {
            CheckArgumentShapes(tensors);

            Tensor intensor_key = tensors[0], intensor_value = tensors[1];
            Tensor outtensor_key = tensors[2], outtensor_value = tensors[3];

            TensorShaderCudaBackend.ArrayManipulation.SortWithKey((uint)Stride, (uint)AxisLength, (uint)(outtensor_key.Length / (Stride * AxisLength)), intensor_key.Buffer, intensor_value.Buffer, outtensor_key.Buffer, outtensor_value.Buffer);
        }

        /// <summary>操作を実行</summary>
        public void Execute(Tensor intensor_key, Tensor intensor_value, Tensor outtensor_key, Tensor outtensor_value) {
            Execute(new Tensor[] { intensor_key, intensor_value, outtensor_key, outtensor_value });
        }
    }
}
