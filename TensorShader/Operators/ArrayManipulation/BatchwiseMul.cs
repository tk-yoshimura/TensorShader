using System.Collections.Generic;

namespace TensorShader.Operators.ArrayManipulation {
    /// <summary>バッチ方向に乗算</summary>
    internal class BatchwiseMul : Operator {
        /// <summary>コンストラクタ</summary>
        public BatchwiseMul(Shape shape) {
            this.arguments = new List<(ArgumentType type, Shape shape)>{
                (ArgumentType.In, shape),
                (ArgumentType.In, Shape.Vector(shape.Batch)),
                (ArgumentType.Out, shape),
            };
        }

        /// <summary>操作を実行</summary>
        public override void Execute(params Tensor[] tensors) {
            CheckArgumentShapes(tensors);

            Tensor inmap = tensors[0], invector = tensors[1], outmap = tensors[2];

            TensorShaderCudaBackend.Batchwise.Mul((uint)invector.Length, (uint)inmap.Length, invector.Buffer, inmap.Buffer, outmap.Buffer);
        }

        /// <summary>操作を実行</summary>
        public void Execute(Tensor inmap, Tensor invector, Tensor outmap) {
            Execute(new Tensor[] { inmap, invector, outmap });
        }
    }
}
