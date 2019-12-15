using System.Collections.Generic;

namespace TensorShader.Operators.ArrayManipulation {
    /// <summary>要素コピー</summary>
    internal class CopyTo : Operator {
        /// <summary>形状</summary>
        public Shape Shape { private set; get; }

        /// <summary>コンストラクタ</summary>
        public CopyTo(Shape shape) {
            this.arguments = new List<(ArgumentType type, Shape shape)>{
                (ArgumentType.In, shape),
                (ArgumentType.Out, shape),
            };

            this.Shape = shape;
        }

        /// <summary>操作を実行</summary>
        public override void Execute(params Tensor[] tensors) {
            CheckArgumentShapes(tensors);

            Tensor inmap = tensors[0], outmap = tensors[1];

            if (inmap != outmap) {
                inmap.CopyTo(outmap);
            }
        }

        /// <summary>操作を実行</summary>
        public void Execute(Tensor inmap, Tensor outmap) {
            Execute(new Tensor[] { inmap, outmap });
        }
    }
}
