using System;
using System.Collections.Generic;

namespace TensorShader.Operators.ArrayManipulation {
    /// <summary>チャネル拡張</summary>
    internal class ExpandChannel : Operator {
        /// <summary>形状</summary>
        public Shape Shape { private set; get; }

        /// <summary>拡張数</summary>
        public int Expands { private set; get; }

        /// <summary>コンストラクタ</summary>
        public ExpandChannel(Shape shape, int expands) {
            if (shape.Ndim <= 0) {
                throw new ArgumentException(ExceptionMessage.Shape("Ndim", shape));
            }

            if (expands < 1) {
                throw new ArgumentException(nameof(expands));
            }

            int[] s = shape;

            s[0] *= expands;

            this.arguments = new List<(ArgumentType type, Shape shape)>{
                (ArgumentType.In, shape),
                (ArgumentType.Out, new Shape(shape.Type, s)),
            };

            this.Shape = shape;
            this.Expands = expands;
        }

        /// <summary>操作を実行</summary>
        public override void Execute(params Tensor[] tensors) {
            CheckArgumentShapes(tensors);

            Tensor inmap = tensors[0], outmap = tensors[1];

            if (Expands <= 1) {
                inmap.CopyTo(outmap);
            }
            else {
                TensorShaderCudaBackend.ArrayManipulation.Broadcast(1, inmap.Buffer, (uint)Expands, outmap.Buffer, (uint)inmap.Length);
            }
        }

        /// <summary>操作を実行</summary>
        public void Execute(Tensor inmap, Tensor outmap) {
            Execute(new Tensor[] { inmap, outmap });
        }
    }
}
