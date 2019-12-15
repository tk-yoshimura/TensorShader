using System;
using System.Collections.Generic;
using System.Linq;

namespace TensorShader.Operators.TrivectorCast {
    /// <summary>実数から3次元ベクトルを構成</summary>
    internal class TrivectorCast : Operator {
        /// <summary>形状</summary>
        public Shape Shape { private set; get; }

        /// <summary>コンストラクタ</summary>
        public TrivectorCast(Shape inshape) {
            if (!new ShapeType[] { ShapeType.Vector, ShapeType.Map, ShapeType.Kernel }.Contains(inshape.Type)) {
                throw new ArgumentException(ExceptionMessage.Shape("Type", inshape));
            }

            int[] s = inshape;
            s[0] *= 3;

            this.arguments = new List<(ArgumentType type, Shape shape)>{
                (ArgumentType.In, inshape),
                (ArgumentType.In, inshape),
                (ArgumentType.In, inshape),
                (ArgumentType.Out, new Shape(inshape.Type, s)),
            };

            this.Shape = inshape;
        }

        /// <summary>操作を実行</summary>
        public override void Execute(params Tensor[] tensors) {
            CheckArgumentShapes(tensors);

            Tensor inmap1 = tensors[0], inmap2 = tensors[1], inmap3 = tensors[2], outmap = tensors[3];

            TensorShaderCudaBackend.Trivector.Cast((uint)outmap.Length, inmap1.Buffer, inmap2.Buffer, inmap3.Buffer, outmap.Buffer);
        }

        /// <summary>操作を実行</summary>
        public void Execute(Tensor inmap1, Tensor inmap2, Tensor inmap3, Tensor outmap) {
            Execute(new Tensor[] { inmap1, inmap2, inmap3, outmap });
        }
    }
}
