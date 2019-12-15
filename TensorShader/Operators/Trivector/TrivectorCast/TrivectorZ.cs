using System;
using System.Collections.Generic;
using System.Linq;

namespace TensorShader.Operators.TrivectorCast {
    /// <summary>3次元ベクトルZ成分</summary>
    internal class TrivectorZ : Operator {
        /// <summary>形状</summary>
        public Shape Shape { private set; get; }

        /// <summary>コンストラクタ</summary>
        public TrivectorZ(Shape inshape) {
            if (!new ShapeType[] { ShapeType.Vector, ShapeType.Map, ShapeType.Kernel }.Contains(inshape.Type)) {
                throw new ArgumentException(ExceptionMessage.Shape("Type", inshape));
            }

            int[] s = inshape;
            s[0] /= 3;

            this.arguments = new List<(ArgumentType type, Shape shape)>{
                (ArgumentType.In, inshape),
                (ArgumentType.Out, new Shape(inshape.Type, s)),
            };

            this.Shape = inshape;
        }

        /// <summary>操作を実行</summary>
        public override void Execute(params Tensor[] tensors) {
            CheckArgumentShapes(tensors);

            Tensor inmap = tensors[0], outmap = tensors[1];

            TensorShaderCudaBackend.Trivector.Z((uint)inmap.Length, inmap.Buffer, outmap.Buffer);
        }

        /// <summary>操作を実行</summary>
        public void Execute(Tensor inmap, Tensor outmap) {
            Execute(new Tensor[] { inmap, outmap });
        }
    }
}
