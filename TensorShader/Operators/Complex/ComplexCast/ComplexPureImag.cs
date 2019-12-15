using System;
using System.Collections.Generic;
using System.Linq;

namespace TensorShader.Operators.ComplexCast {
    /// <summary>複素純虚数</summary>
    internal class ComplexPureImag : Operator {
        /// <summary>形状</summary>
        public Shape Shape { private set; get; }

        /// <summary>コンストラクタ</summary>
        public ComplexPureImag(Shape inshape) {
            if (!new ShapeType[] { ShapeType.Vector, ShapeType.Map, ShapeType.Kernel }.Contains(inshape.Type)) {
                throw new ArgumentException(ExceptionMessage.Shape("Type", inshape));
            }

            int[] s = inshape;
            s[0] *= 2;

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

            TensorShaderCudaBackend.Complex.PureImag((uint)outmap.Length, inmap.Buffer, outmap.Buffer);
        }

        /// <summary>操作を実行</summary>
        public void Execute(Tensor inmap, Tensor outmap) {
            Execute(new Tensor[] { inmap, outmap });
        }
    }
}
