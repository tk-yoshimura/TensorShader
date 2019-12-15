using System;
using System.Collections.Generic;
using System.Linq;

namespace TensorShader.Operators.ComplexCast {
    /// <summary>実数から複素数を構成</summary>
    internal class ComplexCast : Operator {
        /// <summary>形状</summary>
        public Shape Shape { private set; get; }

        /// <summary>コンストラクタ</summary>
        public ComplexCast(Shape inshape) {
            if (!new ShapeType[] { ShapeType.Vector, ShapeType.Map, ShapeType.Kernel }.Contains(inshape.Type)) {
                throw new ArgumentException(ExceptionMessage.Shape("Type", inshape));
            }

            int[] s = inshape;
            s[0] *= 2;

            this.arguments = new List<(ArgumentType type, Shape shape)>{
                (ArgumentType.In, inshape),
                (ArgumentType.In, inshape),
                (ArgumentType.Out, new Shape(inshape.Type, s)),
            };

            this.Shape = inshape;
        }

        /// <summary>操作を実行</summary>
        public override void Execute(params Tensor[] tensors) {
            CheckArgumentShapes(tensors);

            Tensor inmap1 = tensors[0], inmap2 = tensors[1], outmap = tensors[2];

            TensorShaderCudaBackend.Complex.Cast((uint)outmap.Length, inmap1.Buffer, inmap2.Buffer, outmap.Buffer);
        }

        /// <summary>操作を実行</summary>
        public void Execute(Tensor inmap1, Tensor inmap2, Tensor outmap) {
            Execute(new Tensor[] { inmap1, inmap2, outmap });
        }
    }
}
