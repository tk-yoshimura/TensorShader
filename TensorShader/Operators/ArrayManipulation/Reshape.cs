using System;
using System.Collections.Generic;

namespace TensorShader.Operators.ArrayManipulation {
    /// <summary>形状変更</summary>
    internal class Reshape : Operator {
        /// <summary>コンストラクタ</summary>
        public Reshape(Shape inshape, Shape outshape) {
            if (inshape.Length != outshape.Length) {
                throw new ArgumentException($"{nameof(inshape)}, {nameof(outshape)}");
            }

            this.arguments = new List<(ArgumentType type, Shape shape)>{
                (ArgumentType.In, inshape),
                (ArgumentType.Out, outshape),
            };
        }

        /// <summary>操作を実行</summary>
        public override void Execute(params Tensor[] tensors) {
            CheckArgumentShapes(tensors);

            Tensor inmap = tensors[0], outmap = tensors[1];

            if (inmap != outmap) {
                inmap.CopyTo(outmap);
            }
            outmap.Reshape(arguments[1].shape);
        }

        /// <summary>操作を実行</summary>
        public void Execute(Tensor inmap, Tensor outmap) {
            Execute(new Tensor[] { inmap, outmap });
        }
    }
}
