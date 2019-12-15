using System;
using System.Collections.Generic;

namespace TensorShader.Operators.ArrayManipulation {
    /// <summary>インデクサ</summary>
    internal class BatchIndexer : Operator {
        /// <summary>形状</summary>
        public Shape Shape { private set; get; }

        /// <summary>インデクス</summary>
        public int Index { private set; get; }

        /// <summary>コンストラクタ</summary>
        public BatchIndexer(Shape shape, int index) {
            if (shape.Ndim < 2 || shape.Type != ShapeType.Map) {
                throw new ArgumentException(nameof(shape));
            }

            if (index < 0 || index >= shape.Batch) {
                throw new ArgumentOutOfRangeException(nameof(index));
            }

            int[] s = shape;
            s[s.Length - 1] = 1;

            this.arguments = new List<(ArgumentType type, Shape shape)>{
                (ArgumentType.In, shape),
                (ArgumentType.Out, new Shape(ShapeType.Map, s)),
            };

            this.Shape = shape;
        }

        /// <summary>操作を実行</summary>
        public override void Execute(params Tensor[] tensors) {
            CheckArgumentShapes(tensors);

            Tensor inmap = tensors[0], outmap = tensors[1];

            inmap.RegionCopyTo(outmap, (uint)(Shape.DataSize * Index), 0, (uint)Shape.DataSize);
        }

        /// <summary>操作を実行</summary>
        public void Execute(Tensor inmap, Tensor outmap) {
            Execute(new Tensor[] { inmap, outmap });
        }
    }
}
