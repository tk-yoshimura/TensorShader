using System;
using System.Collections.Generic;
using System.Linq;

namespace TensorShader.Operators.QuaternionUnaryArithmetric {
    /// <summary>四元数1項演算</summary>
    internal abstract class QuaternionUnaryArithmetric : Operator {
        /// <summary>形状</summary>
        public Shape Shape { private set; get; }

        /// <summary>要素数</summary>
        public int Elements { private set; get; }

        /// <summary>コンストラクタ</summary>
        public QuaternionUnaryArithmetric(Shape shape) {
            if (!new ShapeType[] { ShapeType.Vector, ShapeType.Map, ShapeType.Kernel }.Contains(shape.Type)) {
                throw new ArgumentException(ExceptionMessage.Shape("Type", shape));
            }

            if (shape.Channels % 4 != 0) {
                throw new ArgumentException(ExceptionMessage.TensorLengthMultiple("Channels", shape, shape.Channels, 4));
            }
            if (shape.InChannels % 4 != 0) {
                throw new ArgumentException(ExceptionMessage.TensorLengthMultiple("InChannels", shape, shape.InChannels, 4));
            }

            this.Elements = shape.Length / 4;

            this.arguments = new List<(ArgumentType type, Shape shape)>{
                (ArgumentType.In, shape),
                (ArgumentType.Out, shape),
            };

            this.Shape = shape;
        }

        /// <summary>操作を実行</summary>
        public void Execute(Tensor inmap, Tensor outmap) {
            Execute(new Tensor[] { inmap, outmap });
        }
    }
}
