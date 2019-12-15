using System;
using System.Collections.Generic;
using System.Linq;

namespace TensorShader.Operators.TrivectorUnaryArithmetric {
    /// <summary>3次元ベクトル1項演算</summary>
    internal abstract class TrivectorUnaryArithmetric : Operator {
        /// <summary>形状</summary>
        public Shape Shape { private set; get; }

        /// <summary>要素数</summary>
        public int Elements { private set; get; }

        /// <summary>コンストラクタ</summary>
        public TrivectorUnaryArithmetric(Shape shape) {
            if (!new ShapeType[] { ShapeType.Vector, ShapeType.Map, ShapeType.Kernel }.Contains(shape.Type)) {
                throw new ArgumentException(ExceptionMessage.Shape("Type", shape));
            }

            if (shape.Channels % 3 != 0) {
                throw new ArgumentException(ExceptionMessage.TensorLengthMultiple("Channels", shape, shape.Channels, 3));
            }
            if (shape.InChannels % 3 != 0) {
                throw new ArgumentException(ExceptionMessage.TensorLengthMultiple("InChannels", shape, shape.InChannels, 3));
            }

            this.Elements = shape.Length / 3;

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
