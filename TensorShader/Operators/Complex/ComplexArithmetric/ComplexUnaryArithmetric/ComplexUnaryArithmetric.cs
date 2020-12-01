using System;
using System.Collections.Generic;
using System.Linq;

namespace TensorShader.Operators.ComplexUnaryArithmetric {
    /// <summary>複素1項演算</summary>
    internal abstract class ComplexUnaryArithmetric : Operator {
        /// <summary>形状</summary>
        public Shape Shape { private set; get; }

        /// <summary>要素数</summary>
        public int Elements { private set; get; }

        /// <summary>コンストラクタ</summary>
        public ComplexUnaryArithmetric(Shape shape) {
            if (!new ShapeType[] { ShapeType.Vector, ShapeType.Map, ShapeType.Kernel }.Contains(shape.Type)) {
                throw new ArgumentException(ExceptionMessage.Shape("Type", shape));
            }

            if (shape.Channels % 2 != 0) {
                throw new ArgumentException(ExceptionMessage.LengthMultiple("Channels", shape, shape.Channels, 2));
            }
            if (shape.InChannels % 2 != 0) {
                throw new ArgumentException(ExceptionMessage.LengthMultiple("InChannels", shape, shape.InChannels, 2));
            }

            this.Elements = shape.Length / 2;

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
