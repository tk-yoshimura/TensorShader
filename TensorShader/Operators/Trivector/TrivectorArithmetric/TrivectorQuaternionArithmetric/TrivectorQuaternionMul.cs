using System;
using System.Collections.Generic;
using System.Linq;

namespace TensorShader.Operators.TrivectorQuaternionArithmetric {
    /// <summary>3次元ベクトル四元数回転積</summary>
    internal class TrivectorQuaternionMul : Operator {
        /// <summary>形状</summary>
        public Shape Shape { private set; get; }

        /// <summary>コンストラクタ</summary>
        public TrivectorQuaternionMul(Shape vectorshape) {
            if (!new ShapeType[] { ShapeType.Vector, ShapeType.Map, ShapeType.Kernel }.Contains(vectorshape.Type)) {
                throw new ArgumentException(ExceptionMessage.Shape("Type", vectorshape));
            }

            if (vectorshape.Channels % 3 != 0) {
                throw new ArgumentException(ExceptionMessage.LengthMultiple("Channels", vectorshape, vectorshape.Channels, 3));
            }
            if (vectorshape.InChannels % 3 != 0) {
                throw new ArgumentException(ExceptionMessage.LengthMultiple("InChannels", vectorshape, vectorshape.InChannels, 3));
            }

            int[] s = vectorshape;
            s[0] = s[0] / 3 * 4;

            Shape quatshape = new(vectorshape.Type, s);

            this.arguments = new List<(ArgumentType type, Shape shape)>{
                (ArgumentType.In, vectorshape),
                (ArgumentType.In, quatshape),
                (ArgumentType.Out, vectorshape),
            };

            this.Shape = vectorshape;
        }

        /// <summary>操作を実行</summary>
        public override void Execute(params Tensor[] tensors) {
            Tensor inmap1 = tensors[0], inmap2 = tensors[1], outmap = tensors[2];

            TensorShaderCudaBackend.Trivector.Mul((uint)inmap1.Length, inmap1.Buffer, inmap2.Buffer, outmap.Buffer);
        }

        /// <summary>操作を実行</summary>
        public void Execute(Tensor inmap1, Tensor inmap2, Tensor outmap) {
            Execute(new Tensor[] { inmap1, inmap2, outmap });
        }

    }
}
