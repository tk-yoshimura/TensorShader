using System;
using System.Collections.Generic;
using System.Linq;

namespace TensorShader.Operators.TrivectorQuaternionArithmetric {
    /// <summary>3次元ベクトル四元数回転積四元数勾配</summary>
    internal class TrivectorQuaternionMulQGrad : Operator {
        /// <summary>形状</summary>
        public Shape Shape { private set; get; }

        /// <summary>コンストラクタ</summary>
        public TrivectorQuaternionMulQGrad(Shape vectorshape) {
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
                (ArgumentType.In, vectorshape),
                (ArgumentType.In, quatshape),
                (ArgumentType.Out, quatshape),
            };

            this.Shape = vectorshape;
        }

        /// <summary>操作を実行</summary>
        public override void Execute(params Tensor[] tensors) {
            Tensor inmap1 = tensors[0], inmap2 = tensors[1], inmap3 = tensors[2], outmap = tensors[3];

            TensorShaderCudaBackend.Trivector.MulQGrad((uint)inmap1.Length, inmap1.Buffer, inmap2.Buffer, inmap3.Buffer, outmap.Buffer);
        }

        /// <summary>操作を実行</summary>
        public void Execute(Tensor inmap1, Tensor inmap2, Tensor inmap3, Tensor outmap) {
            Execute(new Tensor[] { inmap1, inmap2, inmap3, outmap });
        }

    }
}
