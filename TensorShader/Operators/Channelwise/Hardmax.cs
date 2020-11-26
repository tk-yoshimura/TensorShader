using System;
using System.Collections.Generic;

namespace TensorShader.Operators.Channelwise {
    /// <summary>Hardmax</summary>
    internal class Hardmax : Operator {
        /// <summary>チャネル</summary>
        public int Channels { private set; get; }

        /// <summary>形状</summary>
        public Shape Shape { private set; get; }

        /// <summary>コンストラクタ</summary>
        public Hardmax(int channels, Shape shape) {
            this.arguments = new List<(ArgumentType type, Shape shape)>{
                (ArgumentType.In, shape),
                (ArgumentType.Out, shape),
            };

            this.Channels = channels;
            this.Shape = shape;
        }

        /// <summary>操作を実行</summary>
        public override void Execute(params Tensor[] tensors) {
            CheckArgumentShapes(tensors);

            Tensor inmap = tensors[0], outmap = tensors[1];

            TensorShaderCudaBackend.Channelwise.Hardmax((uint)outmap.Length, (uint)Channels, inmap.Buffer, outmap.Buffer);
        }

        /// <summary>操作を実行</summary>
        public void Execute(Tensor inmap, Tensor outmap) {
            Execute(new Tensor[] { inmap, outmap });
        }
    }
}
