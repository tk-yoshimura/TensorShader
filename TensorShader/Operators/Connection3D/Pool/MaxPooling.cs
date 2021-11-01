using System;
using System.Collections.Generic;

namespace TensorShader.Operators.Connection3D {
    /// <summary>最大値プーリング</summary>
    internal class MaxPooling : Operator {
        /// <summary>チャネル数</summary>
        public int Channels { private set; get; }

        /// <summary>ストライド</summary>
        public int Stride { private set; get; }

        /// <summary>バッチサイズ</summary>
        public int Batch { private set; get; }

        /// <summary>コンストラクタ</summary>
        public MaxPooling(int inwidth, int inheight, int indepth, int channels, int stride, int batch = 1) {
            if (stride < 2) {
                throw new ArgumentException(null, nameof(stride));
            }

            this.arguments = new List<(ArgumentType type, Shape shape)>{
                (ArgumentType.In, Shape.Map3D(channels, inwidth, inheight, indepth, batch)),
                (ArgumentType.Out, Shape.Map3D(channels, inwidth / stride, inheight / stride, indepth / stride, batch)),
            };

            this.Channels = channels;
            this.Stride = stride;
            this.Batch = batch;
        }

        /// <summary>操作を実行</summary>
        public override void Execute(params Tensor[] tensors) {
            CheckArgumentShapes(tensors);

            Tensor inmap = tensors[0], outmap = tensors[1];

            TensorShaderCudaBackend.Pool.MaxPool3D((uint)Channels, (uint)inmap.Width, (uint)inmap.Height, (uint)inmap.Depth, (uint)Batch, (uint)Stride, inmap.Buffer, outmap.Buffer);
        }
    }
}
