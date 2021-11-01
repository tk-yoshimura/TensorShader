using System;
using System.Collections.Generic;

namespace TensorShader.Operators.Connection1D {
    /// <summary>平均値逆プーリング</summary>
    internal class AverageUnpooling : Operator {
        /// <summary>チャネル数</summary>
        public int Channels { private set; get; }

        /// <summary>ストライド</summary>
        public int Stride { private set; get; }

        /// <summary>バッチサイズ</summary>
        public int Batch { private set; get; }

        /// <summary>コンストラクタ</summary>
        public AverageUnpooling(int outwidth, int channels, int stride, int batch = 1) {
            if (stride < 2) {
                throw new ArgumentException(null, nameof(stride));
            }

            this.arguments = new List<(ArgumentType type, Shape shape)>{
                (ArgumentType.In, Shape.Map1D(channels, outwidth / stride, batch)),
                (ArgumentType.Out, Shape.Map1D(channels, outwidth, batch)),
            };

            this.Channels = channels;
            this.Stride = stride;
            this.Batch = batch;
        }

        /// <summary>操作を実行</summary>
        public override void Execute(params Tensor[] tensors) {
            CheckArgumentShapes(tensors);

            Tensor inmap = tensors[0], outmap = tensors[1];

            TensorShaderCudaBackend.Pool.AverageUnpool1D((uint)Channels, (uint)outmap.Width, (uint)Batch, (uint)Stride, inmap.Buffer, outmap.Buffer);
        }

        /// <summary>操作を実行</summary>
        public void Execute(Tensor inmap, Tensor outmap) {
            Execute(new Tensor[] { inmap, outmap });
        }
    }
}
