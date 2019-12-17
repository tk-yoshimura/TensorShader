using System;
using System.Collections.Generic;

namespace TensorShader.Operators.Connection3D {
    /// <summary>ストライド逆プーリング</summary>
    internal class StrideUnpooling : Operator {
        /// <summary>チャネル数</summary>
        public int Channels { private set; get; }

        /// <summary>ストライド</summary>
        public int Stride { private set; get; }

        /// <summary>バッチサイズ</summary>
        public int Batch { private set; get; }

        /// <summary>コンストラクタ</summary>
        public StrideUnpooling(int outwidth, int outheight, int outdepth, int channels, int stride, int batch = 1) {
            if (stride < 2) {
                throw new ArgumentException(nameof(stride));
            }

            this.arguments = new List<(ArgumentType type, Shape shape)>{
                (ArgumentType.In, Shape.Map3D(channels, outwidth / stride, outheight / stride, outdepth / stride, batch)),
                (ArgumentType.Out, Shape.Map3D(channels, outwidth, outheight, outdepth, batch)),
            };

            this.Channels = channels;
            this.Stride = stride;
            this.Batch = batch;
        }

        /// <summary>操作を実行</summary>
        public override void Execute(params Tensor[] tensors) {
            CheckArgumentShapes(tensors);

            Tensor inmap = tensors[0], outmap = tensors[1];

            TensorShaderCudaBackend.Pool.StrideUnpool3D((uint)Channels, (uint)outmap.Width, (uint)outmap.Height, (uint)outmap.Depth, (uint)Batch, (uint)Stride, inmap.Buffer, outmap.Buffer);
        }

        /// <summary>操作を実行</summary>
        public void Execute(Tensor inmap, Tensor outmap) {
            Execute(new Tensor[] { inmap, outmap });
        }
    }
}
