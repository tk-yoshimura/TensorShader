using System;
using System.Collections.Generic;

namespace TensorShader.Operators.Connection3D {
    /// <summary>最大値逆プーリング</summary>
    internal class MaxUnpooling : Operator {
        /// <summary>チャネル数</summary>
        public int Channels { private set; get; }

        /// <summary>ストライド</summary>
        public int Stride { private set; get; }

        /// <summary>バッチサイズ</summary>
        public int Batch { private set; get; }

        /// <summary>コンストラクタ</summary>
        public MaxUnpooling(int outwidth, int outheight, int outdepth, int channels, int stride, int batch = 1) {
            if (stride < 2) {
                throw new ArgumentException(nameof(stride));
            }

            this.arguments = new List<(ArgumentType type, Shape shape)>{
                (ArgumentType.In, Shape.Map3D(channels, outwidth / stride, outheight / stride, outdepth / stride, batch)),
                (ArgumentType.In, Shape.Map3D(channels, outwidth, outheight, outdepth, batch)),
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

            Tensor gradmap = tensors[0], valmap = tensors[1], poolmap = tensors[2], outmap = tensors[3];

            TensorShaderCudaBackend.Pool.MaxUnpool3D((uint)Channels, (uint)outmap.Width, (uint)outmap.Height, (uint)outmap.Depth, (uint)Batch, (uint)Stride, gradmap.Buffer, poolmap.Buffer, valmap.Buffer, outmap.Buffer);
        }

        /// <summary>操作を実行</summary>
        public void Execute(Tensor gradmap, Tensor valmap, Tensor poolmap, Tensor outmap) {
            Execute(new Tensor[] { gradmap, valmap, poolmap, outmap });
        }
    }
}
