using System;
using System.Collections.Generic;
using System.Threading.Tasks;

namespace TensorShader.Operators.Connection2D {
    /// <summary>チャネルごとの2次元逆畳み込み</summary>
    internal class ChannelwiseDeconvolution : Operator {
        /// <summary>チャネル</summary>
        public int Channels { private set; get; }

        /// <summary>フィルタサイズ</summary>
        /// <remarks>奇数を指定すること</remarks>
        public int KernelWidth { private set; get; }

        /// <summary>フィルタサイズ</summary>
        /// <remarks>奇数を指定すること</remarks>
        public int KernelHeight { private set; get; }

        /// <summary>ストライド</summary>
        public int Stride { private set; get; }

        /// <summary>バッチサイズ</summary>
        public int Batch { private set; get; }

        /// <summary>コンストラクタ</summary>
        public ChannelwiseDeconvolution(int outwidth, int outheight, int channels, int kwidth, int kheight, int stride, int batch = 1) {
            if (stride < 1) {
                throw new ArgumentException(nameof(stride));
            }

            int inwidth = (outwidth - kwidth) / stride + 1;
            int inheight = (outheight - kheight) / stride + 1;

            this.arguments = new List<(ArgumentType type, Shape shape)>{
                (ArgumentType.In, Shape.Map2D(channels, inwidth, inheight, batch)),
                (ArgumentType.In, Shape.Kernel2D(channels, 1, kwidth, kheight)),
                (ArgumentType.Out, Shape.Map2D(channels, outwidth, outheight, batch)),
            };

            this.Channels = channels;
            this.KernelWidth = kwidth;
            this.KernelHeight = kheight;
            this.Stride = stride;
            this.Batch = batch;
        }

        /// <summary>操作を実行</summary>
        public override void Execute(params Tensor[] tensors) {
            CheckArgumentShapes(tensors);

            Tensor inmap = tensors[0], infilter = tensors[1], outmap = tensors[2];

            Parallel.For(0, Batch, (th) => {
                TensorShaderCudaBackend.Convolution.ChannelwiseDeconvolution2D((uint)Channels,
                                                                              (uint)outmap.Width, (uint)outmap.Height, (uint)Batch, (uint)th,
                                                                              (uint)KernelWidth, (uint)KernelHeight, (uint)Stride,
                                                                              inmap.Buffer, infilter.Buffer, outmap.Buffer);
            });
        }

        /// <summary>操作を実行</summary>
        public void Execute(Tensor inmap, Tensor infilter, Tensor outmap) {
            Execute(new Tensor[] { inmap, infilter, outmap });
        }
    }
}
