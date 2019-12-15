using System;
using System.Collections.Generic;

namespace TensorShader.Operators.Connection3D {
    /// <summary>チャネルごとのカーネル積</summary>
    internal class ChannelwiseKernelProduct : Operator {
        /// <summary>チャネル</summary>
        public int Channels { private set; get; }

        /// <summary>フィルタサイズ</summary>
        /// <remarks>奇数を指定すること</remarks>
        public int KernelWidth { private set; get; }

        /// <summary>フィルタサイズ</summary>
        /// <remarks>奇数を指定すること</remarks>
        public int KernelHeight { private set; get; }

        /// <summary>フィルタサイズ</summary>
        /// <remarks>奇数を指定すること</remarks>
        public int KernelDepth { private set; get; }

        /// <summary>ストライド</summary>
        public int Stride { private set; get; }

        /// <summary>バッチサイズ</summary>
        public int Batch { private set; get; }

        /// <summary>コンストラクタ</summary>
        public ChannelwiseKernelProduct(int inwidth, int inheight, int indepth, int channels, int kwidth, int kheight, int kdepth, int stride, int batch = 1) {
            if (stride < 1) {
                throw new ArgumentException(nameof(stride));
            }

            int outwidth = (inwidth - kwidth) / stride + 1;
            int outheight = (inheight - kheight) / stride + 1;
            int outdepth = (indepth - kdepth) / stride + 1;

            this.arguments = new List<(ArgumentType type, Shape shape)>{
                (ArgumentType.In, Shape.Map3D(channels, inwidth, inheight, indepth, batch)),
                (ArgumentType.In, Shape.Map3D(channels, outwidth, outheight, outdepth, batch)),
                (ArgumentType.Out, Shape.Kernel3D(channels, 1, kwidth, kheight, kdepth)),
            };

            this.Channels = channels;
            this.KernelWidth = kwidth;
            this.KernelHeight = kheight;
            this.KernelDepth = kdepth;
            this.Stride = stride;
            this.Batch = batch;
        }

        /// <summary>操作を実行</summary>
        public override void Execute(params Tensor[] tensors) {
            CheckArgumentShapes(tensors);

            Tensor inmap1 = tensors[0], inmap2 = tensors[1], outfilter = tensors[2];

            TensorShaderCudaBackend.Convolution.ChannelwiseKernelProduct3D((uint)Channels,
                                                                          (uint)inmap1.Width, (uint)inmap1.Height, (uint)inmap1.Depth,
                                                                          (uint)Batch,
                                                                          (uint)KernelWidth, (uint)KernelHeight, (uint)KernelDepth, (uint)Stride,
                                                                          inmap1.Buffer, inmap2.Buffer, outfilter.Buffer);
        }

        /// <summary>操作を実行</summary>
        public void Execute(Tensor inmap1, Tensor inmap2, Tensor outfilter) {
            Execute(new Tensor[] { inmap1, inmap2, outfilter });
        }
    }
}
