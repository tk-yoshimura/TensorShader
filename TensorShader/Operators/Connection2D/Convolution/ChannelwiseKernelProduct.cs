using System.Collections.Generic;

namespace TensorShader.Operators.Connection2D {
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

        /// <summary>バッチサイズ</summary>
        public int Batch { private set; get; }

        /// <summary>コンストラクタ</summary>
        public ChannelwiseKernelProduct(int inwidth, int inheight, int channels, int kwidth, int kheight, int batch = 1) {
            int outwidth = inwidth - kwidth + 1;
            int outheight = inheight - kheight + 1;

            this.arguments = new List<(ArgumentType type, Shape shape)>{
                (ArgumentType.In, Shape.Map2D(channels, inwidth, inheight, batch)),
                (ArgumentType.In, Shape.Map2D(channels, outwidth, outheight, batch)),
                (ArgumentType.Out, Shape.Kernel2D(channels, 1, kwidth, kheight)),
            };

            this.Channels = channels;
            this.KernelWidth = kwidth;
            this.KernelHeight = kheight;
            this.Batch = batch;
        }

        /// <summary>操作を実行</summary>
        public override void Execute(params Tensor[] tensors) {
            CheckArgumentShapes(tensors);

            Tensor inmap1 = tensors[0], inmap2 = tensors[1], outfilter = tensors[2];

            TensorShaderCudaBackend.Convolution.ChannelwiseKernelProduct2D((uint)Channels,
                                                                           (uint)inmap1.Width, (uint)inmap1.Height,
                                                                           (uint)Batch,
                                                                           (uint)KernelWidth, (uint)KernelHeight,
                                                                           inmap1.Buffer, inmap2.Buffer, outfilter.Buffer);
        }

        /// <summary>操作を実行</summary>
        public void Execute(Tensor inmap1, Tensor inmap2, Tensor outfilter) {
            Execute(new Tensor[] { inmap1, inmap2, outfilter });
        }
    }
}
