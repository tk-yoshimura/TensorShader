using System.Collections.Generic;

namespace TensorShader.Operators.Connection1D {
    /// <summary>チャネルごとのカーネル積</summary>
    internal class ChannelwiseKernelProduct : Operator {
        /// <summary>チャネル</summary>
        public int Channels { private set; get; }

        /// <summary>フィルタサイズ</summary>
        /// <remarks>奇数を指定すること</remarks>
        public int KernelWidth { private set; get; }

        /// <summary>バッチサイズ</summary>
        public int Batch { private set; get; }

        /// <summary>コンストラクタ</summary>
        public ChannelwiseKernelProduct(int inwidth, int channels, int kwidth, int batch = 1) {
            int outwidth = inwidth - kwidth + 1;

            this.arguments = new List<(ArgumentType type, Shape shape)>{
                (ArgumentType.In, Shape.Map1D(channels, inwidth, batch)),
                (ArgumentType.In, Shape.Map1D(channels, outwidth, batch)),
                (ArgumentType.Out, Shape.Kernel1D(channels, 1, kwidth)),
            };

            this.Channels = channels;
            this.KernelWidth = kwidth;
            this.Batch = batch;
        }

        /// <summary>操作を実行</summary>
        public override void Execute(params Tensor[] tensors) {
            CheckArgumentShapes(tensors);

            Tensor inmap1 = tensors[0], inmap2 = tensors[1], outfilter = tensors[2];

            TensorShaderCudaBackend.Convolution.ChannelwiseKernelProduct1D((uint)Channels,
                                                                           (uint)inmap1.Width,
                                                                           (uint)Batch,
                                                                           (uint)KernelWidth, 
                                                                           inmap1.Buffer, inmap2.Buffer, outfilter.Buffer);
        }

        /// <summary>操作を実行</summary>
        public void Execute(Tensor inmap1, Tensor inmap2, Tensor outfilter) {
            Execute(new Tensor[] { inmap1, inmap2, outfilter });
        }
    }
}
