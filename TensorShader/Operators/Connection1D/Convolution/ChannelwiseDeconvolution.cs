using System.Collections.Generic;

namespace TensorShader.Operators.Connection1D {
    /// <summary>チャネルごとの1次元逆畳み込み</summary>
    internal class ChannelwiseDeconvolution : Operator {
        /// <summary>チャネル</summary>
        public int Channels { private set; get; }

        /// <summary>フィルタサイズ</summary>
        /// <remarks>奇数を指定すること</remarks>
        public int KernelWidth { private set; get; }

        /// <summary>バッチサイズ</summary>
        public int Batch { private set; get; }

        /// <summary>コンストラクタ</summary>
        public ChannelwiseDeconvolution(int outwidth, int channels, int kwidth, int batch = 1) {
            int inwidth = outwidth - kwidth + 1;

            this.arguments = new List<(ArgumentType type, Shape shape)>{
                (ArgumentType.In, Shape.Map1D(channels, inwidth, batch)),
                (ArgumentType.In, Shape.Kernel1D(channels, 1, kwidth)),
                (ArgumentType.Out, Shape.Map1D(channels, outwidth, batch)),
            };

            this.Channels = channels;
            this.KernelWidth = kwidth;
            this.Batch = batch;
        }

        /// <summary>操作を実行</summary>
        public override void Execute(params Tensor[] tensors) {
            CheckArgumentShapes(tensors);

            Tensor inmap = tensors[0], infilter = tensors[1], outmap = tensors[2];

            TensorShaderCudaBackend.Convolution.ChannelwiseDeconvolution1D((uint)Channels,
                                                                           (uint)outmap.Width, (uint)Batch,
                                                                           (uint)KernelWidth,
                                                                           inmap.Buffer, infilter.Buffer, outmap.Buffer);
        }

        /// <summary>操作を実行</summary>
        public void Execute(Tensor inmap, Tensor infilter, Tensor outmap) {
            Execute(new Tensor[] { inmap, infilter, outmap });
        }
    }
}
