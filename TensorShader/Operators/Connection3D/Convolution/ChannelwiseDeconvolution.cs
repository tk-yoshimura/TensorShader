using System.Collections.Generic;

namespace TensorShader.Operators.Connection3D {
    /// <summary>チャネルごとの3次元逆畳み込み</summary>
    internal class ChannelwiseDeconvolution : Operator {
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

        /// <summary>バッチサイズ</summary>
        public int Batch { private set; get; }

        /// <summary>コンストラクタ</summary>
        public ChannelwiseDeconvolution(int outwidth, int outheight, int outdepth, int channels, int kwidth, int kheight, int kdepth, int batch = 1) {
            int inwidth = outwidth - kwidth + 1;
            int inheight = outheight - kheight + 1;
            int indepth = outdepth - kdepth + 1;

            this.arguments = new List<(ArgumentType type, Shape shape)>{
                (ArgumentType.In, Shape.Map3D(channels, inwidth, inheight, indepth, batch)),
                (ArgumentType.In, Shape.Kernel3D(channels, 1, kwidth, kheight, kdepth)),
                (ArgumentType.Out, Shape.Map3D(channels, outwidth, outheight, outdepth, batch)),
            };

            this.Channels = channels;
            this.KernelWidth = kwidth;
            this.KernelHeight = kheight;
            this.KernelDepth = kdepth;
            this.Batch = batch;
        }

        /// <summary>操作を実行</summary>
        public override void Execute(params Tensor[] tensors) {
            CheckArgumentShapes(tensors);

            Tensor inmap = tensors[0], infilter = tensors[1], outmap = tensors[2];

            TensorShaderCudaBackend.Convolution.ChannelwiseDeconvolution3D((uint)Channels,
                                                                           (uint)outmap.Width, (uint)outmap.Height, (uint)outmap.Depth, (uint)Batch, 
                                                                           (uint)KernelWidth, (uint)KernelHeight, (uint)KernelDepth,
                                                                           inmap.Buffer, infilter.Buffer, outmap.Buffer);
        }

        /// <summary>操作を実行</summary>
        public void Execute(Tensor inmap, Tensor infilter, Tensor outmap) {
            Execute(new Tensor[] { inmap, infilter, outmap });
        }
    }
}
