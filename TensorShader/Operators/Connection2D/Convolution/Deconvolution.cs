using System.Collections.Generic;

namespace TensorShader.Operators.Connection2D {
    /// <summary>2次元逆畳み込み</summary>
    internal class Deconvolution : Operator {
        /// <summary>入力チャネル</summary>
        public int InChannels { private set; get; }

        /// <summary>出力チャネル</summary>
        public int OutChannels { private set; get; }

        /// <summary>フィルタサイズ</summary>
        /// <remarks>奇数を指定すること</remarks>
        public int KernelWidth { private set; get; }

        /// <summary>フィルタサイズ</summary>
        /// <remarks>奇数を指定すること</remarks>
        public int KernelHeight { private set; get; }

        /// <summary>バッチサイズ</summary>
        public int Batch { private set; get; }

        /// <summary>コンストラクタ</summary>
        public Deconvolution(int outwidth, int outheight, int inchannels, int outchannels, int kwidth, int kheight, int batch = 1) {
            
            int inwidth = outwidth - kwidth + 1;
            int inheight = outheight - kheight + 1;

            this.arguments = new List<(ArgumentType type, Shape shape)>{
                (ArgumentType.In, Shape.Map2D(inchannels, inwidth, inheight, batch)),
                (ArgumentType.In, Shape.Kernel2D(outchannels, inchannels, kwidth, kheight)),
                (ArgumentType.Out, Shape.Map2D(outchannels, outwidth, outheight, batch)),
            };

            this.InChannels = inchannels;
            this.OutChannels = outchannels;
            this.KernelWidth = kwidth;
            this.KernelHeight = kheight;
            this.Batch = batch;
        }

        /// <summary>操作を実行</summary>
        public override void Execute(params Tensor[] tensors) {
            CheckArgumentShapes(tensors);

            Tensor inmap = tensors[0], infilter = tensors[1], outmap = tensors[2];

            TensorShaderCudaBackend.Convolution.Deconvolution2D((uint)InChannels, (uint)OutChannels,
                                                                (uint)outmap.Width, (uint)outmap.Height, (uint)Batch, 
                                                                (uint)KernelWidth, (uint)KernelHeight, 
                                                                inmap.Buffer, infilter.Buffer, outmap.Buffer);
        }

        /// <summary>操作を実行</summary>
        public void Execute(Tensor inmap, Tensor infilter, Tensor outmap) {
            Execute(new Tensor[] { inmap, infilter, outmap });
        }
    }
}
