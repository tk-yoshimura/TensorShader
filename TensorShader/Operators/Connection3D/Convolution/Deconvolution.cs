using System.Collections.Generic;

namespace TensorShader.Operators.Connection3D {
    /// <summary>3次元逆畳み込み</summary>
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

        /// <summary>フィルタサイズ</summary>
        /// <remarks>奇数を指定すること</remarks>
        public int KernelDepth { private set; get; }

        /// <summary>バッチサイズ</summary>
        public int Batch { private set; get; }

        /// <summary>コンストラクタ</summary>
        public Deconvolution(int outwidth, int outheight, int outdepth, int inchannels, int outchannels, int kwidth, int kheight, int kdepth, int batch = 1) {
            
            int inwidth = outwidth - kwidth + 1;
            int inheight = outheight - kheight + 1;
            int indepth = outdepth - kdepth + 1;

            this.arguments = new List<(ArgumentType type, Shape shape)>{
                (ArgumentType.In, Shape.Map3D(inchannels, inwidth, inheight, indepth, batch)),
                (ArgumentType.In, Shape.Kernel3D(outchannels, inchannels, kwidth, kheight, kdepth)),
                (ArgumentType.Out, Shape.Map3D(outchannels, outwidth, outheight, outdepth, batch)),
            };

            this.InChannels = inchannels;
            this.OutChannels = outchannels;
            this.KernelWidth = kwidth;
            this.KernelHeight = kheight;
            this.KernelDepth = kdepth;
            this.Batch = batch;
        }

        /// <summary>操作を実行</summary>
        public override void Execute(params Tensor[] tensors) {
            CheckArgumentShapes(tensors);

            Tensor inmap = tensors[0], infilter = tensors[1], outmap = tensors[2];

            TensorShaderCudaBackend.Convolution.Deconvolution3D((uint)InChannels, (uint)OutChannels,
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
