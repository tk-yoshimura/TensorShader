using System;
using System.Collections.Generic;

namespace TensorShader.Operators.TrivectorConvolution {
    /// <summary>3次元ベクトル3次元逆畳み込み</summary>
    internal class TrivectorDeconvolution3D : Operator {
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

        /// <summary>勾配</summary>
        public bool GradMode { private set; get; }

        /// <summary>バッチサイズ</summary>
        public int Batch { private set; get; }

        /// <summary>コンストラクタ</summary>
        public TrivectorDeconvolution3D(int inwidth, int inheight, int indepth, int inchannels, int outchannels, int kwidth, int kheight, int kdepth, bool gradmode = false, int batch = 1) {
            if (inchannels % 3 != 0) {
                throw new ArgumentException(ExceptionMessage.ArgumentMultiple(nameof(inchannels), inchannels, 3));
            }
            if (outchannels % 3 != 0) {
                throw new ArgumentException(ExceptionMessage.ArgumentMultiple(nameof(outchannels), outchannels, 3));
            }

            int outwidth = inwidth + kwidth - 1;
            int outheight = inheight + kheight - 1;
            int outdepth = indepth + kdepth - 1;

            this.arguments = new List<(ArgumentType type, Shape shape)>{
                (ArgumentType.In, Shape.Map3D(inchannels, inwidth, inheight, indepth, batch)),
                (ArgumentType.In, Shape.Kernel3D(outchannels / 3 * 4, inchannels / 3, kwidth, kheight, kdepth)),
                (ArgumentType.Out, Shape.Map3D(outchannels, outwidth, outheight, outdepth, batch)),
            };

            this.InChannels = inchannels;
            this.OutChannels = outchannels;
            this.KernelWidth = kwidth;
            this.KernelHeight = kheight;
            this.KernelDepth = kdepth;
            this.GradMode = gradmode;
            this.Batch = batch;
        }

        /// <summary>操作を実行</summary>
        public override void Execute(params Tensor[] tensors) {
            CheckArgumentShapes(tensors);

            Tensor inmap = tensors[0], infilter = tensors[1], outmap = tensors[2];

            TensorShaderCudaBackend.Trivector.Deconvolution3D((uint)InChannels, (uint)OutChannels,
                                                              (uint)inmap.Width, (uint)inmap.Height, (uint)inmap.Depth,
                                                              (uint)Batch, 
                                                              (uint)KernelWidth, (uint)KernelHeight, (uint)KernelDepth,
                                                              GradMode,
                                                              inmap.Buffer, infilter.Buffer, outmap.Buffer);
        }

        /// <summary>操作を実行</summary>
        public void Execute(Tensor inmap, Tensor infilter, Tensor outmap) {
            Execute(new Tensor[] { inmap, infilter, outmap });
        }
    }
}
