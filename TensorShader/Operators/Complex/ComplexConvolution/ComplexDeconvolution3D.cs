using System;
using System.Collections.Generic;

namespace TensorShader.Operators.ComplexConvolution {
    /// <summary>複素3次元逆畳み込み</summary>
    internal class ComplexDeconvolution3D : Operator {
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
        public ComplexDeconvolution3D(int outwidth, int outheight, int outdepth, int inchannels, int outchannels, int kwidth, int kheight, int kdepth, bool gradmode = false, int batch = 1) {
            if (inchannels % 2 != 0) {
                throw new ArgumentException(ExceptionMessage.ArgumentMultiple(nameof(inchannels), inchannels, 2));
            }
            if (outchannels % 2 != 0) {
                throw new ArgumentException(ExceptionMessage.ArgumentMultiple(nameof(outchannels), outchannels, 2));
            }

            int inwidth = outwidth - kwidth + 1;
            int inheight = outheight - kheight + 1;
            int indepth = outdepth - kdepth + 1;

            this.arguments = new List<(ArgumentType type, Shape shape)>{
                (ArgumentType.In, Shape.Map3D(inchannels, inwidth, inheight, indepth, batch)),
                (ArgumentType.In, Shape.Kernel3D(outchannels, inchannels / 2, kwidth, kheight, kdepth)),
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

            TensorShaderCudaBackend.Complex.Deconvolution3D((uint)InChannels, (uint)OutChannels,
                                                            (uint)outmap.Width, (uint)outmap.Height, (uint)outmap.Depth,
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
