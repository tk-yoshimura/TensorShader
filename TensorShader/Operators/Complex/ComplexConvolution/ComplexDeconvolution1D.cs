using System;
using System.Collections.Generic;
using System.Threading.Tasks;

namespace TensorShader.Operators.ComplexConvolution {
    /// <summary>複素1次元逆畳み込み</summary>
    internal class ComplexDeconvolution1D : Operator {
        /// <summary>入力チャネル</summary>
        public int InChannels { private set; get; }

        /// <summary>出力チャネル</summary>
        public int OutChannels { private set; get; }

        /// <summary>フィルタサイズ</summary>
        /// <remarks>奇数を指定すること</remarks>
        public int KernelWidth { private set; get; }

        /// <summary>ストライド</summary>
        public int Stride { private set; get; }

        /// <summary>勾配</summary>
        public bool GradMode { private set; get; }

        /// <summary>バッチサイズ</summary>
        public int Batch { private set; get; }

        /// <summary>コンストラクタ</summary>
        public ComplexDeconvolution1D(int outwidth, int inchannels, int outchannels, int kwidth, int stride, bool gradmode = false, int batch = 1) {
            if (stride < 1) {
                throw new ArgumentException(nameof(stride));
            }

            if (inchannels % 2 != 0) {
                throw new ArgumentException(ExceptionMessage.ArgumentMultiple(nameof(inchannels), inchannels, 2));
            }
            if (outchannels % 2 != 0) {
                throw new ArgumentException(ExceptionMessage.ArgumentMultiple(nameof(outchannels), outchannels, 2));
            }

            int inwidth = (outwidth - kwidth) / stride + 1;

            this.arguments = new List<(ArgumentType type, Shape shape)>{
                (ArgumentType.In, Shape.Map1D(inchannels, inwidth, batch)),
                (ArgumentType.In, Shape.Kernel1D(outchannels, inchannels / 2, kwidth)),
                (ArgumentType.Out, Shape.Map1D(outchannels, outwidth, batch)),
            };

            this.InChannels = inchannels;
            this.OutChannels = outchannels;
            this.KernelWidth = kwidth;
            this.Stride = stride;
            this.GradMode = gradmode;
            this.Batch = batch;
        }

        /// <summary>操作を実行</summary>
        public override void Execute(params Tensor[] tensors) {
            CheckArgumentShapes(tensors);

            Tensor inmap = tensors[0], infilter = tensors[1], outmap = tensors[2];

            Parallel.For(0, Batch, (th) => {
                TensorShaderCudaBackend.Complex.Deconvolution1D((uint)InChannels, (uint)OutChannels,
                                                               (uint)outmap.Width,
                                                               (uint)Batch, (uint)th,
                                                               (uint)KernelWidth,
                                                               (uint)Stride, GradMode,
                                                               inmap.Buffer, infilter.Buffer, outmap.Buffer);
            });
        }

        /// <summary>操作を実行</summary>
        public void Execute(Tensor inmap, Tensor infilter, Tensor outmap) {
            Execute(new Tensor[] { inmap, infilter, outmap });
        }
    }
}
