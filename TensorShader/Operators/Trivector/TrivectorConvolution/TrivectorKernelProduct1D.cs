using System;
using System.Collections.Generic;

namespace TensorShader.Operators.TrivectorConvolution {
    /// <summary>3次元ベクトルカーネル積</summary>
    internal class TrivectorKernelProduct1D : Operator {
        /// <summary>入力チャネル</summary>
        public int InChannels { private set; get; }

        /// <summary>出力チャネル</summary>
        public int OutChannels { private set; get; }

        /// <summary>フィルタサイズ</summary>
        /// <remarks>奇数を指定すること</remarks>
        public int KernelWidth { private set; get; }

        /// <summary>転置</summary>
        public bool Transpose { private set; get; }

        /// <summary>バッチサイズ</summary>
        public int Batch { private set; get; }

        /// <summary>コンストラクタ</summary>
        public TrivectorKernelProduct1D(int inwidth, int inchannels, int outchannels, int kwidth, bool transpose = false, int batch = 1) {
            if (inchannels % 3 != 0) {
                throw new ArgumentException(ExceptionMessage.ArgumentMultiple(nameof(inchannels), inchannels, 3));
            }
            if (outchannels % 3 != 0) {
                throw new ArgumentException(ExceptionMessage.ArgumentMultiple(nameof(outchannels), outchannels, 3));
            }

            int outwidth = inwidth - kwidth + 1;

            this.arguments = new List<(ArgumentType type, Shape shape)>{
                (ArgumentType.In, Shape.Map1D(inchannels, inwidth, batch)),
                (ArgumentType.In, Shape.Map1D(outchannels, outwidth, batch)),
                (ArgumentType.In, Shape.Kernel1D(inchannels / 3 * 4, outchannels / 3, kwidth)),
                (ArgumentType.Out, Shape.Kernel1D(inchannels / 3 * 4, outchannels / 3, kwidth)),
            };

            this.InChannels = inchannels;
            this.OutChannels = outchannels;
            this.KernelWidth = kwidth;
            this.Transpose = transpose;
            this.Batch = batch;
        }

        /// <summary>操作を実行</summary>
        public override void Execute(params Tensor[] tensors) {
            CheckArgumentShapes(tensors);

            Tensor inmap1 = tensors[0], inmap2 = tensors[1], inmap3 = tensors[2], outfilter = tensors[3];

            TensorShaderCudaBackend.Trivector.KernelProduct1D((uint)InChannels, (uint)OutChannels,
                                                              (uint)inmap1.Width,
                                                              (uint)Batch,
                                                              (uint)KernelWidth,
                                                              Transpose,
                                                              inmap1.Buffer, inmap2.Buffer, inmap3.Buffer, outfilter.Buffer);
        }

        /// <summary>操作を実行</summary>
        public void Execute(Tensor inmap1, Tensor inmap2, Tensor inmap3, Tensor outfilter) {
            Execute(new Tensor[] { inmap1, inmap2, inmap3, outfilter });
        }
    }
}
