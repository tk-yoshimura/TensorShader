using System;
using System.Collections.Generic;
using System.Threading.Tasks;

namespace TensorShader.Operators.ComplexConvolution {
    /// <summary>複素全結合カーネル積</summary>
    internal class ComplexKernelProductDense : Operator {
        /// <summary>入力チャネル</summary>
        public int InChannels { private set; get; }

        /// <summary>出力チャネル</summary>
        public int OutChannels { private set; get; }

        /// <summary>転置</summary>
        public bool Transpose { private set; get; }

        /// <summary>バッチサイズ</summary>
        public int Batch { private set; get; }

        /// <summary>コンストラクタ</summary>
        public ComplexKernelProductDense(int inchannels, int outchannels, bool transpose = false, int batch = 1) {
            if (inchannels % 2 != 0) {
                throw new ArgumentException(ExceptionMessage.ArgumentMultiple(nameof(inchannels), inchannels, 2));
            }
            if (outchannels % 2 != 0) {
                throw new ArgumentException(ExceptionMessage.ArgumentMultiple(nameof(outchannels), outchannels, 2));
            }

            this.arguments = new List<(ArgumentType type, Shape shape)>{
                (ArgumentType.In, Shape.Map0D(inchannels, batch)),
                (ArgumentType.In, Shape.Map0D(outchannels, batch)),
                (ArgumentType.Out, Shape.Kernel0D(inchannels, outchannels / 2))
            };

            this.InChannels = inchannels;
            this.OutChannels = outchannels;
            this.Transpose = transpose;
            this.Batch = batch;
        }

        /// <summary>操作を実行</summary>
        public override void Execute(params Tensor[] tensors) {
            CheckArgumentShapes(tensors);

            Tensor inmap1 = tensors[0], inmap2 = tensors[1], outfilter = tensors[2];

            Parallel.For(0, OutChannels / 2, (outch) => {
                TensorShaderCudaBackend.Complex.KernelProductDense((uint)InChannels, (uint)OutChannels, (uint)Batch, (uint)outch * 2, Transpose, inmap1.Buffer, inmap2.Buffer, outfilter.Buffer);
            });
        }

        /// <summary>操作を実行</summary>
        public void Execute(Tensor inmap1, Tensor inmap2, Tensor outfilter) {
            Execute(new Tensor[] { inmap1, inmap2, outfilter });
        }
    }
}
