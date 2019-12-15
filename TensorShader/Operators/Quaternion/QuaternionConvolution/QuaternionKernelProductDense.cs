using System;
using System.Collections.Generic;
using System.Threading.Tasks;

namespace TensorShader.Operators.QuaternionConvolution {
    /// <summary>四元数全結合カーネル積</summary>
    internal class QuaternionKernelProductDense : Operator {
        /// <summary>入力チャネル</summary>
        public int InChannels { private set; get; }

        /// <summary>出力チャネル</summary>
        public int OutChannels { private set; get; }

        /// <summary>転置</summary>
        public bool Transpose { private set; get; }

        /// <summary>バッチサイズ</summary>
        public int Batch { private set; get; }

        /// <summary>コンストラクタ</summary>
        public QuaternionKernelProductDense(int inchannels, int outchannels, bool transpose = false, int batch = 1) {
            if (inchannels % 4 != 0) {
                throw new ArgumentException(ExceptionMessage.ArgumentMultiple(nameof(inchannels), inchannels, 4));
            }
            if (outchannels % 4 != 0) {
                throw new ArgumentException(ExceptionMessage.ArgumentMultiple(nameof(outchannels), outchannels, 4));
            }

            this.arguments = new List<(ArgumentType type, Shape shape)>{
                (ArgumentType.In, Shape.Map0D(inchannels, batch)),
                (ArgumentType.In, Shape.Map0D(outchannels, batch)),
                (ArgumentType.Out, Shape.Kernel0D(inchannels, outchannels / 4))
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

            Parallel.For(0, OutChannels / 4, (outch) => {
                TensorShaderCudaBackend.Quaternion.KernelProductDense((uint)InChannels, (uint)OutChannels, (uint)Batch, (uint)outch * 4, Transpose, inmap1.Buffer, inmap2.Buffer, outfilter.Buffer);
            });
        }

        /// <summary>操作を実行</summary>
        public void Execute(Tensor inmap1, Tensor inmap2, Tensor outfilter) {
            Execute(new Tensor[] { inmap1, inmap2, outfilter });
        }
    }
}
