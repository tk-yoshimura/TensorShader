using System;
using System.Collections.Generic;
using System.Threading.Tasks;

namespace TensorShader.Operators.ComplexConvolution {
    /// <summary>複素全結合</summary>
    internal class ComplexDense : Operator {
        /// <summary>入力チャネル</summary>
        public int InChannels { private set; get; }

        /// <summary>出力チャネル</summary>
        public int OutChannels { private set; get; }

        /// <summary>勾配</summary>
        public bool GradMode { private set; get; }

        /// <summary>バッチサイズ</summary>
        public int Batch { private set; get; }

        /// <summary>コンストラクタ</summary>
        public ComplexDense(int inchannels, int outchannels, bool gradmode = false, int batch = 1) {
            if (inchannels % 2 != 0) {
                throw new ArgumentException(ExceptionMessage.ArgumentMultiple(nameof(inchannels), inchannels, 2));
            }
            if (outchannels % 2 != 0) {
                throw new ArgumentException(ExceptionMessage.ArgumentMultiple(nameof(outchannels), outchannels, 2));
            }

            this.arguments = new List<(ArgumentType type, Shape shape)>{
                (ArgumentType.In, Shape.Map0D(inchannels, batch)),
                (ArgumentType.In, Shape.Kernel0D(inchannels, outchannels / 2)),
                (ArgumentType.Out, Shape.Map0D(outchannels, batch)),
            };

            this.InChannels = inchannels;
            this.OutChannels = outchannels;
            this.GradMode = gradmode;
            this.Batch = batch;
        }

        /// <summary>操作を実行</summary>
        public override void Execute(params Tensor[] tensors) {
            CheckArgumentShapes(tensors);

            Tensor inmap = tensors[0], infilter = tensors[1], outmap = tensors[2];

            Parallel.For(0, Batch, (th) => {
                TensorShaderCudaBackend.Complex.Dense((uint)InChannels, (uint)OutChannels, (uint)Batch, (uint)th, GradMode, inmap.Buffer, infilter.Buffer, outmap.Buffer);
            });
        }

        /// <summary>操作を実行</summary>
        public void Execute(Tensor inmap, Tensor infilter, Tensor outmap) {
            Execute(new Tensor[] { inmap, infilter, outmap });
        }
    }
}
