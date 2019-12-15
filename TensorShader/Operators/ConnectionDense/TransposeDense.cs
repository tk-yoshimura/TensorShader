using System.Collections.Generic;

namespace TensorShader.Operators.ConnectionDense {
    /// <summary>転置全結合</summary>
    internal class TransposeDense : Operator {
        /// <summary>入力チャネル</summary>
        public int InChannels { private set; get; }

        /// <summary>出力チャネル</summary>
        public int OutChannels { private set; get; }

        /// <summary>バッチサイズ</summary>
        public int Batch { private set; get; }

        /// <summary>コンストラクタ</summary>
        public TransposeDense(int inchannels, int outchannels, int batch = 1) {
            this.arguments = new List<(ArgumentType type, Shape shape)>{
                (ArgumentType.In, Shape.Map0D(inchannels, batch)),
                (ArgumentType.In, Shape.Kernel0D(outchannels, inchannels)),
                (ArgumentType.Out, Shape.Map0D(outchannels, batch)),
            };

            this.InChannels = inchannels;
            this.OutChannels = outchannels;
            this.Batch = batch;
        }

        /// <summary>操作を実行</summary>
        public override void Execute(params Tensor[] tensors) {
            CheckArgumentShapes(tensors);

            Tensor inmap = tensors[0], infilter = tensors[1], outmap = tensors[2];

            TensorShaderCudaBackend.Convolution.TransposeDense((uint)InChannels, (uint)OutChannels, (uint)Batch, inmap.Buffer, infilter.Buffer, outmap.Buffer);
        }

        /// <summary>操作を実行</summary>
        public void Execute(Tensor inmap, Tensor infilter, Tensor outmap) {
            Execute(new Tensor[] { inmap, infilter, outmap });
        }
    }
}
