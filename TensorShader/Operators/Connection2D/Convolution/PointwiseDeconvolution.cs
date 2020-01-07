using System.Collections.Generic;

namespace TensorShader.Operators.Connection2D {
    /// <summary>ポイントごとの2次元逆畳み込み</summary>
    internal class PointwiseDeconvolution : Operator {
        /// <summary>入力チャネル</summary>
        public int InChannels { private set; get; }

        /// <summary>出力チャネル</summary>
        public int OutChannels { private set; get; }

        /// <summary>バッチサイズ</summary>
        public int Batch { private set; get; }

        /// <summary>コンストラクタ</summary>
        public PointwiseDeconvolution(int width, int height, int inchannels, int outchannels, int batch = 1) {
            this.arguments = new List<(ArgumentType type, Shape shape)>{
                (ArgumentType.In, Shape.Map2D(inchannels, width, height, batch)),
                (ArgumentType.In, Shape.Kernel0D(outchannels, inchannels)),
                (ArgumentType.Out, Shape.Map2D(outchannels, width, height, batch)),
            };

            this.InChannels = inchannels;
            this.OutChannels = outchannels;
            this.Batch = batch;
        }

        /// <summary>操作を実行</summary>
        public override void Execute(params Tensor[] tensors) {
            CheckArgumentShapes(tensors);

            Tensor inmap = tensors[0], infilter = tensors[1], outmap = tensors[2];

            TensorShaderCudaBackend.Convolution.PointwiseDeconvolution((uint)InChannels, (uint)OutChannels,
                                                                       (uint)(inmap.Width * inmap.Height), (uint)Batch,
                                                                       inmap.Buffer, infilter.Buffer, outmap.Buffer);
        }

        /// <summary>操作を実行</summary>
        public void Execute(Tensor inmap, Tensor infilter, Tensor outmap) {
            Execute(new Tensor[] { inmap, infilter, outmap });
        }
    }
}
