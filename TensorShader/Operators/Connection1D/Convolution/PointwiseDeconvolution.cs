using System.Collections.Generic;
using System.Threading.Tasks;

namespace TensorShader.Operators.Connection1D {
    /// <summary>ポイントごとの2次元逆畳み込み</summary>
    internal class PointwiseDeconvolution : Operator {
        /// <summary>入力チャネル</summary>
        public int InChannels { private set; get; }

        /// <summary>出力チャネル</summary>
        public int OutChannels { private set; get; }

        /// <summary>バッチサイズ</summary>
        public int Batch { private set; get; }

        /// <summary>コンストラクタ</summary>
        public PointwiseDeconvolution(int width, int inchannels, int outchannels, int batch = 1) {
            this.arguments = new List<(ArgumentType type, Shape shape)>{
                (ArgumentType.In, Shape.Map1D(inchannels, width, batch)),
                (ArgumentType.In, Shape.Kernel0D(outchannels, inchannels)),
                (ArgumentType.Out, Shape.Map1D(outchannels, width, batch)),
            };

            this.InChannels = inchannels;
            this.OutChannels = outchannels;
            this.Batch = batch;
        }

        /// <summary>操作を実行</summary>
        public override void Execute(params Tensor[] tensors) {
            CheckArgumentShapes(tensors);

            Tensor inmap = tensors[0], infilter = tensors[1], outmap = tensors[2];

            Parallel.For(0, Batch, (th) => {
                TensorShaderCudaBackend.Convolution.PointwiseDeconvolution((uint)InChannels, (uint)OutChannels,
                                                                    (uint)inmap.Width, (uint)Batch, (uint)th,
                                                                    inmap.Buffer, infilter.Buffer, outmap.Buffer);
            });
        }

        /// <summary>操作を実行</summary>
        public void Execute(Tensor inmap, Tensor infilter, Tensor outmap) {
            Execute(new Tensor[] { inmap, infilter, outmap });
        }
    }
}
