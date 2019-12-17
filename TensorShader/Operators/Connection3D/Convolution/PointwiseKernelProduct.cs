using System.Collections.Generic;

namespace TensorShader.Operators.Connection3D {
    /// <summary>ポイントごとのカーネル積</summary>
    internal class PointwiseKernelProduct : Operator {
        /// <summary>入力チャネル</summary>
        public int InChannels { private set; get; }

        /// <summary>出力チャネル</summary>
        public int OutChannels { private set; get; }

        /// <summary>バッチサイズ</summary>
        public int Batch { private set; get; }

        /// <summary>コンストラクタ</summary>
        public PointwiseKernelProduct(int width, int height, int depth, int inchannels, int outchannels, int batch = 1) {
            this.arguments = new List<(ArgumentType type, Shape shape)>{
                (ArgumentType.In, Shape.Map3D(inchannels, width, height, depth, batch)),
                (ArgumentType.In, Shape.Map3D(outchannels, width, height, depth, batch)),
                (ArgumentType.Out, Shape.Kernel0D(inchannels, outchannels)),
            };

            this.InChannels = inchannels;
            this.OutChannels = outchannels;
            this.Batch = batch;
        }

        /// <summary>操作を実行</summary>
        public override void Execute(params Tensor[] tensors) {
            CheckArgumentShapes(tensors);

            Tensor inmap1 = tensors[0], inmap2 = tensors[1], outfilter = tensors[2];

            TensorShaderCudaBackend.Convolution.PointwiseKernelProduct((uint)InChannels, (uint)OutChannels,
                                                                        (uint)(inmap1.Width * inmap1.Height * inmap1.Depth),
                                                                        (uint)Batch, 
                                                                        inmap1.Buffer, inmap2.Buffer, outfilter.Buffer);
        }

        /// <summary>操作を実行</summary>
        public void Execute(Tensor inmap1, Tensor inmap2, Tensor outfilter) {
            Execute(new Tensor[] { inmap1, inmap2, outfilter });
        }
    }
}
