using System;
using System.Linq;

using static TensorShaderCudaBackend.Transpose;

namespace TensorShaderCudaBackend.Shaders.Convolution.CudnnImplement {

    /// <summary>1次元逆畳み込み</summary>
    public sealed class Deconvolution1D : Shader {
        private readonly Cudnn.CudnnController controller;

        /// <summary>識別子</summary>
        public override sealed string Signature => $"{GetType().Name.Split(',').Last()}";

        /// <summary>コンストラクタ</summary>
        public Deconvolution1D(Cudnn.CudnnController controller) {
            this.controller = controller;
        }

        /// <summary>実行</summary>
        public override void Execute(Stream stream, params object[] args) {
            CheckArgument(args);

            if (controller.Stream != stream) {
                throw new ArgumentException(null, nameof(stream));
            }

            CudaArray<float> inmap = args[0] as CudaArray<float>;
            CudaArray<float> outmap = args[1] as CudaArray<float>;
            CudaArray<float> filter = args[2] as CudaArray<float>;

            uint inchannels = (args[3] as uint?).Value;
            uint outchannels = (args[4] as uint?).Value;
            uint kwidth = (args[5] as uint?).Value;

            uint inwidth = (args[6] as uint?).Value;
            uint batches = (args[7] as uint?).Value;

            uint outwidth = inwidth + kwidth - 1;

            using Cudnn.TensorDescriptor indesc = new(
                Cudnn.TensorFormat.NHWC, Cudnn.DataType.Float,
                (int)batches, (int)inchannels, 1, (int)inwidth
            );

            using Cudnn.TensorDescriptor outdesc = new(
                Cudnn.TensorFormat.NHWC, Cudnn.DataType.Float,
                (int)batches, (int)outchannels, 1, (int)outwidth
            );

            using Cudnn.FilterDescriptor filterdesc = new(
                Cudnn.TensorFormat.NHWC, Cudnn.DataType.Float,
                (int)inchannels, (int)outchannels, 1, (int)kwidth
            );

            using Cudnn.ConvolutionDescriptor convdesc = new(
                Cudnn.DataType.Float, pad: (0, 0), stride: (1, 1), dilation: (1, 1)
            );

            CudaArray<float> transpose_filter =
                WorkspaceReserver<float>.Request(stream, filter.DeviceID, index: 0, inchannels * outchannels * kwidth);

            BlockTranspose(outchannels, inchannels, kwidth, filter, transpose_filter, stream);

            controller.ConvolutionBackwardData(
                transpose_filter, filterdesc, inmap, indesc, convdesc, outmap, outdesc,
                Cudnn.ConvolutionBwdDataAlgo.Algo0
            );
        }

        /// <summary>引数チェック</summary>
        protected override void CheckArgument(params object[] args) {
            if (args is null || args.Length != 8) {
                throw new ArgumentException(null, nameof(args));
            }

            if (args[3] is not uint inchannels || !Limits.CheckChannels(inchannels)) {
                throw new ArgumentException(nameof(inchannels));
            }

            if (args[4] is not uint outchannels || !Limits.CheckChannels(outchannels)) {
                throw new ArgumentException(nameof(outchannels));
            }

            if (args[5] is not uint kwidth || !Limits.CheckKernelSize(kwidth)) {
                throw new ArgumentException(nameof(kwidth));
            }

            if (args[6] is not uint inwidth || !Limits.CheckWidth(inwidth)) {
                throw new ArgumentException(nameof(inwidth));
            }

            if (args[7] is not uint batches || !Limits.CheckBatches(batches)) {
                throw new ArgumentException(nameof(batches));
            }

            uint outwidth = inwidth + kwidth - 1;

            if (args[0] is not CudaArray<float> inmap || inmap.Length < inchannels * inwidth * batches) {
                throw new ArgumentException(nameof(inmap));
            }

            if (args[1] is not CudaArray<float> outmap || outmap.Length < outchannels * outwidth * batches) {
                throw new ArgumentException(nameof(outmap));
            }

            if (args[2] is not CudaArray<float> filter || filter.Length < inchannels * outchannels * kwidth) {
                throw new ArgumentException(nameof(filter));
            }
        }
    }
}
