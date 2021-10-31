using System;
using System.Linq;

using static TensorShaderCudaBackend.Transpose;

namespace TensorShaderCudaBackend.Shaders.Convolution.CudnnImplement {

    /// <summary>1次元畳み込み</summary>
    public sealed class Convolution1D : Shader {
        private readonly Cudnn.CudnnController controller;

        /// <summary>識別子</summary>
        public override sealed string Signature => $"{GetType().Name.Split(',').Last()}";

        /// <summary>コンストラクタ</summary>
        public Convolution1D(Cudnn.CudnnController controller) {
            this.controller = controller;
        }

        /// <summary>実行</summary>
        public override void Execute(Stream stream, params object[] args) {
            CheckArgument(args);

            if (controller.Stream != stream) {
                throw new ArgumentException(nameof(stream));
            }

            CudaArray<float> inmap = args[0] as CudaArray<float>;
            CudaArray<float> outmap = args[1] as CudaArray<float>;
            CudaArray<float> filter = args[2] as CudaArray<float>;

            uint inchannels = (args[3] as uint?).Value;
            uint outchannels = (args[4] as uint?).Value;
            uint kwidth = (args[5] as uint?).Value;

            uint inwidth = (args[6] as uint?).Value;
            uint batches = (args[7] as uint?).Value;

            uint outwidth = inwidth + 1 - kwidth;

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
                (int)outchannels, (int)inchannels, 1, (int)kwidth
            );

            using Cudnn.ConvolutionDescriptor convdesc = new(
                Cudnn.DataType.Float, pad: (0, 0), stride: (1, 1), dilation: (1, 1)
            );

            CudaArray<float> transpose_filter =
                WorkspaceReserver<float>.Request(stream, filter.DeviceID, index: 0, inchannels * outchannels * kwidth);

            BlockTranspose(inchannels, outchannels, kwidth, filter, transpose_filter, stream);

            controller.ConvolutionForward(inmap, indesc, transpose_filter, filterdesc, convdesc, outmap, outdesc);
        }

        /// <summary>引数チェック</summary>
        protected override void CheckArgument(params object[] args) {
            if (args is null || args.Length != 8) {
                throw new ArgumentException(nameof(args));
            }

            if (!(args[3] is uint inchannels) || !Limits.CheckChannels(inchannels)) {
                throw new ArgumentException(nameof(inchannels));
            }

            if (!(args[4] is uint outchannels) || !Limits.CheckChannels(outchannels)) {
                throw new ArgumentException(nameof(outchannels));
            }

            if (!(args[5] is uint kwidth) || !Limits.CheckKernelSize(kwidth)) {
                throw new ArgumentException(nameof(kwidth));
            }

            if (!(args[6] is uint inwidth) || !Limits.CheckWidth(inwidth, kwidth)) {
                throw new ArgumentException(nameof(inwidth));
            }

            if (!(args[7] is uint batches) || !Limits.CheckBatches(batches)) {
                throw new ArgumentException(nameof(batches));
            }

            uint outwidth = inwidth + 1 - kwidth;

            if (!(args[0] is CudaArray<float> inmap) || inmap.Length < inchannels * inwidth * batches) {
                throw new ArgumentException(nameof(inmap));
            }

            if (!(args[1] is CudaArray<float> outmap) || outmap.Length < outchannels * outwidth * batches) {
                throw new ArgumentException(nameof(outmap));
            }

            if (!(args[2] is CudaArray<float> filter) || filter.Length < inchannels * outchannels * kwidth) {
                throw new ArgumentException(nameof(filter));
            }
        }
    }
}
