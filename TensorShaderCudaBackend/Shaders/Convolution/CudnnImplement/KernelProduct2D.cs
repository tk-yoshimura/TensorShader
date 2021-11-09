using System;
using System.Linq;

using static TensorShaderCudaBackend.Transpose;

namespace TensorShaderCudaBackend.Shaders.Convolution.CudnnImplement {

    /// <summary>カーネル積</summary>
    public sealed class KernelProduct2D : Shader {
        private readonly Cudnn.CudnnController controller;

        /// <summary>識別子</summary>
        public override sealed string Signature => $"{GetType().Name.Split(',').Last()}";

        /// <summary>コンストラクタ</summary>
        public KernelProduct2D(Cudnn.CudnnController controller) {
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
            uint kheight = (args[6] as uint?).Value;

            uint inwidth = (args[7] as uint?).Value;
            uint inheight = (args[8] as uint?).Value;
            uint batches = (args[9] as uint?).Value;

            uint outwidth = inwidth + 1 - kwidth;
            uint outheight = inheight + 1 - kheight;

            using Cudnn.TensorDescriptor indesc = new(
                Cudnn.TensorFormat.NHWC, Cudnn.DataType.Float,
                (int)batches, (int)inchannels, (int)inheight, (int)inwidth
            );

            using Cudnn.TensorDescriptor outdesc = new(
                Cudnn.TensorFormat.NHWC, Cudnn.DataType.Float,
                (int)batches, (int)outchannels, (int)outheight, (int)outwidth
            );

            using Cudnn.FilterDescriptor filterdesc = new(
                Cudnn.TensorFormat.NHWC, Cudnn.DataType.Float,
                (int)outchannels, (int)inchannels, (int)kheight, (int)kwidth
            );

            using Cudnn.ConvolutionDescriptor convdesc = new(
                Cudnn.DataType.Float, pad: (0, 0), stride: (1, 1), dilation: (1, 1)
            );

            CudaArray<float> transpose_filter =
                WorkspaceReserver<float>.Request(stream, filter.DeviceID, index: 0, inchannels * outchannels * kwidth * kheight);

            controller.ConvolutionBackwardFilter(
                inmap, indesc, outmap, outdesc, convdesc, transpose_filter, filterdesc,
                Cudnn.ConvolutionBwdFilterAlgo.Algo0
            );

            BlockTranspose(inchannels, kwidth * kheight, outchannels, transpose_filter, filter, stream);
        }

        /// <summary>引数チェック</summary>
        protected override void CheckArgument(params object[] args) {
            if (args is null || args.Length != 10) {
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

            if (args[6] is not uint kheight || !Limits.CheckKernelSize(kheight)) {
                throw new ArgumentException(nameof(kheight));
            }

            if (args[7] is not uint inwidth || !Limits.CheckWidth(inwidth, kwidth)) {
                throw new ArgumentException(nameof(inwidth));
            }

            if (args[8] is not uint inheight || !Limits.CheckHeight(inheight, kheight)) {
                throw new ArgumentException(nameof(inheight));
            }

            if (args[9] is not uint batches || !Limits.CheckBatches(batches)) {
                throw new ArgumentException(nameof(batches));
            }

            uint outwidth = inwidth + 1 - kwidth;
            uint outheight = inheight + 1 - kheight;

            if (args[0] is not CudaArray<float> inmap || inmap.Length < inchannels * inwidth * inheight * batches) {
                throw new ArgumentException(nameof(inmap));
            }

            if (args[1] is not CudaArray<float> outmap || outmap.Length < outchannels * outwidth * outheight * batches) {
                throw new ArgumentException(nameof(outmap));
            }

            if (args[2] is not CudaArray<float> filter || filter.Length < inchannels * outchannels * kwidth * kheight) {
                throw new ArgumentException(nameof(filter));
            }
        }
    }
}
