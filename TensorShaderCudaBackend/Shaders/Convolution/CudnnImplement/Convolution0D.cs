using System;
using System.Linq;

namespace TensorShaderCudaBackend.Shaders.Convolution.CudnnImplement {

    /// <summary>ポイントごとの畳み込み</summary>
    public sealed class Convolution0D : Shader {
        private readonly Cudnn.CudnnController controller;

        /// <summary>識別子</summary>
        public override sealed string Signature => $"{GetType().Name.Split(',').Last()}";

        /// <summary>コンストラクタ</summary>
        public Convolution0D(Cudnn.CudnnController controller) {
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

            uint pts = (args[5] as uint?).Value;

            using Cudnn.TensorDescriptor indesc = new(
                Cudnn.TensorFormat.NHWC, Cudnn.DataType.Float,
                (int)pts, (int)inchannels, 1, 1
            );

            using Cudnn.TensorDescriptor outdesc = new(
                Cudnn.TensorFormat.NHWC, Cudnn.DataType.Float,
                (int)pts, (int)outchannels, 1, 1
            );

            using Cudnn.FilterDescriptor filterdesc = new(
                Cudnn.TensorFormat.NHWC, Cudnn.DataType.Float,
                (int)outchannels, (int)inchannels, 1, 1
            );

            using Cudnn.ConvolutionDescriptor convdesc = new(
                Cudnn.DataType.Float, pad: (0, 0), stride: (1, 1), dilation: (1, 1)
            );

            controller.ConvolutionForward(inmap, indesc, filter, filterdesc, convdesc, outmap, outdesc);
        }

        /// <summary>引数チェック</summary>
        protected override void CheckArgument(params object[] args) {
            if (args is null || args.Length != 6) {
                throw new ArgumentException(nameof(args));
            }

            if (!(args[3] is uint inchannels) || !Limits.CheckChannels(inchannels)) {
                throw new ArgumentException(nameof(inchannels));
            }

            if (!(args[4] is uint outchannels) || !Limits.CheckChannels(outchannels)) {
                throw new ArgumentException(nameof(outchannels));
            }

            if (!(args[5] is uint pts) || pts < 1) {
                throw new ArgumentException(nameof(pts));
            }

            if (!(args[0] is CudaArray<float> inmap) || inmap.Length < inchannels * pts) {
                throw new ArgumentException(nameof(inmap));
            }

            if (!(args[1] is CudaArray<float> outmap) || outmap.Length < outchannels * pts) {
                throw new ArgumentException(nameof(outmap));
            }

            if (!(args[2] is CudaArray<float> filter) || filter.Length < inchannels * outchannels) {
                throw new ArgumentException(nameof(filter));
            }
        }
    }
}
