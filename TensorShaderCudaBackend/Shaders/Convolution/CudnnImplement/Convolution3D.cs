﻿using System;
using System.Linq;

using static TensorShaderCudaBackend.Transpose;

namespace TensorShaderCudaBackend.Shaders.Convolution.CudnnImplement {

    /// <summary>3次元畳み込み</summary>
    public sealed class Convolution3D : Shader {
        private readonly Cudnn.CudnnController controller;

        /// <summary>識別子</summary>
        public override sealed string Signature => $"{GetType().Name.Split(',').Last()}";

        /// <summary>コンストラクタ</summary>
        public Convolution3D(Cudnn.CudnnController controller) {
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
            uint kheight = (args[6] as uint?).Value;
            uint kdepth = (args[7] as uint?).Value;

            uint inwidth = (args[8] as uint?).Value;
            uint inheight = (args[9] as uint?).Value;
            uint indepth = (args[10] as uint?).Value;
            uint batches = (args[11] as uint?).Value;

            uint outwidth = inwidth + 1 - kwidth;
            uint outheight = inheight + 1 - kheight;
            uint outdepth = indepth + 1 - kdepth;

            using Cudnn.TensorDescriptor indesc = new(
                Cudnn.TensorFormat.NHWC, Cudnn.DataType.Float,
                (int)(batches * outdepth), (int)inchannels, (int)inheight, (int)inwidth
            );

            using Cudnn.TensorDescriptor outdesc = new(
                Cudnn.TensorFormat.NHWC, Cudnn.DataType.Float,
                (int)(batches * outdepth), (int)outchannels, (int)outheight, (int)outwidth
            );

            using Cudnn.FilterDescriptor filterdesc = new(
                Cudnn.TensorFormat.NHWC, Cudnn.DataType.Float,
                (int)outchannels, (int)inchannels, (int)kheight, (int)kwidth
            );

            using Cudnn.ConvolutionDescriptor convdesc = new(
                Cudnn.DataType.Float, pad: (0, 0), stride: (1, 1), dilation: (1, 1)
            );

            outmap.ZerosetAsync(stream, outchannels * outwidth * outheight * outdepth * batches);

            uint slice_inmap_length = inchannels * inwidth * inheight * outdepth;
            uint slice_filter_length = inchannels * outchannels * kwidth * kheight;

            CudaArray<float> slice_inmap =
                WorkspaceReserver<float>.Request(stream, filter.DeviceID, index: 0, slice_inmap_length * batches);
            CudaArray<float> transpose_filter =
                WorkspaceReserver<float>.Request(stream, filter.DeviceID, index: 1, slice_filter_length);
            CudaArray<float> slice_filter =
                WorkspaceReserver<float>.Request(stream, filter.DeviceID, index: 2, slice_filter_length);

            for (uint depth_index = 0; depth_index < kdepth; depth_index++) {
                filter.CopyToAsync(
                    stream,
                    slice_filter_length * depth_index,
                    slice_filter,
                    0u,
                    slice_filter_length
                );

                BlockTranspose(inchannels, outchannels, kwidth * kheight, slice_filter, transpose_filter, stream);

                for (uint th = 0; th < batches; th++) {
                    inmap.CopyToAsync(
                        stream,
                        inchannels * inwidth * inheight * (depth_index + indepth * th),
                        slice_inmap,
                        inchannels * inwidth * inheight * outdepth * th,
                        inchannels * inwidth * inheight * outdepth
                    );
                }

                controller.ConvolutionForward(slice_inmap, indesc, transpose_filter, filterdesc, convdesc, outmap, outdesc, beta: 1);
            }
        }

        /// <summary>引数チェック</summary>
        protected override void CheckArgument(params object[] args) {
            if (args is null || args.Length != 12) {
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

            if (!(args[6] is uint kheight) || !Limits.CheckKernelSize(kheight)) {
                throw new ArgumentException(nameof(kheight));
            }

            if (!(args[7] is uint kdepth) || !Limits.CheckKernelSize(kdepth)) {
                throw new ArgumentException(nameof(kdepth));
            }

            if (!(args[8] is uint inwidth) || !Limits.CheckWidth(inwidth, kwidth)) {
                throw new ArgumentException(nameof(inwidth));
            }

            if (!(args[9] is uint inheight) || !Limits.CheckHeight(inheight, kheight)) {
                throw new ArgumentException(nameof(inheight));
            }

            if (!(args[10] is uint indepth) || !Limits.CheckHeight(indepth, kdepth)) {
                throw new ArgumentException(nameof(indepth));
            }

            if (!(args[11] is uint batches) || !Limits.CheckBatches(batches)) {
                throw new ArgumentException(nameof(batches));
            }

            uint outwidth = inwidth + 1 - kwidth;
            uint outheight = inheight + 1 - kheight;
            uint outdepth = indepth + 1 - kdepth;

            if (!(args[0] is CudaArray<float> inmap) || inmap.Length < inchannels * inwidth * inheight * indepth * batches) {
                throw new ArgumentException(nameof(inmap));
            }

            if (!(args[1] is CudaArray<float> outmap) || outmap.Length < outchannels * outwidth * outheight * outdepth * batches) {
                throw new ArgumentException(nameof(outmap));
            }

            if (!(args[2] is CudaArray<float> filter) || filter.Length < inchannels * outchannels * kwidth * kheight * kdepth) {
                throw new ArgumentException(nameof(filter));
            }
        }
    }
}
