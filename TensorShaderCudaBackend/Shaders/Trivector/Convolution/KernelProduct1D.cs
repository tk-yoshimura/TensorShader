﻿using System;
using System.Linq;

using static TensorShaderCudaBackend.Elementwise;
using static TensorShaderCudaBackend.ArrayManipulation;

namespace TensorShaderCudaBackend.Shaders.Trivector.Convolution {

    /// <summary>カーネル積</summary>
    public sealed class KernelProduct1D : Shader {

        /// <summary>入力チャネル数</summary>
        public uint InChannels { private set; get; }

        /// <summary>出力チャネル数</summary>
        public uint OutChannels { private set; get; }

        /// <summary>フィルタサイズ</summary>
        public uint KernelWidth { private set; get; }

        /// <summary>転置</summary>
        public bool Transpose { private set; get; }

        /// <summary>ブロックサイズ</summary>
        public (uint x, uint y) BlockSize { private set; get; }

        /// <summary>1スレッドで処理する対象ピクセル数</summary>
        private static uint BatchPixels => 16;

        /// <summary>識別子</summary>
        public override sealed string Signature =>
            $"{GetType().Name.Split(',').Last()} {nameof(InChannels)} = {InChannels} {nameof(OutChannels)} = {OutChannels} " +
            $"{nameof(KernelWidth)} = {KernelWidth} {nameof(Transpose)} = {Transpose}";

        /// <summary>コンストラクタ</summary>
        public KernelProduct1D(uint inchannels, uint outchannels, uint kwidth, bool transpose) {
            if (!Limits.CheckChannels(inchannels, outchannels) || !Limits.CheckMultipleNum(multiple: 3, inchannels, outchannels)) {
                throw new ArgumentException($"{nameof(inchannels)}, {nameof(outchannels)}");
            }
            if (!Limits.CheckKernelSize(kwidth)) {
                throw new ArgumentException(nameof(kwidth));
            }

            this.InChannels = inchannels / 3;
            this.OutChannels = outchannels / 3;
            this.KernelWidth = kwidth;
            this.Transpose = transpose;

            this.BlockSize = Kernel.MinimizeGridsBlockSize((InChannels, OutChannels));

            string code = $@"

            {Defines.CtorFloat4}
            {Defines.FloatFloatAdd}
            {Defines.Trivector.KernelProd}
            {Defines.Quaternion.AtomicAdd}

            __global__ void trivector_kernelproduct_1d(float3 *inmap, float3 *outmap, float4 *filter_value, float4 *filter_grad,
                                                       unsigned int outwidth) {{

                unsigned int inch = {Defines.IndexX}, outch = {Defines.IndexY};
                unsigned int ox_offset = {Defines.BlockIndexZ} * {BatchPixels};
                unsigned int tidx = {Defines.ThreadIdX}, tidy = {Defines.ThreadIdY};

                __shared__ float3 us[{BlockSize.x}], vs[{BlockSize.y}];

                for(unsigned int kx = 0; kx < {KernelWidth}; kx++){{
                    unsigned int filter_index = (inch + {InChannels} * (outch + {OutChannels} * kx));

                    float4 q = filter_value[filter_index];

                    float4 gq_hi = ctor_float4(0.0, 0.0, 0.0, 0.0), gq_lo = ctor_float4(0.0, 0.0, 0.0, 0.0);

                    for(unsigned int ox = ox_offset, ix = ox + kx; ox < ox_offset + {BatchPixels} && ox < outwidth; ox++, ix++){{
                        if(tidx == 0 && outch < {OutChannels}){{
                            vs[tidy] = outmap[outch + {OutChannels} * ox];
                        }}
                        if(tidy == 0 && inch < {InChannels}){{
                            us[tidx] = inmap[inch + {InChannels} * ix];
                        }}
                        __syncthreads();

                        if(inch < {InChannels} && outch < {OutChannels}){{
                            float3 u = us[tidx];
                            float3 v = vs[tidy];

                            trivector_quaternion_kernelprod(gq_hi, gq_lo, {(Transpose ? "v, u" : "u, v")}, q);
                        }}
                        __syncthreads();
                    }}

                    if(inch < {InChannels} && outch < {OutChannels}){{
                        floatfloat_atomicadd(filter_grad + filter_index * 2, gq_hi, gq_lo);
                    }}
                }}
            }}";

            this.Kernel = new Kernel(code, "trivector_kernelproduct_1d");
            this.Kernel.SetCacheAllocationFromUsageSharedMemory((BlockSize.x + BlockSize.y) * 3 * 4);
        }

        /// <summary>実行</summary>
        public override void Execute(Stream stream, params object[] args) {
            CheckArgument(args);

            CudaArray<float> inmap = args[0] as CudaArray<float>;
            CudaArray<float> outmap = args[1] as CudaArray<float>;
            CudaArray<float> filter_value = args[2] as CudaArray<float>;
            CudaArray<float> filter_grad = args[3] as CudaArray<float>;

            uint inwidth = (args[4] as uint?).Value;
            uint batches = (args[5] as uint?).Value;

            uint outwidth = inwidth + 1 - KernelWidth;

            CudaArray<float> dfloat_filter =
                CudaArrayReserver<float>.Request(stream, inmap.DeviceID, index: 0, InChannels * OutChannels * KernelWidth * 8);
            dfloat_filter.ZerosetAsync(stream, InChannels * OutChannels * KernelWidth * 8);

            uint xsets = (outwidth + BatchPixels - 1) / BatchPixels;

            for (uint th = 0; th < batches; th++) {
                Kernel.Execute(
                    indexes: (InChannels, OutChannels, xsets),
                    block: (BlockSize.x, BlockSize.y, 1),
                    dynamic_shared_memory_bytes: 0,
                    stream,
                    inmap.ElementPtr(th * InChannels * inwidth * 3),
                    outmap.ElementPtr(th * OutChannels * outwidth * 3),
                    filter_value,
                    dfloat_filter,
                    outwidth
                );
            }

            HorizontalAdd(InChannels * OutChannels * KernelWidth * 4, dfloat_filter, filter_grad, stream);
            MulConstant(InChannels * OutChannels * KernelWidth * 4, 2, filter_grad, filter_grad, stream);
        }

        /// <summary>引数チェック</summary>
        protected override void CheckArgument(params object[] args) {
            if (args == null || args.Length != 6) {
                throw new ArgumentException(nameof(args));
            }

            if (!(args[4] is uint inwidth) || !Limits.CheckWidth(inwidth, KernelWidth)) {
                throw new ArgumentException(nameof(inwidth));
            }

            if (!(args[5] is uint batches) || !Limits.CheckBatches(batches)) {
                throw new ArgumentException(nameof(batches));
            }

            uint outwidth = inwidth + 1 - KernelWidth;

            if (!(args[0] is CudaArray<float> inmap) || inmap.Length < InChannels * inwidth * batches * 3) {
                throw new ArgumentException(nameof(inmap));
            }

            if (!(args[1] is CudaArray<float> outmap) || outmap.Length < OutChannels * outwidth * batches * 3) {
                throw new ArgumentException(nameof(outmap));
            }

            if (!(args[2] is CudaArray<float> filter_value) || filter_value.Length < InChannels * OutChannels * KernelWidth * 4) {
                throw new ArgumentException(nameof(filter_value));
            }

            if (!(args[3] is CudaArray<float> filter_grad) || filter_grad.Length < InChannels * OutChannels * KernelWidth * 4) {
                throw new ArgumentException(nameof(filter_grad));
            }
        }
    }
}
