﻿using System;
using System.Linq;

using static TensorShaderCudaBackend.Transpose;

namespace TensorShaderCudaBackend.Shaders.Trivector.Convolution.FloatFloatPrecision {

    /// <summary>1次元畳み込み</summary>
    public sealed class Convolution1D : Shader {

        /// <summary>入力チャネル数</summary>
        public uint InChannels { private set; get; }

        /// <summary>出力チャネル数</summary>
        public uint OutChannels { private set; get; }

        /// <summary>フィルタサイズ</summary>
        public uint KernelWidth { private set; get; }

        /// <summary>勾配</summary>
        public bool GradMode { private set; get; }

        /// <summary>Xスレッド数</summary>
        private uint ThreadsX { set; get; }

        /// <summary>識別子</summary>
        public override sealed string Signature =>
            $"{GetType().Name.Split(',').Last()} {nameof(InChannels)} = {InChannels} {nameof(OutChannels)} = {OutChannels} " +
            $"{nameof(KernelWidth)} = {KernelWidth} {nameof(GradMode)} = {GradMode}";

        /// <summary>コンストラクタ</summary>
        public Convolution1D(uint inchannels, uint outchannels, uint kwidth, bool gradmode) {
            if (!Limits.CheckChannels(inchannels, outchannels) || !Limits.CheckMultipleNum(multiple: 3, inchannels, outchannels)) {
                throw new ArgumentException($"{nameof(inchannels)}, {nameof(outchannels)}");
            }
            if (!Limits.CheckKernelSize(kwidth)) {
                throw new ArgumentException(null, nameof(kwidth));
            }

            this.InChannels = inchannels / 3;
            this.OutChannels = outchannels / 3;
            this.KernelWidth = kwidth;
            this.GradMode = gradmode;

            this.ThreadsX = Kernel.DefaultBlockSize(OutChannels);

            string code = $@"

            {Defines.CtorFloat3}
            {Defines.FloatFloat.Fma}
            {Defines.FloatFloat.HiLoAdd}
            {Defines.FloatFloat.Trivector.Mul}
            {Defines.FloatFloat.Trivector.MulGrad}
            {Defines.StoreFloatSharedMemory(elemsize: 3, InChannels, ThreadsX)}

            __global__ void trivector_convolution_1d(const float3* __restrict__ inmap, float3* __restrict__ outmap, const float4* __restrict__ filter) {{

                unsigned int outch = {Defines.IndexX}, tid = {Defines.ThreadIdX};
                unsigned int ox = {Defines.BlockIndexY};

                __shared__ float3 vs[{InChannels}];
                float3 vq_hi = ctor_float3(0.0, 0.0, 0.0), vq_lo = ctor_float3(0.0, 0.0, 0.0);

                unsigned int inmap_idx = {InChannels} * ox;
                unsigned int filter_idx = outch;

                { (KernelWidth <= 7 ? "#pragma unroll" : "") }
                for(unsigned int kx = 0, ix = ox; kx < {KernelWidth}; kx++, ix++){{

                    store_smem(inmap + inmap_idx, vs, tid);
                    inmap_idx += {InChannels};

                    { (OutChannels % ThreadsX != 0 ? $"if(outch < {OutChannels}){{" : "") }

                        #pragma unroll 4
                        for(unsigned int inch = 0; inch < {InChannels}; inch++){{
                            float3 v = vs[inch];
                            float4 q = filter[filter_idx];

                            {(GradMode ? "trivector_quaternion_mulgrad" : "trivector_quaternion_mul")}(vq_hi, vq_lo, v, q);

                            filter_idx += {OutChannels};
                        }}

                    { (OutChannels % ThreadsX != 0 ? "}" : "") }
                    __syncthreads();
                }}

                { (OutChannels % ThreadsX != 0 ? $"if(outch < {OutChannels}){{" : "") }
                    unsigned int outmap_idx = outch + {OutChannels} * ox;

                    outmap[outmap_idx] = ctor_float3(vq_hi.x, vq_hi.y, vq_hi.z);
                { (OutChannels % ThreadsX != 0 ? "}" : "") }
            }}";

            this.Kernel = new Kernel(code, "trivector_convolution_1d");
        }

        /// <summary>実行</summary>
        public override void Execute(Stream stream, params object[] args) {
            CheckArgument(args);

            CudaArray<float> inmap = args[0] as CudaArray<float>;
            CudaArray<float> outmap = args[1] as CudaArray<float>;
            CudaArray<float> filter = args[2] as CudaArray<float>;

            uint inwidth = (args[3] as uint?).Value;
            uint batches = (args[4] as uint?).Value;

            uint outwidth = inwidth + 1 - KernelWidth;

            CudaArray<float> transpose_filter =
                WorkspaceReserver<float>.Request(stream, filter.DeviceID, index: 0, InChannels * OutChannels * KernelWidth * 4);

            TransposeQuaternionKernelChannel(InChannels * 4, OutChannels * 4, KernelWidth, filter, transpose_filter, stream);

            for (uint th = 0; th < batches; th++) {
                Kernel.Execute(
                    indexes: (OutChannels, outwidth),
                    block: (ThreadsX, 1),
                    dynamic_shared_memory_bytes: 0, stream,
                    inmap.ElementPtr(th * InChannels * inwidth * 3),
                    outmap.ElementPtr(th * OutChannels * outwidth * 3),
                    transpose_filter
                );
            }
        }

        /// <summary>引数チェック</summary>
        protected override void CheckArgument(params object[] args) {
            if (args is null || args.Length != 5) {
                throw new ArgumentException(null, nameof(args));
            }

            if (args[3] is not uint inwidth || !Limits.CheckWidth(inwidth, KernelWidth)) {
                throw new ArgumentException(nameof(inwidth));
            }

            if (args[4] is not uint batches || !Limits.CheckBatches(batches)) {
                throw new ArgumentException(nameof(batches));
            }

            uint outwidth = inwidth + 1 - KernelWidth;

            if (args[0] is not CudaArray<float> inmap || inmap.Length < InChannels * inwidth * batches * 3) {
                throw new ArgumentException(nameof(inmap));
            }

            if (args[1] is not CudaArray<float> outmap || outmap.Length < OutChannels * outwidth * batches * 3) {
                throw new ArgumentException(nameof(outmap));
            }

            if (args[2] is not CudaArray<float> filter || filter.Length < InChannels * OutChannels * KernelWidth * 4) {
                throw new ArgumentException(nameof(filter));
            }
        }
    }
}
