﻿using System;
using System.Linq;

namespace TensorShaderCudaBackend.Shaders.Quaternion.Convolution.FloatFloatPrecision {

    /// <summary>2次元逆畳み込み</summary>
    public sealed class Deconvolution2D : Shader {

        /// <summary>入力チャネル数</summary>
        public uint InChannels { private set; get; }

        /// <summary>出力チャネル数</summary>
        public uint OutChannels { private set; get; }

        /// <summary>フィルタサイズ</summary>
        public uint KernelWidth { private set; get; }

        /// <summary>フィルタサイズ</summary>
        public uint KernelHeight { private set; get; }

        /// <summary>勾配</summary>
        public bool GradMode { private set; get; }

        /// <summary>実行あたりの積数(2^30=1073741824‬)</summary>
        public static ulong MulPerExecute => 0x40000000;

        /// <summary>Xスレッド数</summary>
        private uint ThreadsX { set; get; }

        /// <summary>識別子</summary>
        public override sealed string Signature =>
            $"{GetType().Name.Split(',').Last()} {nameof(InChannels)} = {InChannels} {nameof(OutChannels)} = {OutChannels} " +
            $"{nameof(KernelWidth)} = {KernelWidth} {nameof(KernelHeight)} = {KernelHeight} {nameof(GradMode)} = {GradMode}";

        /// <summary>コンストラクタ</summary>
        public Deconvolution2D(uint inchannels, uint outchannels, uint kwidth, uint kheight, bool gradmode) {
            if (!Limits.CheckChannels(inchannels, outchannels) || !Limits.CheckMultipleNum(multiple: 4, inchannels, outchannels)) {
                throw new ArgumentException($"{nameof(inchannels)}, {nameof(outchannels)}");
            }
            if (!Limits.CheckKernelSize(kwidth, kheight)) {
                throw new ArgumentException($"{nameof(kwidth)}, {nameof(kheight)}");
            }

            this.InChannels = inchannels / 4;
            this.OutChannels = outchannels / 4;
            this.KernelWidth = kwidth;
            this.KernelHeight = kheight;
            this.GradMode = gradmode;

            this.ThreadsX = Kernel.DefaultBlockSize(OutChannels);

            string code = $@"

            {Defines.CtorFloat4}
            {Defines.FloatFloat.Fma}
            {Defines.FloatFloat.Fms}
            {Defines.FloatFloat.HiLoAdd}
            {Defines.FloatFloat.Quaternion.Mul}
            {Defines.FloatFloat.Quaternion.MulGrad}
            {Defines.StoreFloatSharedMemory(elemsize: 4, InChannels, ThreadsX)}

            __global__ void quaternion_deconvolution_2d(const float4* __restrict__ inmap, float4* __restrict__ outmap, const float4* __restrict__ filter,
                                                        unsigned int oy_offset,
                                                        unsigned int inwidth, unsigned int outwidth,
                                                        unsigned int inheight) {{

                unsigned int outch = {Defines.IndexX}, tid = {Defines.ThreadIdX};
                unsigned int ox = {Defines.BlockIndexY}, oy = oy_offset + {Defines.BlockIndexZ};

                __shared__ float4 us[{InChannels}];
                float4 uv_hi = ctor_float4(0.0, 0.0, 0.0, 0.0), uv_lo = ctor_float4(0.0, 0.0, 0.0, 0.0);

                for(unsigned int ky = 0, iy = oy - {KernelHeight - 1}; ky < {KernelHeight}; ky++, iy++){{
                    if(iy >= inheight){{
                        continue;
                    }}

                    for(unsigned int kx = 0, ix = ox - {KernelWidth - 1}; kx < {KernelWidth}; kx++, ix++){{
                        if(ix >= inwidth){{
                            continue;
                        }}

                        unsigned int inmap_idx = {InChannels} * (ix + inwidth * iy);
                        unsigned int filter_idx = outch + {InChannels * OutChannels} *
                                                  (({KernelWidth - 1} - kx) + {KernelWidth} * ({KernelHeight - 1} - ky));

                        store_smem(inmap + inmap_idx, us, tid);

                        { (OutChannels % ThreadsX != 0 ? $"if(outch < {OutChannels}){{" : "") }

                            #pragma unroll 8
                            for(unsigned int inch = 0; inch < {InChannels}; inch++){{
                                float4 u = us[inch];
                                float4 v = filter[filter_idx];

                                {(GradMode ? "quaternion_mulgrad" : "quaternion_mul")}(uv_hi, uv_lo, u, v);

                                filter_idx += {OutChannels};
                            }}

                        { (OutChannels % ThreadsX != 0 ? "}" : "") }
                        __syncthreads();
                    }}
                }}

                { (OutChannels % ThreadsX != 0 ? $"if(outch < {OutChannels}){{" : "") }
                    unsigned int outmap_idx = outch + {OutChannels} * (ox + outwidth * oy);

                    outmap[outmap_idx] = ctor_float4(uv_hi.x, uv_hi.y, uv_hi.z, uv_hi.w);
                { (OutChannels % ThreadsX != 0 ? "}" : "") }
            }}";

            this.Kernel = new Kernel(code, "quaternion_deconvolution_2d");
        }

        /// <summary>実行</summary>
        public override void Execute(Stream stream, params object[] args) {
            CheckArgument(args);

            CudaArray<float> inmap = args[0] as CudaArray<float>;
            CudaArray<float> outmap = args[1] as CudaArray<float>;
            CudaArray<float> filter = args[2] as CudaArray<float>;

            uint inwidth = (args[3] as uint?).Value;
            uint inheight = (args[4] as uint?).Value;
            uint batches = (args[5] as uint?).Value;

            uint outwidth = inwidth + KernelWidth - 1;
            uint outheight = inheight + KernelHeight - 1;

            ulong mul_per_line = (ulong)InChannels * OutChannels * KernelWidth * KernelHeight * outwidth * 16;

            uint lines_per_execute = (uint)(MulPerExecute / mul_per_line + 1);

            for (uint th = 0; th < batches; th++) {
                for (uint oy_offset = 0; oy_offset < outheight; oy_offset += lines_per_execute) {
                    uint lines = Math.Min(lines_per_execute, outheight - oy_offset);

                    Kernel.Execute(
                        indexes: (OutChannels, outwidth, lines),
                        block: (ThreadsX, 1, 1),
                        dynamic_shared_memory_bytes: 0,
                        stream,
                        inmap.ElementPtr(th * InChannels * inwidth * inheight * 4),
                        outmap.ElementPtr(th * OutChannels * outwidth * outheight * 4),
                        filter,
                        oy_offset,
                        inwidth, outwidth, inheight
                    );
                }
            }
        }

        /// <summary>引数チェック</summary>
        protected override void CheckArgument(params object[] args) {
            if (args is null || args.Length != 6) {
                throw new ArgumentException(null, nameof(args));
            }

            if (args[3] is not uint inwidth || !Limits.CheckWidth(inwidth)) {
                throw new ArgumentException(nameof(inwidth));
            }

            if (args[4] is not uint inheight || !Limits.CheckHeight(inheight)) {
                throw new ArgumentException(nameof(inheight));
            }

            if (args[5] is not uint batches || !Limits.CheckBatches(batches)) {
                throw new ArgumentException(nameof(batches));
            }

            uint outwidth = inwidth + KernelWidth - 1;
            uint outheight = inheight + KernelHeight - 1;

            if (args[0] is not CudaArray<float> inmap || inmap.Length < InChannels * inwidth * inheight * batches * 4) {
                throw new ArgumentException(nameof(inmap));
            }

            if (args[1] is not CudaArray<float> outmap || outmap.Length < OutChannels * outwidth * outheight * batches * 4) {
                throw new ArgumentException(nameof(outmap));
            }

            if (args[2] is not CudaArray<float> filter || filter.Length < InChannels * OutChannels * KernelWidth * KernelHeight * 4) {
                throw new ArgumentException(nameof(filter));
            }
        }
    }
}
