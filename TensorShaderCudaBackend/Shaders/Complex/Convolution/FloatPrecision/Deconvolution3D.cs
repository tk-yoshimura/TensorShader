﻿using System;
using System.Linq;

namespace TensorShaderCudaBackend.Shaders.Complex.Convolution.FloatPrecision {

    /// <summary>3次元逆畳み込み</summary>
    public sealed class Deconvolution3D : Shader {

        /// <summary>入力チャネル数</summary>
        public uint InChannels { private set; get; }

        /// <summary>出力チャネル数</summary>
        public uint OutChannels { private set; get; }

        /// <summary>フィルタサイズ</summary>
        public uint KernelWidth { private set; get; }

        /// <summary>フィルタサイズ</summary>
        public uint KernelHeight { private set; get; }

        /// <summary>フィルタサイズ</summary>
        public uint KernelDepth { private set; get; }

        /// <summary>勾配</summary>
        public bool GradMode { private set; get; }

        /// <summary>実行あたりの積数(2^30=1073741824‬)</summary>
        public static ulong MulPerExecute => 0x40000000;

        /// <summary>Xスレッド数</summary>
        private uint ThreadsX { set; get; }

        /// <summary>識別子</summary>
        public override sealed string Signature =>
            $"{GetType().Name.Split(',').Last()} {nameof(InChannels)} = {InChannels} {nameof(OutChannels)} = {OutChannels} " +
            $"{nameof(KernelWidth)} = {KernelWidth} {nameof(KernelHeight)} = {KernelHeight} {nameof(KernelDepth)} = {KernelDepth} " +
            $"{nameof(GradMode)} = {GradMode}";

        /// <summary>コンストラクタ</summary>
        public Deconvolution3D(uint inchannels, uint outchannels, uint kwidth, uint kheight, uint kdepth, bool gradmode) {
            if (!Limits.CheckChannels(inchannels, outchannels) || !Limits.CheckMultipleNum(multiple: 2, inchannels, outchannels)) {
                throw new ArgumentException($"{nameof(inchannels)}, {nameof(outchannels)}");
            }
            if (!Limits.CheckKernelSize(kwidth, kheight, kdepth)) {
                throw new ArgumentException($"{nameof(kwidth)}, {nameof(kheight)}, {nameof(kdepth)}");
            }

            this.InChannels = inchannels / 2;
            this.OutChannels = outchannels / 2;
            this.KernelWidth = kwidth;
            this.KernelHeight = kheight;
            this.KernelDepth = kdepth;
            this.GradMode = gradmode;

            this.ThreadsX = Kernel.DefaultBlockSize(OutChannels);

            string code = $@"

            {Defines.CtorFloat2}
            {Defines.Float.Fma}
            {Defines.Float.Fms}
            {Defines.Float.Complex.Mul}
            {Defines.Float.Complex.MulGrad}
            {Defines.StoreFloatSharedMemory(elemsize: 2, InChannels, ThreadsX)}

            __global__ void complex_deconvolution_3d(const float2* __restrict__ inmap, float2* __restrict__ outmap, const float2* __restrict__ filter,
                                                     unsigned int oy_offset, unsigned int oz,
                                                     unsigned int inwidth, unsigned int outwidth,
                                                     unsigned int inheight, unsigned int outheight,
                                                     unsigned int indepth) {{

                unsigned int outch = {Defines.IndexX}, tid = {Defines.ThreadIdX};
                unsigned int ox = {Defines.BlockIndexY}, oy = oy_offset + {Defines.BlockIndexZ};

                __shared__ float2 us[{InChannels}];
                float2 uv = ctor_float2(0.0, 0.0);

                for(unsigned int kz = 0, iz = oz - {KernelDepth - 1}; kz < {KernelDepth}; kz++, iz++){{
                    if(iz >= indepth){{
                        continue;
                    }}

                    for(unsigned int ky = 0, iy = oy - {KernelHeight - 1}; ky < {KernelHeight}; ky++, iy++){{
                        if(iy >= inheight){{
                            continue;
                        }}

                        for(unsigned int kx = 0, ix = ox - {KernelWidth - 1}; kx < {KernelWidth}; kx++, ix++){{
                            if(ix >= inwidth){{
                                continue;
                            }}

                            unsigned int inmap_idx = {InChannels} * (ix + inwidth * (iy + inheight * iz));
                            unsigned int filter_idx = outch + {InChannels * OutChannels} *
                                                      (({KernelWidth - 1} - kx) + {KernelWidth} * (({KernelHeight - 1} - ky) + {KernelHeight} * ({KernelDepth - 1} - kz)));

                            store_smem(inmap + inmap_idx, us, tid);

                            { (OutChannels % ThreadsX != 0 ? $"if(outch < {OutChannels}){{" : "") }

                                #pragma unroll 8
                                for(unsigned int inch = 0; inch < {InChannels}; inch++){{
                                    float2 u = us[inch];
                                    float2 v = filter[filter_idx];

                                    {(GradMode ? "complex_mulgrad" : "complex_mul")}(uv, u, v);

                                    filter_idx += {OutChannels};
                                }}

                            { (OutChannels % ThreadsX != 0 ? "}" : "") }
                            __syncthreads();
                        }}
                    }}
                }}

                { (OutChannels % ThreadsX != 0 ? $"if(outch < {OutChannels}){{" : "") }
                    unsigned int outmap_idx = outch + {OutChannels} * (ox + outwidth * (oy + outheight * oz));

                    outmap[outmap_idx] = uv;
                { (OutChannels % ThreadsX != 0 ? "}" : "") }
            }}";

            this.Kernel = new Kernel(code, "complex_deconvolution_3d");
        }

        /// <summary>実行</summary>
        public override void Execute(Stream stream, params object[] args) {
            CheckArgument(args);

            CudaArray<float> inmap = args[0] as CudaArray<float>;
            CudaArray<float> outmap = args[1] as CudaArray<float>;
            CudaArray<float> filter = args[2] as CudaArray<float>;

            uint inwidth = (args[3] as uint?).Value;
            uint inheight = (args[4] as uint?).Value;
            uint indepth = (args[5] as uint?).Value;
            uint batches = (args[6] as uint?).Value;

            uint outwidth = inwidth + KernelWidth - 1;
            uint outheight = inheight + KernelHeight - 1;
            uint outdepth = indepth + KernelDepth - 1;

            ulong mul_per_line = (ulong)InChannels * OutChannels * KernelWidth * KernelHeight * KernelDepth * outwidth * 4;

            uint lines_per_execute = (uint)(MulPerExecute / mul_per_line + 1);

            for (uint th = 0; th < batches; th++) {
                for (uint oz = 0; oz < outdepth; oz++) {
                    for (uint oy_offset = 0; oy_offset < outheight; oy_offset += lines_per_execute) {
                        uint lines = Math.Min(lines_per_execute, outheight - oy_offset);

                        Kernel.Execute(
                            indexes: (OutChannels, outwidth, lines),
                            block: (ThreadsX, 1, 1),
                            dynamic_shared_memory_bytes: 0,
                            stream,
                            inmap.ElementPtr(th * InChannels * inwidth * inheight * indepth * 2),
                            outmap.ElementPtr(th * OutChannels * outwidth * outheight * outdepth * 2),
                            filter,
                            oy_offset, oz,
                            inwidth, outwidth, inheight, outheight, indepth
                        );
                    }
                }
            }
        }

        /// <summary>引数チェック</summary>
        protected override void CheckArgument(params object[] args) {
            if (args is null || args.Length != 7) {
                throw new ArgumentException(null, nameof(args));
            }

            if (args[3] is not uint inwidth || !Limits.CheckWidth(inwidth)) {
                throw new ArgumentException(nameof(inwidth));
            }

            if (args[4] is not uint inheight || !Limits.CheckHeight(inheight)) {
                throw new ArgumentException(nameof(inheight));
            }

            if (args[5] is not uint indepth || !Limits.CheckDepth(indepth)) {
                throw new ArgumentException(nameof(indepth));
            }

            if (args[6] is not uint batches || !Limits.CheckBatches(batches)) {
                throw new ArgumentException(nameof(batches));
            }

            uint outwidth = inwidth + KernelWidth - 1;
            uint outheight = inheight + KernelHeight - 1;
            uint outdepth = indepth + KernelDepth - 1;

            if (args[0] is not CudaArray<float> inmap || inmap.Length < InChannels * inwidth * inheight * indepth * batches * 2) {
                throw new ArgumentException(nameof(inmap));
            }

            if (args[1] is not CudaArray<float> outmap || outmap.Length < OutChannels * outwidth * outheight * outdepth * batches * 2) {
                throw new ArgumentException(nameof(outmap));
            }

            if (args[2] is not CudaArray<float> filter || filter.Length < InChannels * OutChannels * KernelWidth * KernelHeight * KernelDepth * 2) {
                throw new ArgumentException(nameof(filter));
            }
        }
    }
}
