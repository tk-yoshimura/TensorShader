﻿using System;
using System.Linq;

using static TensorShaderCudaBackend.Transpose;

namespace TensorShaderCudaBackend.Shaders.Convolution.FloatFloatPrecision {

    /// <summary>3次元畳み込み</summary>
    public sealed class Convolution3D : Shader {

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

        /// <summary>実行あたりの積数(2^30=1073741824‬)</summary>
        public static ulong MulPerExecute => 0x40000000;

        /// <summary>Xスレッド数</summary>
        private uint ThreadsX { set; get; }

        /// <summary>識別子</summary>
        public override sealed string Signature =>
            $"{GetType().Name.Split(',').Last()} {nameof(InChannels)} = {InChannels} {nameof(OutChannels)} = {OutChannels} " +
            $"{nameof(KernelWidth)} = {KernelWidth} {nameof(KernelHeight)} = {KernelHeight} {nameof(KernelDepth)} = {KernelDepth}";

        /// <summary>コンストラクタ</summary>
        public Convolution3D(uint inchannels, uint outchannels, uint kwidth, uint kheight, uint kdepth) {
            if (!Limits.CheckChannels(inchannels, outchannels)) {
                throw new ArgumentException($"{nameof(inchannels)}, {nameof(outchannels)}");
            }
            if (!Limits.CheckKernelSize(kwidth, kheight, kdepth)) {
                throw new ArgumentException($"{nameof(kwidth)}, {nameof(kheight)}, {nameof(kdepth)}");
            }

            this.InChannels = inchannels;
            this.OutChannels = outchannels;
            this.KernelWidth = kwidth;
            this.KernelHeight = kheight;
            this.KernelDepth = kdepth;

            this.ThreadsX = Kernel.DefaultBlockSize(OutChannels);

            string code = $@"

            {Defines.FloatFloat.Fma}
            {Defines.StoreFloatSharedMemory(elemsize: 1, InChannels, ThreadsX)}

            __global__ void convolution_3d(const float* __restrict__ inmap, float* __restrict__ outmap, const float* __restrict__ filter,
                                           unsigned int oy_offset, unsigned int oz,
                                           unsigned int inwidth, unsigned int outwidth,
                                           unsigned int inheight, unsigned int outheight) {{

                unsigned int outch = {Defines.IndexX}, tid = {Defines.ThreadIdX};
                unsigned int ox = {Defines.BlockIndexY}, oy = oy_offset + {Defines.BlockIndexZ};

                __shared__ float us[{InChannels}];
                float uv_hi = 0.0, uv_lo = 0.0;

                unsigned int filter_idx = outch;

                for(unsigned int kz = 0, iz = oz; kz < {KernelDepth}; kz++, iz++){{
                    for(unsigned int ky = 0, iy = oy; ky < {KernelHeight}; ky++, iy++){{

                        unsigned int inmap_idx = {InChannels} * (ox + inwidth * (iy + inheight * iz));

                        { (KernelWidth <= 7 ? "#pragma unroll" : "") }
                        for(unsigned int kx = 0, ix = ox; kx < {KernelWidth}; kx++, ix++){{

                            store_smem(inmap + inmap_idx, us, tid);
                            inmap_idx += {InChannels};

                            { (OutChannels % ThreadsX != 0 ? $"if(outch < {OutChannels}){{" : "") }

                                #pragma unroll 8
                                for(unsigned int inch = 0; inch < {InChannels}; inch++){{
                                    float u = us[inch];
                                    float v = filter[filter_idx];

                                    floatfloat_fma(uv_hi, uv_lo, u, v);

                                    filter_idx += {OutChannels};
                                }}

                            { (OutChannels % ThreadsX != 0 ? "}" : "") }
                            __syncthreads();
                        }}
                    }}
                }}

                { (OutChannels % ThreadsX != 0 ? $"if(outch < {OutChannels}){{" : "") }
                    unsigned int outmap_idx = outch + {OutChannels} * (ox + outwidth * (oy + outheight * oz));

                    outmap[outmap_idx] = uv_hi;
                { (OutChannels % ThreadsX != 0 ? "}" : "") }
            }}";

            this.Kernel = new Kernel(code, "convolution_3d");
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

            uint outwidth = inwidth + 1 - KernelWidth;
            uint outheight = inheight + 1 - KernelHeight;
            uint outdepth = indepth + 1 - KernelDepth;

            ulong mul_per_line = (ulong)InChannels * OutChannels * KernelWidth * KernelHeight * KernelDepth * outwidth;

            uint lines_per_execute = (uint)(MulPerExecute / mul_per_line + 1);

            CudaArray<float> transpose_filter =
                WorkspaceReserver<float>.Request(stream, filter.DeviceID, index: 0, InChannels * OutChannels * KernelWidth * KernelHeight * KernelDepth);

            TransposeKernelChannel(InChannels, OutChannels, KernelWidth * KernelHeight * KernelDepth, filter, transpose_filter, stream);

            for (uint th = 0; th < batches; th++) {
                for (uint oz = 0; oz < outdepth; oz++) {
                    for (uint oy_offset = 0; oy_offset < outheight; oy_offset += lines_per_execute) {
                        uint lines = Math.Min(lines_per_execute, outheight - oy_offset);

                        Kernel.Execute(
                            indexes: (OutChannels, outwidth, lines),
                            block: (ThreadsX, 1, 1),
                            dynamic_shared_memory_bytes: 0,
                            stream,
                            inmap.ElementPtr(th * InChannels * inwidth * inheight * indepth),
                            outmap.ElementPtr(th * OutChannels * outwidth * outheight * outdepth),
                            transpose_filter,
                            oy_offset, oz,
                            inwidth, outwidth, inheight, outheight
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

            if (args[3] is not uint inwidth || !Limits.CheckWidth(inwidth, KernelWidth)) {
                throw new ArgumentException($"{nameof(inwidth)}");
            }

            if (args[4] is not uint inheight || !Limits.CheckHeight(inheight, KernelHeight)) {
                throw new ArgumentException($"{nameof(inheight)}");
            }

            if (args[5] is not uint indepth || !Limits.CheckDepth(indepth, KernelDepth)) {
                throw new ArgumentException($"{nameof(indepth)}");
            }

            if (args[6] is not uint batches || !Limits.CheckBatches(batches)) {
                throw new ArgumentException($"{nameof(batches)}");
            }

            uint outwidth = inwidth + 1 - KernelWidth;
            uint outheight = inheight + 1 - KernelHeight;
            uint outdepth = indepth + 1 - KernelDepth;

            if (args[0] is not CudaArray<float> inmap || inmap.Length < InChannels * inwidth * inheight * indepth * batches) {
                throw new ArgumentException($"{nameof(inmap)}");
            }

            if (args[1] is not CudaArray<float> outmap || outmap.Length < OutChannels * outwidth * outheight * outdepth * batches) {
                throw new ArgumentException($"{nameof(outmap)}");
            }

            if (args[2] is not CudaArray<float> filter || filter.Length < InChannels * OutChannels * KernelWidth * KernelHeight * KernelDepth) {
                throw new ArgumentException($"{nameof(filter)}");
            }
        }
    }
}
