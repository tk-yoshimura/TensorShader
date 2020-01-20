using System;
using System.Linq;

using static TensorShaderCudaBackend.Transpose;

namespace TensorShaderCudaBackend.Shaders.Complex.Convolution {

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

        /// <summary>勾配</summary>
        public bool GradMode { private set; get; }

        /// <summary>実行あたりの積数(2^30=1073741824‬)</summary>
        public static ulong MulPerExecute => 0x40000000;

        /// <summary>識別子</summary>
        public override sealed string Signature =>
            $"{GetType().Name.Split(',').Last()} {nameof(InChannels)} = {InChannels} {nameof(OutChannels)} = {OutChannels} " +
            $"{nameof(KernelWidth)} = {KernelWidth} {nameof(KernelHeight)} = {KernelHeight} {nameof(KernelDepth)} = {KernelDepth} " +
            $"{nameof(GradMode)} = {GradMode}";

        /// <summary>コンストラクタ</summary>
        public Convolution3D(uint inchannels, uint outchannels, uint kwidth, uint kheight, uint kdepth, bool gradmode) {
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

            string code = $@"

            {Defines.CtorFloat2}
            {Defines.FloatFloatAdd}
            {Defines.FloatFloatSub}
            {Defines.Complex.Mul}
            {Defines.Complex.MulGrad}
            {Defines.StoreSharedMemory("float2", InChannels)}

            __global__ void complex_convolution_3d(float2 *inmap, float2 *outmap, float2 *filter,
                                                   unsigned int oy_offset, unsigned int oz,
                                                   unsigned int inwidth, unsigned int outwidth,
                                                   unsigned int inheight, unsigned int outheight) {{

                unsigned int outch = {Defines.IndexX}, tid = {Defines.ThreadIdX}, threads = {Defines.ThreadsX};
                unsigned int ox = {Defines.BlockIndexY}, oy = oy_offset + {Defines.BlockIndexZ};

                __shared__ float2 us[{InChannels}];
                float2 uv_hi = ctor_float2(0.0, 0.0), uv_lo = ctor_float2(0.0, 0.0);

                for(unsigned int kz = 0, iz = oz; kz < {KernelDepth}; kz++, iz++){{
                    for(unsigned int ky = 0, iy = oy; ky < {KernelHeight}; ky++, iy++){{
                        for(unsigned int kx = 0, ix = ox; kx < {KernelWidth}; kx++, ix++){{

                            unsigned int inmap_idx = {InChannels} * (ix + inwidth * (iy + inheight * iz));
                            unsigned int filter_idx = outch + {InChannels * OutChannels} * (kx + {KernelWidth} * (ky + {KernelHeight} * kz));

                            store_smem(inmap + inmap_idx, us, tid, threads);

                            if(outch < {OutChannels}){{
                                for(unsigned int inch = 0; inch < {InChannels}; inch++){{
                                    float2 u = us[inch];
                                    float2 v = filter[filter_idx];

                                    {(GradMode ? "complex_mulgrad" : "complex_mul")}(uv_hi, uv_lo, u, v);

                                    filter_idx += {OutChannels};
                                }}

                            }}
                            __syncthreads();
                        }}
                    }}
                }}

                if(outch < {OutChannels}){{
                    unsigned int outmap_idx = outch + {OutChannels} * (ox + outwidth * (oy + outheight * oz));

                    outmap[outmap_idx] = ctor_float2(uv_hi.x + uv_lo.x, uv_hi.y + uv_lo.y);
                }}
            }}";

            this.Kernel = new Kernel(code, "complex_convolution_3d");
            this.Kernel.SetCacheAllocationFromUsageSharedMemory(InChannels * 2 * 4);
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

            ulong mul_per_line = (ulong)InChannels * OutChannels * KernelWidth * KernelHeight * KernelDepth * outwidth * 4;

            uint lines_per_execute = (uint)(MulPerExecute / mul_per_line + 1);

            CudaArray<float> transpose_filter =
                CudaArrayReserver<float>.Request(stream, filter.DeviceID, index: 0, InChannels * OutChannels * KernelWidth * KernelHeight * KernelDepth * 2);

            TransposeComplexKernelChannel(InChannels * 2, OutChannels * 2, KernelWidth * KernelHeight * KernelDepth, filter, transpose_filter, stream);

            for (uint th = 0; th < batches; th++) {
                for (uint oz = 0; oz < outdepth; oz++) {
                    for (uint oy_offset = 0; oy_offset < outheight; oy_offset += lines_per_execute) {
                        uint lines = Math.Min(lines_per_execute, outheight - oy_offset);

                        Kernel.Execute(
                            indexes: (OutChannels, outwidth, lines),
                            block: (Kernel.DefaultBlockSize(OutChannels), 1, 1),
                            dynamic_shared_memory_bytes: 0, stream,
                            inmap.ElementPtr(th * InChannels * inwidth * inheight * indepth * 2),
                            outmap.ElementPtr(th * OutChannels * outwidth * outheight * outdepth * 2),
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
            if (args == null || args.Length != 7) {
                throw new ArgumentException(nameof(args));
            }

            if (!(args[3] is uint inwidth) || !Limits.CheckWidth(inwidth, KernelWidth)) {
                throw new ArgumentException(nameof(inwidth));
            }

            if (!(args[4] is uint inheight) || !Limits.CheckHeight(inheight, KernelHeight)) {
                throw new ArgumentException(nameof(inheight));
            }

            if (!(args[5] is uint indepth) || !Limits.CheckDepth(indepth, KernelDepth)) {
                throw new ArgumentException(nameof(indepth));
            }

            if (!(args[6] is uint batches) || !Limits.CheckBatches(batches)) {
                throw new ArgumentException(nameof(batches));
            }

            uint outwidth = inwidth + 1 - KernelWidth;
            uint outheight = inheight + 1 - KernelHeight;
            uint outdepth = indepth + 1 - KernelDepth;

            if (!(args[0] is CudaArray<float> inmap) || inmap.Length < InChannels * inwidth * inheight * indepth * batches * 2) {
                throw new ArgumentException(nameof(inmap));
            }

            if (!(args[1] is CudaArray<float> outmap) || outmap.Length < OutChannels * outwidth * outheight * outdepth * batches * 2) {
                throw new ArgumentException(nameof(outmap));
            }

            if (!(args[2] is CudaArray<float> filter) || filter.Length < InChannels * OutChannels * KernelWidth * KernelHeight * KernelDepth * 2) {
                throw new ArgumentException(nameof(filter));
            }
        }
    }
}
