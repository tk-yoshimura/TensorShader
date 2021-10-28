using System;
using System.Linq;

namespace TensorShaderCudaBackend.Shaders.Quaternion.Convolution.FloatFloatPrecision {

    /// <summary>1次元逆畳み込み</summary>
    public sealed class Deconvolution1D : Shader {

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
        public Deconvolution1D(uint inchannels, uint outchannels, uint kwidth, bool gradmode) {
            if (!Limits.CheckChannels(inchannels, outchannels) || !Limits.CheckMultipleNum(multiple: 4, inchannels, outchannels)) {
                throw new ArgumentException($"{nameof(inchannels)}, {nameof(outchannels)}");
            }
            if (!Limits.CheckKernelSize(kwidth)) {
                throw new ArgumentException(nameof(kwidth));
            }

            this.InChannels = inchannels / 4;
            this.OutChannels = outchannels / 4;
            this.KernelWidth = kwidth;
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

            __global__ void quaternion_deconvolution_1d(const float4* __restrict__ inmap, float4* __restrict__ outmap, const float4* __restrict__ filter,
                                                        unsigned int inwidth) {{

                unsigned int outch = {Defines.IndexX}, tid = {Defines.ThreadIdX};
                unsigned int ox = {Defines.BlockIndexY};

                __shared__ float4 us[{InChannels}];
                float4 uv_hi = ctor_float4(0.0, 0.0, 0.0, 0.0), uv_lo = ctor_float4(0.0, 0.0, 0.0, 0.0);

                for(unsigned int kx = 0, ix = ox - {KernelWidth - 1}; kx < {KernelWidth}; kx++, ix++){{
                    if(ix >= inwidth){{
                        continue;
                    }}

                    unsigned int inmap_idx = {InChannels} * ix;
                    unsigned int filter_idx = outch + {InChannels * OutChannels} * ({KernelWidth - 1} - kx);

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

                { (OutChannels % ThreadsX != 0 ? $"if(outch < {OutChannels}){{" : "") }
                    unsigned int outmap_idx = outch + {OutChannels} * ox;

                    outmap[outmap_idx] = ctor_float4(uv_hi.x + uv_lo.x, uv_hi.y + uv_lo.y, uv_hi.z + uv_lo.z, uv_hi.w + uv_lo.w);
                { (OutChannels % ThreadsX != 0 ? "}" : "") }
            }}";

            this.Kernel = new Kernel(code, "quaternion_deconvolution_1d");
        }

        /// <summary>実行</summary>
        public override void Execute(Stream stream, params object[] args) {
            CheckArgument(args);

            CudaArray<float> inmap = args[0] as CudaArray<float>;
            CudaArray<float> outmap = args[1] as CudaArray<float>;
            CudaArray<float> filter = args[2] as CudaArray<float>;

            uint inwidth = (args[3] as uint?).Value;
            uint batches = (args[4] as uint?).Value;

            uint outwidth = inwidth + KernelWidth - 1;

            for (uint th = 0; th < batches; th++) {
                Kernel.Execute(
                    indexes: (OutChannels, outwidth),
                    block: (ThreadsX, 1),
                    dynamic_shared_memory_bytes: 0,
                    stream,
                    inmap.ElementPtr(th * InChannels * inwidth * 4),
                    outmap.ElementPtr(th * OutChannels * outwidth * 4),
                    filter,
                    inwidth
                );
            }
        }

        /// <summary>引数チェック</summary>
        protected override void CheckArgument(params object[] args) {
            if (args is null || args.Length != 5) {
                throw new ArgumentException(nameof(args));
            }

            if (!(args[3] is uint inwidth) || !Limits.CheckWidth(inwidth)) {
                throw new ArgumentException(nameof(inwidth));
            }

            if (!(args[4] is uint batches) || !Limits.CheckBatches(batches)) {
                throw new ArgumentException(nameof(batches));
            }

            uint outwidth = inwidth + KernelWidth - 1;

            if (!(args[0] is CudaArray<float> inmap) || inmap.Length < InChannels * inwidth * batches * 4) {
                throw new ArgumentException(nameof(inmap));
            }

            if (!(args[1] is CudaArray<float> outmap) || outmap.Length < OutChannels * outwidth * batches * 4) {
                throw new ArgumentException(nameof(outmap));
            }

            if (!(args[2] is CudaArray<float> filter) || filter.Length < InChannels * OutChannels * KernelWidth * 4) {
                throw new ArgumentException(nameof(filter));
            }
        }
    }
}
