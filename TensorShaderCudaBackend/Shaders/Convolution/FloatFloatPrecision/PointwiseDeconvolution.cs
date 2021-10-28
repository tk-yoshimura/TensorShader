using System;
using System.Linq;

namespace TensorShaderCudaBackend.Shaders.Convolution.FloatFloatPrecision {

    /// <summary>ポイントごとの逆畳み込み</summary>
    public sealed class PointwiseDeconvolution : Shader {

        /// <summary>入力チャネル数</summary>
        public uint InChannels { private set; get; }

        /// <summary>出力チャネル数</summary>
        public uint OutChannels { private set; get; }

        /// <summary>実行あたりの積数(2^30=1073741824‬)</summary>
        public static ulong MulPerExecute => 0x40000000;

        /// <summary>実行あたりのポイント数(2^14=16384‬)</summary>
        public static uint PointsPerExecute => 0x4000;

        /// <summary>Xスレッド数</summary>
        private uint ThreadsX { set; get; }

        /// <summary>識別子</summary>
        public override sealed string Signature =>
            $"{GetType().Name.Split(',').Last()} {nameof(InChannels)} = {InChannels} {nameof(OutChannels)} = {OutChannels}";

        /// <summary>コンストラクタ</summary>
        public PointwiseDeconvolution(uint inchannels, uint outchannels) {
            if (!Limits.CheckChannels(inchannels, outchannels)) {
                throw new ArgumentException($"{nameof(inchannels)}, {nameof(outchannels)}");
            }

            this.InChannels = inchannels;
            this.OutChannels = outchannels;

            this.ThreadsX = Kernel.DefaultBlockSize(OutChannels);

            string code = $@"

            {Defines.FloatFloat.Fma}
            {Defines.StoreFloatSharedMemory(elemsize: 1, InChannels, ThreadsX)}

            __global__ void ptwise_deconvolution(const float* __restrict__ inmap, float* __restrict__ outmap, const float* __restrict__ filter) {{

                unsigned int outch = {Defines.IndexX}, tid = {Defines.ThreadIdX};
                unsigned int i = {Defines.BlockIndexY};

                __shared__ float us[{InChannels}];
                float uv_hi = 0.0, uv_lo = 0.0;

                unsigned int filter_idx = outch;
                unsigned int inmap_idx = {InChannels} * i;

                store_smem(inmap + inmap_idx, us, tid);

                { (OutChannels % ThreadsX != 0 ? $"if(outch < {OutChannels}){{" : "") }

                    #pragma unroll 8
                    for(unsigned int inch = 0; inch < {InChannels}; inch++){{
                        float u = us[inch];
                        float v = filter[filter_idx];

                        floatfloat_fma(uv_hi, uv_lo, u, v);

                        filter_idx += {OutChannels};
                    }}

                    unsigned int outmap_idx = outch + {OutChannels} * i;

                    outmap[outmap_idx] = uv_hi + uv_lo;
                { (OutChannels % ThreadsX != 0 ? "}" : "") }
            }}";

            this.Kernel = new Kernel(code, "ptwise_deconvolution");
        }

        /// <summary>実行</summary>
        public override void Execute(Stream stream, params object[] args) {
            CheckArgument(args);

            CudaArray<float> inmap = args[0] as CudaArray<float>;
            CudaArray<float> outmap = args[1] as CudaArray<float>;
            CudaArray<float> filter = args[2] as CudaArray<float>;

            uint pts = (args[3] as uint?).Value;

            ulong mul_per_point = (ulong)InChannels * OutChannels;
            uint points_per_execute_mul = (uint)(MulPerExecute / mul_per_point + 1);
            uint points_per_execute = Math.Min(PointsPerExecute, points_per_execute_mul);

            for (uint p = 0; p < pts; p += points_per_execute) {
                uint pl = Math.Min(points_per_execute, pts - p);

                Kernel.Execute(
                    indexes: (OutChannels, pl),
                    block: (ThreadsX, 1),
                    dynamic_shared_memory_bytes: 0,
                    stream,
                    inmap.ElementPtr(p * InChannels),
                    outmap.ElementPtr(p * OutChannels),
                    filter
                );
            }
        }

        /// <summary>引数チェック</summary>
        protected override void CheckArgument(params object[] args) {
            if (args is null || args.Length != 4) {
                throw new ArgumentException(nameof(args));
            }

            if (!(args[3] is uint pts) || pts < 1) {
                throw new ArgumentException(nameof(pts));
            }

            if (!(args[0] is CudaArray<float> inmap) || inmap.Length < InChannels * pts) {
                throw new ArgumentException(nameof(inmap));
            }

            if (!(args[1] is CudaArray<float> outmap) || outmap.Length < OutChannels * pts) {
                throw new ArgumentException(nameof(outmap));
            }

            if (!(args[2] is CudaArray<float> filter) || filter.Length < InChannels * OutChannels) {
                throw new ArgumentException(nameof(filter));
            }
        }
    }
}
