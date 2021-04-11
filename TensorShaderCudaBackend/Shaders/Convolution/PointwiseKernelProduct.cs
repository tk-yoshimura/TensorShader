using System;
using System.Linq;

using static TensorShaderCudaBackend.ArrayManipulation;

namespace TensorShaderCudaBackend.Shaders.Convolution {

    /// <summary>カーネル積</summary>
    public sealed class PointwiseKernelProduct : Shader {
        /// <summary>入力チャネル数</summary>
        public uint InChannels { private set; get; }

        /// <summary>出力チャネル数</summary>
        public uint OutChannels { private set; get; }

        /// <summary>ブロックサイズ</summary>
        public (uint x, uint y) BlockSize { private set; get; }

        /// <summary>1スレッドで処理する対象ピクセル数</summary>
        private static uint BatchPixels => 32;

        /// <summary>実行あたりの積数(2^29=536870912)</summary>
        public static ulong MulPerExecute => 0x20000000;

        /// <summary>実行あたりのポイント数(2^14=16384‬)</summary>
        public static uint PointsPerExecute => 0x4000;

        /// <summary>識別子</summary>
        public override sealed string Signature =>
            $"{GetType().Name.Split(',').Last()} {nameof(InChannels)} = {InChannels} {nameof(OutChannels)} = {OutChannels}";

        /// <summary>コンストラクタ</summary>
        public PointwiseKernelProduct(uint inchannels, uint outchannels) {
            if (!Limits.CheckChannels(inchannels, outchannels)) {
                throw new ArgumentException($"{nameof(inchannels)}, {nameof(outchannels)}");
            }

            this.InChannels = inchannels;
            this.OutChannels = outchannels;

            this.BlockSize = Kernel.MinimizeGridsBlockSize((InChannels, OutChannels));

            string code = $@"

            {Defines.FloatFloatFma}
            {Defines.AtomicAdd}

            __global__ void ptwise_kernelproduct(const float* __restrict__ inmap, const float* __restrict__ outmap, float* __restrict__ filter, 
                                                 unsigned int pl) {{

                unsigned int inch = {Defines.IndexX}, outch = {Defines.IndexY};
                unsigned int i_offset = {Defines.BlockIndexZ} * {BatchPixels};
                unsigned int tidx = {Defines.ThreadIdX}, tidy = {Defines.ThreadIdY};

                __shared__ float us[{BlockSize.x}], vs[{BlockSize.y}];

                unsigned int filter_index = (inch + {InChannels} * outch) * 2;

                float uv_hi = 0.0, uv_lo = 0.0;

                for(unsigned int i = i_offset; i < i_offset + {BatchPixels} && i < pl; i++){{
                    { (OutChannels % BlockSize.y != 0 ? $"if(tidx == 0 && outch < {OutChannels}){{" : "if(tidx == 0){") }
                        vs[tidy] = outmap[outch + {OutChannels} * i];
                    }}
                    { (InChannels % BlockSize.x != 0 ? $"if(tidy == 0 && inch < {InChannels}){{" : "if(tidy == 0){") }
                        us[tidx] = inmap[inch + {InChannels} * i];
                    }}
                    __syncthreads();

                    { (InChannels % BlockSize.x != 0 ? $"if(inch < {InChannels}){{" : "") }
                    { (OutChannels % BlockSize.y != 0 ? $"if(outch < {OutChannels}){{" : "") }

                        float u = us[tidx];
                        float v = vs[tidy];

                        floatfloat_fma(uv_hi, uv_lo, u, v);

                    { (InChannels % BlockSize.x != 0 ? "}" : "") }
                    { (OutChannels % BlockSize.y != 0 ? "}" : "") }

                    __syncthreads();
                }}

                { (InChannels % BlockSize.x != 0 ? $"if(inch < {InChannels}){{" : "") }
                { (OutChannels % BlockSize.y != 0 ? $"if(outch < {OutChannels}){{" : "") }

                    floatfloat_atomicadd(filter + filter_index, uv_hi, uv_lo);

                { (InChannels % BlockSize.x != 0 ? "}" : "") }
                { (OutChannels % BlockSize.y != 0 ? "}" : "") }
            }}";

            this.Kernel = new Kernel(code, "ptwise_kernelproduct");
        }

        /// <summary>実行</summary>
        public override void Execute(Stream stream, params object[] args) {
            CheckArgument(args);

            CudaArray<float> inmap = args[0] as CudaArray<float>;
            CudaArray<float> outmap = args[1] as CudaArray<float>;
            CudaArray<float> filter = args[2] as CudaArray<float>;

            uint pts = (args[3] as uint?).Value;

            CudaArray<float> dfloat_filter =
                WorkspaceReserver<float>.Request(stream, inmap.DeviceID, index: 0, InChannels * OutChannels * 2);
            dfloat_filter.ZerosetAsync(stream, InChannels * OutChannels * 2);

            ulong mul_per_point = (ulong)InChannels * OutChannels;
            uint points_per_execute_mul = (uint)(MulPerExecute / mul_per_point + 1);
            uint points_per_execute = Math.Min(PointsPerExecute, points_per_execute_mul);

            for (uint p = 0; p < pts; p += points_per_execute) {
                uint pl = Math.Min(points_per_execute, pts - p);
                uint xsets = (pl + BatchPixels - 1) / BatchPixels;

                Kernel.Execute(
                    indexes: (InChannels, OutChannels, xsets),
                    block: (BlockSize.x, BlockSize.y, 1),
                    dynamic_shared_memory_bytes: 0, stream,
                    inmap.ElementPtr(p * InChannels),
                    outmap.ElementPtr(p * OutChannels),
                    dfloat_filter,
                    pl
                );
            }

            HorizontalAdd(InChannels * OutChannels, dfloat_filter, filter, stream);
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
