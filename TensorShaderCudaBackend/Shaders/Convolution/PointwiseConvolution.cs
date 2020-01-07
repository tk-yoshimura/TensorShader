﻿using System;
using System.Linq;

using static TensorShaderCudaBackend.Transpose;

namespace TensorShaderCudaBackend.Shaders.Convolution {

    /// <summary>ポイントごとの畳み込み</summary>
    public sealed class PointwiseConvolution : Shader {

        /// <summary>入力チャネル数</summary>
        public uint InChannels { private set; get; }

        /// <summary>出力チャネル数</summary>
        public uint OutChannels { private set; get; }

        /// <summary>実行あたりの積数(2^25=33554432‬)</summary>
        public static uint MulPerExecute => 0x2000000;

        /// <summary>実行あたりのポイント数(2^14=16384‬)</summary>
        public static uint PointsPerExecute => 0x4000;

        /// <summary>識別子</summary>
        public override sealed string Signature =>
            $"{GetType().Name.Split(',').Last()} {nameof(InChannels)} = {InChannels} {nameof(OutChannels)} = {OutChannels}";

        /// <summary>コンストラクタ</summary>
        public PointwiseConvolution(uint inchannels, uint outchannels) {
            if (!Limits.CheckChannels(inchannels, outchannels)) {
                throw new ArgumentException($"{nameof(inchannels)}, {nameof(outchannels)}");
            }

            this.InChannels = inchannels;
            this.OutChannels = outchannels;

            string code = $@"

            static __inline__ __device__ void floatfloat_add(float &hi, float &lo, float val){{
                float tmp = hi;
                hi += val;
                lo -= (hi - tmp) - val;
            }}

            __global__ void ptwise_convolution(float *inmap, float *outmap, float *filter) {{

                unsigned int outch = {Defines.IndexX}, tid = {Defines.ThreadIdX}, threads = {Defines.ThreadsX};
                unsigned int i = {Defines.BlockIndexY};

                __shared__ float us[{InChannels}];
                float uv_hi = 0.0, uv_lo = 0.0;

                unsigned int inmap_idx = {InChannels} * i;
                unsigned int filter_idx = outch;

                for(unsigned int inch = tid; inch < {InChannels}; inch += threads){{
                    us[inch] = inmap[inch + inmap_idx];
                }}
                __syncthreads();

                if(outch < {OutChannels}){{
                    for(unsigned int inch = 0; inch < {InChannels}; inch++){{                            
                        float u = us[inch];
                        float v = filter[filter_idx];

                        floatfloat_add(uv_hi, uv_lo, u * v);

                        filter_idx += {OutChannels};
                    }}

                    unsigned int outmap_idx = outch + {OutChannels} * i;

                    outmap[outmap_idx] = uv_hi + uv_lo;
                }}
            }}";

            this.Kernel = new Kernel(code, "ptwise_convolution");
        }

        /// <summary>実行</summary>
        public override void Execute(Stream stream, params object[] args) {
            CheckArgument(args);

            CudaArray<float> inmap = args[0] as CudaArray<float>;
            CudaArray<float> outmap = args[1] as CudaArray<float>;
            CudaArray<float> filter = args[2] as CudaArray<float>;

            uint pts = (args[3] as uint?).Value;

            CudaArray<float> transpose_filter =
                CudaArrayReserver<float>.Request(stream, filter.DeviceID, index: 0, InChannels * OutChannels);

            TransposeKernelChannel(InChannels, OutChannels, 1u, filter, transpose_filter, stream);

            uint mul_per_point = InChannels * OutChannels;
            uint points_per_execute_mul = MulPerExecute / mul_per_point + 1;
            uint points_per_execute = Math.Min(PointsPerExecute, points_per_execute_mul);

            for (uint p = 0; p < pts; p += points_per_execute) {
                uint pl = Math.Min(points_per_execute, pts - p);

                Kernel.Execute(
                    indexes: (OutChannels, pl),
                    block: (Kernel.DefaultBlockSize(OutChannels), 1),
                    dynamic_shared_memory_bytes: 0,
                    stream,
                    inmap.ElementPtr(p * InChannels),
                    outmap.ElementPtr(p * OutChannels),
                    transpose_filter
                );
            }
        }

        /// <summary>引数チェック</summary>
        protected override void CheckArgument(params object[] args) {
            if (args == null || args.Length != 4) {
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