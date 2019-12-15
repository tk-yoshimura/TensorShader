﻿using System;
using System.Linq;

namespace TensorShaderCudaBackend.Shaders.Convolution {

    /// <summary>転置全結合</summary>
    public sealed class TransposeDense : Shader {

        /// <summary>入力チャネル数</summary>
        public uint InChannels { private set; get; }

        /// <summary>出力チャネル数</summary>
        public uint OutChannels { private set; get; }
                
        /// <summary>識別子</summary>
        public override sealed string Signature => 
            $"{GetType().Name.Split(',').Last()} {nameof(InChannels)} = {InChannels} {nameof(OutChannels)} = {OutChannels}";

        /// <summary>コンストラクタ</summary>
        public TransposeDense(uint inchannels, uint outchannels) { 
            if (inchannels < 1 || inchannels > Limits.Channels) {
                throw new ArgumentException(nameof(inchannels));
            }
            if (outchannels < 1 || outchannels > Limits.Channels) {
                throw new ArgumentException(nameof(outchannels));
            }

            this.InChannels = inchannels;
            this.OutChannels = outchannels;

            string code = $@"

            __global__ void transpose_dense(const float* __restrict__ inmap, float *outmap, float *filter) {{

                unsigned int outch = {Defines.IndexX}, tid = {Defines.ThreadIdX}, threads = {Defines.ThreadsX};

                __shared__ float us[{InChannels}];
                float uv_hi = 0, uv_lo = 0;

                unsigned int filter_idx = outch;

                for(unsigned int inch = tid; inch < {InChannels}; inch += threads){{
                    us[inch] = inmap[inch];
                }}
                __syncthreads();

                if(outch < {OutChannels}){{
                        
                    for(unsigned int inch = 0; inch < {InChannels}; inch++){{                            
                        float u = us[inch];
                        float v = filter[filter_idx];

                        float uv = u * v;
                        float tmp = uv_hi;
                        uv_hi += uv;
                        uv_lo -= (uv_hi - tmp) - uv;

                        filter_idx += {OutChannels};
                    }}

                    outmap[outch] = uv_hi + uv_lo;
                }}
            }}";

            this.Kernel = new Kernel(code, "transpose_dense");
        }

        /// <summary>実行</summary>
        public override void Execute(Stream stream, params object[] args) {
            CheckArgument(args);

            CudaArray<float> inmap = args[0] as CudaArray<float>;
            CudaArray<float> outmap = args[1] as CudaArray<float>;
            CudaArray<float> filter = args[2] as CudaArray<float>;
           
            uint batches = (args[3] as uint?).Value;

            for (uint th = 0; th < batches; th++) {
                Kernel.Execute(OutChannels,
                    dynamic_shared_memory_bytes: 0, stream,
                    inmap.ElementPtr(th * InChannels), 
                    outmap.ElementPtr(th * OutChannels),
                    filter
                );
            }
        }

        /// <summary>引数チェック</summary>
        protected override void CheckArgument(params object[] args) {
            if (args == null || args.Length != 4) {
                throw new ArgumentException(nameof(args));
            }

            if (!(args[3] is uint batches) || batches < 1) {
                throw new ArgumentException($"{nameof(args)}[3]");
            }

            if (!(args[0] is CudaArray<float> inmap) || inmap.Length < InChannels * batches) {
                throw new ArgumentException($"{nameof(args)}[0]");
            }

            if (!(args[1] is CudaArray<float> outmap) || outmap.Length < OutChannels * batches) {
                throw new ArgumentException($"{nameof(args)}[1]");
            }

            if (!(args[2] is CudaArray<float> filter) || filter.Length < InChannels * OutChannels) {
                throw new ArgumentException($"{nameof(args)}[2]");
            }
        }
    }
}
