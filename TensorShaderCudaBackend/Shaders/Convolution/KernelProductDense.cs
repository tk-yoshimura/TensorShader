using System;
using System.Linq;

namespace TensorShaderCudaBackend.Shaders.Convolution {

    /// <summary>カーネル積</summary>
    public sealed class KernelProductDense : Shader {
        private static ArrayManipulation.HorizontalAdd hadd;

        /// <summary>入力チャネル数</summary>
        public uint InChannels { private set; get; }

        /// <summary>出力チャネル数</summary>
        public uint OutChannels { private set; get; }

        /// <summary>ブロックサイズ</summary>
        public (uint x, uint y) BlockSize { private set; get; }
                
        /// <summary>識別子</summary>
        public override sealed string Signature => 
            $"{GetType().Name.Split(',').Last()} {nameof(InChannels)} = {InChannels} {nameof(OutChannels)} = {OutChannels}";
        
        /// <summary>コンストラクタ</summary>
        public KernelProductDense(uint inchannels, uint outchannels) { 
            if (inchannels < 1 || inchannels > Limits.Channels) {
                throw new ArgumentException(nameof(inchannels));
            }
            if (outchannels < 1 || outchannels > Limits.Channels) {
                throw new ArgumentException(nameof(outchannels));
            }

            this.InChannels = inchannels;
            this.OutChannels = outchannels;

            this.BlockSize = Kernel.MinimizeGridsBlockSize((InChannels, OutChannels));

            string code = $@"
            __global__ void kernelproduct_dense(const float* __restrict__ inmap, const float* __restrict__ outmap, float *filter) {{

                unsigned int inch = {Defines.IndexX}, outch = {Defines.IndexY};
                unsigned int tidx = {Defines.ThreadIdX}, tidy = {Defines.ThreadIdY};

                __shared__ float us[{BlockSize.x}], vs[{BlockSize.y}];
                unsigned int filter_offset = (inch + {InChannels} * outch) * 2;
                filter += filter_offset;

                if(tidy == 0 && inch < {InChannels}){{
                    us[tidx] = inmap[inch];
                }}
                if(tidx == 0 && outch < {OutChannels}){{
                    vs[tidy] = outmap[outch];
                }}
                __syncthreads();

                if(inch < {InChannels} && outch < {OutChannels}){{
                    float u = us[tidx];
                    float v = vs[tidy];

                    float uv = u * v;

                    int uvi = ((*(int*)(&uv)) & 0xFFFFF000);

                    float uv_hi = *(float*)(&uvi);
                    float uv_lo = uv - uv_hi;

                    filter[0] += uv_hi;
                    filter[1] += uv_lo;
                }}
            }}";

            this.Kernel = new Kernel(code, "kernelproduct_dense");
        }

        /// <summary>実行</summary>
        public override void Execute(Stream stream, params object[] args) {
            CheckArgument(args);

            if(hadd == null) { 
                hadd = new ArrayManipulation.HorizontalAdd();
            }

            CudaArray<float> inmap = args[0] as CudaArray<float>;
            CudaArray<float> outmap = args[1] as CudaArray<float>;
            CudaArray<float> filter = args[2] as CudaArray<float>;
           
            uint batches = (args[3] as uint?).Value;

            CudaArray<float> dfloat_filter = 
                CudaArrayReserver<float>.Request(stream, inmap.DeviceID, index:0, InChannels * OutChannels * 2);
            dfloat_filter.ZerosetAsync(stream, InChannels * OutChannels * 2);

            for (uint th = 0; th < batches; th++) {
                Kernel.Execute((InChannels, OutChannels), BlockSize,
                    dynamic_shared_memory_bytes: 0, stream,
                    inmap.ElementPtr(th * InChannels), 
                    outmap.ElementPtr(th * OutChannels),
                    dfloat_filter
                );
            }

            hadd.Execute(stream, dfloat_filter, filter, InChannels * OutChannels);
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
