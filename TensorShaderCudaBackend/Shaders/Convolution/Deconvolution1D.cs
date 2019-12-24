using System;
using System.Linq;

namespace TensorShaderCudaBackend.Shaders.Convolution {

    /// <summary>1次元逆畳み込み</summary>
    public sealed class Deconvolution1D : Shader {

        /// <summary>入力チャネル数</summary>
        public uint InChannels { private set; get; }

        /// <summary>出力チャネル数</summary>
        public uint OutChannels { private set; get; }

        /// <summary>フィルタサイズ</summary>
        public uint KernelWidth { private set; get; }
                
        /// <summary>識別子</summary>
        public override sealed string Signature => 
            $"{GetType().Name.Split(',').Last()} {nameof(InChannels)} = {InChannels} {nameof(OutChannels)} = {OutChannels} " +
            $"{nameof(KernelWidth)} = {KernelWidth}";
        
        /// <summary>コンストラクタ</summary>
        public Deconvolution1D(uint inchannels, uint outchannels, uint kwidth) { 
            if (!Limits.CheckChannels(inchannels, outchannels)) {
                throw new ArgumentException($"{nameof(inchannels)}, {nameof(outchannels)}");
            }
            if (!Limits.CheckKernelSize(kwidth)) { 
                throw new ArgumentException(nameof(kwidth));
            }

            this.InChannels = inchannels;
            this.OutChannels = outchannels;
            this.KernelWidth = kwidth;

            string code = $@"

            static __inline__ __device__ void floatfloat_add(float &hi, float &lo, float val){{
                float tmp = hi;
                hi += val;
                lo -= (hi - tmp) - val;
            }}

            __global__ void deconvolution_1d(float *inmap, float *outmap, float *filter,
                                             unsigned int inwidth) {{

                unsigned int outch = {Defines.IndexX}, tid = {Defines.ThreadIdX}, threads = {Defines.ThreadsX};
                unsigned int ox = {Defines.BlockIndexY};

                __shared__ float us[{InChannels}];
                float uv_hi = 0.0, uv_lo = 0.0;

                for(unsigned int kx = 0, ix = ox - {KernelWidth - 1}; kx < {KernelWidth}; kx++, ix++){{ 
                    if(ix >= inwidth){{
                        continue;
                    }}

                    unsigned int inmap_idx = {InChannels} * ix;
                    unsigned int filter_idx = outch + {InChannels * OutChannels} * ({KernelWidth - 1} - kx);

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

                    }}
                    __syncthreads();
                }}

                if(outch < {OutChannels}){{
                    unsigned int outmap_idx = outch + {OutChannels} * ox;

                    outmap[outmap_idx] = uv_hi + uv_lo;
                }}
            }}";

            this.Kernel = new Kernel(code, "deconvolution_1d");
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
                    indexes:(OutChannels, outwidth), 
                    block:(Kernel.DefaultBlockSize(OutChannels), 1),
                    dynamic_shared_memory_bytes: 0, 
                    stream,
                    inmap.ElementPtr(th * InChannels * inwidth), 
                    outmap.ElementPtr(th * OutChannels * outwidth),
                    filter,
                    inwidth
                );
            }
        }

        /// <summary>引数チェック</summary>
        protected override void CheckArgument(params object[] args) {
            if (args == null || args.Length != 5) {
                throw new ArgumentException(nameof(args));
            }

            if (!(args[3] is uint inwidth) || !Limits.CheckWidth(inwidth)) {
                throw new ArgumentException(nameof(inwidth));
            }

            if (!(args[4] is uint batches) || !Limits.CheckBatches(batches)) {
                throw new ArgumentException(nameof(batches));
            }

            uint outwidth = inwidth + KernelWidth - 1;

            if (!(args[0] is CudaArray<float> inmap) || inmap.Length < InChannels * inwidth * batches) {
                throw new ArgumentException(nameof(inmap));
            }

            if (!(args[1] is CudaArray<float> outmap) || outmap.Length < OutChannels * outwidth * batches) {
                throw new ArgumentException(nameof(outmap));
            }

            if (!(args[2] is CudaArray<float> filter) || filter.Length < InChannels * OutChannels * KernelWidth) {
                throw new ArgumentException(nameof(filter));
            }
        }
    }
}
