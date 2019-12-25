using System;
using System.Linq;

namespace TensorShaderCudaBackend.Shaders.Quaternion.Convolution {

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
                
        /// <summary>識別子</summary>
        public override sealed string Signature => 
            $"{GetType().Name.Split(',').Last()} {nameof(InChannels)} = {InChannels} {nameof(OutChannels)} = {OutChannels} " +
            $"{nameof(KernelWidth)} = {KernelWidth} {nameof(GradMode)} = {GradMode}";
        
        /// <summary>コンストラクタ</summary>
        public Deconvolution1D(uint inchannels, uint outchannels, uint kwidth, bool gradmode) { 
            if (!Limits.CheckChannels(inchannels, outchannels) || !Limits.CheckMultipleNum(multiple:4, inchannels, outchannels)) {
                throw new ArgumentException($"{nameof(inchannels)}, {nameof(outchannels)}");
            }
            if (!Limits.CheckKernelSize(kwidth)) { 
                throw new ArgumentException(nameof(kwidth));
            }

            this.InChannels = inchannels / 4;
            this.OutChannels = outchannels / 4;
            this.KernelWidth = kwidth;
            this.GradMode = gradmode;

            string code = $@"

            static __inline__ __device__ float4 ctor_float4(float x, float y, float z, float w){{
                float4 t; t.x = x; t.y = y; t.z = z; t.w = w; return t;
            }}

            static __inline__ __device__ void floatfloat_add(float &hi, float &lo, float val){{
                float tmp = hi;
                hi += val;
                lo -= (hi - tmp) - val;
            }}

            static __inline__ __device__ void floatfloat_sub(float &hi, float &lo, float val){{
                float tmp = hi;
                hi -= val;
                lo -= (hi - tmp) + val;
            }}

            static __inline__ __device__ void quaternion_mul(float4 &hi, float4 &lo, float4 x1, float4 x2){{
                floatfloat_add(hi.x, lo.x, x1.x * x2.x);
                floatfloat_sub(hi.x, lo.x, x1.y * x2.y);
                floatfloat_sub(hi.x, lo.x, x1.z * x2.z);
                floatfloat_sub(hi.x, lo.x, x1.w * x2.w);

                floatfloat_add(hi.y, lo.y, x1.x * x2.y);
                floatfloat_add(hi.y, lo.y, x1.y * x2.x);
                floatfloat_sub(hi.y, lo.y, x1.z * x2.w);
                floatfloat_add(hi.y, lo.y, x1.w * x2.z);

                floatfloat_add(hi.z, lo.z, x1.x * x2.z);
                floatfloat_add(hi.z, lo.z, x1.y * x2.w);
                floatfloat_add(hi.z, lo.z, x1.z * x2.x);
                floatfloat_sub(hi.z, lo.z, x1.w * x2.y);

                floatfloat_add(hi.w, lo.w, x1.x * x2.w);
                floatfloat_sub(hi.w, lo.w, x1.y * x2.z);
                floatfloat_add(hi.w, lo.w, x1.z * x2.y);
                floatfloat_add(hi.w, lo.w, x1.w * x2.x);
            }}

            static __inline__ __device__ void quaternion_mulgrad(float4 &hi, float4 &lo, float4 x1, float4 x2){{
                floatfloat_add(hi.x, lo.x, x1.x * x2.x);
                floatfloat_add(hi.x, lo.x, x1.y * x2.y);
                floatfloat_add(hi.x, lo.x, x1.z * x2.z);
                floatfloat_add(hi.x, lo.x, x1.w * x2.w);

                floatfloat_add(hi.y, lo.y, x1.x * x2.y);
                floatfloat_sub(hi.y, lo.y, x1.y * x2.x);
                floatfloat_add(hi.y, lo.y, x1.z * x2.w);
                floatfloat_sub(hi.y, lo.y, x1.w * x2.z);

                floatfloat_add(hi.z, lo.z, x1.x * x2.z);
                floatfloat_sub(hi.z, lo.z, x1.y * x2.w);
                floatfloat_sub(hi.z, lo.z, x1.z * x2.x);
                floatfloat_add(hi.z, lo.z, x1.w * x2.y);

                floatfloat_add(hi.w, lo.w, x1.x * x2.w);
                floatfloat_add(hi.w, lo.w, x1.y * x2.z);
                floatfloat_sub(hi.w, lo.w, x1.z * x2.y);
                floatfloat_sub(hi.w, lo.w, x1.w * x2.x);
            }}

            __global__ void quaternion_deconvolution_1d(float4 *inmap, float4 *outmap, float4 *filter,
                                                        unsigned int inwidth) {{

                unsigned int outch = {Defines.IndexX}, tid = {Defines.ThreadIdX}, threads = {Defines.ThreadsX};
                unsigned int ox = {Defines.BlockIndexY};

                __shared__ float4 us[{InChannels}];
                float4 vu_hi = ctor_float4(0.0, 0.0, 0.0, 0.0), vu_lo = ctor_float4(0.0, 0.0, 0.0, 0.0);

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
                            float4 u = us[inch];
                            float4 v = filter[filter_idx];

                            {(GradMode ? "quaternion_mulgrad" : "quaternion_mul")}(vu_hi, vu_lo, v, u);

                            filter_idx += {OutChannels};
                        }}

                    }}
                    __syncthreads();
                }}

                if(outch < {OutChannels}){{
                    unsigned int outmap_idx = outch + {OutChannels} * ox;

                    outmap[outmap_idx] = ctor_float4(vu_hi.x + vu_lo.x, vu_hi.y + vu_lo.y, vu_hi.z + vu_lo.z, vu_hi.w + vu_lo.w);
                }}
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
                    indexes:(OutChannels, outwidth), 
                    block:(Kernel.DefaultBlockSize(OutChannels), 1),
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
