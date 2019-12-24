using System;
using System.Linq;

namespace TensorShaderCudaBackend.Shaders.Trivector.Convolution {

    /// <summary>2次元畳み込み</summary>
    public sealed class Convolution2D : Shader {
        private readonly Transpose.TransposeQuaternionKernelChannel transpose;

        /// <summary>入力チャネル数</summary>
        public uint InChannels { private set; get; }

        /// <summary>出力チャネル数</summary>
        public uint OutChannels { private set; get; }

        /// <summary>フィルタサイズ</summary>
        public uint KernelWidth { private set; get; }

        /// <summary>フィルタサイズ</summary>
        public uint KernelHeight { private set; get; }

        /// <summary>勾配</summary>
        public bool GradMode { private set; get; }

        /// <summary>実行あたりの積数(2^25=33554432‬)</summary>
        public static uint MulPerExecute => 0x2000000;

        /// <summary>識別子</summary>
        public override sealed string Signature => 
            $"{GetType().Name.Split(',').Last()} {nameof(InChannels)} = {InChannels} {nameof(OutChannels)} = {OutChannels} " +
            $"{nameof(KernelWidth)} = {KernelWidth} {nameof(KernelHeight)} = {KernelHeight} {nameof(GradMode)} = {GradMode}";
        
        /// <summary>コンストラクタ</summary>
        public Convolution2D(uint inchannels, uint outchannels, uint kwidth, uint kheight, bool gradmode) { 
            if (!Limits.CheckChannels(inchannels, outchannels) || !Limits.CheckMultipleNum(multiple:3, inchannels, outchannels)) {
                throw new ArgumentException($"{nameof(inchannels)}, {nameof(outchannels)}");
            }
            if (!Limits.CheckKernelSize(kwidth, kheight)) { 
                throw new ArgumentException($"{nameof(kwidth)}, {nameof(kheight)}");
            }

            this.InChannels = inchannels / 3;
            this.OutChannels = outchannels / 3;
            this.KernelWidth = kwidth;
            this.KernelHeight = kheight;
            this.GradMode = gradmode;
            this.transpose = new Transpose.TransposeQuaternionKernelChannel(inchannels / 3 * 4, outchannels / 3 * 4);

            string code = $@"

            static __inline__ __device__ float3 ctor_float3(float x, float y, float z){{
                float3 t; t.x = x; t.y = y; t.z = z; return t;
            }}

            static __inline__ __device__ void floatfloat_add(float &hi, float &lo, float val){{
                float tmp = hi;
                hi += val;
                lo -= (hi - tmp) - val;
            }}

            static __inline__ __device__ void trivector_quaternion_mul(float3 &hi, float3 &lo, float3 v, float4 q){{
                float sx = q.x * q.x, sy = q.y * q.y, sz = q.z * q.z, sw = q.w * q.w; 
                float mx = q.y * q.z, my = q.z * q.w, mz = q.w * q.y; 
                float nx = q.x * q.y, ny = q.x * q.z, nz = q.x * q.w;

                floatfloat_add(hi.x, lo.x, v.x * (sx + sy - sz - sw));
                floatfloat_add(hi.x, lo.x, 2.0 * (v.y * (mx - nz)));
                floatfloat_add(hi.x, lo.x, 2.0 * (v.z * (mz + ny)));

                floatfloat_add(hi.y, lo.y, v.y * (sx - sy + sz - sw));
                floatfloat_add(hi.y, lo.y, 2.0 * (v.z * (my - nx)));
                floatfloat_add(hi.y, lo.y, 2.0 * (v.x * (mx + nz)));

                floatfloat_add(hi.z, lo.z, v.z * (sx - sy - sz + sw));
                floatfloat_add(hi.z, lo.z, 2.0 * (v.x * (mz - ny)));
                floatfloat_add(hi.z, lo.z, 2.0 * (v.y * (my + nx)));
            }}

            static __inline__ __device__ void trivector_quaternion_mulgrad(float3 &hi, float3 &lo, float3 v, float4 q){{
                float sx = q.x * q.x, sy = q.y * q.y, sz = q.z * q.z, sw = q.w * q.w; 
                float mx = q.y * q.z, my = q.z * q.w, mz = q.w * q.y; 
                float nx = q.x * q.y, ny = q.x * q.z, nz = q.x * q.w;

                floatfloat_add(hi.x, lo.x, v.x * (sx + sy - sz - sw));
                floatfloat_add(hi.x, lo.x, 2.0 * (v.y * (mx + nz)));
                floatfloat_add(hi.x, lo.x, 2.0 * (v.z * (mz - ny)));

                floatfloat_add(hi.y, lo.y, v.y * (sx - sy + sz - sw));
                floatfloat_add(hi.y, lo.y, 2.0 * (v.z * (my + nx)));
                floatfloat_add(hi.y, lo.y, 2.0 * (v.x * (mx - nz)));

                floatfloat_add(hi.z, lo.z, v.z * (sx - sy - sz + sw));
                floatfloat_add(hi.z, lo.z, 2.0 * (v.x * (mz + ny)));
                floatfloat_add(hi.z, lo.z, 2.0 * (v.y * (my - nx)));
            }}

            __global__ void trivector_convolution_2d(float3 *inmap, float3 *outmap, float4 *filter, 
                                                     unsigned int oy_offset,
                                                     unsigned int inwidth, unsigned int outwidth) {{

                unsigned int outch = {Defines.IndexX}, tid = {Defines.ThreadIdX}, threads = {Defines.ThreadsX};
                unsigned int ox = {Defines.BlockIndexY}, oy = oy_offset + {Defines.BlockIndexZ};

                __shared__ float3 vs[{InChannels}];
                float3 vq_hi = ctor_float3(0.0, 0.0, 0.0), vq_lo = ctor_float3(0.0, 0.0, 0.0);

                for(unsigned int ky = 0, iy = oy; ky < {KernelHeight}; ky++, iy++){{
                    for(unsigned int kx = 0, ix = ox; kx < {KernelWidth}; kx++, ix++){{ 

                        unsigned int inmap_idx = {InChannels} * (ix + inwidth * iy);
                        unsigned int filter_idx = outch + {InChannels * OutChannels} * (kx + {KernelWidth} * ky);

                        for(unsigned int inch = tid; inch < {InChannels}; inch += threads){{
                            vs[inch] = inmap[inch + inmap_idx];
                        }}
                        __syncthreads();

                        if(outch < {OutChannels}){{                        
                            for(unsigned int inch = 0; inch < {InChannels}; inch++){{                            
                                float3 v = vs[inch];
                                float4 q = filter[filter_idx];

                                {(GradMode ? "trivector_quaternion_mulgrad" : "trivector_quaternion_mul")}(vq_hi, vq_lo, v, q);

                                filter_idx += {OutChannels};
                            }}

                        }}
                        __syncthreads();
                    }}
                }}

                if(outch < {OutChannels}){{
                    unsigned int outmap_idx = outch + {OutChannels} * (ox + outwidth * oy);

                    outmap[outmap_idx] = ctor_float3(vq_hi.x + vq_lo.x, vq_hi.y + vq_lo.y, vq_hi.z + vq_lo.z);
                }}
            }}";

            this.Kernel = new Kernel(code, "trivector_convolution_2d");
        }

        /// <summary>実行</summary>
        public override void Execute(Stream stream, params object[] args) {
            CheckArgument(args);

            CudaArray<float> inmap = args[0] as CudaArray<float>;
            CudaArray<float> outmap = args[1] as CudaArray<float>;
            CudaArray<float> filter = args[2] as CudaArray<float>;
           
            uint inwidth = (args[3] as uint?).Value;
            uint inheight = (args[4] as uint?).Value;
            uint batches = (args[5] as uint?).Value;

            uint outwidth = inwidth + 1 - KernelWidth;
            uint outheight = inheight + 1 - KernelHeight;

            uint mul_per_line = InChannels * OutChannels * KernelWidth * KernelHeight * outwidth * 16;

            uint lines_per_execute = MulPerExecute / mul_per_line + 1;

            CudaArray<float> transpose_filter = 
                CudaArrayReserver<float>.Request(stream, filter.DeviceID, index:0, InChannels * OutChannels * KernelWidth * KernelHeight * 4);
            transpose.Execute(stream, filter, transpose_filter, KernelWidth * KernelHeight);
            
            for (uint th = 0; th < batches; th++) {
                for (uint oy_offset = 0; oy_offset < outheight; oy_offset += lines_per_execute) {
                    uint lines = Math.Min(lines_per_execute, outheight - oy_offset);

                    Kernel.Execute(
                        indexes:(OutChannels, outwidth, lines), 
                        block:(Kernel.DefaultBlockSize(OutChannels), 1, 1),
                        dynamic_shared_memory_bytes: 0, stream,
                        inmap.ElementPtr(th * InChannels * inwidth * inheight * 3), 
                        outmap.ElementPtr(th * OutChannels * outwidth * outheight * 3),
                        transpose_filter,
                        oy_offset,
                        inwidth, outwidth
                    );
                }
            }
        }

        /// <summary>引数チェック</summary>
        protected override void CheckArgument(params object[] args) {
            if (args == null || args.Length != 6) {
                throw new ArgumentException(nameof(args));
            }

            if (!(args[3] is uint inwidth) || !Limits.CheckWidth(inwidth, KernelWidth)) {
                throw new ArgumentException(nameof(inwidth));
            }

            if (!(args[4] is uint inheight) || !Limits.CheckHeight(inheight, KernelHeight)) {
                throw new ArgumentException(nameof(inheight));
            }

            if (!(args[5] is uint batches) || !Limits.CheckBatches(batches)) {
                throw new ArgumentException(nameof(batches));
            }

            uint outwidth = inwidth + 1 - KernelWidth;
            uint outheight = inheight + 1 - KernelHeight;

            if (!(args[0] is CudaArray<float> inmap) || inmap.Length < InChannels * inwidth * inheight * batches * 3) {
                throw new ArgumentException(nameof(inmap));
            }

            if (!(args[1] is CudaArray<float> outmap) || outmap.Length < OutChannels * outwidth * outheight * batches * 3) {
                throw new ArgumentException(nameof(outmap));
            }

            if (!(args[2] is CudaArray<float> filter) || filter.Length < InChannels * OutChannels * KernelWidth * KernelHeight * 4) {
                throw new ArgumentException(nameof(filter));
            }
        }
    }
}
