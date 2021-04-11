using System;
using System.Linq;

namespace TensorShaderCudaBackend.Shaders.ArrayManipulation {

    /// <summary>ソート(共有メモリ使用版)</summary>
    public sealed class SortUseSharedMemory : Shader {

        /// <summary>最大軸長さ</summary>
        public static uint MaxAxisLength =>
            (uint)(API.Cuda.CurrectDeviceProperty.SharedMemoryBytesPerBlock - 256) / sizeof(float);

        /// <summary>識別子</summary>
        public override sealed string Signature => $"{GetType().Name.Split(',').Last()}";

        /// <summary>実行あたりの要素数(2^20=1048576)</summary>
        public static uint ElementsPerExecute => 0x100000;

        /// <summary>実行あたりのスライド数(2^14=16384)</summary>
        public static uint SlidesPerExecute => 0x4000;

        /// <summary>コンストラクタ</summary>
        public SortUseSharedMemory() {
            string code = $@"

            __global__ void sort(const float* __restrict__ inmap, float* __restrict__ outmap,
                                 unsigned int stride, unsigned int axislength, unsigned int slides) {{

                extern __shared__ float s[];
                __shared__ int is_swaped_threads;

                unsigned int tid = {Defines.ThreadIdX}, threads = {Defines.ThreadsX};
                unsigned int m = {Defines.BlockIndexX}, n = {Defines.BlockIndexY};
                if (m >= stride || n >= slides) {{
                    return;
                }}

                unsigned int packet = (axislength + threads - 1) / threads;

                unsigned int offset = m + n * stride * axislength;

                for(unsigned int i = 0, j = tid, inmap_idx = offset + j * stride;
                    i < packet && j < axislength; i++, j += threads, inmap_idx += threads * stride) {{

                    s[j] = inmap[inmap_idx];
                }};

                __syncthreads();

                for (int h = (axislength + 1) / 2; h >= 2; h = h * 4 / 5) {{
                    for (int j = tid; ; j += threads) {{
                        int i = j % h + (j / h) * 2 * h;
                        if(i + h >= axislength) break;

                        float a = s[i], b = s[i + h];

                        if (a > b) {{
                            s[i] = b;
                            s[i + h] = a;
                        }}
                    }}

                    __syncthreads();

                    for (int j = tid; ; j += threads) {{
                        int i = j % h + (j / h) * 2 * h + h;
                        if(i + h >= axislength) break;

                        float a = s[i], b = s[i + h];

                        if (a > b) {{
                            s[i] = b;
                            s[i + h] = a;
                        }}
                    }}

                    __syncthreads();
                }}

                if(tid == 0){{
                    is_swaped_threads = 1;
                }}
                __syncthreads();

                while(is_swaped_threads){{
                    int is_swaped = 0;

                    for (int j = tid; ; j += threads) {{
                        int i = j * 2;
                        if(i + 1 >= axislength) break;

                        float a = s[i], b = s[i + 1];

                        if (a > b) {{
                            s[i] = b;
                            s[i + 1] = a;
                            is_swaped = 1;
                        }}
                    }}

                    __syncthreads();

                    for (int j = tid; ; j += threads) {{
                        int i = j * 2 + 1;
                        if(i + 1 >= axislength) break;

                        float a = s[i], b = s[i + 1];

                        if (a > b) {{
                            s[i] = b;
                            s[i + 1] = a;
                            is_swaped = 1;
                        }}
                    }}

                    __syncthreads();

                    if(tid == 0){{
                        is_swaped_threads = 0;
                    }}

                    __syncthreads();

                    if(__any_sync(0xFFFFFFFF, is_swaped)){{
                        if(tid % {API.Cuda.CurrectDeviceProperty.WarpSize} == 0){{
                            is_swaped_threads = 1;
                        }}
                    }}

                    __syncthreads();
                }}

                for(unsigned int i = 0, j = tid, outmap_idx = offset + j * stride;
                    i < packet && j < axislength; i++, j += threads, outmap_idx += threads * stride) {{

                    outmap[outmap_idx] = s[j];
                }};
            }}";

            this.Kernel = new Kernel(code, "sort");
        }

        /// <summary>実行</summary>
        public override sealed void Execute(Stream stream, params object[] args) {
            CheckArgument(args);

            CudaArray<float> inmap = args[0] as CudaArray<float>;
            CudaArray<float> outmap = args[1] as CudaArray<float>;

            uint stride = (args[2] as uint?).Value;
            uint axislength = (args[3] as uint?).Value;
            uint slides = (args[4] as uint?).Value;

            uint blocksize = 1;
            while (blocksize < axislength / 4) {
                blocksize *= 2;
            }

            blocksize = Math.Min(blocksize, Kernel.MaxBlockSize);

            uint batch_slides = 1;
            while (batch_slides * 2 <= SlidesPerExecute) {
                if (stride * batch_slides * axislength >= ElementsPerExecute) {
                    break;
                }
                batch_slides *= 2;
            }

            for (uint s = 0; s < slides; s += batch_slides) {
                uint sl = Math.Min(batch_slides, slides - s);

                Kernel.Execute(
                    indexes: (stride * blocksize, sl),
                    block: (blocksize, 1),
                    dynamic_shared_memory_bytes: sizeof(float) * axislength,
                    stream,
                    inmap.ElementPtr(s * stride * axislength),
                    outmap.ElementPtr(s * stride * axislength),
                    stride, axislength, sl
                );
            }

        }

        /// <summary>引数チェック</summary>
        protected override sealed void CheckArgument(params object[] args) {
            if (args is null || args.Length != 5) {
                throw new ArgumentException(nameof(args));
            }

            if (!(args[2] is uint stride) || stride < 1) {
                throw new ArgumentException(nameof(stride));
            }

            if (!(args[3] is uint axislength) || axislength < 1 || axislength > MaxAxisLength) {
                throw new ArgumentException(nameof(axislength));
            }

            if (!(args[4] is uint slides) || slides < 0) {
                throw new ArgumentException(nameof(slides));
            }

            uint length = stride * axislength * slides;

            if (!(args[0] is CudaArray<float> inmap) || inmap.Length < length) {
                throw new ArgumentException(nameof(inmap));
            }

            if (!(args[1] is CudaArray<float> outmap) || outmap.Length < length) {
                throw new ArgumentException(nameof(outmap));
            }
        }
    }
}