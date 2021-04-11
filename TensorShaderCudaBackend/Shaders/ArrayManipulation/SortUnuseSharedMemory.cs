using System;
using System.Linq;

namespace TensorShaderCudaBackend.Shaders.ArrayManipulation {

    /// <summary>ソート(共有メモリ不使用版)</summary>
    public sealed class SortUnuseSharedMemory : Shader {

        /// <summary>識別子</summary>
        public override sealed string Signature => $"{GetType().Name.Split(',').Last()}";

        /// <summary>実行あたりの要素数(2^20=1048576)</summary>
        public static uint ElementsPerExecute => 0x100000;

        /// <summary>実行あたりのスライド数(2^14=16384)</summary>
        public static uint SlidesPerExecute => 0x4000;

        /// <summary>コンストラクタ</summary>
        public SortUnuseSharedMemory() {
            string code = $@"

            __global__ void sort(float* __restrict__ map,
                                 unsigned int stride, unsigned int axislength, unsigned int slides) {{

                __shared__ int is_swaped_threads;

                unsigned int tid = {Defines.ThreadIdX}, threads = {Defines.ThreadsX};
                unsigned int m = {Defines.BlockIndexX}, n = {Defines.BlockIndexY};
                if (m >= stride || n >= slides) {{
                    return;
                }}

                unsigned int offset = m + n * stride * axislength;

                map += offset;

                for (int h = (axislength + 1) / 2; h >= 2; h = h * 4 / 5) {{
                    for (int j = tid; ; j += threads) {{
                        int i = j % h + (j / h) * 2 * h;
                        if(i + h >= axislength) break;

                        float a = map[i * stride], b = map[(i + h) * stride];

                        if (a > b) {{
                            map[i * stride] = b;
                            map[(i + h) * stride] = a;
                        }}
                    }}

                    __syncthreads();

                    for (int j = tid; ; j += threads) {{
                        int i = j % h + (j / h) * 2 * h + h;
                        if(i + h >= axislength) break;

                        float a = map[i * stride], b = map[(i + h) * stride];

                        if (a > b) {{
                            map[i * stride] = b;
                            map[(i + h) * stride] = a;
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

                        float a = map[i * stride], b = map[(i + 1) * stride];

                        if (a > b) {{
                            map[i * stride] = b;
                            map[(i + 1) * stride] = a;
                            is_swaped = 1;
                        }}
                    }}

                    __syncthreads();

                    for (int j = tid; ; j += threads) {{
                        int i = j * 2 + 1;
                        if(i + 1 >= axislength) break;

                        float a = map[i * stride], b = map[(i + 1) * stride];

                        if (a > b) {{
                            map[i * stride] = b;
                            map[(i + 1) * stride] = a;
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

            inmap.CopyToAsync(stream, outmap, stride * axislength * slides);

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
                    dynamic_shared_memory_bytes: 0,
                    stream,
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

            if (!(args[3] is uint axislength) || axislength < 1) {
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