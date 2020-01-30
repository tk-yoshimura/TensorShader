using System;
using System.Linq;

namespace TensorShaderCudaBackend.Shaders.Batchwise {

    /// <summary>バッチごとに乗算</summary>
    public class BatchwiseMul : Shader {

        /// <summary>バッチ数</summary>
        public uint Batches { private set; get; }

        /// <summary>定数メモリを使用するか</summary>
        public bool UseConstMemory { private set; get; }

        /// <summary>識別子</summary>
        public override sealed string Signature => $"{GetType().Name.Split(',').Last()} {nameof(Batches)} = {Batches}";

        /// <summary>コンストラクタ</summary>
        /// <param name="batches">バッチ数</param>
        public BatchwiseMul(uint batches) {
            if (batches < 1) {
                throw new ArgumentException(nameof(batches));
            }

            this.Batches = batches;
            this.UseConstMemory = (batches * sizeof(float)) <= API.Cuda.CurrectDeviceProperty.ConstMemoryBytes;

            string code;

            if (UseConstMemory) {
                code = $@"

                __constant__ float v[{Batches}];

                __global__ void mul_bw(const float* __restrict__ x, float* __restrict__ y, unsigned int map_stride) {{
                    unsigned int i = {Defines.IndexX}, j = {Defines.IndexY};
                    if (i >= map_stride || j >= {Batches}) {{
                        return;
                    }}
                    y[i + map_stride * j] = v[j] * x[i + map_stride * j];
                }}";
            }
            else {
                code = $@"

                __global__ void mul_bw(const float* __restrict__ v, const float* __restrict__ x, 
                                       float* __restrict__ y, unsigned int map_stride) {{
                    unsigned int i = {Defines.IndexX}, j = {Defines.IndexY};
                    if (i >= map_stride || j >= {Batches}) {{
                        return;
                    }}
                    y[i + map_stride * j] = v[j] * x[i + map_stride * j];
                }}";
            }

            this.Kernel = new Kernel(code, "mul_bw");
        }

        /// <summary>実行</summary>
        public override sealed void Execute(Stream stream, params object[] args) {
            CheckArgument(args);

            uint map_stride = (args.Last() as uint?).Value;

            if (UseConstMemory) {
                CudaArray<float> v = args[0] as CudaArray<float>;

                if (stream != null) {
                    Kernel.StoreConstMemoryAsync(stream, "v", v, Batches);
                }
                else {
                    Kernel.StoreConstMemory("v", v, Batches);
                }

                args = args.Skip(1).ToArray();
            }

            Kernel.Execute(indexes: (map_stride, Batches), dynamic_shared_memory_bytes: 0, stream, args);
        }

        /// <summary>引数チェック</summary>
        protected override sealed void CheckArgument(params object[] args) {
            if (args == null || args.Length != 4) {
                throw new ArgumentException(nameof(args));
            }

            if (!(args[3] is uint map_stride)) {
                throw new ArgumentException(nameof(map_stride));
            }

            if (!(args[0] is CudaArray<float> vectorarr) || vectorarr.Length < (ulong)Batches) {
                throw new ArgumentException(nameof(vectorarr));
            }

            if (!(args[1] is CudaArray<float> inmaparr) || inmaparr.Length < (ulong)map_stride * Batches) {
                throw new ArgumentException(nameof(inmaparr));
            }

            if (!(args[2] is CudaArray<float> outmaparr) || outmaparr.Length < (ulong)map_stride * Batches) {
                throw new ArgumentException(nameof(outmaparr));
            }
        }
    }
}
