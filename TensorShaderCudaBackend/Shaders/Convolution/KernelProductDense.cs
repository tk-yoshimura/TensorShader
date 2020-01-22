using System;
using System.Linq;

using static TensorShaderCudaBackend.ArrayManipulation;

namespace TensorShaderCudaBackend.Shaders.Convolution {

    /// <summary>カーネル積</summary>
    public sealed class KernelProductDense : Shader {

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
            if (!Limits.CheckChannels(inchannels, outchannels)) {
                throw new ArgumentException($"{nameof(inchannels)}, {nameof(outchannels)}");
            }

            this.InChannels = inchannels;
            this.OutChannels = outchannels;

            this.BlockSize = Kernel.MinimizeGridsBlockSize((InChannels, OutChannels));

            string code = $@"

            static __inline__ __device__ void floatfloat_atomicadd(float *ptr, float val){{
                float tmp = atomicAdd(ptr, val);
                atomicAdd(ptr + 1, -(((tmp + val) - tmp) - val));
            }}

            __global__ void kernelproduct_dense(float *inmap, float *outmap, float *filter) {{

                unsigned int inch = {Defines.IndexX}, outch = {Defines.IndexY}, th = {Defines.BlockIndexZ};
                unsigned int tidx = {Defines.ThreadIdX}, tidy = {Defines.ThreadIdY};

                unsigned int inmap_offset = {InChannels} * th;
                inmap += inmap_offset;

                unsigned int outmap_offset = {OutChannels} * th;
                outmap += outmap_offset;

                unsigned int filter_offset = (inch + {InChannels} * outch) * 2;
                filter += filter_offset;

                __shared__ float us[{BlockSize.x}], vs[{BlockSize.y}];

                { (InChannels % BlockSize.x != 0 ? $"if(tidy == 0 && inch < {InChannels}){{" : "if(tidy == 0){") }
                    us[tidx] = inmap[inch];
                }}
                { (OutChannels % BlockSize.y != 0 ? $"if(tidx == 0 && outch < {OutChannels}){{" : "if(tidx == 0){") }
                    vs[tidy] = outmap[outch];
                }}
                __syncthreads();

                { (InChannels % BlockSize.x != 0 ? $"if(inch < {InChannels}){{" : "") }
                { (OutChannels % BlockSize.y != 0 ? $"if(outch < {OutChannels}){{" : "") }

                    float u = us[tidx];
                    float v = vs[tidy];

                    floatfloat_atomicadd(filter, u * v);

                { (InChannels % BlockSize.x != 0 ? "}" : "") }
                { (OutChannels % BlockSize.y != 0 ? "}" : "") }
            }}";

            this.Kernel = new Kernel(code, "kernelproduct_dense");
        }

        /// <summary>実行</summary>
        public override void Execute(Stream stream, params object[] args) {
            CheckArgument(args);

            CudaArray<float> inmap = args[0] as CudaArray<float>;
            CudaArray<float> outmap = args[1] as CudaArray<float>;
            CudaArray<float> filter = args[2] as CudaArray<float>;

            uint batches = (args[3] as uint?).Value;

            CudaArray<float> dfloat_filter =
                CudaArrayReserver<float>.Request(stream, inmap.DeviceID, index: 0, InChannels * OutChannels * 2);
            dfloat_filter.ZerosetAsync(stream, InChannels * OutChannels * 2);

            Kernel.Execute(
                indexes: (InChannels, OutChannels, batches),
                block: (BlockSize.x, BlockSize.y, 1),
                dynamic_shared_memory_bytes: 0, stream,
                inmap,
                outmap,
                dfloat_filter
            );

            HorizontalAdd(InChannels * OutChannels, dfloat_filter, filter, stream);
        }

        /// <summary>引数チェック</summary>
        protected override void CheckArgument(params object[] args) {
            if (args == null || args.Length != 4) {
                throw new ArgumentException(nameof(args));
            }

            if (!(args[3] is uint batches) || !Limits.CheckBatches(batches)) {
                throw new ArgumentException(nameof(batches));
            }

            if (!(args[0] is CudaArray<float> inmap) || inmap.Length < InChannels * batches) {
                throw new ArgumentException(nameof(inmap));
            }

            if (!(args[1] is CudaArray<float> outmap) || outmap.Length < OutChannels * batches) {
                throw new ArgumentException(nameof(outmap));
            }

            if (!(args[2] is CudaArray<float> filter) || filter.Length < InChannels * OutChannels) {
                throw new ArgumentException(nameof(filter));
            }
        }
    }
}
