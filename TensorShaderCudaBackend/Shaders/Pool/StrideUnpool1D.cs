namespace TensorShaderCudaBackend.Shaders.Pool {

    /// <summary>1次元ストライド逆プール</summary>
    public sealed class StrideUnpool1D : Unpool1D {

        /// <summary>コンストラクタ</summary>
        /// <param name="channels">チャネル数</param>
        /// <param name="stride">ストライド</param>
        public StrideUnpool1D(uint channels, uint stride)
            : base(channels, stride) {

            string code = $@"

            __global__ void strideunpool_1d(float *inmap, float *outmap, 
                                            unsigned int inwidth) {{

                unsigned int ch = {Defines.IndexX}, ix = {Defines.IndexY};

                if (ch >= {Channels} || ix >= inwidth) {{
                    return;
                }}

                unsigned int ox = ix * {Stride};

                unsigned int inmap_idx = ch + {Channels} * ix;
                unsigned int outmap_idx = ch + {Channels} * ox;
                
                outmap[outmap_idx] = inmap[inmap_idx];
            }}";

            this.Kernel = new Kernel(code, "strideunpool_1d");
        }

        /// <summary>実行</summary>
        public override void Execute(Stream stream, params object[] args) {
            CheckArgument(args);

            CudaArray<float> inmap = args[0] as CudaArray<float>;
            CudaArray<float> outmap = args[1] as CudaArray<float>;
            uint outwidth = (args[2] as uint?).Value;
            uint batches = (args[3] as uint?).Value;

            uint inwidth = outwidth / Stride;

            outmap.ZerosetAsync(stream, Channels * outwidth * batches);

            for (uint th = 0; th < batches; th++) {
                Kernel.Execute(
                    indexes: (Channels, inwidth),
                    dynamic_shared_memory_bytes: 0,
                    stream,
                    inmap.ElementPtr(th * Channels * inwidth),
                    outmap.ElementPtr(th * Channels * outwidth),
                    inwidth
                );
            }
        }
    }
}
