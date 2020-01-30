namespace TensorShaderCudaBackend.Shaders.Pool {

    /// <summary>1次元平均値逆プール</summary>
    public sealed class AverageUnpool1D : Unpool1D {

        /// <summary>コンストラクタ</summary>
        /// <param name="channels">チャネル数</param>
        /// <param name="stride">ストライド</param>
        public AverageUnpool1D(uint channels, uint stride)
            : base(channels, stride) {

            string code = $@"

            __global__ void averageunpool_1d(const float* __restrict__ inmap, float* __restrict__ outmap,
                                             unsigned int inwidth) {{

                const float inv = 1.0 / {Stride};

                unsigned int ch = {Defines.IndexX}, ix = {Defines.IndexY};

                if (ch >= {Channels} || ix >= inwidth) {{
                    return;
                }}

                unsigned int ox = ix * {Stride};

                unsigned int inmap_idx = ch + {Channels} * ix;
                unsigned int outmap_idx = ch + {Channels} * ox;

                float v = inmap[inmap_idx] * inv;

                for(int kx = 0; kx < {Stride}; kx++){{
                    outmap[outmap_idx] = v;
                    outmap_idx += {Channels};
                }}
            }}";

            this.Kernel = new Kernel(code, "averageunpool_1d");
        }
    }
}
