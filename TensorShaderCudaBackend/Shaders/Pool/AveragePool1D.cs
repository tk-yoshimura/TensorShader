namespace TensorShaderCudaBackend.Shaders.Pool {

    /// <summary>1次元平均値プール</summary>
    public sealed class AveragePool1D : Pool1D {

        /// <summary>コンストラクタ</summary>
        /// <param name="channels">チャネル数</param>
        /// <param name="stride">ストライド</param>
        public AveragePool1D(uint channels, uint stride)
            : base(channels, stride) {

            string code = $@"

            __global__ void averagepool_1d(float *inmap, float *outmap,
                                           unsigned int outwidth) {{

                unsigned int ch = {Defines.IndexX}, ox = {Defines.IndexY};

                if (ch >= {Channels} || ox >= outwidth) {{
                    return;
                }}

                float vsum = 0;

                unsigned int ix = ox * {Stride};
                unsigned int inmap_idx = ch + {Channels} * ix;

                for(int kx = 0; kx < {Stride}; kx++){{
                    vsum += inmap[inmap_idx];
                    inmap_idx += {Channels};
                }}

                unsigned int outmap_idx = ch + {Channels} * ox;
                outmap[outmap_idx] = vsum / {Stride};
            }}";

            this.Kernel = new Kernel(code, "averagepool_1d");
        }
    }
}
