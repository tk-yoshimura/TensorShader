namespace TensorShaderCudaBackend.Shaders.Pool {

    /// <summary>2次元平均値逆プール</summary>
    public sealed class AverageUnpool2D : Unpool2D {

        /// <summary>コンストラクタ</summary>
        /// <param name="channels">チャネル数</param>
        /// <param name="stride">ストライド</param>
        public AverageUnpool2D(uint channels, uint stride)
            : base(channels, stride) {

            string code = $@"

            __global__ void averageunpool_2d(float *inmap, float *outmap,
                                             unsigned int inwidth, unsigned int outwidth,
                                             unsigned int inheight) {{

                unsigned int ch = {Defines.IndexX}, ix = {Defines.IndexY}, iy = {Defines.IndexZ};

                if (ch >= {Channels} || ix >= inwidth || iy >= inheight) {{
                    return;
                }}

                unsigned int ox = ix * {Stride}, oy = iy * {Stride};

                unsigned int inmap_idx = ch + {Channels} * (ix + inwidth * iy);

                float v = inmap[inmap_idx] / {Stride * Stride};

                for(int ky = 0; ky < {Stride}; ky++){{
                    unsigned int outmap_idx = ch + {Channels} * (ox + outwidth * (oy + ky));

                    for(int kx = 0; kx < {Stride}; kx++){{

                        outmap[outmap_idx] = v;
                        outmap_idx += {Channels};
                    }}
                }}
            }}";

            this.Kernel = new Kernel(code, "averageunpool_2d");
        }
    }
}
