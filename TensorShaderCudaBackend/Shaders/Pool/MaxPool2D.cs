namespace TensorShaderCudaBackend.Shaders.Pool {

    /// <summary>2次元最大値プール</summary>
    public sealed class MaxPool2D : Pool2D {

        /// <summary>コンストラクタ</summary>
        /// <param name="channels">チャネル数</param>
        /// <param name="stride">ストライド</param>
        public MaxPool2D(uint channels, uint stride)
            : base(channels, stride) {

            string code = $@"
            #define NEGATIVE_INF __int_as_float(0xff800000)

            __global__ void maxpool_2d(float *inmap, float *outmap, 
                                       unsigned int inwidth, unsigned int outwidth, 
                                       unsigned int outheight) {{

                unsigned int ch = {Defines.IndexX}, ox = {Defines.IndexY}, oy = {Defines.IndexZ};

                if (ch >= {Channels} || ox >= outwidth || oy >= outheight) {{
                    return;
                }}

                float vmax = NEGATIVE_INF;

                unsigned int ix = ox * {Stride}, iy = oy * {Stride};

                for(int ky = 0; ky < {Stride}; ky++){{
                    unsigned int inmap_idx = ch + {Channels} * (ix + inwidth * (iy + ky));
                    
                    for(int kx = 0; kx < {Stride}; kx++){{
                        vmax = max(inmap[inmap_idx], vmax);
                        inmap_idx += {Channels};
                    }}
                }}

                unsigned int outmap_idx = ch + {Channels} * (ox + outwidth * oy);
                outmap[outmap_idx] = vmax;
            }}";

            this.Kernel = new Kernel(code, "maxpool_2d");
        }
    }
}
