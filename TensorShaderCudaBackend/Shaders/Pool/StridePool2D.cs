namespace TensorShaderCudaBackend.Shaders.Pool {

    /// <summary>2次元ストライドプール</summary>
    public sealed class StridePool2D : Pool2D {

        /// <summary>コンストラクタ</summary>
        /// <param name="channels">チャネル数</param>
        /// <param name="stride">ストライド</param>
        public StridePool2D(uint channels, uint stride)
            : base(channels, stride) {

            string code = $@"

            __global__ void stridepool_2d(float *inmap, float *outmap,
                                          unsigned int inwidth, unsigned int outwidth,
                                          unsigned int outheight) {{

                unsigned int ch = {Defines.IndexX}, ox = {Defines.IndexY}, oy = {Defines.IndexZ};

                if (ch >= {Channels} || ox >= outwidth || oy >= outheight) {{
                    return;
                }}

                unsigned int ix = ox * {Stride}, iy = oy * {Stride};

                unsigned int inmap_idx = ch + {Channels} * (ix + inwidth * iy);
                unsigned int outmap_idx = ch + {Channels} * (ox + outwidth * oy);

                outmap[outmap_idx] = inmap[inmap_idx];
            }}";

            this.Kernel = new Kernel(code, "stridepool_2d");
        }
    }
}
