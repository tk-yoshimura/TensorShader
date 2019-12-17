namespace TensorShaderCudaBackend.Shaders.Pool {

    /// <summary>3次元ストライドプール</summary>
    public sealed class StridePool3D : Pool3D {

        /// <summary>コンストラクタ</summary>
        /// <param name="channels">チャネル数</param>
        /// <param name="stride">ストライド</param>
        public StridePool3D(uint channels, uint stride)
            : base(channels, stride) {

            string code = $@"

            __global__ void stridepool_3d(float *inmap, float *outmap, 
                                          unsigned int oz,
                                          unsigned int inwidth, unsigned int outwidth, 
                                          unsigned int inheight, unsigned int outheight) {{

                unsigned int ch = {Defines.IndexX}, ox = {Defines.IndexY}, oy = {Defines.IndexZ};

                if (ch >= {Channels} || ox >= outwidth || oy >= outheight) {{
                    return;
                }}

                unsigned int ix = ox * {Stride}, iy = oy * {Stride}, iz = oz * {Stride};

                unsigned int inmap_idx = ch + {Channels} * (ix + inwidth * (iy + inheight * iz));
                unsigned int outmap_idx = ch + {Channels} * (ox + outwidth * (oy + outheight * oz));

                outmap[outmap_idx] = inmap[inmap_idx];
            }}";

            this.Kernel = new Kernel(code, "stridepool_3d");
        }
    }
}
