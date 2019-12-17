namespace TensorShaderCudaBackend.Shaders.Pool {

    /// <summary>3次元平均値プール</summary>
    public sealed class AveragePool3D : Pool3D {

        /// <summary>コンストラクタ</summary>
        /// <param name="channels">チャネル数</param>
        /// <param name="stride">ストライド</param>
        public AveragePool3D(uint channels, uint stride)
            : base(channels, stride) {

            string code = $@"

            __global__ void averagepool_3d(float *inmap, float *outmap, 
                                           unsigned int oz,
                                           unsigned int inwidth, unsigned int outwidth, 
                                           unsigned int inheight, unsigned int outheight) {{

                unsigned int ch = {Defines.IndexX}, ox = {Defines.IndexY}, oy = {Defines.IndexZ};

                if (ch >= {Channels} || ox >= outwidth || oy >= outheight) {{
                    return;
                }}

                float vsum = 0;

                unsigned int ix = ox * {Stride}, iy = oy * {Stride}, iz = oz * {Stride};

                for(int kz = 0; kz < {Stride}; kz++){{
                    for(int ky = 0; ky < {Stride}; ky++){{
                        unsigned int inmap_idx = ch + {Channels} * (ix + inwidth * ((iy + ky) + inheight * (iz + kz)));
                    
                        for(int kx = 0; kx < {Stride}; kx++){{
                            vsum += inmap[inmap_idx];
                            inmap_idx += {Channels};
                        }}
                    }}
                }}

                unsigned int outmap_idx = ch + {Channels} * (ox + outwidth * (oy + outheight * oz));
                outmap[outmap_idx] = vsum / {Stride * Stride * Stride};
            }}";

            this.Kernel = new Kernel(code, "averagepool_3d");
        }
    }
}
