namespace TensorShaderCudaBackend.Shaders.Pool {

    /// <summary>3次元平均値逆プール</summary>
    public sealed class AverageUnpool3D : Unpool3D {

        /// <summary>コンストラクタ</summary>
        /// <param name="channels">チャネル数</param>
        /// <param name="stride">ストライド</param>
        public AverageUnpool3D(uint channels, uint stride)
            : base(channels, stride) {

            string code = $@"

            __global__ void averageunpool3d(float *inmap, float *outmap, 
                                            unsigned int iz, 
                                            unsigned int inwidth, unsigned int outwidth, 
                                            unsigned int inheight, unsigned int outheight) {{

                unsigned int ch = {Defines.IndexX}, ix = {Defines.IndexY}, iy = {Defines.IndexZ};

                if (ch >= {Channels} || ix >= inwidth || iy >= inheight) {{
                    return;
                }}

                unsigned int ox = ix * {Stride}, oy = iy * {Stride}, oz = iz * {Stride};

                unsigned int inmap_idx = ch + {Channels} * (ix + inwidth * (iy + inheight * iz));
                
                float v = inmap[inmap_idx] / {Stride * Stride * Stride};

                for(int kz = 0; kz < {Stride}; kz++){{
                    for(int ky = 0; ky < {Stride}; ky++){{
                        unsigned int outmap_idx = ch + {Channels} * (ox + outwidth * ((oy + ky) + outheight * (oz + kz)));
                    
                        for(int kx = 0; kx < {Stride}; kx++){{
                    
                            outmap[outmap_idx] = v;
                            outmap_idx += {Channels};
                        }}
                    }}
                }}
            }}";

            this.Kernel = new Kernel(code, "averageunpool3d");
        }
    }
}
