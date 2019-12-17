namespace TensorShaderCudaBackend.Shaders.Pool {

    /// <summary>1次元最大値プール</summary>
    public sealed class MaxPool1D : Pool1D {

        /// <summary>コンストラクタ</summary>
        /// <param name="channels">チャネル数</param>
        /// <param name="stride">ストライド</param>
        public MaxPool1D(uint channels, uint stride)
            : base(channels, stride) {

            string code = $@"
            #define NEGATIVE_INF __int_as_float(0xff800000)

            __global__ void maxpool_1d(float *inmap, float *outmap, 
                                       unsigned int outwidth) {{

                unsigned int ch = {Defines.IndexX}, ox = {Defines.IndexY};

                if (ch >= {Channels} || ox >= outwidth) {{
                    return;
                }}

                float vmax = NEGATIVE_INF;

                unsigned int ix = ox * {Stride};
                unsigned int inmap_idx = ch + {Channels} * ix;

                for(int kx = 0; kx < {Stride}; kx++){{
                    vmax = max(inmap[inmap_idx], vmax);
                    inmap_idx += {Channels};
                }}

                unsigned int outmap_idx = ch + {Channels} * ox;
                outmap[outmap_idx] = vmax;
            }}";

            this.Kernel = new Kernel(code, "maxpool_1d");
        }
    }
}
