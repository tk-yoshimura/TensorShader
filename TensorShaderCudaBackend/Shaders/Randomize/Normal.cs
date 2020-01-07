namespace TensorShaderCudaBackend.Shaders.Randomize {

    /// <summary>正規乱数(Box-Muller Method)</summary>
    public sealed class Normal : Randomize {

        /// <summary>コンストラクタ</summary>
        public Normal() {
            string code = $@"
            __global__ void normal_random(float *y, unsigned int length, unsigned int warps,
                                          unsigned int seed1, unsigned int seed2, unsigned int seed3) {{
                unsigned int j = {Defines.IndexX}, k = {Defines.IndexY};
                if (k >= warps) {{
                    return;
                }}

                unsigned int sx, sy, sz, sw;

                sx = seed1 + j * 0x1010u - k * 0x0110u;
                sy = seed2 + j * 0x1001u - k * 0x0011u;
                sz = seed3 + j * 0x1100u - k * 0x0101u;

                for(unsigned int i = 0; i < {Dumps}; i++){{
                    sw = (sx ^ (sx << 3)) ^ (sy ^ (sy >> 19)) ^ (sz ^ (sz << 6));
                    sx = sy; sy = sz; sz = sw;
                }}
                
                for(unsigned int i = 0, idx = j + {RandomPerWarp} * k; i < {RandomPerThread} && idx < length; i++, idx += {WarpSize}){{
                    sw = (sx ^ (sx << 3)) ^ (sy ^ (sy >> 19)) ^ (sz ^ (sz << 6));
                    sx = sy; sy = sz; sz = sw;

                    float vy = (sy + 1.0) * 2.328306436538696e-10, vz = sz * 2.328306436538696e-10 * 6.283185307179586;

                    y[idx] = sqrtf(-2 * logf(vy)) * cosf(vz);
                }}
            }}";

            this.Kernel = new Kernel(code, "normal_random");
        }
    }
}
