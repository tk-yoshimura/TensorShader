using System;

namespace TensorShaderCudaBackend.Shaders.Randomize {

    /// <summary>ベルヌーイ分布に従う2値</summary>
    public sealed class Binary : Randomize {

        /// <summary>コンストラクタ</summary>
        public Binary() :
            base(exparams: 1) {
            string code = $@"
            __global__ void binary_random(float *y, unsigned int length, unsigned int warps,
                                           unsigned int seed1, unsigned int seed2, unsigned int seed3, float thr) {{
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

                    y[idx] = (sw * 2.328306436538696e-10) < thr ? 1.0 : 0.0;
                }}
            }}";

            this.Kernel = new Kernel(code, "binary_random");
        }

        /// <summary>実行</summary>
        public override void Execute(Stream stream, params object[] args) {
            if (args.Length != 4) {
                throw new ArgumentException(nameof(args));
            }

            if (!(args[3] is float prob) || prob < 0 || !(prob <= 1)) {
                throw new ArgumentException(nameof(prob));
            }

            base.Execute(stream, args);
        }
    }
}
