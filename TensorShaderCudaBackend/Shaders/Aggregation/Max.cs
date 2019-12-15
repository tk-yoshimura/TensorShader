﻿namespace TensorShaderCudaBackend.Shaders.Aggregation {

    /// <summary>最大値</summary>
    public sealed class Max : Aggregation {

        /// <summary>コンストラクタ</summary>
        public Max()
            : base(shared_memory_lines: 1) {

            string code = $@"

            __global__ void aggregate_max(float *inmap, float *outmap, 
                                          unsigned int axislength, unsigned int shared_memory_length, 
                                          unsigned int stride, unsigned int slide) {{

                unsigned int i = {Defines.ThreadIdX}, m = {Defines.BlockIndexX}, n = {Defines.BlockIndexY};

                if (i >= shared_memory_length || m >= stride || n >= slide) {{
                    return;
                }}

                extern __shared__ float s[];

                unsigned int inmap_offset = m + n * stride * axislength;
                inmap += inmap_offset;

                s[i] = inmap[i * stride];
                
                for(unsigned int j = i + shared_memory_length; j < axislength; j += shared_memory_length){{
                    s[i] = fmaxf(s[i], inmap[j * stride]);
                }}

                __syncthreads();

                for(unsigned int k = shared_memory_length / 2; k > 0; k /= 2) {{
                    if (i < k) {{
                        s[i] = fmaxf(s[i], s[i + k]);
                    }}
                    __syncthreads();
                }}

                if(i == 0){{
                    unsigned int outmap_index = m + n * stride;
                    outmap[outmap_index] = s[0];
                }}
            }}";

            this.Kernel = new Kernel(code, "aggregate_max");
        }
    }
}
