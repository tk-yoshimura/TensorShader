namespace TensorShaderCudaBackend.Shaders.Aggregation {

    /// <summary>総和</summary>
    public sealed class Sum : Aggregation {

        /// <summary>コンストラクタ</summary>
        public Sum()
            : base(shared_memory_lines: 2) {

            string code = $@"

            __global__ void aggregate_sum(float *inmap, float *outmap, 
                                          unsigned int axislength, unsigned int shared_memory_length, 
                                          unsigned int stride, unsigned int slide) {{

                unsigned int i = {Defines.ThreadIdX}, m = {Defines.BlockIndexX}, n = {Defines.BlockIndexY};

                if (i >= shared_memory_length || m >= stride || n >= slide) {{
                    return;
                }}

                extern __shared__ float s[];

                float *s_hi = s, *s_lo = s + shared_memory_length;

                unsigned int inmap_offset = m + n * stride * axislength;
                inmap += inmap_offset;

                s_hi[i] = inmap[i * stride];
                s_lo[i] = 0.0;

                for(unsigned int j = i + shared_memory_length; j < axislength; j += shared_memory_length){{
                    float x = inmap[j * stride];
                    float y = s_hi[i];
                    s_hi[i] += x;
                    s_lo[i] -= (s_hi[i] - x) - y;
                }}

                __syncthreads();

                for(unsigned int k = shared_memory_length / 2; k > 0; k /= 2) {{
                    if (i < k) {{
                        float x = s_hi[i + k];
                        float y = s_hi[i];
                        s_hi[i] += x;
                        s_lo[i] -= ((s_hi[i] - x) - y) - s_lo[i + k];
                    }}
                    __syncthreads();
                }}

                if(i == 0){{
                    unsigned int outmap_index = m + n * stride;
                    outmap[outmap_index] = s_hi[0] + s_lo[0];
                }}
            }}";

            this.Kernel = new Kernel(code, "aggregate_sum");
        }
    }
}
