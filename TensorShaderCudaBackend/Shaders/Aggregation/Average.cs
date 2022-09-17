namespace TensorShaderCudaBackend.Shaders.Aggregation {

    /// <summary>平均</summary>
    public sealed class Average : Aggregation {

        /// <summary>コンストラクタ</summary>
        public Average()
            : base(shared_memory_lines: 1) {

            string code = $@"

            __global__ void aggregate_average(const float* __restrict__ inmap, float* __restrict__ outmap,
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
                    s[i] += inmap[j * stride];
                }}

                __syncthreads();

                for(unsigned int k = shared_memory_length / 2; k > 0; k /= 2) {{
                    if (i < k) {{
                        s[i] += s[i + k];
                    }}
                    __syncthreads();
                }}

                if(i == 0){{
                    unsigned int outmap_index = m + n * stride;
                    outmap[outmap_index] = s[0] / axislength;
                }}
            }}";

            this.Kernel = new Kernel(code, "aggregate_average");
        }
    }
}
