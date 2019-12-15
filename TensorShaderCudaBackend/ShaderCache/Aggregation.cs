using System.Collections.Generic;

namespace TensorShaderCudaBackend {

    /// <summary>集計</summary>
    public static class Aggregation {
        private readonly static Dictionary<string, Shader> shaders = new Dictionary<string, Shader>();
        
        /// <summary>総和</summary>
        public static void Sum(uint src_stride, CudaArray<float> src, uint dst_stride, CudaArray<float> dst, uint slides) {
            string key = "sum";
            
            if (!shaders.ContainsKey(key)) {
                shaders.Add(key, new Shaders.Aggregation.Sum());
            }

            Shader shader = shaders[key];

            shader.Execute(Shader.DefaultStream, src, dst, src_stride, dst_stride, slides);
        }

        /// <summary>平均</summary>
        public static void Average(uint src_stride, CudaArray<float> src, uint dst_stride, CudaArray<float> dst, uint slides) {
            string key = "average";
            
            if (!shaders.ContainsKey(key)) {
                shaders.Add(key, new Shaders.Aggregation.Average());
            }

            Shader shader = shaders[key];

            shader.Execute(Shader.DefaultStream, src, dst, src_stride, dst_stride, slides);
        }

        /// <summary>最小値</summary>
        public static void Min(uint src_stride, CudaArray<float> src, uint dst_stride, CudaArray<float> dst, uint slides) {
            string key = "min";
            
            if (!shaders.ContainsKey(key)) {
                shaders.Add(key, new Shaders.Aggregation.Min());
            }

            Shader shader = shaders[key];

            shader.Execute(Shader.DefaultStream, src, dst, src_stride, dst_stride, slides);
        }

        /// <summary>最大値</summary>
        public static void Max(uint src_stride, CudaArray<float> src, uint dst_stride, CudaArray<float> dst, uint slides) {
            string key = "max";
            
            if (!shaders.ContainsKey(key)) {
                shaders.Add(key, new Shaders.Aggregation.Max());
            }

            Shader shader = shaders[key];

            shader.Execute(Shader.DefaultStream, src, dst, src_stride, dst_stride, slides);
        }
    } 
}