using System.Collections.Generic;

namespace TensorShaderCudaBackend {

    /// <summary>バッチ独立演算</summary>
    public static class Batchwise {
        private readonly static Dictionary<string, Shader> shaders = new Dictionary<string, Shader>();

        /// <summary>乗算</summary>
        public static void Mul(uint vector_length, uint map_length,
                               CudaArray<float> srcvector, CudaArray<float> srcmap, CudaArray<float> dstmap,
                               Stream stream = null) {

            string key = $"mul_bw " +
                         $"batches={vector_length}";

            if (!shaders.ContainsKey(key)) {
                shaders.Add(key, new Shaders.Batchwise.BatchwiseMul(vector_length));
            }

            Shader shader = shaders[key];

            if (stream == null) {
                stream = Shader.DefaultStream;
            }

            shader.Execute(stream, srcvector, srcmap, dstmap, map_length / vector_length);
        }
    }
}
