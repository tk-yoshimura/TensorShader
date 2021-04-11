using System;
using System.Collections.Generic;

namespace TensorShaderCudaBackend {

    /// <summary>乱数生成</summary>
    public static class Randomize {
        private readonly static Dictionary<string, Shader> shaders = new();

        /// <summary>一様乱数</summary>
        /// <remarks>値域 : [0, 1)</remarks>
        public static void Uniform(uint length, CudaArray<float> dst, Random random, Stream stream = null) {
            string key = "uniform_random";

            if (!shaders.ContainsKey(key)) {
                shaders.Add(key, new Shaders.Randomize.Uniform());
            }

            Shader shader = shaders[key];

            if (stream is null) {
                stream = Shader.DefaultStream;
            }

            shader.Execute(stream, dst, length, random);
        }

        /// <summary>正規乱数(Box-Muller Method)</summary>
        public static void Normal(uint length, CudaArray<float> dst, Random random, Stream stream = null) {
            string key = "normal_random";

            if (!shaders.ContainsKey(key)) {
                shaders.Add(key, new Shaders.Randomize.Normal());
            }

            Shader shader = shaders[key];

            if (stream is null) {
                stream = Shader.DefaultStream;
            }

            shader.Execute(stream, dst, length, random);
        }

        /// <summary>ベルヌーイ分布に従う2値</summary>
        public static void Bernoulli(uint length, double prob, CudaArray<float> dst, Random random, Stream stream = null) {
            string key = "binary_random";

            if (!shaders.ContainsKey(key)) {
                shaders.Add(key, new Shaders.Randomize.Binary());
            }

            Shader shader = shaders[key];

            if (stream is null) {
                stream = Shader.DefaultStream;
            }

            shader.Execute(stream, dst, length, random, (float)prob);
        }
    }
}
