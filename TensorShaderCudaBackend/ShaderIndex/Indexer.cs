using System.Collections.Generic;

namespace TensorShaderCudaBackend {

    /// <summary>インデクサ</summary>
    public static class Indexer {
        private readonly static Dictionary<string, Shader> shaders = new Dictionary<string, Shader>();

        /// <summary>OneHotVector</summary>
        public static void OneHotVector(uint length, uint channels, CudaArray<float> src, CudaArray<float> dst, Stream stream = null) {
            string key = $"onehotvector {nameof(channels)}={channels}";

            if (!shaders.ContainsKey(key)) {
                shaders.Add(key, new Shaders.Indexer.OneHotVector(channels));
            }

            Shader shader = shaders[key];

            if (stream == null) {
                stream = Shader.DefaultStream;
            }

            shader.Execute(stream, src, dst, length * channels, length);
        }

        /// <summary>ArgMin</summary>
        public static void ArgMin(uint length, uint channels, CudaArray<float> src, CudaArray<float> dst, Stream stream = null) {
            string key = $"argmin {nameof(channels)}={channels}";

            if (!shaders.ContainsKey(key)) {
                shaders.Add(key, new Shaders.Indexer.ArgMin(channels));
            }

            Shader shader = shaders[key];

            if (stream == null) {
                stream = Shader.DefaultStream;
            }

            shader.Execute(stream, src, dst, length * channels, length);
        }

        /// <summary>ArgMax</summary>
        public static void ArgMax(uint length, uint channels, CudaArray<float> src, CudaArray<float> dst, Stream stream = null) {
            string key = $"argmax {nameof(channels)}={channels}";

            if (!shaders.ContainsKey(key)) {
                shaders.Add(key, new Shaders.Indexer.ArgMax(channels));
            }

            Shader shader = shaders[key];

            if (stream == null) {
                stream = Shader.DefaultStream;
            }

            shader.Execute(stream, src, dst, length * channels, length);
        }

        /// <summary>Index</summary>
        public static void Index(uint stride, uint axislength, uint clones, CudaArray<float> dst, Stream stream = null) {
            string key = "index";

            if (!shaders.ContainsKey(key)) {
                shaders.Add(key, new Shaders.Indexer.Index());
            }

            Shader shader = shaders[key];

            if (stream == null) {
                stream = Shader.DefaultStream;
            }

            shader.Execute(stream, dst, stride * axislength * clones, stride, axislength);
        }
    }
}
