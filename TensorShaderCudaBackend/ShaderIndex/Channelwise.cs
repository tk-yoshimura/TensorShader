using System.Collections.Generic;

namespace TensorShaderCudaBackend {

    /// <summary>チャネル独立演算</summary>
    public static class Channelwise {
        private readonly static Dictionary<string, Shader> shaders = new Dictionary<string, Shader>();

        private static Shader BinaryArithmetric(string name, string func, uint channels) {
            string key = $"{name} {nameof(channels)}={channels}";

            if (!shaders.ContainsKey(key)) {
                shaders.Add(key, new Shaders.Channelwise.ChannelwiseBinaryArithmetric(name, func, channels));
            }

            return shaders[key];
        }

        /// <summary>加算</summary>
        public static void Add(uint vector_length, uint map_length,
                               CudaArray<float> srcvector, CudaArray<float> srcmap, CudaArray<float> dstmap,
                               Stream stream = null) {

            Shader shader = BinaryArithmetric("add_cw", "#y = #v + #x;", vector_length);

            if (stream == null) {
                stream = Shader.DefaultStream;
            }

            shader.Execute(stream, srcvector, srcmap, dstmap, map_length);
        }

        /// <summary>乗算</summary>
        public static void Mul(uint vector_length, uint map_length,
                               CudaArray<float> srcvector, CudaArray<float> srcmap, CudaArray<float> dstmap,
                               Stream stream = null) {

            Shader shader = BinaryArithmetric("mul_cw", "#y = #v * #x;", vector_length);

            if (stream == null) {
                stream = Shader.DefaultStream;
            }

            shader.Execute(stream, srcvector, srcmap, dstmap, map_length);
        }

        /// <summary>減算(左ベクトル)</summary>
        public static void SubLVector(uint vector_length, uint map_length,
                                      CudaArray<float> srcvector, CudaArray<float> srcmap, CudaArray<float> dstmap,
                                      Stream stream = null) {

            Shader shader = BinaryArithmetric("subl_cw", "#y = #v - #x;", vector_length);

            if (stream == null) {
                stream = Shader.DefaultStream;
            }

            shader.Execute(stream, srcvector, srcmap, dstmap, map_length);
        }

        /// <summary>減算(右ベクトル)</summary>
        public static void SubRVector(uint vector_length, uint map_length,
                                      CudaArray<float> srcvector, CudaArray<float> srcmap, CudaArray<float> dstmap,
                                      Stream stream = null) {

            Shader shader = BinaryArithmetric("subr_cw", "#y = #x - #v;", vector_length);

            if (stream == null) {
                stream = Shader.DefaultStream;
            }

            shader.Execute(stream, srcvector, srcmap, dstmap, map_length);
        }

        /// <summary>除算(左ベクトル)</summary>
        public static void DivLVector(uint vector_length, uint map_length,
                                      CudaArray<float> srcvector, CudaArray<float> srcmap, CudaArray<float> dstmap,
                                      Stream stream = null) {

            Shader shader = BinaryArithmetric("divl_cw", "#y = #v / #x;", vector_length);

            if (stream == null) {
                stream = Shader.DefaultStream;
            }

            shader.Execute(stream, srcvector, srcmap, dstmap, map_length);
        }

        /// <summary>除算(右ベクトル)</summary>
        public static void DivRVector(uint vector_length, uint map_length,
                                      CudaArray<float> srcvector, CudaArray<float> srcmap, CudaArray<float> dstmap,
                                      Stream stream = null) {

            Shader shader = BinaryArithmetric("divr_cw", "#y = #x / #v;", vector_length);

            if (stream == null) {
                stream = Shader.DefaultStream;
            }

            shader.Execute(stream, srcvector, srcmap, dstmap, map_length);
        }

        /// <summary>Softmax</summary>
        public static void Softmax(uint length, uint channels, CudaArray<float> src, CudaArray<float> dst, Stream stream = null) {
            string key = $"softmax " +
                         $"{nameof(channels)}={channels}";

            if (!shaders.ContainsKey(key)) {
                shaders.Add(key, new Shaders.Channelwise.Softmax(channels));
            }

            Shader shader = shaders[key];

            if (stream == null) {
                stream = Shader.DefaultStream;
            }

            shader.Execute(stream, src, dst, length, length / channels);
        }

        /// <summary>Hardmax</summary>
        public static void Hardmax(uint length, uint channels, CudaArray<float> src, CudaArray<float> dst, Stream stream = null) {
            string key = $"hardmax " +
                         $"{nameof(channels)}={channels}";

            if (!shaders.ContainsKey(key)) {
                shaders.Add(key, new Shaders.Channelwise.Hardmax(channels));
            }

            Shader shader = shaders[key];

            if (stream == null) {
                stream = Shader.DefaultStream;
            }

            shader.Execute(stream, src, dst, length, length / channels);
        }
    }
}
