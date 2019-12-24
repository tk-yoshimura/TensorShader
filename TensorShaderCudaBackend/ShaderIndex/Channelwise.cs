using System.Collections.Generic;

namespace TensorShaderCudaBackend {

    /// <summary>チャネル独立演算</summary>
    public static class Channelwise {
        private readonly static Dictionary<string, Shader> shaders = new Dictionary<string, Shader>();

        private static Shader BinaryArithmetric(string name, string func, uint channels) {
            string key = $"{name} channels={channels}";

            if (!shaders.ContainsKey(key)) {
                shaders.Add(key, new Shaders.Channelwise.ChannelwiseBinaryArithmetric(name, func, channels));
            }

            return shaders[key];
        }

        /// <summary>加算</summary>
        public static void Add(uint vector_length, uint map_length, CudaArray<float> srcvector, CudaArray<float> srcmap, CudaArray<float> dstmap) {
            Shader shader = BinaryArithmetric("add_cw", "#y = #v + #x;", vector_length);
            shader.Execute(Shader.DefaultStream, srcvector, srcmap, dstmap, map_length);
        }

        /// <summary>乗算</summary>
        public static void Mul(uint vector_length, uint map_length, CudaArray<float> srcvector, CudaArray<float> srcmap, CudaArray<float> dstmap) {
            Shader shader = BinaryArithmetric("mul_cw", "#y = #v * #x;", vector_length);
            shader.Execute(Shader.DefaultStream, srcvector, srcmap, dstmap, map_length);
        }

        /// <summary>減算(左ベクトル)</summary>
        public static void SubLVector(uint vector_length, uint map_length, CudaArray<float> srcvector, CudaArray<float> srcmap, CudaArray<float> dstmap) {
            Shader shader = BinaryArithmetric("subl_cw", "#y = #v - #x;", vector_length);
            shader.Execute(Shader.DefaultStream, srcvector, srcmap, dstmap, map_length);
        }

        /// <summary>減算(右ベクトル)</summary>
        public static void SubRVector(uint vector_length, uint map_length, CudaArray<float> srcvector, CudaArray<float> srcmap, CudaArray<float> dstmap) {
            Shader shader = BinaryArithmetric("subr_cw", "#y = #x - #v;", vector_length);
            shader.Execute(Shader.DefaultStream, srcvector, srcmap, dstmap, map_length);
        }

        /// <summary>除算(左ベクトル)</summary>
        public static void DivLVector(uint vector_length, uint map_length, CudaArray<float> srcvector, CudaArray<float> srcmap, CudaArray<float> dstmap) {
            Shader shader = BinaryArithmetric("divl_cw", "#y = #v / #x;", vector_length);
            shader.Execute(Shader.DefaultStream, srcvector, srcmap, dstmap, map_length);
        }

        /// <summary>除算(右ベクトル)</summary>
        public static void DivRVector(uint vector_length, uint map_length, CudaArray<float> srcvector, CudaArray<float> srcmap, CudaArray<float> dstmap) {
            Shader shader = BinaryArithmetric("divr_cw", "#y = #x / #v;", vector_length);
            shader.Execute(Shader.DefaultStream, srcvector, srcmap, dstmap, map_length);
        }
    }
}
