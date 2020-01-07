using System.Collections.Generic;

namespace TensorShaderCudaBackend {

    /// <summary>転置</summary>
    public static class Transpose {
        private readonly static Dictionary<string, Shader> shaders = new Dictionary<string, Shader>();

        /// <summary>フィルタ行列入出力チャネル軸入れ替え</summary>
        public static void TransposeKernelChannel(uint inchannels, uint outchannels, uint pts,
                                                  CudaArray<float> inmap, CudaArray<float> outmap,
                                                  Stream stream = null) {

            string key = $"transpose_kernel_channel {nameof(inchannels)}={inchannels} {nameof(outchannels)}={outchannels}";

            if (!shaders.ContainsKey(key)) {
                shaders.Add(key, new Shaders.Transpose.TransposeKernelChannel(inchannels, outchannels));
            }

            Shader shader = shaders[key];

            if (stream == null) {
                stream = Shader.DefaultStream;
            }

            shader.Execute(stream, inmap, outmap, pts);
        }

        /// <summary>フィルタ行列入出力チャネル軸入れ替え</summary>
        public static void TransposeComplexKernelChannel(uint inchannels, uint outchannels, uint pts,
                                                         CudaArray<float> inmap, CudaArray<float> outmap,
                                                         Stream stream = null) {

            string key = $"transpose_complex_kernel_channel {nameof(inchannels)}={inchannels} {nameof(outchannels)}={outchannels}";

            if (!shaders.ContainsKey(key)) {
                shaders.Add(key, new Shaders.Transpose.TransposeComplexKernelChannel(inchannels, outchannels));
            }

            Shader shader = shaders[key];

            if (stream == null) {
                stream = Shader.DefaultStream;
            }

            shader.Execute(stream, inmap, outmap, pts);
        }

        /// <summary>フィルタ行列入出力チャネル軸入れ替え</summary>
        public static void TransposeQuaternionKernelChannel(uint inchannels, uint outchannels, uint pts,
                                                            CudaArray<float> inmap, CudaArray<float> outmap,
                                                            Stream stream = null) {

            string key = $"transpose_quaternion_kernel_channel {nameof(inchannels)}={inchannels} {nameof(outchannels)}={outchannels}";

            if (!shaders.ContainsKey(key)) {
                shaders.Add(key, new Shaders.Transpose.TransposeQuaternionKernelChannel(inchannels, outchannels));
            }

            Shader shader = shaders[key];

            if (stream == null) {
                stream = Shader.DefaultStream;
            }

            shader.Execute(stream, inmap, outmap, pts);
        }
    }
}