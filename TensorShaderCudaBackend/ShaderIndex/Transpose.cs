using System.Collections.Generic;

namespace TensorShaderCudaBackend {

    /// <summary>転置</summary>
    public static class Transpose {
        private readonly static Dictionary<string, Shader> shaders = new();

        /// <summary>フィルタ行列入出力チャネル軸入れ替え</summary>
        public static void TransposeKernelChannel(uint inchannels, uint outchannels, uint pts,
                                                  CudaArray<float> inmap, CudaArray<float> outmap,
                                                  Stream stream = null) {

            string key = $"transpose_kernel_channel " +
                         $"{nameof(inchannels)}={inchannels} {nameof(outchannels)}={outchannels}";

            if (!shaders.ContainsKey(key)) {
                shaders.Add(key, new Shaders.Transpose.TransposeKernelChannel(inchannels, outchannels));
            }

            Shader shader = shaders[key];

            if (stream is null) {
                stream = Shader.DefaultStream;
            }

            shader.Execute(stream, inmap, outmap, pts);
        }

        /// <summary>フィルタ行列入出力チャネル軸入れ替え</summary>
        public static void TransposeComplexKernelChannel(uint inchannels, uint outchannels, uint pts,
                                                         CudaArray<float> inmap, CudaArray<float> outmap,
                                                         Stream stream = null) {

            string key = $"transpose_complex_kernel_channel " +
                         $"{nameof(inchannels)}={inchannels} {nameof(outchannels)}={outchannels}";

            if (!shaders.ContainsKey(key)) {
                shaders.Add(key, new Shaders.Transpose.TransposeComplexKernelChannel(inchannels, outchannels));
            }

            Shader shader = shaders[key];

            if (stream is null) {
                stream = Shader.DefaultStream;
            }

            shader.Execute(stream, inmap, outmap, pts);
        }

        /// <summary>フィルタ行列入出力チャネル軸入れ替え</summary>
        public static void TransposeQuaternionKernelChannel(uint inchannels, uint outchannels, uint pts,
                                                            CudaArray<float> inmap, CudaArray<float> outmap,
                                                            Stream stream = null) {

            string key = $"transpose_quaternion_kernel_channel " +
                         $"{nameof(inchannels)}={inchannels} {nameof(outchannels)}={outchannels}";

            if (!shaders.ContainsKey(key)) {
                shaders.Add(key, new Shaders.Transpose.TransposeQuaternionKernelChannel(inchannels, outchannels));
            }

            Shader shader = shaders[key];

            if (stream is null) {
                stream = Shader.DefaultStream;
            }

            shader.Execute(stream, inmap, outmap, pts);
        }

        /// <summary>ブロックごとに転置</summary>
        public static void BlockTranspose(uint block, uint n, uint m,
                                          CudaArray<float> inmap, CudaArray<float> outmap,
                                          Stream stream = null) {

            string key = $"block_transpose " +
                         $"{nameof(block)}={block} {nameof(n)}={n} {nameof(m)}={m}";

            if (!shaders.ContainsKey(key)) {
                shaders.Add(key, new Shaders.Transpose.BlockTranspose(block, n, m));
            }

            Shader shader = shaders[key];

            if (stream is null) {
                stream = Shader.DefaultStream;
            }

            shader.Execute(stream, inmap, outmap);
        }
    }
}