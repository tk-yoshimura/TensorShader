using System.Collections.Generic;

namespace TensorShaderCudaBackend {

    /// <summary>トリミング</summary>
    public static class Trimming {
        private readonly static Dictionary<string, Shader> shaders = new Dictionary<string, Shader>();

        /// <summary>1次元トリミング</summary>
        public static void Trimming1D(uint channels, uint outwidth,
                                      uint batch,
                                      uint trim_left, uint trim_right,
                                      CudaArray<float> inmap, CudaArray<float> outmap,
                                      Stream stream = null) {

            string key = $"trimming_1d " +
                         $"{nameof(channels)}={channels} " +
                         $"{nameof(trim_left)}={trim_left} {nameof(trim_right)}={trim_right}";

            if (!shaders.ContainsKey(key)) {
                shaders.Add(key, new Shaders.Trimming.Trimming1D(channels, trim_left, trim_right));
            }

            Shader shader = shaders[key];

            if (stream == null) {
                stream = Shader.DefaultStream;
            }

            shader.Execute(stream, inmap, outmap, outwidth, batch);
        }

        /// <summary>2次元トリミング</summary>
        public static void Trimming2D(uint channels, uint outwidth, uint outheight,
                                      uint batch,
                                      uint trim_left, uint trim_right,
                                      uint trim_top, uint trim_bottom,
                                      CudaArray<float> inmap, CudaArray<float> outmap,
                                      Stream stream = null) {

            string key = $"trimming_2d " +
                         $"{nameof(channels)}={channels} " +
                         $"{nameof(trim_left)}={trim_left} {nameof(trim_right)}={trim_right} " +
                         $"{nameof(trim_top)}={trim_top} {nameof(trim_bottom)}={trim_bottom}";

            if (!shaders.ContainsKey(key)) {
                shaders.Add(key, new Shaders.Trimming.Trimming2D(channels, trim_left, trim_right, trim_top, trim_bottom));
            }

            Shader shader = shaders[key];

            if (stream == null) {
                stream = Shader.DefaultStream;
            }

            shader.Execute(stream, inmap, outmap, outwidth, outheight, batch);
        }

        /// <summary>3次元トリミング</summary>
        public static void Trimming3D(uint channels, uint outwidth, uint outheight, uint outdepth,
                                      uint batch,
                                      uint trim_left, uint trim_right,
                                      uint trim_top, uint trim_bottom,
                                      uint trim_front, uint trim_rear,
                                      CudaArray<float> inmap, CudaArray<float> outmap,
                                      Stream stream = null) {

            string key = $"trimming_3d " +
                         $"{nameof(channels)}={channels} " +
                         $"{nameof(trim_left)}={trim_left} {nameof(trim_right)}={trim_right} " +
                         $"{nameof(trim_top)}={trim_top} {nameof(trim_bottom)}={trim_bottom} " +
                         $"{nameof(trim_front)}={trim_front} {nameof(trim_rear)}={trim_rear}";

            if (!shaders.ContainsKey(key)) {
                shaders.Add(key, new Shaders.Trimming.Trimming3D(channels,
                    trim_left, trim_right, trim_top, trim_bottom, trim_front, trim_rear));
            }

            Shader shader = shaders[key];

            if (stream == null) {
                stream = Shader.DefaultStream;
            }

            shader.Execute(stream, inmap, outmap, outwidth, outheight, outdepth, batch);
        }

    }
}