using System.Collections.Generic;

namespace TensorShaderCudaBackend {

    /// <summary>拡大</summary>
    public static class Zoom {
        private readonly static Dictionary<string, Shader> shaders = new Dictionary<string, Shader>();

        /// <summary>最近傍補間</summary>
        public static void NeighborZoom1D(uint channels, uint inwidth,
                                          uint batch,
                                          CudaArray<float> inmap, CudaArray<float> outmap,
                                          Stream stream = null) {

            string key = $"neighborzoom_1d " +
                         $"{nameof(channels)}={channels} scale=2";

            if (!shaders.ContainsKey(key)) {
                shaders.Add(key, new Shaders.Zoom.NeighborZoom1D(channels));
            }

            Shader shader = shaders[key];

            if (stream == null) {
                stream = Shader.DefaultStream;
            }

            shader.Execute(stream, inmap, outmap, inwidth, batch);
        }

        /// <summary>最近傍補間</summary>
        public static void NeighborZoom2D(uint channels, uint inwidth, uint inheight,
                                          uint batch,
                                          CudaArray<float> inmap, CudaArray<float> outmap,
                                          Stream stream = null) {

            string key = $"neighborzoom_2d " +
                         $"{nameof(channels)}={channels} scale=2";

            if (!shaders.ContainsKey(key)) {
                shaders.Add(key, new Shaders.Zoom.NeighborZoom2D(channels));
            }

            Shader shader = shaders[key];

            if (stream == null) {
                stream = Shader.DefaultStream;
            }

            shader.Execute(stream, inmap, outmap, inwidth, inheight, batch);
        }

        /// <summary>最近傍補間</summary>
        public static void NeighborZoom3D(uint channels, uint inwidth, uint inheight, uint indepth,
                                          uint batch,
                                          CudaArray<float> inmap, CudaArray<float> outmap,
                                          Stream stream = null) {

            string key = $"neighborzoom_3d " +
                         $"{nameof(channels)}={channels} scale=2";

            if (!shaders.ContainsKey(key)) {
                shaders.Add(key, new Shaders.Zoom.NeighborZoom3D(channels));
            }

            Shader shader = shaders[key];

            if (stream == null) {
                stream = Shader.DefaultStream;
            }

            shader.Execute(stream, inmap, outmap, inwidth, inheight, indepth, batch);
        }

        /// <summary>線形補間</summary>
        public static void LinearZoom1D(uint channels, uint inwidth,
                                        uint batch,
                                        CudaArray<float> inmap, CudaArray<float> outmap,
                                        Stream stream = null) {

            string key = $"linearzoom_1d " +
                         $"{nameof(channels)}={channels} scale=2";

            if (!shaders.ContainsKey(key)) {
                shaders.Add(key, new Shaders.Zoom.LinearZoom1D(channels));
            }

            Shader shader = shaders[key];

            if (stream == null) {
                stream = Shader.DefaultStream;
            }

            shader.Execute(stream, inmap, outmap, inwidth, batch);
        }

        /// <summary>線形補間</summary>
        public static void LinearZoom2D(uint channels, uint inwidth, uint inheight,
                                        uint batch,
                                        CudaArray<float> inmap, CudaArray<float> outmap,
                                        Stream stream = null) {

            string key = $"linearzoom_2d " +
                         $"{nameof(channels)}={channels} scale=2";

            if (!shaders.ContainsKey(key)) {
                shaders.Add(key, new Shaders.Zoom.LinearZoom2D(channels));
            }

            Shader shader = shaders[key];

            if (stream == null) {
                stream = Shader.DefaultStream;
            }

            shader.Execute(stream, inmap, outmap, inwidth, inheight, batch);
        }

        /// <summary>線形補間</summary>
        public static void LinearZoom3D(uint channels, uint inwidth, uint inheight, uint indepth,
                                        uint batch,
                                        CudaArray<float> inmap, CudaArray<float> outmap,
                                        Stream stream = null) {

            string key = $"linearzoom_3d " +
                         $"{nameof(channels)}={channels} scale=2";

            if (!shaders.ContainsKey(key)) {
                shaders.Add(key, new Shaders.Zoom.LinearZoom3D(channels));
            }

            Shader shader = shaders[key];

            if (stream == null) {
                stream = Shader.DefaultStream;
            }

            shader.Execute(stream, inmap, outmap, inwidth, inheight, indepth, batch);
        }
    }
}