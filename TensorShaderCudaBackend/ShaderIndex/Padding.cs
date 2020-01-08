using System.Collections.Generic;

namespace TensorShaderCudaBackend {

    /// <summary>パディング</summary>
    public static class Padding {
        private readonly static Dictionary<string, Shader> shaders = new Dictionary<string, Shader>();

        /// <summary>1次元ゼロパディング</summary>
        public static void ZeroPadding1D(uint channels, uint inwidth,
                                         uint batch,
                                         uint pad_left, uint pad_right,
                                         CudaArray<float> inmap, CudaArray<float> outmap,
                                         Stream stream = null) {

            string key = $"zeropadding_1d " +
                         $"{nameof(channels)}={channels} {nameof(pad_left)}={pad_left} {nameof(pad_right)}={pad_right}";

            if (!shaders.ContainsKey(key)) {
                shaders.Add(key, new Shaders.Padding.ZeroPadding1D(channels, pad_left, pad_right));
            }

            Shader shader = shaders[key];

            if (stream == null) {
                stream = Shader.DefaultStream;
            }

            shader.Execute(stream, inmap, outmap, inwidth, batch);
        }

        /// <summary>1次元エッジパディング</summary>
        public static void EdgePadding1D(uint channels, uint inwidth,
                                         uint batch,
                                         uint pad_left, uint pad_right,
                                         CudaArray<float> inmap, CudaArray<float> outmap,
                                         Stream stream = null) {

            string key = $"edgepadding_1d " +
                         $"{nameof(channels)}={channels} " +
                         $"{nameof(pad_left)}={pad_left} {nameof(pad_right)}={pad_right}";

            if (!shaders.ContainsKey(key)) {
                shaders.Add(key, new Shaders.Padding.EdgePadding1D(channels, pad_left, pad_right));
            }

            Shader shader = shaders[key];

            if (stream == null) {
                stream = Shader.DefaultStream;
            }

            shader.Execute(stream, inmap, outmap, inwidth, batch);
        }

        /// <summary>2次元ゼロパディング</summary>
        public static void ZeroPadding2D(uint channels, uint inwidth, uint inheight,
                                         uint batch,
                                         uint pad_left, uint pad_right,
                                         uint pad_top, uint pad_bottom,
                                         CudaArray<float> inmap, CudaArray<float> outmap,
                                         Stream stream = null) {

            string key = $"zeropadding_2d " +
                         $"{nameof(channels)}={channels} " +
                         $"{nameof(pad_left)}={pad_left} {nameof(pad_right)}={pad_right} " +
                         $"{nameof(pad_top)}={pad_top} {nameof(pad_bottom)}={pad_bottom}";

            if (!shaders.ContainsKey(key)) {
                shaders.Add(key, new Shaders.Padding.ZeroPadding2D(channels, pad_left, pad_right, pad_top, pad_bottom));
            }

            Shader shader = shaders[key];

            if (stream == null) {
                stream = Shader.DefaultStream;
            }

            shader.Execute(stream, inmap, outmap, inwidth, inheight, batch);
        }

        /// <summary>2次元エッジパディング</summary>
        public static void EdgePadding2D(uint channels, uint inwidth, uint inheight,
                                         uint batch,
                                         uint pad_left, uint pad_right,
                                         uint pad_top, uint pad_bottom,
                                         CudaArray<float> inmap, CudaArray<float> outmap,
                                         Stream stream = null) {

            string key = $"edgepadding_2d " +
                         $"{nameof(channels)}={channels} " +
                         $"{nameof(pad_left)}={pad_left} {nameof(pad_right)}={pad_right} " +
                         $"{nameof(pad_top)}={pad_top} {nameof(pad_bottom)}={pad_bottom}";

            if (!shaders.ContainsKey(key)) {
                shaders.Add(key, new Shaders.Padding.EdgePadding2D(channels, pad_left, pad_right, pad_top, pad_bottom));
            }

            Shader shader = shaders[key];

            if (stream == null) {
                stream = Shader.DefaultStream;
            }

            shader.Execute(stream, inmap, outmap, inwidth, inheight, batch);
        }

        /// <summary>3次元ゼロパディング</summary>
        public static void ZeroPadding3D(uint channels, uint inwidth, uint inheight, uint indepth,
                                         uint batch,
                                         uint pad_left, uint pad_right,
                                         uint pad_top, uint pad_bottom,
                                         uint pad_front, uint pad_rear,
                                         CudaArray<float> inmap, CudaArray<float> outmap,
                                         Stream stream = null) {

            string key = $"zeropadding_3d " +
                         $"{nameof(channels)}={channels} " +
                         $"{nameof(pad_left)}={pad_left} {nameof(pad_right)}={pad_right} " +
                         $"{nameof(pad_top)}={pad_top} {nameof(pad_bottom)}={pad_bottom} " +
                         $"{nameof(pad_front)}={pad_front} {nameof(pad_rear)}={pad_rear}";

            if (!shaders.ContainsKey(key)) {
                shaders.Add(key, new Shaders.Padding.ZeroPadding3D(channels,
                    pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_rear));
            }

            Shader shader = shaders[key];

            if (stream == null) {
                stream = Shader.DefaultStream;
            }

            shader.Execute(stream, inmap, outmap, inwidth, inheight, indepth, batch);
        }

        /// <summary>3次元エッジパディング</summary>
        public static void EdgePadding3D(uint channels, uint inwidth, uint inheight, uint indepth,
                                         uint batch,
                                         uint pad_left, uint pad_right,
                                         uint pad_top, uint pad_bottom,
                                         uint pad_front, uint pad_rear,
                                         CudaArray<float> inmap, CudaArray<float> outmap,
                                         Stream stream = null) {

            string key = $"edgepadding_3d " +
                         $"{nameof(channels)}={channels} " +
                         $"{nameof(pad_left)}={pad_left} {nameof(pad_right)}={pad_right} " +
                         $"{nameof(pad_top)}={pad_top} {nameof(pad_bottom)}={pad_bottom} " +
                         $"{nameof(pad_front)}={pad_front} {nameof(pad_rear)}={pad_rear}";

            if (!shaders.ContainsKey(key)) {
                shaders.Add(key, new Shaders.Padding.EdgePadding3D(channels,
                    pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_rear));
            }

            Shader shader = shaders[key];

            if (stream == null) {
                stream = Shader.DefaultStream;
            }

            shader.Execute(stream, inmap, outmap, inwidth, inheight, indepth, batch);
        }
    }
}