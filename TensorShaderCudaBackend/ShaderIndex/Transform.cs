using System.Collections.Generic;

namespace TensorShaderCudaBackend {

    /// <summary>変形</summary>
    public static class Transform {
        private readonly static Dictionary<string, Shader> shaders = new Dictionary<string, Shader>();

        /// <summary>チャネル次元を空間方向に展開</summary>
        public static void ChannelToSpace2D(uint outchannels,
                                            uint inwidth, uint inheight,
                                            uint batch, uint scale,
                                            CudaArray<float> inmap, CudaArray<float> outmap,
                                            Stream stream = null) {

            string key = $"channel_to_space_2d {nameof(outchannels)}={outchannels} {nameof(scale)}={scale}";

            if (!shaders.ContainsKey(key)) {
                shaders.Add(key, new Shaders.Transform.ChannelToSpace2D(scale, outchannels));
            }

            Shader shader = shaders[key];

            if (stream == null) {
                stream = Shader.DefaultStream;
            }

            shader.Execute(stream, inmap, outmap, inwidth, inheight, batch);
        }

        /// <summary>空間方向をチャネル次元に展開</summary>
        public static void SpaceToChannel2D(uint inchannels, uint outwidth, uint outheight,
                                            uint batch, uint scale,
                                            CudaArray<float> inmap, CudaArray<float> outmap,
                                            Stream stream = null) {

            string key = $"space_to_channel_2d {nameof(inchannels)}={inchannels} {nameof(scale)}={scale}";

            if (!shaders.ContainsKey(key)) {
                shaders.Add(key, new Shaders.Transform.SpaceToChannel2D(scale, inchannels));
            }

            Shader shader = shaders[key];

            if (stream == null) {
                stream = Shader.DefaultStream;
            }

            shader.Execute(stream, inmap, outmap, outwidth, outheight, batch);
        }

        /// <summary>チャネル次元を空間方向に展開</summary>
        public static void ChannelToSpace3D(uint outchannels, uint inwidth, uint inheight, uint indepth,
                                            uint batch, uint scale,
                                            CudaArray<float> inmap, CudaArray<float> outmap,
                                            Stream stream = null) {

            string key = $"channel_to_space_3d {nameof(outchannels)}={outchannels} {nameof(scale)}={scale}";

            if (!shaders.ContainsKey(key)) {
                shaders.Add(key, new Shaders.Transform.ChannelToSpace3D(scale, outchannels));
            }

            Shader shader = shaders[key];

            if (stream == null) {
                stream = Shader.DefaultStream;
            }

            shader.Execute(stream, inmap, outmap, inwidth, inheight, indepth, batch);
        }

        /// <summary>空間方向をチャネル次元に展開</summary>
        public static void SpaceToChannel3D(uint inchannels, uint outwidth, uint outheight, uint outdepth,
                                            uint batch, uint scale,
                                            CudaArray<float> inmap, CudaArray<float> outmap,
                                            Stream stream = null) {

            string key = $"space_to_channel_3d {nameof(inchannels)}={inchannels} {nameof(scale)}={scale}";

            if (!shaders.ContainsKey(key)) {
                shaders.Add(key, new Shaders.Transform.SpaceToChannel3D(scale, inchannels));
            }

            Shader shader = shaders[key];

            if (stream == null) {
                stream = Shader.DefaultStream;
            }

            shader.Execute(stream, inmap, outmap, outwidth, outheight, outdepth, batch);
        }

        /// <summary>ColumnToImage変換</summary>
        public static void ColumnToImage1D(uint channels, uint inwidth,
                                           uint batch, uint kwidth,
                                           CudaArray<float> inmap, CudaArray<float> outmap,
                                           Stream stream = null) {

            string key = $"column_to_image_1d {nameof(channels)}={channels} " +
                         $"{nameof(kwidth)}={kwidth}";

            if (!shaders.ContainsKey(key)) {
                shaders.Add(key, new Shaders.Transform.ColumnToImage1D(channels, kwidth));
            }

            Shader shader = shaders[key];

            if (stream == null) {
                stream = Shader.DefaultStream;
            }

            shader.Execute(stream, inmap, outmap, inwidth, batch);
        }

        /// <summary>ImageToColumn変換</summary>
        public static void ImageToColumn1D(uint channels, uint inwidth,
                                           uint batch, uint kwidth,
                                           CudaArray<float> inmap, CudaArray<float> outmap,
                                           Stream stream = null) {

            string key = $"image_to_column_1d {nameof(channels)}={channels} " +
                         $"{nameof(kwidth)}={kwidth}";

            if (!shaders.ContainsKey(key)) {
                shaders.Add(key, new Shaders.Transform.ImageToColumn1D(channels, kwidth));
            }

            Shader shader = shaders[key];

            if (stream == null) {
                stream = Shader.DefaultStream;
            }

            shader.Execute(stream, inmap, outmap, inwidth, batch);
        }

        /// <summary>ColumnToImage変換</summary>
        public static void ColumnToImage2D(uint channels, uint inwidth, uint inheight,
                                           uint batch, uint kwidth, uint kheight,
                                           CudaArray<float> inmap, CudaArray<float> outmap,
                                           Stream stream = null) {

            string key = $"column_to_image_2d {nameof(channels)}={channels} " +
                         $"{nameof(kwidth)}={kwidth} {nameof(kheight)}={kheight}";

            if (!shaders.ContainsKey(key)) {
                shaders.Add(key, new Shaders.Transform.ColumnToImage2D(channels, kwidth, kheight));
            }

            Shader shader = shaders[key];

            if (stream == null) {
                stream = Shader.DefaultStream;
            }

            shader.Execute(stream, inmap, outmap, inwidth, inheight, batch);
        }

        /// <summary>ImageToColumn変換</summary>
        public static void ImageToColumn2D(uint channels, uint inwidth, uint inheight,
                                           uint batch, uint kwidth, uint kheight,
                                           CudaArray<float> inmap, CudaArray<float> outmap,
                                           Stream stream = null) {

            string key = $"image_to_column_2d {nameof(channels)}={channels} " +
                         $"{nameof(kwidth)}={kwidth} {nameof(kheight)}={kheight}";

            if (!shaders.ContainsKey(key)) {
                shaders.Add(key, new Shaders.Transform.ImageToColumn2D(channels, kwidth, kheight));
            }

            Shader shader = shaders[key];

            if (stream == null) {
                stream = Shader.DefaultStream;
            }

            shader.Execute(stream, inmap, outmap, inwidth, inheight, batch);
        }

        /// <summary>ColumnToImage変換</summary>
        public static void ColumnToImage3D(uint channels, uint inwidth, uint inheight, uint indepth,
                                           uint batch, uint kwidth, uint kheight, uint kdepth,
                                           CudaArray<float> inmap, CudaArray<float> outmap,
                                           Stream stream = null) {

            string key = $"column_to_image_3d {nameof(channels)}={channels} " +
                         $"{nameof(kwidth)}={kwidth} {nameof(kheight)}={kheight} {nameof(kdepth)}={kdepth}";

            if (!shaders.ContainsKey(key)) {
                shaders.Add(key, new Shaders.Transform.ColumnToImage3D(channels, kwidth, kheight, kdepth));
            }

            Shader shader = shaders[key];

            if (stream == null) {
                stream = Shader.DefaultStream;
            }

            shader.Execute(stream, inmap, outmap, inwidth, inheight, indepth, batch);
        }

        /// <summary>ImageToColumn変換</summary>
        public static void ImageToColumn3D(uint channels, uint inwidth, uint inheight, uint indepth,
                                           uint batch, uint kwidth, uint kheight, uint kdepth,
                                           CudaArray<float> inmap, CudaArray<float> outmap,
                                           Stream stream = null) {

            string key = $"image_to_column_3d {nameof(channels)}={channels} " +
                         $"{nameof(kwidth)}={kwidth} {nameof(kheight)}={kheight} {nameof(kdepth)}={kdepth}";

            if (!shaders.ContainsKey(key)) {
                shaders.Add(key, new Shaders.Transform.ImageToColumn3D(channels, kwidth, kheight, kdepth));
            }

            Shader shader = shaders[key];

            if (stream == null) {
                stream = Shader.DefaultStream;
            }

            shader.Execute(stream, inmap, outmap, inwidth, inheight, indepth, batch);
        }
    }
}