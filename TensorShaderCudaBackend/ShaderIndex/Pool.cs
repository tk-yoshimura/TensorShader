using System.Collections.Generic;

namespace TensorShaderCudaBackend {

    /// <summary>プーリング</summary>
    public static class Pool {
        private readonly static Dictionary<string, Shader> shaders = new Dictionary<string, Shader>();

        /// <summary>1次元最大値プール</summary>
        public static void MaxPool1D(uint channels, uint inwidth,
                                     uint batch, uint stride,
                                     CudaArray<float> inmap, CudaArray<float> outmap,
                                     Stream stream = null) {

            string key = $"maxpool_1d {nameof(channels)}={channels} {nameof(stride)}={stride}";

            if (!shaders.ContainsKey(key)) {
                shaders.Add(key, new Shaders.Pool.MaxPool1D(channels, stride));
            }

            Shader shader = shaders[key];

            if (stream == null) {
                stream = Shader.DefaultStream;
            }

            shader.Execute(stream, inmap, outmap, inwidth, batch);
        }

        /// <summary>2次元最大値プール</summary>
        public static void MaxPool2D(uint channels, uint inwidth, uint inheight,
                                     uint batch, uint stride,
                                     CudaArray<float> inmap, CudaArray<float> outmap,
                                     Stream stream = null) {

            string key = $"maxpool_2d {nameof(channels)}={channels} {nameof(stride)}={stride}";

            if (!shaders.ContainsKey(key)) {
                shaders.Add(key, new Shaders.Pool.MaxPool2D(channels, stride));
            }

            Shader shader = shaders[key];

            if (stream == null) {
                stream = Shader.DefaultStream;
            }

            shader.Execute(stream, inmap, outmap, inwidth, inheight, batch);
        }

        /// <summary>3次元最大値プール</summary>
        public static void MaxPool3D(uint channels, uint inwidth, uint inheight, uint indepth,
                                     uint batch, uint stride,
                                     CudaArray<float> inmap, CudaArray<float> outmap,
                                     Stream stream = null) {

            string key = $"maxpool_3d {nameof(channels)}={channels} {nameof(stride)}={stride}";

            if (!shaders.ContainsKey(key)) {
                shaders.Add(key, new Shaders.Pool.MaxPool3D(channels, stride));
            }

            Shader shader = shaders[key];

            if (stream == null) {
                stream = Shader.DefaultStream;
            }

            shader.Execute(stream, inmap, outmap, inwidth, inheight, indepth, batch);
        }

        /// <summary>1次元平均値プール</summary>
        public static void AveragePool1D(uint channels, uint inwidth,
                                         uint batch, uint stride,
                                         CudaArray<float> inmap, CudaArray<float> outmap,
                                         Stream stream = null) {

            string key = $"averagepool_1d {nameof(channels)}={channels} {nameof(stride)}={stride}";

            if (!shaders.ContainsKey(key)) {
                shaders.Add(key, new Shaders.Pool.AveragePool1D(channels, stride));
            }

            Shader shader = shaders[key];

            if (stream == null) {
                stream = Shader.DefaultStream;
            }

            shader.Execute(stream, inmap, outmap, inwidth, batch);
        }

        /// <summary>2次元平均値プール</summary>
        public static void AveragePool2D(uint channels, uint inwidth, uint inheight,
                                         uint batch, uint stride,
                                         CudaArray<float> inmap, CudaArray<float> outmap,
                                         Stream stream = null) {

            string key = $"averagepool_2d {nameof(channels)}={channels} {nameof(stride)}={stride}";

            if (!shaders.ContainsKey(key)) {
                shaders.Add(key, new Shaders.Pool.AveragePool2D(channels, stride));
            }

            Shader shader = shaders[key];

            if (stream == null) {
                stream = Shader.DefaultStream;
            }

            shader.Execute(stream, inmap, outmap, inwidth, inheight, batch);
        }

        /// <summary>3次元平均値プール</summary>
        public static void AveragePool3D(uint channels, uint inwidth, uint inheight, uint indepth,
                                         uint batch, uint stride,
                                         CudaArray<float> inmap, CudaArray<float> outmap,
                                         Stream stream = null) {

            string key = $"averagepool_3d {nameof(channels)}={channels} {nameof(stride)}={stride}";

            if (!shaders.ContainsKey(key)) {
                shaders.Add(key, new Shaders.Pool.AveragePool3D(channels, stride));
            }

            Shader shader = shaders[key];

            if (stream == null) {
                stream = Shader.DefaultStream;
            }

            shader.Execute(stream, inmap, outmap, inwidth, inheight, indepth, batch);
        }

        /// <summary>1次元ストライドプール</summary>
        public static void StridePool1D(uint channels, uint inwidth,
                                        uint batch, uint stride,
                                        CudaArray<float> inmap, CudaArray<float> outmap,
                                        Stream stream = null) {

            string key = $"stridepool_1d {nameof(channels)}={channels} {nameof(stride)}={stride}";

            if (!shaders.ContainsKey(key)) {
                shaders.Add(key, new Shaders.Pool.StridePool1D(channels, stride));
            }

            Shader shader = shaders[key];

            if (stream == null) {
                stream = Shader.DefaultStream;
            }

            shader.Execute(stream, inmap, outmap, inwidth, batch);
        }

        /// <summary>2次元ストライドプール</summary>
        public static void StridePool2D(uint channels, uint inwidth, uint inheight,
                                        uint batch, uint stride,
                                        CudaArray<float> inmap, CudaArray<float> outmap,
                                        Stream stream = null) {

            string key = $"stridepool_2d {nameof(channels)}={channels} {nameof(stride)}={stride}";

            if (!shaders.ContainsKey(key)) {
                shaders.Add(key, new Shaders.Pool.StridePool2D(channels, stride));
            }

            Shader shader = shaders[key];

            if (stream == null) {
                stream = Shader.DefaultStream;
            }

            shader.Execute(stream, inmap, outmap, inwidth, inheight, batch);
        }

        /// <summary>3次元ストライドプール</summary>
        public static void StridePool3D(uint channels, uint inwidth, uint inheight, uint indepth,
                                        uint batch, uint stride,
                                        CudaArray<float> inmap, CudaArray<float> outmap,
                                        Stream stream = null) {

            string key = $"stridepool_3d {nameof(channels)}={channels} {nameof(stride)}={stride}";

            if (!shaders.ContainsKey(key)) {
                shaders.Add(key, new Shaders.Pool.StridePool3D(channels, stride));
            }

            Shader shader = shaders[key];

            if (stream == null) {
                stream = Shader.DefaultStream;
            }

            shader.Execute(stream, inmap, outmap, inwidth, inheight, indepth, batch);
        }

        /// <summary>1次元最大値逆プール</summary>
        public static void MaxUnpool1D(uint channels, uint outwidth,
                                       uint batch, uint stride,
                                       CudaArray<float> ingrad, CudaArray<float> inpool, CudaArray<float> inmap, CudaArray<float> outmap,
                                       Stream stream = null) {

            string key = $"maxunpool_1d {nameof(channels)}={channels} {nameof(stride)}={stride}";

            if (!shaders.ContainsKey(key)) {
                shaders.Add(key, new Shaders.Pool.MaxUnpool1D(channels, stride));
            }

            Shader shader = shaders[key];

            if (stream == null) {
                stream = Shader.DefaultStream;
            }

            shader.Execute(stream, ingrad, inpool, inmap, outmap, outwidth, batch);
        }

        /// <summary>2次元最大値逆プール</summary>
        public static void MaxUnpool2D(uint channels, uint outwidth, uint outheight,
                                       uint batch, uint stride,
                                       CudaArray<float> ingrad, CudaArray<float> inpool, CudaArray<float> inmap, CudaArray<float> outmap,
                                       Stream stream = null) {

            string key = $"maxunpool_2d {nameof(channels)}={channels} {nameof(stride)}={stride}";

            if (!shaders.ContainsKey(key)) {
                shaders.Add(key, new Shaders.Pool.MaxUnpool2D(channels, stride));
            }

            Shader shader = shaders[key];

            if (stream == null) {
                stream = Shader.DefaultStream;
            }

            shader.Execute(stream, ingrad, inpool, inmap, outmap, outwidth, outheight, batch);
        }

        /// <summary>3次元最大値逆プール</summary>
        public static void MaxUnpool3D(uint channels, uint outwidth, uint outheight, uint outdepth,
                                       uint batch, uint stride,
                                       CudaArray<float> ingrad, CudaArray<float> inpool, CudaArray<float> inmap, CudaArray<float> outmap,
                                       Stream stream = null) {

            string key = $"maxunpool_3d {nameof(channels)}={channels} {nameof(stride)}={stride}";

            if (!shaders.ContainsKey(key)) {
                shaders.Add(key, new Shaders.Pool.MaxUnpool3D(channels, stride));
            }

            Shader shader = shaders[key];

            if (stream == null) {
                stream = Shader.DefaultStream;
            }

            shader.Execute(stream, ingrad, inpool, inmap, outmap, outwidth, outheight, outdepth, batch);
        }

        /// <summary>1次元平均値逆プール</summary>
        public static void AverageUnpool1D(uint channels, uint outwidth,
                                           uint batch, uint stride,
                                           CudaArray<float> inmap, CudaArray<float> outmap,
                                           Stream stream = null) {

            string key = $"averageunpool_1d {nameof(channels)}={channels} {nameof(stride)}={stride}";

            if (!shaders.ContainsKey(key)) {
                shaders.Add(key, new Shaders.Pool.AverageUnpool1D(channels, stride));
            }

            Shader shader = shaders[key];

            if (stream == null) {
                stream = Shader.DefaultStream;
            }

            shader.Execute(stream, inmap, outmap, outwidth, batch);
        }

        /// <summary>2次元平均値逆プール</summary>
        public static void AverageUnpool2D(uint channels, uint outwidth, uint outheight,
                                           uint batch, uint stride,
                                           CudaArray<float> inmap, CudaArray<float> outmap,
                                           Stream stream = null) {

            string key = $"averageunpool_2d {nameof(channels)}={channels} {nameof(stride)}={stride}";

            if (!shaders.ContainsKey(key)) {
                shaders.Add(key, new Shaders.Pool.AverageUnpool2D(channels, stride));
            }

            Shader shader = shaders[key];

            if (stream == null) {
                stream = Shader.DefaultStream;
            }

            shader.Execute(stream, inmap, outmap, outwidth, outheight, batch);
        }

        /// <summary>3次元平均値逆プール</summary>
        public static void AverageUnpool3D(uint channels, uint outwidth, uint outheight, uint outdepth,
                                           uint batch, uint stride,
                                           CudaArray<float> inmap, CudaArray<float> outmap,
                                           Stream stream = null) {

            string key = $"averageunpool_3d {nameof(channels)}={channels} {nameof(stride)}={stride}";

            if (!shaders.ContainsKey(key)) {
                shaders.Add(key, new Shaders.Pool.AverageUnpool3D(channels, stride));
            }

            Shader shader = shaders[key];

            if (stream == null) {
                stream = Shader.DefaultStream;
            }

            shader.Execute(stream, inmap, outmap, outwidth, outheight, outdepth, batch);
        }

        /// <summary>1次元ストライド逆プール</summary>
        public static void StrideUnpool1D(uint channels, uint outwidth,
                                          uint batch, uint stride,
                                          CudaArray<float> inmap, CudaArray<float> outmap,
                                          Stream stream = null) {

            string key = $"strideunpool_1d {nameof(channels)}={channels} {nameof(stride)}={stride}";

            if (!shaders.ContainsKey(key)) {
                shaders.Add(key, new Shaders.Pool.StrideUnpool1D(channels, stride));
            }

            Shader shader = shaders[key];

            if (stream == null) {
                stream = Shader.DefaultStream;
            }

            shader.Execute(stream, inmap, outmap, outwidth, batch);
        }

        /// <summary>2次元ストライド逆プール</summary>
        public static void StrideUnpool2D(uint channels, uint outwidth, uint outheight,
                                          uint batch, uint stride,
                                          CudaArray<float> inmap, CudaArray<float> outmap,
                                          Stream stream = null) {

            string key = $"strideunpool_2d {nameof(channels)}={channels} {nameof(stride)}={stride}";

            if (!shaders.ContainsKey(key)) {
                shaders.Add(key, new Shaders.Pool.StrideUnpool2D(channels, stride));
            }

            Shader shader = shaders[key];

            if (stream == null) {
                stream = Shader.DefaultStream;
            }

            shader.Execute(stream, inmap, outmap, outwidth, outheight, batch);
        }

        /// <summary>3次元ストライド逆プール</summary>
        public static void StrideUnpool3D(uint channels, uint outwidth, uint outheight, uint outdepth,
                                          uint batch, uint stride,
                                          CudaArray<float> inmap, CudaArray<float> outmap,
                                          Stream stream = null) {

            string key = $"strideunpool_3d {nameof(channels)}={channels} {nameof(stride)}={stride}";

            if (!shaders.ContainsKey(key)) {
                shaders.Add(key, new Shaders.Pool.StrideUnpool3D(channels, stride));
            }

            Shader shader = shaders[key];

            if (stream == null) {
                stream = Shader.DefaultStream;
            }

            shader.Execute(stream, inmap, outmap, outwidth, outheight, outdepth, batch);
        }
    }
}