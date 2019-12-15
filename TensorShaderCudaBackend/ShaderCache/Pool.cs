using System.Collections.Generic;

namespace TensorShaderCudaBackend {

    /// <summary>プーリング</summary>
    public static class Pool {
        private readonly static Dictionary<string, Shader> shaders = new Dictionary<string, Shader>();
        
        /// <summary>1次元最大値プール</summary>
        public static void MaxPool1D(uint channels, uint inwidth, uint batch, uint stride, CudaArray<float> inmap, CudaArray<float> outmap) {
            string key = $"maxpool1d channels={channels} stride={stride}";
            
            if (!shaders.ContainsKey(key)) {
                shaders.Add(key, new Shaders.Pool.MaxPool1D(channels, stride));
            }

            Shader shader = shaders[key];

            shader.Execute(Shader.DefaultStream, inmap, outmap, inwidth, batch);
        }

        /// <summary>2次元最大値プール</summary>
        public static void MaxPool2D(uint channels, uint inwidth, uint inheight, uint batch, uint stride, CudaArray<float> inmap, CudaArray<float> outmap) {
            string key = $"maxpool2d channels={channels} stride={stride}";
            
            if (!shaders.ContainsKey(key)) {
                shaders.Add(key, new Shaders.Pool.MaxPool2D(channels, stride));
            }

            Shader shader = shaders[key];

            shader.Execute(Shader.DefaultStream, inmap, outmap, inwidth, inheight, batch);
        }

        /// <summary>3次元最大値プール</summary>
        public static void MaxPool3D(uint channels, uint inwidth, uint inheight, uint indepth, uint batch,  uint stride, CudaArray<float> inmap, CudaArray<float> outmap) {
            string key = $"maxpool3d channels={channels} stride={stride}";
            
            if (!shaders.ContainsKey(key)) {
                shaders.Add(key, new Shaders.Pool.MaxPool3D(channels, stride));
            }

            Shader shader = shaders[key];

            shader.Execute(Shader.DefaultStream, inmap, outmap, inwidth, inheight, indepth, batch);
        }

        /// <summary>1次元平均値プール</summary>
        public static void AveragePool1D(uint channels, uint inwidth, uint batch, uint stride, CudaArray<float> inmap, CudaArray<float> outmap) {
            string key = $"averagepool1d channels={channels} stride={stride}";
            
            if (!shaders.ContainsKey(key)) {
                shaders.Add(key, new Shaders.Pool.AveragePool1D(channels, stride));
            }

            Shader shader = shaders[key];

            shader.Execute(Shader.DefaultStream, inmap, outmap, inwidth, batch);
        }

        /// <summary>2次元平均値プール</summary>
        public static void AveragePool2D(uint channels, uint inwidth, uint inheight, uint batch, uint stride, CudaArray<float> inmap, CudaArray<float> outmap) {
            string key = $"averagepool2d channels={channels} stride={stride}";
            
            if (!shaders.ContainsKey(key)) {
                shaders.Add(key, new Shaders.Pool.AveragePool2D(channels, stride));
            }

            Shader shader = shaders[key];

            shader.Execute(Shader.DefaultStream, inmap, outmap, inwidth, inheight, batch);
        }

        /// <summary>3次元平均値プール</summary>
        public static void AveragePool3D(uint channels, uint inwidth, uint inheight, uint indepth, uint batch, uint stride, CudaArray<float> inmap, CudaArray<float> outmap) {
            string key = $"averagepool3d channels={channels} stride={stride}";
            
            if (!shaders.ContainsKey(key)) {
                shaders.Add(key, new Shaders.Pool.AveragePool3D(channels, stride));
            }

            Shader shader = shaders[key];

            shader.Execute(Shader.DefaultStream, inmap, outmap, inwidth, inheight, indepth, batch);
        }

        /// <summary>1次元最大値逆プール</summary>
        public static void MaxUnpool1D(uint channels, uint outwidth, uint batch, uint stride, CudaArray<float> ingrad, CudaArray<float> inpool, CudaArray<float> inmap, CudaArray<float> outmap) {
            string key = $"maxunpool1d channels={channels} stride={stride}";
            
            if (!shaders.ContainsKey(key)) {
                shaders.Add(key, new Shaders.Pool.MaxUnpool1D(channels, stride));
            }

            Shader shader = shaders[key];

            shader.Execute(Shader.DefaultStream, ingrad, inpool, inmap, outmap, outwidth, batch);
        }

        /// <summary>2次元最大値逆プール</summary>
        public static void MaxUnpool2D(uint channels, uint outwidth, uint outheight, uint batch, uint stride, CudaArray<float> ingrad, CudaArray<float> inpool, CudaArray<float> inmap, CudaArray<float> outmap) {
            string key = $"maxunpool2d channels={channels} stride={stride}";
            
            if (!shaders.ContainsKey(key)) {
                shaders.Add(key, new Shaders.Pool.MaxUnpool2D(channels, stride));
            }

            Shader shader = shaders[key];

            shader.Execute(Shader.DefaultStream, ingrad, inpool, inmap, outmap, outwidth, outheight, batch);
        }

        /// <summary>3次元最大値逆プール</summary>
        public static void MaxUnpool3D(uint channels, uint outwidth, uint outheight, uint outdepth, uint batch, uint stride, CudaArray<float> ingrad, CudaArray<float> inpool, CudaArray<float> inmap, CudaArray<float> outmap) {
            string key = $"maxunpool3d channels={channels} stride={stride}";
            
            if (!shaders.ContainsKey(key)) {
                shaders.Add(key, new Shaders.Pool.MaxUnpool3D(channels, stride));
            }

            Shader shader = shaders[key];

            shader.Execute(Shader.DefaultStream, ingrad, inpool, inmap, outmap, outwidth, outheight, outdepth, batch);
        }

        /// <summary>1次元平均値逆プール</summary>
        public static void AverageUnpool1D(uint channels, uint outwidth, uint batch, uint stride, CudaArray<float> inmap, CudaArray<float> outmap) {
            string key = $"averageunpool1d channels={channels} stride={stride}";
            
            if (!shaders.ContainsKey(key)) {
                shaders.Add(key, new Shaders.Pool.AverageUnpool1D(channels, stride));
            }

            Shader shader = shaders[key];

            shader.Execute(Shader.DefaultStream, inmap, outmap, outwidth, batch);
        }

        /// <summary>2次元平均値逆プール</summary>
        public static void AverageUnpool2D(uint channels, uint outwidth, uint outheight, uint batch, uint stride, CudaArray<float> inmap, CudaArray<float> outmap) {
            string key = $"averageunpool2d channels={channels} stride={stride}";
            
            if (!shaders.ContainsKey(key)) {
                shaders.Add(key, new Shaders.Pool.AverageUnpool2D(channels, stride));
            }

            Shader shader = shaders[key];

            shader.Execute(Shader.DefaultStream, inmap, outmap, outwidth, outheight, batch);
        }

        /// <summary>3次元平均値逆プール</summary>
        public static void AverageUnpool3D(uint channels, uint outwidth, uint outheight, uint outdepth, uint batch, uint stride, CudaArray<float> inmap, CudaArray<float> outmap) {
            string key = $"averageunpool3d channels={channels} stride={stride}";
            
            if (!shaders.ContainsKey(key)) {
                shaders.Add(key, new Shaders.Pool.AverageUnpool3D(channels, stride));
            }

            Shader shader = shaders[key];

            shader.Execute(Shader.DefaultStream, inmap, outmap, outwidth, outheight, outdepth, batch);
        }
    } 
}