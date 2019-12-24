using System.Collections.Generic;

namespace TensorShaderCudaBackend {

    /// <summary>トリミング</summary>
    public static class Trimming {
        private readonly static Dictionary<string, Shader> shaders = new Dictionary<string, Shader>();
        
        /// <summary>1次元トリミング</summary>
        public static void Trimming1D(uint channels, uint outwidth,
                                      uint batch, 
                                      uint trim_left, uint trim_right,
                                      CudaArray<float> inmap, CudaArray<float> outmap) {

            string key = $"trimming_1d channels={channels} trim_left={trim_left} trim_right={trim_right}";
            
            if (!shaders.ContainsKey(key)) {
                shaders.Add(key, new Shaders.Trimming.Trimming1D(channels, trim_left, trim_right));
            }

            Shader shader = shaders[key];

            shader.Execute(Shader.DefaultStream, inmap, outmap, outwidth, batch);
        }

        /// <summary>2次元トリミング</summary>
        public static void Trimming2D(uint channels, uint outwidth, uint outheight,
                                      uint batch,
                                      uint trim_left, uint trim_right,
                                      uint trim_top, uint trim_bottom,
                                      CudaArray<float> inmap, CudaArray<float> outmap) {

            string key = 
                $"trimming_2d channels={channels} " +
                $"trim_left={trim_left} trim_right={trim_right} " +
                $"trim_top={trim_top} trim_bottom={trim_bottom}";
            
            if (!shaders.ContainsKey(key)) {
                shaders.Add(key, new Shaders.Trimming.Trimming2D(channels, trim_left, trim_right, trim_top, trim_bottom));
            }

            Shader shader = shaders[key];

            shader.Execute(Shader.DefaultStream, inmap, outmap, outwidth, outheight, batch);
        }

        /// <summary>3次元トリミング</summary>
        public static void Trimming3D(uint channels, uint outwidth, uint outheight, uint outdepth,
                                      uint batch, 
                                      uint trim_left, uint trim_right,
                                      uint trim_top, uint trim_bottom,
                                      uint trim_front, uint trim_rear,
                                      CudaArray<float> inmap, CudaArray<float> outmap) {

            string key = 
                $"trimming_3d channels={channels} " +
                $"trim_left={trim_left} trim_right={trim_right} " +
                $"trim_top={trim_top} trim_bottom={trim_bottom} " +
                $"trim_front={trim_front} trim_rear={trim_rear}";
            
            if (!shaders.ContainsKey(key)) {
                shaders.Add(key, new Shaders.Trimming.Trimming3D(channels, 
                    trim_left, trim_right, trim_top, trim_bottom, trim_front, trim_rear));
            }

            Shader shader = shaders[key];

            shader.Execute(Shader.DefaultStream, inmap, outmap, outwidth, outheight, outdepth, batch);
        }

    } 
}