using System.Collections.Generic;

namespace TensorShaderCudaBackend {

    /// <summary>変形</summary>
    public static class Transform {
        private readonly static Dictionary<string, Shader> shaders = new Dictionary<string, Shader>();
        
        /// <summary>チャネル次元を空間方向に展開</summary>
        public static void ChannelToSpace2D(uint outchannels, uint inwidth, uint inheight, uint batch, uint scale, CudaArray<float> inmap, CudaArray<float> outmap) {
            string key = $"channel_to_space_2d outchannels={outchannels} scale={scale}";
            
            if (!shaders.ContainsKey(key)) {
                shaders.Add(key, new Shaders.Transform.ChannelToSpace2D(scale, outchannels));
            }

            Shader shader = shaders[key];

            shader.Execute(Shader.DefaultStream, inmap, outmap, inwidth, inheight, batch);
        }

        /// <summary>空間方向をチャネル次元に展開</summary>
        public static void SpaceToChannel2D(uint inchannels, uint outwidth, uint outheight, uint batch, uint scale, CudaArray<float> inmap, CudaArray<float> outmap) {
            string key = $"space_to_channel_2d inchannels={inchannels} scale={scale}";
            
            if (!shaders.ContainsKey(key)) {
                shaders.Add(key, new Shaders.Transform.SpaceToChannel2D(scale, inchannels));
            }

            Shader shader = shaders[key];

            shader.Execute(Shader.DefaultStream, inmap, outmap, outwidth, outheight, batch);
        }

        /// <summary>チャネル次元を空間方向に展開</summary>
        public static void ChannelToSpace3D(uint outchannels, uint inwidth, uint inheight, uint indepth, uint batch, uint scale, CudaArray<float> inmap, CudaArray<float> outmap) {
            string key = $"channel_to_space_3d outchannels={outchannels} scale={scale}";
            
            if (!shaders.ContainsKey(key)) {
                shaders.Add(key, new Shaders.Transform.ChannelToSpace3D(scale, outchannels));
            }

            Shader shader = shaders[key];

            shader.Execute(Shader.DefaultStream, inmap, outmap, inwidth, inheight, indepth, batch);
        }

        /// <summary>空間方向をチャネル次元に展開</summary>
        public static void SpaceToChannel3D(uint inchannels, uint outwidth, uint outheight, uint outdepth, uint batch, uint scale, CudaArray<float> inmap, CudaArray<float> outmap) {
            string key = $"space_to_channel_3d inchannels={inchannels} scale={scale}";
            
            if (!shaders.ContainsKey(key)) {
                shaders.Add(key, new Shaders.Transform.SpaceToChannel3D(scale, inchannels));
            }

            Shader shader = shaders[key];

            shader.Execute(Shader.DefaultStream, inmap, outmap, outwidth, outheight, outdepth, batch);
        }
    } 
}