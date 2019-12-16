using System;
using System.Collections.Generic;

namespace TensorShaderCudaBackend {

    /// <summary>畳み込み</summary>
    public static class Convolution {
        private readonly static Dictionary<string, Shader> shaders = new Dictionary<string, Shader>();
        
        /// <summary>全結合</summary>
        public static void Dense(uint inchannels, uint outchannels,
                                 uint batch,
                                 CudaArray<float> inmap, CudaArray<float> kernel, CudaArray<float> outmap) {

            string key = $"dense inchannels={inchannels} outchannels={outchannels}";
            
            if (!shaders.ContainsKey(key)) {
                shaders.Add(key, new Shaders.Convolution.Dense(inchannels, outchannels));
            }

            Shader shader = shaders[key];

            shader.Execute(Shader.DefaultStream, inmap, outmap, kernel, batch);
        }

        /// <summary>転置全結合</summary>
        public static void TransposeDense(uint inchannels, uint outchannels,
                                          uint batch, 
                                          CudaArray<float> inmap, CudaArray<float> kernel, CudaArray<float> outmap) {

            string key = $"transpose_dense inchannels={inchannels} outchannels={outchannels}";
            
            if (!shaders.ContainsKey(key)) {
                shaders.Add(key, new Shaders.Convolution.TransposeDense(inchannels, outchannels));
            }

            Shader shader = shaders[key];

            shader.Execute(Shader.DefaultStream, inmap, outmap, kernel, batch);
        }

        /// <summary>カーネル積</summary>
        public static void KernelProductDense(uint inchannels, uint outchannels,
                                              uint batch,
                                              CudaArray<float> inmap, CudaArray<float> outmap, CudaArray<float> kernel) {

            string key = $"kernelproduct_dense inchannels={inchannels} outchannels={outchannels}";
            
            if (!shaders.ContainsKey(key)) {
                shaders.Add(key, new Shaders.Convolution.KernelProductDense(inchannels, outchannels));
            }

            Shader shader = shaders[key];

            shader.Execute(Shader.DefaultStream, inmap, outmap, kernel, batch);
        }

        /// <summary>1次元畳み込み</summary>
        public static void Convolution1D(uint inchannels, uint outchannels, uint inwidth,
                                  uint batch, uint kwidth, 
                                  CudaArray<float> inmap, CudaArray<float> kernel, CudaArray<float> outmap) {

            string key = $"convolution1d inchannels={inchannels} outchannels={outchannels} kwidth={kwidth}";
            
            if (!shaders.ContainsKey(key)) {
                shaders.Add(key, new Shaders.Convolution.Convolution1D(inchannels, outchannels, kwidth));
            }

            Shader shader = shaders[key];

            shader.Execute(Shader.DefaultStream, inmap, outmap, kernel, inwidth, batch);
        }

        /// <summary>1次元逆畳み込み</summary>
        public static void Deconvolution1D(uint inchannels, uint outchannels, uint outwidth,
                                    uint batch, uint kwidth, 
                                    CudaArray<float> inmap, CudaArray<float> kernel, CudaArray<float> outmap) {

            string key = $"deconvolution1d inchannels={inchannels} outchannels={outchannels} kwidth={kwidth}";
            
            if (!shaders.ContainsKey(key)) {
                shaders.Add(key, new Shaders.Convolution.Deconvolution1D(inchannels, outchannels, kwidth));
            }

            Shader shader = shaders[key];

            shader.Execute(Shader.DefaultStream, inmap, outmap, kernel, outwidth, batch);
        }

        /// <summary>カーネル積</summary>
        public static void KernelProduct1D(uint inchannels, uint outchannels, uint inwidth,
                                    uint batch, uint kwidth, 
                                    CudaArray<float> inmap, CudaArray<float> outmap, CudaArray<float> kernel) {

            string key = $"kernelproduct_1d inchannels={inchannels} outchannels={outchannels} kwidth={kwidth}";
            
            if (!shaders.ContainsKey(key)) {
                shaders.Add(key, new Shaders.Convolution.KernelProduct1D(inchannels, outchannels, kwidth));
            }

            Shader shader = shaders[key];

            shader.Execute(Shader.DefaultStream, inmap, outmap, kernel, inwidth, batch);
        }

        /// <summary>2次元畳み込み</summary>
        public static void Convolution2D(uint inchannels, uint outchannels, uint inwidth, uint inheight,
                                         uint batch, uint kwidth, uint kheight, 
                                         CudaArray<float> inmap, CudaArray<float> kernel, CudaArray<float> outmap) {

            string key = $"convolution2d inchannels={inchannels} outchannels={outchannels} kwidth={kwidth} kheight={kheight}";
            
            if (!shaders.ContainsKey(key)) {
                shaders.Add(key, new Shaders.Convolution.Convolution2D(inchannels, outchannels, kwidth, kheight));
            }

            Shader shader = shaders[key];

            shader.Execute(Shader.DefaultStream, inmap, outmap, kernel, inwidth, inheight, batch);
        }

        /// <summary>2次元逆畳み込み</summary>
        public static void Deconvolution2D(uint inchannels, uint outchannels, uint outwidth, uint outheight,
                                    uint batch, uint kwidth, uint kheight, 
                                    CudaArray<float> inmap, CudaArray<float> kernel, CudaArray<float> outmap) {

            string key = $"deconvolution2d inchannels={inchannels} outchannels={outchannels} kwidth={kwidth} kheight={kheight}";
            
            if (!shaders.ContainsKey(key)) {
                shaders.Add(key, new Shaders.Convolution.Deconvolution2D(inchannels, outchannels, kwidth, kheight));
            }

            Shader shader = shaders[key];

            shader.Execute(Shader.DefaultStream, inmap, outmap, kernel, outwidth, outheight, batch);
        }

        /// <summary>カーネル積</summary>
        public static void KernelProduct2D(uint inchannels, uint outchannels, uint inwidth, uint inheight,
                                           uint batch, uint kwidth, uint kheight, 
                                           CudaArray<float> inmap, CudaArray<float> outmap, CudaArray<float> kernel) {

            string key = $"kernelproduct_2d inchannels={inchannels} outchannels={outchannels} kwidth={kwidth} kheight={kheight}";
            
            if (!shaders.ContainsKey(key)) {
                shaders.Add(key, new Shaders.Convolution.KernelProduct2D(inchannels, outchannels, kwidth, kheight));
            }

            Shader shader = shaders[key];

            shader.Execute(Shader.DefaultStream, inmap, outmap, kernel, inwidth, inheight, batch);
        }

        /// <summary>3次元畳み込み</summary>
        public static void Convolution3D(uint inchannels, uint outchannels, uint inwidth, uint inheight, uint indepth,
                                         uint batch, uint kwidth, uint kheight, uint kdepth, 
                                         CudaArray<float> inmap, CudaArray<float> kernel, CudaArray<float> outmap) {

            string key = $"convolution3d inchannels={inchannels} outchannels={outchannels} kwidth={kwidth} kheight={kheight} kdepth={kdepth}";
            
            if (!shaders.ContainsKey(key)) {
                shaders.Add(key, new Shaders.Convolution.Convolution3D(inchannels, outchannels, kwidth, kheight, kdepth));
            }

            Shader shader = shaders[key];

            shader.Execute(Shader.DefaultStream, inmap, outmap, kernel, inwidth, inheight, indepth, batch);
        }

        /// <summary>3次元逆畳み込み</summary>
        public static void Deconvolution3D(uint inchannels, uint outchannels, uint outwidth, uint outheight, uint outdepth,
                                           uint batch, uint kwidth, uint kheight, uint kdepth, 
                                           CudaArray<float> inmap, CudaArray<float> kernel, CudaArray<float> outmap) {

            string key = $"deconvolution3d inchannels={inchannels} outchannels={outchannels} kwidth={kwidth} kheight={kheight} kdepth={kdepth}";
            
            if (!shaders.ContainsKey(key)) {
                shaders.Add(key, new Shaders.Convolution.Deconvolution3D(inchannels, outchannels, kwidth, kheight, kdepth));
            }

            Shader shader = shaders[key];

            shader.Execute(Shader.DefaultStream, inmap, outmap, kernel, outwidth, outheight, outdepth, batch);
        }

        /// <summary>カーネル積</summary>
        public static void KernelProduct3D(uint inchannels, uint outchannels, uint inwidth, uint inheight, uint indepth,
                                           uint batch, uint kwidth, uint kheight, uint kdepth, 
                                           CudaArray<float> inmap, CudaArray<float> outmap, CudaArray<float> kernel) {

            string key = $"kernelproduct_3d inchannels={inchannels} outchannels={outchannels} kwidth={kwidth} kheight={kheight} kdepth={kdepth}";
            
            if (!shaders.ContainsKey(key)) {
                shaders.Add(key, new Shaders.Convolution.KernelProduct3D(inchannels, outchannels, kwidth, kheight, kdepth));
            }

            Shader shader = shaders[key];

            shader.Execute(Shader.DefaultStream, inmap, outmap, kernel, inwidth, inheight, indepth, batch);
        }


        public static void ChannelwiseConvolution1D(uint channels, uint inwidth,
                                             uint batch, uint th, uint kwidth, uint stride,
                                             CudaArray<float> inmap, CudaArray<float> kernel, CudaArray<float> outmap) {
            throw new NotImplementedException();
        }


        public static void ChannelwiseDeconvolution1D(uint channels, uint outwidth,
                                               uint batch, uint th, uint kwidth, uint stride,
                                               CudaArray<float> inmap, CudaArray<float> kernel, CudaArray<float> outmap) {
            throw new NotImplementedException();
        }


        public static void ChannelwiseKernelProduct1D(uint channels, uint inwidth,
                                               uint batch, uint kwidth, uint stride,
                                               CudaArray<float> inmap, CudaArray<float> outmap, CudaArray<float> kernel) {
            throw new NotImplementedException();
        }


        public static void ChannelwiseConvolution2D(uint channels, uint inwidth, uint inheight,
                                             uint batch, uint th, uint kwidth, uint kheight, uint stride,
                                             CudaArray<float> inmap, CudaArray<float> kernel, CudaArray<float> outmap) {
            throw new NotImplementedException();
        }


        public static void ChannelwiseDeconvolution2D(uint channels, uint outwidth, uint outheight,
                                               uint batch, uint th, uint kwidth, uint kheight, uint stride,
                                               CudaArray<float> inmap, CudaArray<float> kernel, CudaArray<float> outmap) {
            throw new NotImplementedException();
        }


        public static void ChannelwiseKernelProduct2D(uint channels, uint inwidth, uint inheight,
                                               uint batch, uint kwidth, uint kheight, uint stride,
                                               CudaArray<float> inmap, CudaArray<float> outmap, CudaArray<float> kernel) {
            throw new NotImplementedException();
        }


        public static void ChannelwiseConvolution3D(uint channels, uint inwidth, uint inheight, uint indepth,
                                             uint batch, uint th, uint kwidth, uint kheight, uint kdepth, uint stride,
                                             CudaArray<float> inmap, CudaArray<float> kernel, CudaArray<float> outmap) {
            throw new NotImplementedException();
        }


        public static void ChannelwiseDeconvolution3D(uint channels, uint outwidth, uint outheight, uint outdepth,
                                               uint batch, uint th, uint kwidth, uint kheight, uint kdepth, uint stride,
                                               CudaArray<float> inmap, CudaArray<float> kernel, CudaArray<float> outmap) {
            throw new NotImplementedException();
        }


        public static void ChannelwiseKernelProduct3D(uint channels, uint inwidth, uint inheight, uint indepth,
                                               uint batch, uint kwidth, uint kheight, uint kdepth, uint stride,
                                               CudaArray<float> inmap, CudaArray<float> outmap, CudaArray<float> kernel) {
            throw new NotImplementedException();
        }


        public static void PointwiseConvolution(uint inchannels, uint outchannels, uint points,
                                         uint batch, uint th,
                                         CudaArray<float> inmap, CudaArray<float> kernel, CudaArray<float> outmap) {
            throw new NotImplementedException();
        }


        public static void PointwiseDeconvolution(uint inchannels, uint outchannels, uint points,
                                           uint batch, uint th,
                                           CudaArray<float> inmap, CudaArray<float> kernel, CudaArray<float> outmap) {
            throw new NotImplementedException();
        }


        public static void PointwiseKernelProduct(uint inchannels, uint outchannels, uint points,
                                           uint batch, uint outch,
                                           CudaArray<float> inmap, CudaArray<float> outmap, CudaArray<float> kernel) {
            throw new NotImplementedException();
        }
    } 
}