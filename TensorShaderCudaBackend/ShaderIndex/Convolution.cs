using System;
using System.Collections.Generic;

namespace TensorShaderCudaBackend {

    /// <summary>畳み込み</summary>
    public static class Convolution {
        private readonly static Dictionary<string, Shader> shaders = new Dictionary<string, Shader>();
        
        /// <summary>全結合</summary>
        public static void Dense(uint inchannels, uint outchannels,
                                 uint batch,
                                 CudaArray<float> inmap, CudaArray<float> kernel, CudaArray<float> outmap, 
                                 Stream stream = null) {

            string key = 
                $"dense " +
                $"{nameof(inchannels)}={inchannels} {nameof(outchannels)}={outchannels}";
            
            if (!shaders.ContainsKey(key)) {
                shaders.Add(key, new Shaders.Convolution.Dense(inchannels, outchannels));
            }

            Shader shader = shaders[key];

            if(stream == null) { 
                stream = Shader.DefaultStream;
            }
            
            shader.Execute(stream, inmap, outmap, kernel, batch);
        }

        /// <summary>転置全結合</summary>
        public static void TransposeDense(uint inchannels, uint outchannels,
                                          uint batch, 
                                          CudaArray<float> inmap, CudaArray<float> kernel, CudaArray<float> outmap, 
                                          Stream stream = null) {

            string key = 
                $"transpose_dense " +
                $"{nameof(inchannels)}={inchannels} {nameof(outchannels)}={outchannels}";
            
            if (!shaders.ContainsKey(key)) {
                shaders.Add(key, new Shaders.Convolution.TransposeDense(inchannels, outchannels));
            }

            Shader shader = shaders[key];

            if(stream == null) { 
                stream = Shader.DefaultStream;
            }
            
            shader.Execute(stream, inmap, outmap, kernel, batch);
        }

        /// <summary>カーネル積</summary>
        public static void KernelProductDense(uint inchannels, uint outchannels,
                                              uint batch,
                                              CudaArray<float> inmap, CudaArray<float> outmap, CudaArray<float> kernel, 
                                              Stream stream = null) {

            string key = 
                $"kernelproduct_dense " +
                $"{nameof(inchannels)}={inchannels} {nameof(outchannels)}={outchannels}";
            
            if (!shaders.ContainsKey(key)) {
                shaders.Add(key, new Shaders.Convolution.KernelProductDense(inchannels, outchannels));
            }

            Shader shader = shaders[key];

            if(stream == null) { 
                stream = Shader.DefaultStream;
            }
            
            shader.Execute(stream, inmap, outmap, kernel, batch);
        }

        /// <summary>1次元畳み込み</summary>
        public static void Convolution1D(uint inchannels, uint outchannels, uint inwidth,
                                         uint batch, uint kwidth, 
                                         CudaArray<float> inmap, CudaArray<float> kernel, CudaArray<float> outmap, 
                                         Stream stream = null) {

            string key = 
                $"convolution_1d " +
                $"{nameof(inchannels)}={inchannels} {nameof(outchannels)}={outchannels} " +
                $"{nameof(kwidth)}={kwidth}";
            
            if (!shaders.ContainsKey(key)) {
                shaders.Add(key, new Shaders.Convolution.Convolution1D(inchannels, outchannels, kwidth));
            }

            Shader shader = shaders[key];

            if(stream == null) { 
                stream = Shader.DefaultStream;
            }
            
            shader.Execute(stream, inmap, outmap, kernel, inwidth, batch);
        }

        /// <summary>1次元逆畳み込み</summary>
        public static void Deconvolution1D(uint inchannels, uint outchannels, uint inwidth,
                                           uint batch, uint kwidth, 
                                           CudaArray<float> inmap, CudaArray<float> kernel, CudaArray<float> outmap, 
                                           Stream stream = null) {

            string key =
                $"deconvolution_1d " +
                $"{nameof(inchannels)}={inchannels} {nameof(outchannels)}={outchannels} " +
                $"{nameof(kwidth)}={kwidth}";
            
            if (!shaders.ContainsKey(key)) {
                shaders.Add(key, new Shaders.Convolution.Deconvolution1D(inchannels, outchannels, kwidth));
            }

            Shader shader = shaders[key];

            if(stream == null) { 
                stream = Shader.DefaultStream;
            }
            
            shader.Execute(stream, inmap, outmap, kernel, inwidth, batch);
        }

        /// <summary>カーネル積</summary>
        public static void KernelProduct1D(uint inchannels, uint outchannels, uint inwidth,
                                           uint batch, uint kwidth, 
                                           CudaArray<float> inmap, CudaArray<float> outmap, CudaArray<float> kernel, 
                                           Stream stream = null) {

            string key = 
                $"kernelproduct_1d {nameof(inchannels)}={inchannels} {nameof(outchannels)}={outchannels} " +
                $"{nameof(kwidth)}={kwidth}";
            
            if (!shaders.ContainsKey(key)) {
                shaders.Add(key, new Shaders.Convolution.KernelProduct1D(inchannels, outchannels, kwidth));
            }

            Shader shader = shaders[key];

            if(stream == null) { 
                stream = Shader.DefaultStream;
            }
            
            shader.Execute(stream, inmap, outmap, kernel, inwidth, batch);
        }

        /// <summary>2次元畳み込み</summary>
        public static void Convolution2D(uint inchannels, uint outchannels, uint inwidth, uint inheight,
                                         uint batch, uint kwidth, uint kheight, 
                                         CudaArray<float> inmap, CudaArray<float> kernel, CudaArray<float> outmap, 
                                         Stream stream = null) {

            string key = 
                $"convolution_2d {nameof(inchannels)}={inchannels} {nameof(outchannels)}={outchannels} " +
                $"{nameof(kwidth)}={kwidth} {nameof(kheight)}={kheight}";
            
            if (!shaders.ContainsKey(key)) {
                shaders.Add(key, new Shaders.Convolution.Convolution2D(inchannels, outchannels, kwidth, kheight));
            }

            Shader shader = shaders[key];

            if(stream == null) { 
                stream = Shader.DefaultStream;
            }
            
            shader.Execute(stream, inmap, outmap, kernel, inwidth, inheight, batch);
        }

        /// <summary>2次元逆畳み込み</summary>
        public static void Deconvolution2D(uint inchannels, uint outchannels, uint inwidth, uint inheight,
                                           uint batch, uint kwidth, uint kheight, 
                                           CudaArray<float> inmap, CudaArray<float> kernel, CudaArray<float> outmap, 
                                           Stream stream = null) {

            string key = 
                $"deconvolution_2d {nameof(inchannels)}={inchannels} {nameof(outchannels)}={outchannels} " +
                $"{nameof(kwidth)}={kwidth} {nameof(kheight)}={kheight}";
            
            if (!shaders.ContainsKey(key)) {
                shaders.Add(key, new Shaders.Convolution.Deconvolution2D(inchannels, outchannels, kwidth, kheight));
            }

            Shader shader = shaders[key];

            if(stream == null) { 
                stream = Shader.DefaultStream;
            }
            
            shader.Execute(stream, inmap, outmap, kernel, inwidth, inheight, batch);
        }

        /// <summary>カーネル積</summary>
        public static void KernelProduct2D(uint inchannels, uint outchannels, uint inwidth, uint inheight,
                                           uint batch, uint kwidth, uint kheight, 
                                           CudaArray<float> inmap, CudaArray<float> outmap, CudaArray<float> kernel, 
                                           Stream stream = null) {

            string key = 
                $"kernelproduct_2d {nameof(inchannels)}={inchannels} {nameof(outchannels)}={outchannels} " +
                $"{nameof(kwidth)}={kwidth} {nameof(kheight)}={kheight}";
            
            if (!shaders.ContainsKey(key)) {
                shaders.Add(key, new Shaders.Convolution.KernelProduct2D(inchannels, outchannels, kwidth, kheight));
            }

            Shader shader = shaders[key];

            if(stream == null) { 
                stream = Shader.DefaultStream;
            }
            
            shader.Execute(stream, inmap, outmap, kernel, inwidth, inheight, batch);
        }

        /// <summary>3次元畳み込み</summary>
        public static void Convolution3D(uint inchannels, uint outchannels, uint inwidth, uint inheight, uint indepth,
                                         uint batch, uint kwidth, uint kheight, uint kdepth, 
                                         CudaArray<float> inmap, CudaArray<float> kernel, CudaArray<float> outmap, 
                                         Stream stream = null) {

            string key = 
                $"convolution_3d {nameof(inchannels)}={inchannels} {nameof(outchannels)}={outchannels} " +
                $"{nameof(kwidth)}={kwidth} {nameof(kheight)}={kheight} {nameof(kdepth)}={kdepth}";
            
            if (!shaders.ContainsKey(key)) {
                shaders.Add(key, new Shaders.Convolution.Convolution3D(inchannels, outchannels, kwidth, kheight, kdepth));
            }

            Shader shader = shaders[key];

            if(stream == null) { 
                stream = Shader.DefaultStream;
            }
            
            shader.Execute(stream, inmap, outmap, kernel, inwidth, inheight, indepth, batch);
        }

        /// <summary>3次元逆畳み込み</summary>
        public static void Deconvolution3D(uint inchannels, uint outchannels, uint inwidth, uint inheight, uint indepth,
                                           uint batch, uint kwidth, uint kheight, uint kdepth, 
                                           CudaArray<float> inmap, CudaArray<float> kernel, CudaArray<float> outmap, 
                                           Stream stream = null) {

            string key = 
                $"deconvolution_3d {nameof(inchannels)}={inchannels} {nameof(outchannels)}={outchannels} " +
                $"{nameof(kwidth)}={kwidth} {nameof(kheight)}={kheight} {nameof(kdepth)}={kdepth}";
            
            if (!shaders.ContainsKey(key)) {
                shaders.Add(key, new Shaders.Convolution.Deconvolution3D(inchannels, outchannels, kwidth, kheight, kdepth));
            }

            Shader shader = shaders[key];

            if(stream == null) { 
                stream = Shader.DefaultStream;
            }
            
            shader.Execute(stream, inmap, outmap, kernel, inwidth, inheight, indepth, batch);
        }

        /// <summary>カーネル積</summary>
        public static void KernelProduct3D(uint inchannels, uint outchannels, uint inwidth, uint inheight, uint indepth,
                                           uint batch, uint kwidth, uint kheight, uint kdepth, 
                                           CudaArray<float> inmap, CudaArray<float> outmap, CudaArray<float> kernel, 
                                           Stream stream = null) {

            string key = 
                $"kernelproduct_3d {nameof(inchannels)}={inchannels} {nameof(outchannels)}={outchannels} " +
                $"{nameof(kwidth)}={kwidth} {nameof(kheight)}={kheight} {nameof(kdepth)}={kdepth}";
            
            if (!shaders.ContainsKey(key)) {
                shaders.Add(key, new Shaders.Convolution.KernelProduct3D(inchannels, outchannels, kwidth, kheight, kdepth));
            }

            Shader shader = shaders[key];

            if(stream == null) { 
                stream = Shader.DefaultStream;
            }
            
            shader.Execute(stream, inmap, outmap, kernel, inwidth, inheight, indepth, batch);
        }

        /// <summary>チャネルごとの1次元畳み込み</summary>
        public static void ChannelwiseConvolution1D(uint channels, uint inwidth,
                                                    uint batch, uint kwidth, 
                                                    CudaArray<float> inmap, CudaArray<float> kernel, CudaArray<float> outmap, 
                                                    Stream stream = null) {

            string key = 
                $"chwise_convolution_1d {nameof(channels)}={channels} " +
                $"{nameof(kwidth)}={kwidth}";
            
            if (!shaders.ContainsKey(key)) {
                shaders.Add(key, new Shaders.Convolution.ChannelwiseConvolution1D(channels, kwidth));
            }

            Shader shader = shaders[key];

            if(stream == null) { 
                stream = Shader.DefaultStream;
            }
            
            shader.Execute(stream, inmap, outmap, kernel, inwidth, batch);
        }

        /// <summary>チャネルごとの1次元逆畳み込み</summary>
        public static void ChannelwiseDeconvolution1D(uint channels, uint inwidth,
                                                      uint batch, uint kwidth, 
                                                      CudaArray<float> inmap, CudaArray<float> kernel, CudaArray<float> outmap, 
                                                      Stream stream = null) {

            string key = 
                $"chwise_deconvolution_1d {nameof(channels)}={channels} " +
                $"{nameof(kwidth)}={kwidth}";
            
            if (!shaders.ContainsKey(key)) {
                shaders.Add(key, new Shaders.Convolution.ChannelwiseDeconvolution1D(channels, kwidth));
            }

            Shader shader = shaders[key];

            if(stream == null) { 
                stream = Shader.DefaultStream;
            }
            
            shader.Execute(stream, inmap, outmap, kernel, inwidth, batch);
        }

        /// <summary>カーネル積</summary>
        public static void ChannelwiseKernelProduct1D(uint channels, uint inwidth,
                                                      uint batch, uint kwidth, 
                                                      CudaArray<float> inmap, CudaArray<float> outmap, CudaArray<float> kernel, 
                                                      Stream stream = null) {

            string key = 
                $"chwise_kernelproduct_1d {nameof(channels)}={channels} " +
                $"{nameof(kwidth)}={kwidth}";
            
            if (!shaders.ContainsKey(key)) {
                shaders.Add(key, new Shaders.Convolution.ChannelwiseKernelProduct1D(channels, kwidth));
            }

            Shader shader = shaders[key];

            if(stream == null) { 
                stream = Shader.DefaultStream;
            }
            
            shader.Execute(stream, inmap, outmap, kernel, inwidth, batch);
        }

        /// <summary>チャネルごとの2次元畳み込み</summary>
        public static void ChannelwiseConvolution2D(uint channels, uint inwidth, uint inheight,
                                                    uint batch, uint kwidth, uint kheight, 
                                                    CudaArray<float> inmap, CudaArray<float> kernel, CudaArray<float> outmap, 
                                                    Stream stream = null) {

            string key = 
                $"chwise_convolution_2d {nameof(channels)}={channels} " +
                $"{nameof(kwidth)}={kwidth} {nameof(kheight)}={kheight}";
            
            if (!shaders.ContainsKey(key)) {
                shaders.Add(key, new Shaders.Convolution.ChannelwiseConvolution2D(channels, kwidth, kheight));
            }

            Shader shader = shaders[key];

            if(stream == null) { 
                stream = Shader.DefaultStream;
            }
            
            shader.Execute(stream, inmap, outmap, kernel, inwidth, inheight, batch);
        }

        /// <summary>チャネルごとの2次元逆畳み込み</summary>
        public static void ChannelwiseDeconvolution2D(uint channels, uint inwidth, uint inheight,
                                                      uint batch, uint kwidth, uint kheight, 
                                                      CudaArray<float> inmap, CudaArray<float> kernel, CudaArray<float> outmap, 
                                                      Stream stream = null) {
            
            string key = 
                $"chwise_deconvolution_2d {nameof(channels)}={channels} " +
                $"{nameof(kwidth)}={kwidth} {nameof(kheight)}={kheight}";
            
            if (!shaders.ContainsKey(key)) {
                shaders.Add(key, new Shaders.Convolution.ChannelwiseDeconvolution2D(channels, kwidth, kheight));
            }

            Shader shader = shaders[key];

            if(stream == null) { 
                stream = Shader.DefaultStream;
            }
            
            shader.Execute(stream, inmap, outmap, kernel, inwidth, inheight, batch);
        }

        /// <summary>カーネル積</summary>
        public static void ChannelwiseKernelProduct2D(uint channels, uint inwidth, uint inheight,
                                                      uint batch, uint kwidth, uint kheight, 
                                                      CudaArray<float> inmap, CudaArray<float> outmap, CudaArray<float> kernel, 
                                                      Stream stream = null) {
            
            string key = 
                $"chwise_kernelproduct_2d {nameof(channels)}={channels} " +
                $"{nameof(kwidth)}={kwidth} {nameof(kheight)}={kheight}";
            
            if (!shaders.ContainsKey(key)) {
                shaders.Add(key, new Shaders.Convolution.ChannelwiseKernelProduct2D(channels, kwidth, kheight));
            }

            Shader shader = shaders[key];

            if(stream == null) { 
                stream = Shader.DefaultStream;
            }
            
            shader.Execute(stream, inmap, outmap, kernel, inwidth, inheight, batch);
        }

        /// <summary>チャネルごとの3次元畳み込み</summary>
        public static void ChannelwiseConvolution3D(uint channels, uint inwidth, uint inheight, uint indepth,
                                                    uint batch, uint kwidth, uint kheight, uint kdepth, 
                                                    CudaArray<float> inmap, CudaArray<float> kernel, CudaArray<float> outmap, 
                                                    Stream stream = null) {

            string key = 
                $"chwise_convolution_3d {nameof(channels)}={channels} " +
                $"{nameof(kwidth)}={kwidth} {nameof(kheight)}={kheight} {nameof(kdepth)}={kdepth}";
            
            if (!shaders.ContainsKey(key)) {
                shaders.Add(key, new Shaders.Convolution.ChannelwiseConvolution3D(channels, kwidth, kheight, kdepth));
            }

            Shader shader = shaders[key];

            if(stream == null) { 
                stream = Shader.DefaultStream;
            }
            
            shader.Execute(stream, inmap, outmap, kernel, inwidth, inheight, indepth, batch);
        }

        /// <summary>チャネルごとの3次元逆畳み込み</summary>
        public static void ChannelwiseDeconvolution3D(uint channels, uint inwidth, uint inheight, uint indepth,
                                                      uint batch, uint kwidth, uint kheight, uint kdepth, 
                                                      CudaArray<float> inmap, CudaArray<float> kernel, CudaArray<float> outmap, 
                                                      Stream stream = null) {

            string key = 
                $"chwise_deconvolution_3d {nameof(channels)}={channels} " +
                $"{nameof(kwidth)}={kwidth} {nameof(kheight)}={kheight} {nameof(kdepth)}={kdepth}";
            
            if (!shaders.ContainsKey(key)) {
                shaders.Add(key, new Shaders.Convolution.ChannelwiseDeconvolution3D(channels, kwidth, kheight, kdepth));
            }

            Shader shader = shaders[key];

            if(stream == null) { 
                stream = Shader.DefaultStream;
            }
            
            shader.Execute(stream, inmap, outmap, kernel, inwidth, inheight, indepth, batch);
        }

        /// <summary>カーネル積</summary>
        public static void ChannelwiseKernelProduct3D(uint channels, uint inwidth, uint inheight, uint indepth,
                                                      uint batch, uint kwidth, uint kheight, uint kdepth, 
                                                      CudaArray<float> inmap, CudaArray<float> outmap, CudaArray<float> kernel, 
                                                      Stream stream = null) {

            string key = 
                $"chwise_kernelproduct_3d {nameof(channels)}={channels} " +
                $"{nameof(kwidth)}={kwidth} {nameof(kheight)}={kheight} {nameof(kdepth)}={kdepth}";
            
            if (!shaders.ContainsKey(key)) {
                shaders.Add(key, new Shaders.Convolution.ChannelwiseKernelProduct3D(channels, kwidth, kheight, kdepth));
            }

            Shader shader = shaders[key];

            if(stream == null) { 
                stream = Shader.DefaultStream;
            }
            
            shader.Execute(stream, inmap, outmap, kernel, inwidth, inheight, indepth, batch);
        }

        /// <summary>ポイントごとの畳み込み</summary>
        public static void PointwiseConvolution(uint inchannels, uint outchannels, uint points,
                                                uint batch, 
                                                CudaArray<float> inmap, CudaArray<float> kernel, CudaArray<float> outmap, 
                                                Stream stream = null) {

            string key = 
                $"ptwise_convolution_1d {nameof(inchannels)}={inchannels} {nameof(outchannels)}={outchannels}";
            
            if (!shaders.ContainsKey(key)) {
                shaders.Add(key, new Shaders.Convolution.PointwiseConvolution(inchannels, outchannels));
            }

            Shader shader = shaders[key];

            if(stream == null) { 
                stream = Shader.DefaultStream;
            }
            
            shader.Execute(stream, inmap, outmap, kernel, points * batch);
        }

        /// <summary>ポイントごとの逆畳み込み</summary>
        public static void PointwiseDeconvolution(uint inchannels, uint outchannels, uint points,
                                                  uint batch, 
                                                  CudaArray<float> inmap, CudaArray<float> kernel, CudaArray<float> outmap, 
                                                  Stream stream = null) {
            
            string key = 
                $"ptwise_deconvolution_1d {nameof(inchannels)}={inchannels} {nameof(outchannels)}={outchannels}";
            
            if (!shaders.ContainsKey(key)) {
                shaders.Add(key, new Shaders.Convolution.PointwiseDeconvolution(inchannels, outchannels));
            }

            Shader shader = shaders[key];

            if(stream == null) { 
                stream = Shader.DefaultStream;
            }
            
            shader.Execute(stream, inmap, outmap, kernel, points * batch);
        }

        /// <summary>カーネル積</summary>
        public static void PointwiseKernelProduct(uint inchannels, uint outchannels, uint points,
                                                  uint batch, 
                                                  CudaArray<float> inmap, CudaArray<float> outmap, CudaArray<float> kernel, 
                                                  Stream stream = null) {

            string key = 
                $"ptwise_kernelproduct_1d {nameof(inchannels)}={inchannels} {nameof(outchannels)}={outchannels}";
            
            if (!shaders.ContainsKey(key)) {
                shaders.Add(key, new Shaders.Convolution.PointwiseKernelProduct(inchannels, outchannels));
            }

            Shader shader = shaders[key];

            if(stream == null) { 
                stream = Shader.DefaultStream;
            }
            
            shader.Execute(stream, inmap, outmap, kernel, points * batch);
        }
    } 
}