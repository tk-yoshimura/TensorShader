using System;
using System.Collections.Generic;

namespace TensorShaderCudaBackend {

    /// <summary>畳み込み</summary>
    public static class Convolution {
        private readonly static Dictionary<string, Shader> shaders = new();

        /// <summary>全結合</summary>
        public static void Dense(uint inchannels, uint outchannels,
                                 uint batch,
                                 CudaArray<float> inmap, CudaArray<float> kernel, CudaArray<float> outmap,
                                 Stream stream = null) {

            string key = $"dense " +
                         $"{nameof(inchannels)}={inchannels} {nameof(outchannels)}={outchannels} " + 
                         $"precision={Environment.Precision}";

            if (!shaders.ContainsKey(key)) {
                if (Environment.Precision == Environment.PrecisionMode.Float) {
                    shaders.Add(key, new Shaders.Convolution.FloatPrecision.Dense(inchannels, outchannels));
                }
                else if (Environment.Precision == Environment.PrecisionMode.FloatFloat) {
                    shaders.Add(key, new Shaders.Convolution.FloatFloatPrecision.Dense(inchannels, outchannels));
                }
            }

            Shader shader = shaders[key];

            if (stream is null) {
                stream = Shader.DefaultStream;
            }

            shader.Execute(stream, inmap, outmap, kernel, batch);
        }

        /// <summary>転置全結合</summary>
        public static void TransposeDense(uint inchannels, uint outchannels,
                                          uint batch,
                                          CudaArray<float> inmap, CudaArray<float> kernel, CudaArray<float> outmap,
                                          Stream stream = null) {

            string key = $"transpose_dense " +
                         $"{nameof(inchannels)}={inchannels} {nameof(outchannels)}={outchannels} " + 
                         $"precision={Environment.Precision}";

            if (!shaders.ContainsKey(key)) {
                if (Environment.Precision == Environment.PrecisionMode.Float) {
                    shaders.Add(key, new Shaders.Convolution.FloatPrecision.TransposeDense(inchannels, outchannels));
                }
                else if (Environment.Precision == Environment.PrecisionMode.FloatFloat) {
                    shaders.Add(key, new Shaders.Convolution.FloatFloatPrecision.TransposeDense(inchannels, outchannels));
                }
            }

            Shader shader = shaders[key];

            if (stream is null) {
                stream = Shader.DefaultStream;
            }

            shader.Execute(stream, inmap, outmap, kernel, batch);
        }

        /// <summary>カーネル積</summary>
        public static void KernelProductDense(uint inchannels, uint outchannels,
                                              uint batch,
                                              CudaArray<float> inmap, CudaArray<float> outmap, CudaArray<float> kernel,
                                              Stream stream = null) {

            string key = $"kernelproduct_dense " +
                         $"{nameof(inchannels)}={inchannels} {nameof(outchannels)}={outchannels} " + 
                         $"precision={Environment.Precision}";

            if (!shaders.ContainsKey(key)) {
                if (Environment.Precision == Environment.PrecisionMode.Float) {
                    shaders.Add(key, new Shaders.Convolution.FloatPrecision.KernelProductDense(inchannels, outchannels));
                }
                else if (Environment.Precision == Environment.PrecisionMode.FloatFloat) {
                    shaders.Add(key, new Shaders.Convolution.FloatFloatPrecision.KernelProductDense(inchannels, outchannels));
                }
            }

            Shader shader = shaders[key];

            if (stream is null) {
                stream = Shader.DefaultStream;
            }

            shader.Execute(stream, inmap, outmap, kernel, batch);
        }

        /// <summary>1次元畳み込み</summary>
        public static void Convolution1D(uint inchannels, uint outchannels, uint inwidth,
                                         uint batch, uint kwidth,
                                         CudaArray<float> inmap, CudaArray<float> kernel, CudaArray<float> outmap,
                                         Stream stream = null) {

            string key = $"convolution_1d " +
                         $"{nameof(inchannels)}={inchannels} {nameof(outchannels)}={outchannels} " +
                         $"{nameof(kwidth)}={kwidth} " + 
                         $"precision={Environment.Precision}";

            if (!shaders.ContainsKey(key)) {
                if (Environment.Precision == Environment.PrecisionMode.Float) {
                    shaders.Add(key, new Shaders.Convolution.FloatPrecision.Convolution1D(inchannels, outchannels, kwidth));
                }
                else if (Environment.Precision == Environment.PrecisionMode.FloatFloat) {
                    shaders.Add(key, new Shaders.Convolution.FloatFloatPrecision.Convolution1D(inchannels, outchannels, kwidth));
                }
            }

            Shader shader = shaders[key];

            if (stream is null) {
                stream = Shader.DefaultStream;
            }

            shader.Execute(stream, inmap, outmap, kernel, inwidth, batch);
        }

        /// <summary>1次元逆畳み込み</summary>
        public static void Deconvolution1D(uint inchannels, uint outchannels, uint inwidth,
                                           uint batch, uint kwidth,
                                           CudaArray<float> inmap, CudaArray<float> kernel, CudaArray<float> outmap,
                                           Stream stream = null) {

            string key = $"deconvolution_1d " +
                         $"{nameof(inchannels)}={inchannels} {nameof(outchannels)}={outchannels} " +
                         $"{nameof(kwidth)}={kwidth} " + 
                         $"precision={Environment.Precision}";

            if (!shaders.ContainsKey(key)) {
                if (Environment.Precision == Environment.PrecisionMode.Float) {
                    shaders.Add(key, new Shaders.Convolution.FloatPrecision.Deconvolution1D(inchannels, outchannels, kwidth));
                }
                else if (Environment.Precision == Environment.PrecisionMode.FloatFloat) {
                    shaders.Add(key, new Shaders.Convolution.FloatFloatPrecision.Deconvolution1D(inchannels, outchannels, kwidth));
                }
            }

            Shader shader = shaders[key];

            if (stream is null) {
                stream = Shader.DefaultStream;
            }

            shader.Execute(stream, inmap, outmap, kernel, inwidth, batch);
        }

        /// <summary>カーネル積</summary>
        public static void KernelProduct1D(uint inchannels, uint outchannels, uint inwidth,
                                           uint batch, uint kwidth,
                                           CudaArray<float> inmap, CudaArray<float> outmap, CudaArray<float> kernel,
                                           Stream stream = null) {

            string key = $"kernelproduct_1d " +
                         $"{nameof(inchannels)}={inchannels} {nameof(outchannels)}={outchannels} " +
                         $"{nameof(kwidth)}={kwidth} " + 
                         $"precision={Environment.Precision}";

            if (!shaders.ContainsKey(key)) {
                if (Environment.Precision == Environment.PrecisionMode.Float) {
                    shaders.Add(key, new Shaders.Convolution.FloatPrecision.KernelProduct1D(inchannels, outchannels, kwidth));
                }
                else if (Environment.Precision == Environment.PrecisionMode.FloatFloat) {
                    shaders.Add(key, new Shaders.Convolution.FloatFloatPrecision.KernelProduct1D(inchannels, outchannels, kwidth));
                }
            }

            Shader shader = shaders[key];

            if (stream is null) {
                stream = Shader.DefaultStream;
            }

            shader.Execute(stream, inmap, outmap, kernel, inwidth, batch);
        }

        /// <summary>2次元畳み込み</summary>
        public static void Convolution2D(uint inchannels, uint outchannels, uint inwidth, uint inheight,
                                         uint batch, uint kwidth, uint kheight,
                                         CudaArray<float> inmap, CudaArray<float> kernel, CudaArray<float> outmap,
                                         Stream stream = null) {

            string key = $"convolution_2d " +
                         $"{nameof(inchannels)}={inchannels} {nameof(outchannels)}={outchannels} " +
                         $"{nameof(kwidth)}={kwidth} {nameof(kheight)}={kheight} " + 
                         $"precision={Environment.Precision}";

            if (!shaders.ContainsKey(key)) {
                if (Environment.Precision == Environment.PrecisionMode.Float) {
                    shaders.Add(key, new Shaders.Convolution.FloatPrecision.Convolution2D(inchannels, outchannels, kwidth, kheight));
                }
                else if (Environment.Precision == Environment.PrecisionMode.FloatFloat) {
                    shaders.Add(key, new Shaders.Convolution.FloatFloatPrecision.Convolution2D(inchannels, outchannels, kwidth, kheight));
                }
            }

            Shader shader = shaders[key];

            if (stream is null) {
                stream = Shader.DefaultStream;
            }

            shader.Execute(stream, inmap, outmap, kernel, inwidth, inheight, batch);
        }

        /// <summary>2次元逆畳み込み</summary>
        public static void Deconvolution2D(uint inchannels, uint outchannels, uint inwidth, uint inheight,
                                           uint batch, uint kwidth, uint kheight,
                                           CudaArray<float> inmap, CudaArray<float> kernel, CudaArray<float> outmap,
                                           Stream stream = null) {

            string key = $"deconvolution_2d " +
                         $"{nameof(inchannels)}={inchannels} {nameof(outchannels)}={outchannels} " +
                         $"{nameof(kwidth)}={kwidth} {nameof(kheight)}={kheight} " + 
                         $"precision={Environment.Precision}";

            if (!shaders.ContainsKey(key)) {
                if (Environment.Precision == Environment.PrecisionMode.Float) {
                    shaders.Add(key, new Shaders.Convolution.FloatPrecision.Deconvolution2D(inchannels, outchannels, kwidth, kheight));
                }
                else if (Environment.Precision == Environment.PrecisionMode.FloatFloat) {
                    shaders.Add(key, new Shaders.Convolution.FloatFloatPrecision.Deconvolution2D(inchannels, outchannels, kwidth, kheight));
                }
            }

            Shader shader = shaders[key];

            if (stream is null) {
                stream = Shader.DefaultStream;
            }

            shader.Execute(stream, inmap, outmap, kernel, inwidth, inheight, batch);
        }

        /// <summary>カーネル積</summary>
        public static void KernelProduct2D(uint inchannels, uint outchannels, uint inwidth, uint inheight,
                                           uint batch, uint kwidth, uint kheight,
                                           CudaArray<float> inmap, CudaArray<float> outmap, CudaArray<float> kernel,
                                           Stream stream = null) {

            string key = $"kernelproduct_2d " +
                         $"{nameof(inchannels)}={inchannels} {nameof(outchannels)}={outchannels} " +
                         $"{nameof(kwidth)}={kwidth} {nameof(kheight)}={kheight} " + 
                         $"precision={Environment.Precision}";

            if (!shaders.ContainsKey(key)) {
                if (Environment.Precision == Environment.PrecisionMode.Float) {
                    shaders.Add(key, new Shaders.Convolution.FloatPrecision.KernelProduct2D(inchannels, outchannels, kwidth, kheight));
                }
                else if (Environment.Precision == Environment.PrecisionMode.FloatFloat) {
                    shaders.Add(key, new Shaders.Convolution.FloatFloatPrecision.KernelProduct2D(inchannels, outchannels, kwidth, kheight));
                }
            }

            Shader shader = shaders[key];

            if (stream is null) {
                stream = Shader.DefaultStream;
            }

            shader.Execute(stream, inmap, outmap, kernel, inwidth, inheight, batch);
        }

        /// <summary>3次元畳み込み</summary>
        public static void Convolution3D(uint inchannels, uint outchannels, uint inwidth, uint inheight, uint indepth,
                                         uint batch, uint kwidth, uint kheight, uint kdepth,
                                         CudaArray<float> inmap, CudaArray<float> kernel, CudaArray<float> outmap,
                                         Stream stream = null) {

            string key = $"convolution_3d " +
                         $"{nameof(inchannels)}={inchannels} {nameof(outchannels)}={outchannels} " +
                         $"{nameof(kwidth)}={kwidth} {nameof(kheight)}={kheight} {nameof(kdepth)}={kdepth} " + 
                         $"precision={Environment.Precision}";

            if (!shaders.ContainsKey(key)) {
                if (Environment.Precision == Environment.PrecisionMode.Float) {
                    shaders.Add(key, new Shaders.Convolution.FloatPrecision.Convolution3D(inchannels, outchannels, kwidth, kheight, kdepth));
                }
                else if (Environment.Precision == Environment.PrecisionMode.FloatFloat) {
                    shaders.Add(key, new Shaders.Convolution.FloatFloatPrecision.Convolution3D(inchannels, outchannels, kwidth, kheight, kdepth));
                }
            }

            Shader shader = shaders[key];

            if (stream is null) {
                stream = Shader.DefaultStream;
            }

            shader.Execute(stream, inmap, outmap, kernel, inwidth, inheight, indepth, batch);
        }

        /// <summary>3次元逆畳み込み</summary>
        public static void Deconvolution3D(uint inchannels, uint outchannels, uint inwidth, uint inheight, uint indepth,
                                           uint batch, uint kwidth, uint kheight, uint kdepth,
                                           CudaArray<float> inmap, CudaArray<float> kernel, CudaArray<float> outmap,
                                           Stream stream = null) {

            string key = $"deconvolution_3d " +
                         $"{nameof(inchannels)}={inchannels} {nameof(outchannels)}={outchannels} " +
                         $"{nameof(kwidth)}={kwidth} {nameof(kheight)}={kheight} {nameof(kdepth)}={kdepth} " + 
                         $"precision={Environment.Precision}";

            if (!shaders.ContainsKey(key)) {
                if (Environment.Precision == Environment.PrecisionMode.Float) {
                    shaders.Add(key, new Shaders.Convolution.FloatPrecision.Deconvolution3D(inchannels, outchannels, kwidth, kheight, kdepth));
                }
                else if (Environment.Precision == Environment.PrecisionMode.FloatFloat) {
                    shaders.Add(key, new Shaders.Convolution.FloatFloatPrecision.Deconvolution3D(inchannels, outchannels, kwidth, kheight, kdepth));
                }
            }

            Shader shader = shaders[key];

            if (stream is null) {
                stream = Shader.DefaultStream;
            }

            shader.Execute(stream, inmap, outmap, kernel, inwidth, inheight, indepth, batch);
        }

        /// <summary>カーネル積</summary>
        public static void KernelProduct3D(uint inchannels, uint outchannels, uint inwidth, uint inheight, uint indepth,
                                           uint batch, uint kwidth, uint kheight, uint kdepth,
                                           CudaArray<float> inmap, CudaArray<float> outmap, CudaArray<float> kernel,
                                           Stream stream = null) {

            string key = $"kernelproduct_3d " +
                         $"{nameof(inchannels)}={inchannels} {nameof(outchannels)}={outchannels} " +
                         $"{nameof(kwidth)}={kwidth} {nameof(kheight)}={kheight} {nameof(kdepth)}={kdepth} " + 
                         $"precision={Environment.Precision}";

            if (!shaders.ContainsKey(key)) {
                if (Environment.Precision == Environment.PrecisionMode.Float) {
                    shaders.Add(key, new Shaders.Convolution.FloatPrecision.KernelProduct3D(inchannels, outchannels, kwidth, kheight, kdepth));
                }
                else if (Environment.Precision == Environment.PrecisionMode.FloatFloat) {
                    shaders.Add(key, new Shaders.Convolution.FloatFloatPrecision.KernelProduct3D(inchannels, outchannels, kwidth, kheight, kdepth));
                }
            }

            Shader shader = shaders[key];

            if (stream is null) {
                stream = Shader.DefaultStream;
            }

            shader.Execute(stream, inmap, outmap, kernel, inwidth, inheight, indepth, batch);
        }

        /// <summary>チャネルごとの1次元畳み込み</summary>
        public static void ChannelwiseConvolution1D(uint channels, uint inwidth,
                                                    uint batch, uint kwidth,
                                                    CudaArray<float> inmap, CudaArray<float> kernel, CudaArray<float> outmap,
                                                    Stream stream = null) {

            string key = $"chwise_convolution_1d " +
                         $"{nameof(channels)}={channels} " +
                         $"{nameof(kwidth)}={kwidth} " + 
                         $"precision={Environment.Precision}";

            if (!shaders.ContainsKey(key)) {
                if (Environment.Precision == Environment.PrecisionMode.Float) {
                    shaders.Add(key, new Shaders.Convolution.FloatPrecision.ChannelwiseConvolution1D(channels, kwidth));
                }
                else if (Environment.Precision == Environment.PrecisionMode.FloatFloat) {
                    shaders.Add(key, new Shaders.Convolution.FloatFloatPrecision.ChannelwiseConvolution1D(channels, kwidth));
                }
            }

            Shader shader = shaders[key];

            if (stream is null) {
                stream = Shader.DefaultStream;
            }

            shader.Execute(stream, inmap, outmap, kernel, inwidth, batch);
        }

        /// <summary>チャネルごとの1次元逆畳み込み</summary>
        public static void ChannelwiseDeconvolution1D(uint channels, uint inwidth,
                                                      uint batch, uint kwidth,
                                                      CudaArray<float> inmap, CudaArray<float> kernel, CudaArray<float> outmap,
                                                      Stream stream = null) {

            string key = $"chwise_deconvolution_1d " +
                         $"{nameof(channels)}={channels} " +
                         $"{nameof(kwidth)}={kwidth} " + 
                         $"precision={Environment.Precision}";

            if (!shaders.ContainsKey(key)) {
                if (Environment.Precision == Environment.PrecisionMode.Float) {
                    shaders.Add(key, new Shaders.Convolution.FloatPrecision.ChannelwiseDeconvolution1D(channels, kwidth));
                }
                else if (Environment.Precision == Environment.PrecisionMode.FloatFloat) {
                    shaders.Add(key, new Shaders.Convolution.FloatFloatPrecision.ChannelwiseDeconvolution1D(channels, kwidth));
                }
            }

            Shader shader = shaders[key];

            if (stream is null) {
                stream = Shader.DefaultStream;
            }

            shader.Execute(stream, inmap, outmap, kernel, inwidth, batch);
        }

        /// <summary>カーネル積</summary>
        public static void ChannelwiseKernelProduct1D(uint channels, uint inwidth,
                                                      uint batch, uint kwidth,
                                                      CudaArray<float> inmap, CudaArray<float> outmap, CudaArray<float> kernel,
                                                      Stream stream = null) {

            string key = $"chwise_kernelproduct_1d " +
                         $"{nameof(channels)}={channels} " +
                         $"{nameof(kwidth)}={kwidth} " + 
                         $"precision={Environment.Precision}";

            if (!shaders.ContainsKey(key)) {
                if (Environment.Precision == Environment.PrecisionMode.Float) {
                    shaders.Add(key, new Shaders.Convolution.FloatPrecision.ChannelwiseKernelProduct1D(channels, kwidth));
                }
                else if (Environment.Precision == Environment.PrecisionMode.FloatFloat) {
                    shaders.Add(key, new Shaders.Convolution.FloatFloatPrecision.ChannelwiseKernelProduct1D(channels, kwidth));
                }
            }

            Shader shader = shaders[key];

            if (stream is null) {
                stream = Shader.DefaultStream;
            }

            shader.Execute(stream, inmap, outmap, kernel, inwidth, batch);
        }

        /// <summary>チャネルごとの2次元畳み込み</summary>
        public static void ChannelwiseConvolution2D(uint channels, uint inwidth, uint inheight,
                                                    uint batch, uint kwidth, uint kheight,
                                                    CudaArray<float> inmap, CudaArray<float> kernel, CudaArray<float> outmap,
                                                    Stream stream = null) {

            string key = $"chwise_convolution_2d " +
                         $"{nameof(channels)}={channels} " +
                         $"{nameof(kwidth)}={kwidth} {nameof(kheight)}={kheight} " + 
                         $"precision={Environment.Precision}";

            if (!shaders.ContainsKey(key)) {
                if (Environment.Precision == Environment.PrecisionMode.Float) {
                    shaders.Add(key, new Shaders.Convolution.FloatPrecision.ChannelwiseConvolution2D(channels, kwidth, kheight));
                }
                else if (Environment.Precision == Environment.PrecisionMode.FloatFloat) {
                    shaders.Add(key, new Shaders.Convolution.FloatFloatPrecision.ChannelwiseConvolution2D(channels, kwidth, kheight));
                }
            }

            Shader shader = shaders[key];

            if (stream is null) {
                stream = Shader.DefaultStream;
            }

            shader.Execute(stream, inmap, outmap, kernel, inwidth, inheight, batch);
        }

        /// <summary>チャネルごとの2次元逆畳み込み</summary>
        public static void ChannelwiseDeconvolution2D(uint channels, uint inwidth, uint inheight,
                                                      uint batch, uint kwidth, uint kheight,
                                                      CudaArray<float> inmap, CudaArray<float> kernel, CudaArray<float> outmap,
                                                      Stream stream = null) {

            string key = $"chwise_deconvolution_2d " +
                         $"{nameof(channels)}={channels} " +
                         $"{nameof(kwidth)}={kwidth} {nameof(kheight)}={kheight} " + 
                         $"precision={Environment.Precision}";

            if (!shaders.ContainsKey(key)) {
                if (Environment.Precision == Environment.PrecisionMode.Float) {
                    shaders.Add(key, new Shaders.Convolution.FloatPrecision.ChannelwiseDeconvolution2D(channels, kwidth, kheight));
                }
                else if (Environment.Precision == Environment.PrecisionMode.FloatFloat) {
                    shaders.Add(key, new Shaders.Convolution.FloatFloatPrecision.ChannelwiseDeconvolution2D(channels, kwidth, kheight));
                }
            }

            Shader shader = shaders[key];

            if (stream is null) {
                stream = Shader.DefaultStream;
            }

            shader.Execute(stream, inmap, outmap, kernel, inwidth, inheight, batch);
        }

        /// <summary>カーネル積</summary>
        public static void ChannelwiseKernelProduct2D(uint channels, uint inwidth, uint inheight,
                                                      uint batch, uint kwidth, uint kheight,
                                                      CudaArray<float> inmap, CudaArray<float> outmap, CudaArray<float> kernel,
                                                      Stream stream = null) {

            string key = $"chwise_kernelproduct_2d " +
                         $"{nameof(channels)}={channels} " +
                         $"{nameof(kwidth)}={kwidth} {nameof(kheight)}={kheight} " + 
                         $"precision={Environment.Precision}";

            if (!shaders.ContainsKey(key)) {
                if (Environment.Precision == Environment.PrecisionMode.Float) {
                    shaders.Add(key, new Shaders.Convolution.FloatPrecision.ChannelwiseKernelProduct2D(channels, kwidth, kheight));
                }
                else if (Environment.Precision == Environment.PrecisionMode.FloatFloat) {
                    shaders.Add(key, new Shaders.Convolution.FloatFloatPrecision.ChannelwiseKernelProduct2D(channels, kwidth, kheight));
                }
            }

            Shader shader = shaders[key];

            if (stream is null) {
                stream = Shader.DefaultStream;
            }

            shader.Execute(stream, inmap, outmap, kernel, inwidth, inheight, batch);
        }

        /// <summary>チャネルごとの3次元畳み込み</summary>
        public static void ChannelwiseConvolution3D(uint channels, uint inwidth, uint inheight, uint indepth,
                                                    uint batch, uint kwidth, uint kheight, uint kdepth,
                                                    CudaArray<float> inmap, CudaArray<float> kernel, CudaArray<float> outmap,
                                                    Stream stream = null) {

            string key = $"chwise_convolution_3d " +
                         $"{nameof(channels)}={channels} " +
                         $"{nameof(kwidth)}={kwidth} {nameof(kheight)}={kheight} {nameof(kdepth)}={kdepth} " + 
                         $"precision={Environment.Precision}";

            if (!shaders.ContainsKey(key)) {
                if (Environment.Precision == Environment.PrecisionMode.Float) {
                    shaders.Add(key, new Shaders.Convolution.FloatPrecision.ChannelwiseConvolution3D(channels, kwidth, kheight, kdepth));
                }
                else if (Environment.Precision == Environment.PrecisionMode.FloatFloat) {
                    shaders.Add(key, new Shaders.Convolution.FloatFloatPrecision.ChannelwiseConvolution3D(channels, kwidth, kheight, kdepth));
                }
            }

            Shader shader = shaders[key];

            if (stream is null) {
                stream = Shader.DefaultStream;
            }

            shader.Execute(stream, inmap, outmap, kernel, inwidth, inheight, indepth, batch);
        }

        /// <summary>チャネルごとの3次元逆畳み込み</summary>
        public static void ChannelwiseDeconvolution3D(uint channels, uint inwidth, uint inheight, uint indepth,
                                                      uint batch, uint kwidth, uint kheight, uint kdepth,
                                                      CudaArray<float> inmap, CudaArray<float> kernel, CudaArray<float> outmap,
                                                      Stream stream = null) {

            string key = $"chwise_deconvolution_3d " +
                         $"{nameof(channels)}={channels} " +
                         $"{nameof(kwidth)}={kwidth} {nameof(kheight)}={kheight} {nameof(kdepth)}={kdepth} " + 
                         $"precision={Environment.Precision}";

            if (!shaders.ContainsKey(key)) {
                if (Environment.Precision == Environment.PrecisionMode.Float) {
                    shaders.Add(key, new Shaders.Convolution.FloatPrecision.ChannelwiseDeconvolution3D(channels, kwidth, kheight, kdepth));
                }
                else if (Environment.Precision == Environment.PrecisionMode.FloatFloat) {
                    shaders.Add(key, new Shaders.Convolution.FloatFloatPrecision.ChannelwiseDeconvolution3D(channels, kwidth, kheight, kdepth));
                }
            }

            Shader shader = shaders[key];

            if (stream is null) {
                stream = Shader.DefaultStream;
            }

            shader.Execute(stream, inmap, outmap, kernel, inwidth, inheight, indepth, batch);
        }

        /// <summary>カーネル積</summary>
        public static void ChannelwiseKernelProduct3D(uint channels, uint inwidth, uint inheight, uint indepth,
                                                      uint batch, uint kwidth, uint kheight, uint kdepth,
                                                      CudaArray<float> inmap, CudaArray<float> outmap, CudaArray<float> kernel,
                                                      Stream stream = null) {

            string key = $"chwise_kernelproduct_3d " +
                         $"{nameof(channels)}={channels} " +
                         $"{nameof(kwidth)}={kwidth} {nameof(kheight)}={kheight} {nameof(kdepth)}={kdepth} " + 
                         $"precision={Environment.Precision}";

            if (!shaders.ContainsKey(key)) {
                if (Environment.Precision == Environment.PrecisionMode.Float) {
                    shaders.Add(key, new Shaders.Convolution.FloatPrecision.ChannelwiseKernelProduct3D(channels, kwidth, kheight, kdepth));
                }
                else if (Environment.Precision == Environment.PrecisionMode.FloatFloat) {
                    shaders.Add(key, new Shaders.Convolution.FloatFloatPrecision.ChannelwiseKernelProduct3D(channels, kwidth, kheight, kdepth));
                }
            }

            Shader shader = shaders[key];

            if (stream is null) {
                stream = Shader.DefaultStream;
            }

            shader.Execute(stream, inmap, outmap, kernel, inwidth, inheight, indepth, batch);
        }

        /// <summary>ポイントごとの畳み込み</summary>
        public static void PointwiseConvolution(uint inchannels, uint outchannels, uint points,
                                                uint batch,
                                                CudaArray<float> inmap, CudaArray<float> kernel, CudaArray<float> outmap,
                                                Stream stream = null) {

            string key = $"ptwise_convolution " +
                         $"{nameof(inchannels)}={inchannels} {nameof(outchannels)}={outchannels} " + 
                         $"precision={Environment.Precision}";

            if (!shaders.ContainsKey(key)) {
                if (Environment.Precision == Environment.PrecisionMode.Float) {
                    shaders.Add(key, new Shaders.Convolution.FloatPrecision.PointwiseConvolution(inchannels, outchannels));
                }
                else if (Environment.Precision == Environment.PrecisionMode.FloatFloat) {
                    shaders.Add(key, new Shaders.Convolution.FloatFloatPrecision.PointwiseConvolution(inchannels, outchannels));
                }
            }

            Shader shader = shaders[key];

            if (stream is null) {
                stream = Shader.DefaultStream;
            }

            shader.Execute(stream, inmap, outmap, kernel, points * batch);
        }

        /// <summary>ポイントごとの逆畳み込み</summary>
        public static void PointwiseDeconvolution(uint inchannels, uint outchannels, uint points,
                                                  uint batch,
                                                  CudaArray<float> inmap, CudaArray<float> kernel, CudaArray<float> outmap,
                                                  Stream stream = null) {

            string key = $"ptwise_deconvolution " +
                         $"{nameof(inchannels)}={inchannels} {nameof(outchannels)}={outchannels} " + 
                         $"precision={Environment.Precision}";

            if (!shaders.ContainsKey(key)) {
                if (Environment.Precision == Environment.PrecisionMode.Float) {
                    shaders.Add(key, new Shaders.Convolution.FloatPrecision.PointwiseDeconvolution(inchannels, outchannels));
                }
                else if (Environment.Precision == Environment.PrecisionMode.FloatFloat) {
                    shaders.Add(key, new Shaders.Convolution.FloatFloatPrecision.PointwiseDeconvolution(inchannels, outchannels));
                }
            }

            Shader shader = shaders[key];

            if (stream is null) {
                stream = Shader.DefaultStream;
            }

            shader.Execute(stream, inmap, outmap, kernel, points * batch);
        }

        /// <summary>カーネル積</summary>
        public static void PointwiseKernelProduct(uint inchannels, uint outchannels, uint points,
                                                  uint batch,
                                                  CudaArray<float> inmap, CudaArray<float> outmap, CudaArray<float> kernel,
                                                  Stream stream = null) {

            string key = $"ptwise_kernelproduct " +
                         $"{nameof(inchannels)}={inchannels} {nameof(outchannels)}={outchannels} " + 
                         $"precision={Environment.Precision}";

            if (!shaders.ContainsKey(key)) {
                if (Environment.Precision == Environment.PrecisionMode.Float) {
                    shaders.Add(key, new Shaders.Convolution.FloatPrecision.PointwiseKernelProduct(inchannels, outchannels));
                }
                else if (Environment.Precision == Environment.PrecisionMode.FloatFloat) {
                    shaders.Add(key, new Shaders.Convolution.FloatFloatPrecision.PointwiseKernelProduct(inchannels, outchannels));
                }
            }

            Shader shader = shaders[key];

            if (stream is null) {
                stream = Shader.DefaultStream;
            }

            shader.Execute(stream, inmap, outmap, kernel, points * batch);
        }
    }
}