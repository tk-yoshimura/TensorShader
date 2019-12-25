using System;
using System.Collections.Generic;

namespace TensorShaderCudaBackend {

    /// <summary>複素数</summary>
    public static class Complex {
        private readonly static Dictionary<string, Shader> shaders = new Dictionary<string, Shader>();
        
        private static Shader UnaryArithmetric(string name, string func) {
            if (!shaders.ContainsKey(name)) {
                shaders.Add(name, new Shaders.Complex.Arithmetric.UnaryArithmetric(name, func));
            }

            return shaders[name];
        }

        private static Shader BinaryArithmetric(string name, string func) {
            if (!shaders.ContainsKey(name)) {
                shaders.Add(name, new Shaders.Complex.Arithmetric.BinaryArithmetric(name, func));
            }

            return shaders[name];
        }

        /// <summary>複素積</summary>
        public static void Mul(uint length, CudaArray<float> src1, CudaArray<float> src2, CudaArray<float> dst, Stream stream = null) {
            Shader shader = BinaryArithmetric(
                "complex_mul_ew", 
                "#y.x = #x1.x * #x2.x - #x1.y * #x2.y;" +
                "#y.y = #x1.x * #x2.y + #x1.y * #x2.x;"
            );
            
            if(stream == null) { 
                stream = Shader.DefaultStream;
            }
            
            shader.Execute(stream, src1, src2, dst, length);
        }

        /// <summary>複素積勾配</summary>
        public static void MulGrad(uint length, CudaArray<float> src1, CudaArray<float> src2, CudaArray<float> dst, Stream stream = null) {
            Shader shader = BinaryArithmetric(
                "complex_mulgrad_ew", 
                "#y.x = #x1.x * #x2.x + #x1.y * #x2.y;" +
                "#y.y = #x1.y * #x2.x - #x1.x * #x2.y;"
            );
            
            if(stream == null) { 
                stream = Shader.DefaultStream;
            }
            
            shader.Execute(stream, src1, src2, dst, length);
        }

        /// <summary>RRelu</summary>
        public static void RRelu(uint length, CudaArray<float> src, CudaArray<float> dst, Stream stream = null) {
            Shader shader = UnaryArithmetric(
                "complex_rrelu_ew", 
                "#y.x = fmaxf(0.0, #x.x);" +
                "#y.y = #x.y;"
            );
            
            if(stream == null) { 
                stream = Shader.DefaultStream;
            }
            
            shader.Execute(stream, src, dst, length);
        }

        /// <summary>RRelu勾配</summary>
        public static void RReluGrad(uint length, CudaArray<float> src1, CudaArray<float> src2, CudaArray<float> dst, Stream stream = null) {
            Shader shader = BinaryArithmetric(
                "complex_rrelugrad_ew", 
                "#y.x = #x2.x >= 0.0 ? #x1.x : 0.0;" +
                "#y.y = #x1.y;"
            );
            
            if(stream == null) { 
                stream = Shader.DefaultStream;
            }
            
            shader.Execute(stream, src1, src2, dst, length);
        }
        
        /// <summary>ZRelu</summary>
        public static void ZRelu(uint length, CudaArray<float> src, CudaArray<float> dst, Stream stream = null) {
            Shader shader = UnaryArithmetric(
                "complex_zrelu_ew", 
                "#y = #x.x >= 0.0 && #x.y >= 0.0 ? #x : ctor_float2(0.0, 0.0);"
            );
            
            if(stream == null) { 
                stream = Shader.DefaultStream;
            }
            
            shader.Execute(stream, src, dst, length);
        }

        /// <summary>ZRelu勾配</summary>
        public static void ZReluGrad(uint length, CudaArray<float> src1, CudaArray<float> src2, CudaArray<float> dst, Stream stream = null) {
            Shader shader = BinaryArithmetric(
                "complex_zrelugrad_ew", 
                "#y = #x2.x >= 0.0 && #x2.y >= 0.0 ? #x1 : ctor_float2(0.0, 0.0);"
            );
            
            if(stream == null) { 
                stream = Shader.DefaultStream;
            }
            
            shader.Execute(stream, src1, src2, dst, length);
        }
        
        /// <summary>複素共役</summary>
        public static void Conjugate(uint length, CudaArray<float> src, CudaArray<float> dst, Stream stream = null) {
            Shader shader = UnaryArithmetric(
                "complex_conj_ew", 
                "#y.x = #x.x;" +
                "#y.y = -#x.y;"
            );
            
            if(stream == null) { 
                stream = Shader.DefaultStream;
            }
            
            shader.Execute(stream, src, dst, length);
        }
        
        /// <summary>2乗</summary>
        public static void Square(uint length, CudaArray<float> src, CudaArray<float> dst, Stream stream = null) {
            Shader shader = UnaryArithmetric(
                "complex_squa_ew", 
                "#y.x = #x.x * #x.x - #x.y * #x.y;" +
                "#y.y = 2.0 * #x.x * #x.y;"
            );
            
            if(stream == null) { 
                stream = Shader.DefaultStream;
            }
            
            shader.Execute(stream, src, dst, length);
        }

        /// <summary>2乗勾配</summary>
        public static void SquareGrad(uint length, CudaArray<float> src1, CudaArray<float> src2, CudaArray<float> dst, Stream stream = null) {
            Shader shader = BinaryArithmetric(
                "complex_squagrad_ew", 
                "#y.x = 2.0 * (#x1.x * #x2.x + #x1.y * #x2.y);" +
                "#y.y = 2.0 * (#x1.y * #x2.x - #x1.x * #x2.y);"
            );
            
            if(stream == null) { 
                stream = Shader.DefaultStream;
            }
            
            shader.Execute(stream, src1, src2, dst, length);
        }
        
        /// <summary>複素数減衰</summary>
        public static void Decay(uint length, CudaArray<float> src, CudaArray<float> dst, Stream stream = null) {
            Shader shader = UnaryArithmetric(
                "complex_decay_ew", 
                "float norm = #x.x * #x.x + #x.y * #x.y;" +
                "float s = norm / (norm + 1.0);" + 
                "#y.x = #x.x * s;" +
                "#y.y = #x.y * s;"
            );
            
            if(stream == null) { 
                stream = Shader.DefaultStream;
            }
            
            shader.Execute(stream, src, dst, length);
        }

        /// <summary>複素数減衰勾配</summary>
        public static void DecayGrad(uint length, CudaArray<float> src1, CudaArray<float> src2, CudaArray<float> dst, Stream stream = null) {
            Shader shader = BinaryArithmetric(
                "complex_decaygrad_ew", 
                "float x12x = #x1.x * #x2.x, x12y = #x1.y * #x2.y;" +
                "float sx2x = #x2.x * #x2.x, sx2y = #x2.y * #x2.y;" +
                "float norm = sx2x + sx2y, norm_p1 = norm + 1;" +
                "float norm_norm_p1 = norm_p1 * norm, inv_squa_norm_p1 = 1.0 / (norm_p1 * norm_p1);" +
                "#y.x = (#x1.x * (2.0 * sx2x + norm_norm_p1) + 2.0 * #x2.x * x12y) * inv_squa_norm_p1;" + 
                "#y.y = (#x1.y * (2.0 * sx2y + norm_norm_p1) + 2.0 * #x2.y * x12x) * inv_squa_norm_p1;"
            );
            
            if(stream == null) { 
                stream = Shader.DefaultStream;
            }
            
            shader.Execute(stream, src1, src2, dst, length);
        }
        
        /// <summary>複素数Squash</summary>
        public static void Squash(uint length, CudaArray<float> src, CudaArray<float> dst, Stream stream = null) {
            Shader shader = UnaryArithmetric(
                "complex_squash_ew", 
                "float norm = #x.x * #x.x + #x.y * #x.y;" +
                "float s = 1.0 / (sqrtf(norm) + 1.0);" + 
                "#y.x = #x.x * s;" +
                "#y.y = #x.y * s;"
            );
            
            if(stream == null) { 
                stream = Shader.DefaultStream;
            }
            
            shader.Execute(stream, src, dst, length);
        }

        /// <summary>複素数Squash勾配</summary>
        public static void SquashGrad(uint length, CudaArray<float> src1, CudaArray<float> src2, CudaArray<float> dst, Stream stream = null) {  
            Shader shader = BinaryArithmetric(
                "complex_squashgrad_ew", 
                "float x12x = #x1.x * #x2.x, x12y = #x1.y * #x2.y;" +
                "float sx2x = #x2.x * #x2.x, sx2y = #x2.y * #x2.y;" +
                "float length = sqrtf(sx2x + sx2y), length_p1 = length + 1;" +
                "float length_length_p1 = length_p1 * length, inv_length_squa_length_p1 = 1.0 / (length * length_p1 * length_p1);" +
                "#y.x = (#x1.x * (length_length_p1 - sx2x) - #x2.x * x12y) * inv_length_squa_length_p1;" + 
                "#y.y = (#x1.y * (length_length_p1 - sx2y) - #x2.y * x12x) * inv_length_squa_length_p1;"
            );
            
            if(stream == null) { 
                stream = Shader.DefaultStream;
            }
            
            shader.Execute(stream, src1, src2, dst, length);
        }
        
        /// <summary>複素数正規化</summary>
        public static void Normalize(uint length, CudaArray<float> src, CudaArray<float> dst, Stream stream = null) {
            Shader shader = UnaryArithmetric(
                "complex_normalize_ew", 
                "float norm = #x.x * #x.x + #x.y * #x.y;" +
                "float s = 1.0 / fmaxf(sqrtf(norm), 1e-5);" + 
                "#y.x = #x.x * s;" +
                "#y.y = #x.y * s;"
            );
            
            if(stream == null) { 
                stream = Shader.DefaultStream;
            }
            
            shader.Execute(stream, src, dst, length);
        }

        /// <summary>複素数正規化勾配</summary>
        public static void NormalizeGrad(uint length, CudaArray<float> src1, CudaArray<float> src2, CudaArray<float> dst, Stream stream = null) {
            Shader shader = BinaryArithmetric(
                "complex_normalizegrad_ew", 
                "float x12x = #x1.x * #x2.x, x12y = #x1.y * #x2.y;" +
                "float sx2x = #x2.x * #x2.x, sx2y = #x2.y * #x2.y;" +
                "float length = fmaxf(sqrtf(sx2x + sx2y), 1e-5);" +
                "float inv_cube_length = 1.0 / (length * length * length);" +
                "#y.x = (#x1.x * sx2y - #x2.x * x12y) * inv_cube_length;" + 
                "#y.y = (#x1.y * sx2x - #x2.y * x12x) * inv_cube_length;"
            );
            
            if(stream == null) { 
                stream = Shader.DefaultStream;
            }
            
            shader.Execute(stream, src1, src2, dst, length);
        }

        /// <summary>実数から複素数へキャスト</summary>
        public static void Cast(uint length, CudaArray<float> src_real, CudaArray<float> src_imag, CudaArray<float> dst, Stream stream = null) {
            string key = "complex_cast";
            
            if (!shaders.ContainsKey(key)) {
                shaders.Add(key, new Shaders.ArrayManipulation.Gather(arrays: 2));
            }

            Shader shader = shaders[key];
                        
            if(stream == null) { 
                stream = Shader.DefaultStream;
            }
            
            shader.Execute(stream, src_real, src_imag, dst, length);
        }

        /// <summary>複素実部</summary>
        public static void Real(uint length, CudaArray<float> src, CudaArray<float> dst_real, Stream stream = null) {
            string key = "complex_real";
            
            if (!shaders.ContainsKey(key)) {
                shaders.Add(key, new Shaders.ArrayManipulation.Scatter(arrays: 2, index: 0));
            }

            Shader shader = shaders[key];
                        
            if(stream == null) { 
                stream = Shader.DefaultStream;
            }
            
            shader.Execute(stream, src, dst_real, length);
        }

        /// <summary>複素虚部</summary>
        public static void Imag(uint length, CudaArray<float> src, CudaArray<float> dst_imag, Stream stream = null) {
            string key = "complex_imag";
            
            if (!shaders.ContainsKey(key)) {
                shaders.Add(key, new Shaders.ArrayManipulation.Scatter(arrays: 2, index: 1));
            }

            Shader shader = shaders[key];

            if(stream == null) { 
                stream = Shader.DefaultStream;
            }
            
            shader.Execute(stream, src, dst_imag, length);
        }

        /// <summary>純実部</summary>
        public static void PureReal(uint length, CudaArray<float> src_real, CudaArray<float> dst, Stream stream = null) {
            string key = "complex_purereal";
            
            if (!shaders.ContainsKey(key)) {
                shaders.Add(key, new Shaders.ArrayManipulation.StrideCopy(stride: 2, index: 0));
            }

            Shader shader = shaders[key];

            if(stream == null) { 
                stream = Shader.DefaultStream;
            }
            
            shader.Execute(stream, src_real, dst, length);
        }

        /// <summary>純虚部</summary>
        public static void PureImag(uint length, CudaArray<float> src_imag, CudaArray<float> dst, Stream stream = null) {
            string key = "complex_pureimag";
            
            if (!shaders.ContainsKey(key)) {
                shaders.Add(key, new Shaders.ArrayManipulation.StrideCopy(stride: 2, index: 1));
            }

            Shader shader = shaders[key];

            if(stream == null) { 
                stream = Shader.DefaultStream;
            }
            
            shader.Execute(stream, src_imag, dst, length);
        }

        /// <summary>全結合</summary>
        public static void Dense(uint inchannels, uint outchannels,
                                 uint batch, bool gradmode,
                                 CudaArray<float> inmap, CudaArray<float> kernel, CudaArray<float> outmap, 
                                 Stream stream = null) {

            string key = 
                $"complex_dense " +
                $"{nameof(inchannels)}={inchannels} {nameof(outchannels)}={outchannels} " +
                $"{nameof(gradmode)}={gradmode}";
            
            if (!shaders.ContainsKey(key)) {
                shaders.Add(key, new Shaders.Complex.Convolution.Dense(inchannels, outchannels, gradmode));
            }

            Shader shader = shaders[key];

            if(stream == null) { 
                stream = Shader.DefaultStream;
            }
            
            shader.Execute(stream, inmap, outmap, kernel, batch);
        }

        /// <summary>転置全結合</summary>
        public static void TransposeDense(uint inchannels, uint outchannels,
                                          uint batch, bool gradmode,
                                          CudaArray<float> inmap, CudaArray<float> kernel, CudaArray<float> outmap, 
                                          Stream stream = null) {

            string key = 
                $"complex_transpose_dense " +
                $"{nameof(inchannels)}={inchannels} {nameof(outchannels)}={outchannels} " +
                $"{nameof(gradmode)}={gradmode}";
            
            if (!shaders.ContainsKey(key)) {
                shaders.Add(key, new Shaders.Complex.Convolution.TransposeDense(inchannels, outchannels, gradmode));
            }

            Shader shader = shaders[key];

            if(stream == null) { 
                stream = Shader.DefaultStream;
            }
            
            shader.Execute(stream, inmap, outmap, kernel, batch);
        }

        /// <summary>カーネル積</summary>
        public static void KernelProductDense(uint inchannels, uint outchannels,
                                              uint batch, bool transpose,
                                              CudaArray<float> inmap, CudaArray<float> outmap, CudaArray<float> kernel,
                                              Stream stream = null) {

            string key = 
                $"complex_kernelproduct_dense " +
                $"{nameof(inchannels)}={inchannels} {nameof(outchannels)}={outchannels} " +
                $"{nameof(transpose)}={transpose}";
            
            if (!shaders.ContainsKey(key)) {
                shaders.Add(key, new Shaders.Complex.Convolution.KernelProductDense(inchannels, outchannels, transpose));
            }

            Shader shader = shaders[key];

            if(stream == null) { 
                stream = Shader.DefaultStream;
            }
            
            shader.Execute(stream, inmap, outmap, kernel, batch);
        }

        /// <summary>1次元畳み込み</summary>
        public static void Convolution1D(uint inchannels, uint outchannels, uint inwidth,
                                         uint batch, uint kwidth, bool gradmode,
                                         CudaArray<float> inmap, CudaArray<float> kernel, CudaArray<float> outmap,
                                         Stream stream = null) {

            string key = 
                $"complex_convolution_1d " +
                $"{nameof(inchannels)}={inchannels} {nameof(outchannels)}={outchannels} " +
                $"{nameof(kwidth)}={kwidth} " +
                $"{nameof(gradmode)}={gradmode}";
            
            if (!shaders.ContainsKey(key)) {
                shaders.Add(key, new Shaders.Complex.Convolution.Convolution1D(inchannels, outchannels, kwidth, gradmode));
            }

            Shader shader = shaders[key];

            if(stream == null) { 
                stream = Shader.DefaultStream;
            }
            
            shader.Execute(stream, inmap, outmap, kernel, inwidth, batch);
        }

        /// <summary>1次元逆畳み込み</summary>
        public static void Deconvolution1D(uint inchannels, uint outchannels, uint inwidth,
                                           uint batch, uint kwidth, bool gradmode,
                                           CudaArray<float> inmap, CudaArray<float> kernel, CudaArray<float> outmap, 
                                           Stream stream = null) {

            string key = 
                $"complex_deconvolution_1d " +
                $"{nameof(inchannels)}={inchannels} {nameof(outchannels)}={outchannels} " +
                $"{nameof(kwidth)}={kwidth} " +
                $"{nameof(gradmode)}={gradmode}";
            
            if (!shaders.ContainsKey(key)) {
                shaders.Add(key, new Shaders.Complex.Convolution.Deconvolution1D(inchannels, outchannels, kwidth, gradmode));
            }

            Shader shader = shaders[key];

            if(stream == null) { 
                stream = Shader.DefaultStream;
            }
            
            shader.Execute(stream, inmap, outmap, kernel, inwidth, batch);
        }

        /// <summary>カーネル積</summary>
        public static void KernelProduct1D(uint inchannels, uint outchannels, uint inwidth,
                                           uint batch, uint kwidth, bool transpose,
                                           CudaArray<float> inmap, CudaArray<float> outmap, CudaArray<float> kernel, 
                                           Stream stream = null) {

            string key = 
                $"complex_kernelproduct_1d " +
                $"{nameof(inchannels)}={inchannels} {nameof(outchannels)}={outchannels} " +
                $"{nameof(kwidth)}={kwidth} " +
                $"{nameof(transpose)}={transpose}";
            
            if (!shaders.ContainsKey(key)) {
                shaders.Add(key, new Shaders.Complex.Convolution.KernelProduct1D(inchannels, outchannels, kwidth, transpose));
            }

            Shader shader = shaders[key];

            if(stream == null) { 
                stream = Shader.DefaultStream;
            }
            
            shader.Execute(stream, inmap, outmap, kernel, inwidth, batch);
        }

        /// <summary>2次元畳み込み</summary>
        public static void Convolution2D(uint inchannels, uint outchannels, uint inwidth, uint inheight,
                                         uint batch, uint kwidth, uint kheight, bool gradmode,
                                         CudaArray<float> inmap, CudaArray<float> kernel, CudaArray<float> outmap, 
                                         Stream stream = null) {

            string key = 
                $"complex_convolution_2d " +
                $"{nameof(inchannels)}={inchannels} {nameof(outchannels)}={outchannels} " +
                $"{nameof(kwidth)}={kwidth} {nameof(kheight)}={kheight} " +
                $"{nameof(gradmode)}={gradmode}";
            
            if (!shaders.ContainsKey(key)) {
                shaders.Add(key, new Shaders.Complex.Convolution.Convolution2D(inchannels, outchannels, kwidth, kheight, gradmode));
            }

            Shader shader = shaders[key];

            if(stream == null) { 
                stream = Shader.DefaultStream;
            }
            
            shader.Execute(stream, inmap, outmap, kernel, inwidth, inheight, batch);
        }

        /// <summary>2次元逆畳み込み</summary>
        public static void Deconvolution2D(uint inchannels, uint outchannels, uint inwidth, uint inheight,
                                           uint batch, uint kwidth, uint kheight, bool gradmode,
                                           CudaArray<float> inmap, CudaArray<float> kernel, CudaArray<float> outmap, 
                                           Stream stream = null) {

            string key = 
                $"complex_deconvolution_2d " +
                $"{nameof(inchannels)}={inchannels} {nameof(outchannels)}={outchannels} " +
                $"{nameof(kwidth)}={kwidth} {nameof(kheight)}={kheight} " +
                $"{nameof(gradmode)}={gradmode}";
            
            if (!shaders.ContainsKey(key)) {
                shaders.Add(key, new Shaders.Complex.Convolution.Deconvolution2D(inchannels, outchannels, kwidth, kheight, gradmode));
            }

            Shader shader = shaders[key];

            if(stream == null) { 
                stream = Shader.DefaultStream;
            }
            
            shader.Execute(stream, inmap, outmap, kernel, inwidth, inheight, batch);
        }

        /// <summary>カーネル積</summary>
        public static void KernelProduct2D(uint inchannels, uint outchannels, uint inwidth, uint inheight,
                                           uint batch, uint kwidth, uint kheight, bool transpose,
                                           CudaArray<float> inmap, CudaArray<float> outmap, CudaArray<float> kernel, 
                                           Stream stream = null) {

            string key = 
                $"complex_kernelproduct_2d " +
                $"{nameof(inchannels)}={inchannels} {nameof(outchannels)}={outchannels} " +
                $"{nameof(kwidth)}={kwidth} {nameof(kheight)}={kheight} " +
                $"{nameof(transpose)}={transpose}";
            
            if (!shaders.ContainsKey(key)) {
                shaders.Add(key, new Shaders.Complex.Convolution.KernelProduct2D(inchannels, outchannels, kwidth, kheight, transpose));
            }

            Shader shader = shaders[key];

            if(stream == null) { 
                stream = Shader.DefaultStream;
            }
            
            shader.Execute(stream, inmap, outmap, kernel, inwidth, inheight, batch);
        }

        /// <summary>3次元畳み込み</summary>
        public static void Convolution3D(uint inchannels, uint outchannels, uint inwidth, uint inheight, uint indepth,
                                         uint batch, uint kwidth, uint kheight, uint kdepth, bool gradmode,
                                         CudaArray<float> inmap, CudaArray<float> kernel, CudaArray<float> outmap, 
                                         Stream stream = null) {

            string key = 
                $"complex_convolution_3d " +
                $"{nameof(inchannels)}={inchannels} {nameof(outchannels)}={outchannels} " +
                $"{nameof(kwidth)}={kwidth} {nameof(kheight)}={kheight} {nameof(kdepth)}={kdepth} " +
                $"{nameof(gradmode)}={gradmode}";
            
            if (!shaders.ContainsKey(key)) {
                shaders.Add(key, new Shaders.Complex.Convolution.Convolution3D(inchannels, outchannels, kwidth, kheight, kdepth, gradmode));
            }

            Shader shader = shaders[key];

            if(stream == null) { 
                stream = Shader.DefaultStream;
            }
            
            shader.Execute(stream, inmap, outmap, kernel, inwidth, inheight, indepth, batch);
        }

        /// <summary>3次元逆畳み込み</summary>
        public static void Deconvolution3D(uint inchannels, uint outchannels, uint inwidth, uint inheight, uint indepth,
                                           uint batch, uint kwidth, uint kheight, uint kdepth, bool gradmode,
                                           CudaArray<float> inmap, CudaArray<float> kernel, CudaArray<float> outmap, 
                                           Stream stream = null) {

            string key = 
                $"complex_deconvolution_3d " +
                $"{nameof(inchannels)}={inchannels} {nameof(outchannels)}={outchannels} " +
                $"{nameof(kwidth)}={kwidth} {nameof(kheight)}={kheight} {nameof(kdepth)}={kdepth} " +
                $"{nameof(gradmode)}={gradmode}";
            
            if (!shaders.ContainsKey(key)) {
                shaders.Add(key, new Shaders.Complex.Convolution.Deconvolution3D(inchannels, outchannels, kwidth, kheight, kdepth, gradmode));
            }

            Shader shader = shaders[key];

            if(stream == null) { 
                stream = Shader.DefaultStream;
            }
            
            shader.Execute(stream, inmap, outmap, kernel, inwidth, inheight, indepth, batch);
        }

        /// <summary>カーネル積</summary>
        public static void KernelProduct3D(uint inchannels, uint outchannels, uint inwidth, uint inheight, uint indepth,
                                           uint batch, uint kwidth, uint kheight, uint kdepth, bool transpose,
                                           CudaArray<float> inmap, CudaArray<float> outmap, CudaArray<float> kernel, 
                                           Stream stream = null) {

            string key = 
                $"complex_kernelproduct_3d " +
                $"{nameof(inchannels)}={inchannels} {nameof(outchannels)}={outchannels} " +
                $"{nameof(kwidth)}={kwidth} {nameof(kheight)}={kheight} {nameof(kdepth)}={kdepth} " +
                $"{nameof(transpose)}={transpose}";
            
            if (!shaders.ContainsKey(key)) {
                shaders.Add(key, new Shaders.Complex.Convolution.KernelProduct3D(inchannels, outchannels, kwidth, kheight, kdepth, transpose));
            }

            Shader shader = shaders[key];

            if(stream == null) { 
                stream = Shader.DefaultStream;
            }
            
            shader.Execute(stream, inmap, outmap, kernel, inwidth, inheight, indepth, batch);
        }
    } 
}