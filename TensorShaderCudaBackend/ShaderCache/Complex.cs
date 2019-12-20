using System;
using System.Collections.Generic;

namespace TensorShaderCudaBackend {

    /// <summary>複素数</summary>
    public static class Complex {
        private readonly static Dictionary<string, Shader> shaders = new Dictionary<string, Shader>();
        
        private static Shader UnaryArithmetric(string name, string func) {
            if (!shaders.ContainsKey(name)) {
                shaders.Add(name, new Shaders.Complex.Elementwise.UnaryArithmetric(name, func));
            }

            return shaders[name];
        }

        private static Shader BinaryArithmetric(string name, string func) {
            if (!shaders.ContainsKey(name)) {
                shaders.Add(name, new Shaders.Complex.Elementwise.BinaryArithmetric(name, func));
            }

            return shaders[name];
        }

        /// <summary>複素積</summary>
        public static void Mul(uint length, CudaArray<float> src1, CudaArray<float> src2, CudaArray<float> dst) {
            Shader shader = BinaryArithmetric(
                "complex_mul_ew", 
                "#y.x = #x1.x * #x2.x - #x1.y * #x2.y;" +
                "#y.y = #x1.x * #x2.y + #x1.y * #x2.x;"
            );
            shader.Execute(Shader.DefaultStream, src1, src2, dst, length);
        }

        /// <summary>複素積勾配</summary>
        public static void MulGrad(uint length, CudaArray<float> src1, CudaArray<float> src2, CudaArray<float> dst) {
            Shader shader = BinaryArithmetric(
                "complex_mulgrad_ew", 
                "#y.x = #x1.x * #x2.x + #x1.y * #x2.y;" +
                "#y.y = #x1.y * #x2.x - #x1.x * #x2.y;"
            );
            shader.Execute(Shader.DefaultStream, src1, src2, dst, length);
        }

        /// <summary>RRelu</summary>
        public static void RRelu(uint length, CudaArray<float> src, CudaArray<float> dst) {
            Shader shader = UnaryArithmetric(
                "complex_relu_ew", 
                "#y.x = fmaxf(0.0, #x.x);" +
                "#y.y = #x.y;"
            );
            shader.Execute(Shader.DefaultStream, src, dst, length);
        }

        /// <summary>RRelu勾配</summary>
        public static void RReluGrad(uint length, CudaArray<float> src1, CudaArray<float> src2, CudaArray<float> dst) {
            Shader shader = BinaryArithmetric(
                "complex_relugrad_ew", 
                "#y.x = #x2.x >= 0 ? #x1.x : 0.0;" +
                "#y.y = #x1.y;"
            );
            shader.Execute(Shader.DefaultStream, src1, src2, dst, length);
        }
        
        public static void ZRelu(uint length, CudaArray<float> src, CudaArray<float> dst) {
            throw new NotImplementedException();
        }

        public static void ZReluGrad(uint length, CudaArray<float> src1, CudaArray<float> src2, CudaArray<float> dst) {
            throw new NotImplementedException();
        }
        
        public static void Conjugate(uint length, CudaArray<float> src, CudaArray<float> dst) {
            throw new NotImplementedException();
        }
        
        public static void Square(uint length, CudaArray<float> src, CudaArray<float> dst) {
            throw new NotImplementedException();
        }

        public static void SquareGrad(uint length, CudaArray<float> src1, CudaArray<float> src2, CudaArray<float> dst) {
            throw new NotImplementedException();
        }
        
        public static void Decay(uint length, CudaArray<float> src, CudaArray<float> dst) {
            throw new NotImplementedException();
        }

        public static void DecayGrad(uint length, CudaArray<float> src1, CudaArray<float> src2, CudaArray<float> dst) {
            throw new NotImplementedException();
        }
        
        public static void Squash(uint length, CudaArray<float> src, CudaArray<float> dst) {
            throw new NotImplementedException();
        }

        public static void SquashGrad(uint length, CudaArray<float> src1, CudaArray<float> src2, CudaArray<float> dst) {
            throw new NotImplementedException();
        }
        
        public static void Normalize(uint length, CudaArray<float> src, CudaArray<float> dst) {
            throw new NotImplementedException();
        }

        public static void NormalizeGrad(uint length, CudaArray<float> src1, CudaArray<float> src2, CudaArray<float> dst) {
            throw new NotImplementedException();
        }


        public static void Cast(uint length, CudaArray<float> src_real, CudaArray<float> src_imag, CudaArray<float> dst) {
            throw new NotImplementedException();
        }

        public static void Real(uint length, CudaArray<float> src, CudaArray<float> dst_real) {
            throw new NotImplementedException();
        }

        public static void Imag(uint length, CudaArray<float> src, CudaArray<float> dst_imag) {
            throw new NotImplementedException();
        }

        public static void PureReal(uint length, CudaArray<float> src_real, CudaArray<float> dst) {
            throw new NotImplementedException();
        }

        public static void PureImag(uint length, CudaArray<float> src_imag, CudaArray<float> dst) {
            throw new NotImplementedException();
        }


        public static void Dense(uint inchannels, uint outchannels,
                          uint batch, bool gradmode,
                          CudaArray<float> inmap, CudaArray<float> kernel, CudaArray<float> outmap) {
            throw new NotImplementedException();
        }


        public static void TransposeDense(uint inchannels, uint outchannels,
                                   uint batch, bool gradmode,
                                   CudaArray<float> inmap, CudaArray<float> kernel, CudaArray<float> outmap) {
            throw new NotImplementedException();
        }


        public static void KernelProductDense(uint inchannels, uint outchannels,
                                       uint batch, bool transpose,
                                       CudaArray<float> inmap, CudaArray<float> outmap, CudaArray<float> kernel) {
            throw new NotImplementedException();
        }


        public static void Convolution1D(uint inchannels, uint outchannels, uint inwidth,
                                  uint batch, uint kwidth, bool gradmode,
                                  CudaArray<float> inmap, CudaArray<float> kernel, CudaArray<float> outmap) {
            throw new NotImplementedException();
        }


        public static void Deconvolution1D(uint inchannels, uint outchannels, uint outwidth,
                                    uint batch, uint kwidth, bool gradmode,
                                    CudaArray<float> inmap, CudaArray<float> kernel, CudaArray<float> outmap) {
            throw new NotImplementedException();
        }


        public static void KernelProduct1D(uint inchannels, uint outchannels, uint inwidth,
                                    uint batch, uint kwidth, bool transpose,
                                    CudaArray<float> inmap, CudaArray<float> outmap, CudaArray<float> kernel) {
            throw new NotImplementedException();
        }


        public static void Convolution2D(uint inchannels, uint outchannels, uint inwidth, uint inheight,
                                  uint batch, uint kwidth, uint kheight, bool gradmode,
                                  CudaArray<float> inmap, CudaArray<float> kernel, CudaArray<float> outmap) {
            throw new NotImplementedException();
        }


        public static void Deconvolution2D(uint inchannels, uint outchannels, uint outwidth, uint outheight,
                                    uint batch, uint kwidth, uint kheight, bool gradmode,
                                    CudaArray<float> inmap, CudaArray<float> kernel, CudaArray<float> outmap) {
            throw new NotImplementedException();
        }


        public static void KernelProduct2D(uint inchannels, uint outchannels, uint inwidth, uint inheight,
                                    uint batch, uint kwidth, uint kheight, bool transpose,
                                    CudaArray<float> inmap, CudaArray<float> outmap, CudaArray<float> kernel) {
            throw new NotImplementedException();
        }


        public static void Convolution3D(uint inchannels, uint outchannels, uint inwidth, uint inheight, uint indepth,
                                  uint batch, uint kwidth, uint kheight, uint kdepth, bool gradmode,
                                  CudaArray<float> inmap, CudaArray<float> kernel, CudaArray<float> outmap) {
            throw new NotImplementedException();
        }


        public static void Deconvolution3D(uint inchannels, uint outchannels, uint outwidth, uint outheight, uint outdepth,
                                    uint batch, uint kwidth, uint kheight, uint kdepth, bool gradmode,
                                    CudaArray<float> inmap, CudaArray<float> kernel, CudaArray<float> outmap) {
            throw new NotImplementedException();
        }


        public static void KernelProduct3D(uint inchannels, uint outchannels, uint inwidth, uint inheight, uint indepth,
                                    uint batch, uint kwidth, uint kheight, uint kdepth, bool transpose,
                                    CudaArray<float> inmap, CudaArray<float> outmap, CudaArray<float> kernel) {
            throw new NotImplementedException();
        }
    } 
}