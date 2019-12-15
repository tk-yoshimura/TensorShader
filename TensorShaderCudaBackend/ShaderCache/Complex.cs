using System;
using System.Collections.Generic;

namespace TensorShaderCudaBackend {

    /// <summary>複素数</summary>
    public static class Complex {
        private readonly static Dictionary<string, Shader> shaders = new Dictionary<string, Shader>();
        
        public static void Mul(uint length, CudaArray<float> src1, CudaArray<float> src2, CudaArray<float> dst) {
            throw new NotImplementedException();
        }

        public static void MulGrad(uint length, CudaArray<float> src1, CudaArray<float> src2, CudaArray<float> dst) {
            throw new NotImplementedException();
        }


        public static void RRelu(uint length, CudaArray<float> src, CudaArray<float> dst) {
            throw new NotImplementedException();
        }

        public static void RReluGrad(uint length, CudaArray<float> src1, CudaArray<float> src2, CudaArray<float> dst) {
            throw new NotImplementedException();
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
                          uint batch, uint th, bool gradmode,
                          CudaArray<float> inmap, CudaArray<float> kernel, CudaArray<float> outmap) {
            throw new NotImplementedException();
        }


        public static void TransposeDense(uint inchannels, uint outchannels,
                                   uint batch, uint th, bool gradmode,
                                   CudaArray<float> inmap, CudaArray<float> kernel, CudaArray<float> outmap) {
            throw new NotImplementedException();
        }


        public static void KernelProductDense(uint inchannels, uint outchannels,
                                       uint batch, uint outch, bool transpose,
                                       CudaArray<float> inmap, CudaArray<float> outmap, CudaArray<float> kernel) {
            throw new NotImplementedException();
        }


        public static void Convolution1D(uint inchannels, uint outchannels, uint inwidth,
                                  uint batch, uint th, uint kwidth, uint stride, bool gradmode,
                                  CudaArray<float> inmap, CudaArray<float> kernel, CudaArray<float> outmap) {
            throw new NotImplementedException();
        }


        public static void Deconvolution1D(uint inchannels, uint outchannels, uint outwidth,
                                    uint batch, uint th, uint kwidth, uint stride, bool gradmode,
                                    CudaArray<float> inmap, CudaArray<float> kernel, CudaArray<float> outmap) {
            throw new NotImplementedException();
        }


        public static void KernelProduct1D(uint inchannels, uint outchannels, uint inwidth,
                                    uint batch, uint outch, uint kwidth, uint stride, bool transpose,
                                    CudaArray<float> inmap, CudaArray<float> outmap, CudaArray<float> kernel) {
            throw new NotImplementedException();
        }


        public static void Convolution2D(uint inchannels, uint outchannels, uint inwidth, uint inheight,
                                  uint batch, uint th, uint kwidth, uint kheight, uint stride, bool gradmode,
                                  CudaArray<float> inmap, CudaArray<float> kernel, CudaArray<float> outmap) {
            throw new NotImplementedException();
        }


        public static void Deconvolution2D(uint inchannels, uint outchannels, uint outwidth, uint outheight,
                                    uint batch, uint th, uint kwidth, uint kheight, uint stride, bool gradmode,
                                    CudaArray<float> inmap, CudaArray<float> kernel, CudaArray<float> outmap) {
            throw new NotImplementedException();
        }


        public static void KernelProduct2D(uint inchannels, uint outchannels, uint inwidth, uint inheight,
                                    uint batch, uint outch, uint kwidth, uint kheight, uint stride, bool transpose,
                                    CudaArray<float> inmap, CudaArray<float> outmap, CudaArray<float> kernel) {
            throw new NotImplementedException();
        }


        public static void Convolution3D(uint inchannels, uint outchannels, uint inwidth, uint inheight, uint indepth,
                                  uint batch, uint th, uint kwidth, uint kheight, uint kdepth, uint stride, bool gradmode,
                                  CudaArray<float> inmap, CudaArray<float> kernel, CudaArray<float> outmap) {
            throw new NotImplementedException();
        }


        public static void Deconvolution3D(uint inchannels, uint outchannels, uint outwidth, uint outheight, uint outdepth,
                                    uint batch, uint th, uint kwidth, uint kheight, uint kdepth, uint stride, bool gradmode,
                                    CudaArray<float> inmap, CudaArray<float> kernel, CudaArray<float> outmap) {
            throw new NotImplementedException();
        }


        public static void KernelProduct3D(uint inchannels, uint outchannels, uint inwidth, uint inheight, uint indepth,
                                    uint batch, uint outch, uint kwidth, uint kheight, uint kdepth, uint stride, bool transpose,
                                    CudaArray<float> inmap, CudaArray<float> outmap, CudaArray<float> kernel) {
            throw new NotImplementedException();
        }
    } 
}