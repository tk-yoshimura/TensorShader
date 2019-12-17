using System;
using System.Collections.Generic;

namespace TensorShaderCudaBackend {

    /// <summary>4元数</summary>
    public static class Quaternion {
        private readonly static Dictionary<string, Shader> shaders = new Dictionary<string, Shader>();
        
        public static void Mul(uint length, CudaArray<float> src1, CudaArray<float> src2, CudaArray<float> dst) {
            throw new NotImplementedException();
        }

        public static void MulGrad(uint length, CudaArray<float> src1, CudaArray<float> src2, CudaArray<float> dst) {
            throw new NotImplementedException();
        }

        public static void MulTransposeGrad(uint length, CudaArray<float> src1, CudaArray<float> src2, CudaArray<float> dst) {
            throw new NotImplementedException();
        }


        public static void RRelu(uint length, CudaArray<float> src, CudaArray<float> dst) {
            throw new NotImplementedException();
        }

        public static void RReluGrad(uint length, CudaArray<float> src1, CudaArray<float> src2, CudaArray<float> dst) {
            throw new NotImplementedException();
        }


        public static void Conjugate(uint length, CudaArray<float> src, CudaArray<float> dst) {
            throw new NotImplementedException();
        }


        public static void Square(uint length, CudaArray<float> src, CudaArray<float> dst) {
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


        public static void Cast(uint length, CudaArray<float> src_r, CudaArray<float> src_i, CudaArray<float> src_j, CudaArray<float> src_k, CudaArray<float> dst) {
            throw new NotImplementedException();
        }

        public static void R(uint length, CudaArray<float> src, CudaArray<float> dst_r) {
            throw new NotImplementedException();
        }

        public static void I(uint length, CudaArray<float> src, CudaArray<float> dst_i) {
            throw new NotImplementedException();
        }

        public static void J(uint length, CudaArray<float> src, CudaArray<float> dst_j) {
            throw new NotImplementedException();
        }

        public static void K(uint length, CudaArray<float> src, CudaArray<float> dst_k) {
            throw new NotImplementedException();
        }

        public static void PureR(uint length, CudaArray<float> src_r, CudaArray<float> dst) {
            throw new NotImplementedException();
        }

        public static void PureI(uint length, CudaArray<float> src_i, CudaArray<float> dst) {
            throw new NotImplementedException();
        }

        public static void PureJ(uint length, CudaArray<float> src_j, CudaArray<float> dst) {
            throw new NotImplementedException();
        }

        public static void PureK(uint length, CudaArray<float> src_k, CudaArray<float> dst) {
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