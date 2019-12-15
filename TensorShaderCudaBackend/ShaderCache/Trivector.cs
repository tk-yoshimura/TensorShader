using System;
using System.Collections.Generic;

namespace TensorShaderCudaBackend {

    /// <summary>3次元ベクトル</summary>
    public static class Trivector {
        private readonly static Dictionary<string, Shader> shaders = new Dictionary<string, Shader>();
        
        public static void Mul(uint vectorlength, CudaArray<float> src_vector, CudaArray<float> src_quaternion, CudaArray<float> dst_vector) {
            throw new NotImplementedException();
        }

        public static void MulQGrad(uint vectorlength, CudaArray<float> src_vector_value, CudaArray<float> src_vector_grad, CudaArray<float> src_quaternion, CudaArray<float> dst_quaternion) {
            throw new NotImplementedException();
        }

        public static void MulVGrad(uint vectorlength, CudaArray<float> src_vector, CudaArray<float> src_quaternion, CudaArray<float> dst_vector) {
            throw new NotImplementedException();
        }


        public static void Cross(uint length, CudaArray<float> src1, CudaArray<float> src2, CudaArray<float> dst) {
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


        public static void Cast(uint length, CudaArray<float> src_x, CudaArray<float> src_y, CudaArray<float> src_z, CudaArray<float> dst) {
            throw new NotImplementedException();
        }

        public static void X(uint length, CudaArray<float> src, CudaArray<float> dst_x) {
            throw new NotImplementedException();
        }

        public static void Y(uint length, CudaArray<float> src, CudaArray<float> dst_y) {
            throw new NotImplementedException();
        }

        public static void Z(uint length, CudaArray<float> src, CudaArray<float> dst_z) {
            throw new NotImplementedException();
        }

        public static void PureX(uint length, CudaArray<float> src_x, CudaArray<float> dst) {
            throw new NotImplementedException();
        }

        public static void PureY(uint length, CudaArray<float> src_y, CudaArray<float> dst) {
            throw new NotImplementedException();
        }

        public static void PureZ(uint length, CudaArray<float> src_z, CudaArray<float> dst) {
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
                                       CudaArray<float> inmap, CudaArray<float> outmap, CudaArray<float> kernel_value, CudaArray<float> kernel_grad) {
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
                                    CudaArray<float> inmap, CudaArray<float> outmap, CudaArray<float> kernel_value, CudaArray<float> kernel_grad) {
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
                                    CudaArray<float> inmap, CudaArray<float> outmap, CudaArray<float> kernel_value, CudaArray<float> kernel_grad) {
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
                                    CudaArray<float> inmap, CudaArray<float> outmap, CudaArray<float> kernel_value, CudaArray<float> kernel_grad) {
            throw new NotImplementedException();
        }
    } 
}