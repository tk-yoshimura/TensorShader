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

        /// <summary>実数から3次元ベクトルへキャスト</summary>
        public static void Cast(uint length, CudaArray<float> src_x, CudaArray<float> src_y, CudaArray<float> src_z, CudaArray<float> dst) {
            string key = "trivector_cast";
            
            if (!shaders.ContainsKey(key)) {
                shaders.Add(key, new Shaders.ArrayManipulation.Gather(arrays: 3));
            }

            Shader shader = shaders[key];

            shader.Execute(Shader.DefaultStream, src_x, src_y, src_z, dst, length);
        }

        /// <summary>X成分</summary>
        public static void X(uint length, CudaArray<float> src, CudaArray<float> dst_x) {
            string key = "trivector_x";
            
            if (!shaders.ContainsKey(key)) {
                shaders.Add(key, new Shaders.ArrayManipulation.Scatter(arrays: 3, index: 0));
            }

            Shader shader = shaders[key];

            shader.Execute(Shader.DefaultStream, src, dst_x, length);
        }

        /// <summary>Y成分</summary>
        public static void Y(uint length, CudaArray<float> src, CudaArray<float> dst_y) {
            string key = "trivector_y";
            
            if (!shaders.ContainsKey(key)) {
                shaders.Add(key, new Shaders.ArrayManipulation.Scatter(arrays: 3, index: 1));
            }

            Shader shader = shaders[key];

            shader.Execute(Shader.DefaultStream, src, dst_y, length);
        }

        /// <summary>Z成分</summary>
        public static void Z(uint length, CudaArray<float> src, CudaArray<float> dst_z) {
            string key = "trivector_z";
            
            if (!shaders.ContainsKey(key)) {
                shaders.Add(key, new Shaders.ArrayManipulation.Scatter(arrays: 3, index: 2));
            }

            Shader shader = shaders[key];

            shader.Execute(Shader.DefaultStream, src, dst_z, length);
        }

        /// <summary>純X成分</summary>
        public static void PureX(uint length, CudaArray<float> src_x, CudaArray<float> dst) {
            string key = "trivector_purex";
            
            if (!shaders.ContainsKey(key)) {
                shaders.Add(key, new Shaders.ArrayManipulation.StrideCopy(stride: 3, index: 0));
            }

            Shader shader = shaders[key];

            shader.Execute(Shader.DefaultStream, src_x, dst, length);
        }

        /// <summary>純Y成分</summary>
        public static void PureY(uint length, CudaArray<float> src_y, CudaArray<float> dst) {
            string key = "trivector_purey";
            
            if (!shaders.ContainsKey(key)) {
                shaders.Add(key, new Shaders.ArrayManipulation.StrideCopy(stride: 3, index: 1));
            }

            Shader shader = shaders[key];

            shader.Execute(Shader.DefaultStream, src_y, dst, length);
        }

        /// <summary>純Z成分</summary>
        public static void PureZ(uint length, CudaArray<float> src_z, CudaArray<float> dst) {
            string key = "trivector_purez";
            
            if (!shaders.ContainsKey(key)) {
                shaders.Add(key, new Shaders.ArrayManipulation.StrideCopy(stride: 3, index: 2));
            }

            Shader shader = shaders[key];

            shader.Execute(Shader.DefaultStream, src_z, dst, length);
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
                                       CudaArray<float> inmap, CudaArray<float> outmap, CudaArray<float> kernel_value, CudaArray<float> kernel_grad) {
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
                                    CudaArray<float> inmap, CudaArray<float> outmap, CudaArray<float> kernel_value, CudaArray<float> kernel_grad) {
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
                                    CudaArray<float> inmap, CudaArray<float> outmap, CudaArray<float> kernel_value, CudaArray<float> kernel_grad) {
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
                                    CudaArray<float> inmap, CudaArray<float> outmap, CudaArray<float> kernel_value, CudaArray<float> kernel_grad) {
            throw new NotImplementedException();
        }
    } 
}