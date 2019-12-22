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

        /// <summary>実数から四元数へキャスト</summary>
        public static void Cast(uint length, CudaArray<float> src_r, CudaArray<float> src_i, CudaArray<float> src_j, CudaArray<float> src_k, CudaArray<float> dst) {
            string key = "quaternion_cast";
            
            if (!shaders.ContainsKey(key)) {
                shaders.Add(key, new Shaders.ArrayManipulation.Gather(arrays: 4));
            }

            Shader shader = shaders[key];

            shader.Execute(Shader.DefaultStream, src_r, src_i, src_j, src_k, dst, length);
        }

        /// <summary>R成分</summary>
        public static void R(uint length, CudaArray<float> src, CudaArray<float> dst_r) {
            string key = "quaternion_r";
            
            if (!shaders.ContainsKey(key)) {
                shaders.Add(key, new Shaders.ArrayManipulation.Scatter(arrays: 4, index: 0));
            }

            Shader shader = shaders[key];

            shader.Execute(Shader.DefaultStream, src, dst_r, length);
        }

        /// <summary>I成分</summary>
        public static void I(uint length, CudaArray<float> src, CudaArray<float> dst_i) {
            string key = "quaternion_i";
            
            if (!shaders.ContainsKey(key)) {
                shaders.Add(key, new Shaders.ArrayManipulation.Scatter(arrays: 4, index: 1));
            }

            Shader shader = shaders[key];

            shader.Execute(Shader.DefaultStream, src, dst_i, length);
        }

        /// <summary>J成分</summary>
        public static void J(uint length, CudaArray<float> src, CudaArray<float> dst_j) {
            string key = "quaternion_j";
            
            if (!shaders.ContainsKey(key)) {
                shaders.Add(key, new Shaders.ArrayManipulation.Scatter(arrays: 4, index: 2));
            }

            Shader shader = shaders[key];

            shader.Execute(Shader.DefaultStream, src, dst_j, length);
        }

        /// <summary>K成分</summary>
        public static void K(uint length, CudaArray<float> src, CudaArray<float> dst_k) {
            string key = "quaternion_k";
            
            if (!shaders.ContainsKey(key)) {
                shaders.Add(key, new Shaders.ArrayManipulation.Scatter(arrays: 4, index: 3));
            }

            Shader shader = shaders[key];

            shader.Execute(Shader.DefaultStream, src, dst_k, length);
        }

        /// <summary>純R成分</summary>
        public static void PureR(uint length, CudaArray<float> src_r, CudaArray<float> dst) {
            string key = "quaternion_purer";
            
            if (!shaders.ContainsKey(key)) {
                shaders.Add(key, new Shaders.ArrayManipulation.StrideCopy(stride: 4, index: 0));
            }

            Shader shader = shaders[key];

            shader.Execute(Shader.DefaultStream, src_r, dst, length);
        }

        /// <summary>純I成分</summary>
        public static void PureI(uint length, CudaArray<float> src_i, CudaArray<float> dst) {
            string key = "quaternion_purei";
            
            if (!shaders.ContainsKey(key)) {
                shaders.Add(key, new Shaders.ArrayManipulation.StrideCopy(stride: 4, index: 1));
            }

            Shader shader = shaders[key];

            shader.Execute(Shader.DefaultStream, src_i, dst, length);
        }

        /// <summary>純J成分</summary>
        public static void PureJ(uint length, CudaArray<float> src_j, CudaArray<float> dst) {
            string key = "quaternion_purej";
            
            if (!shaders.ContainsKey(key)) {
                shaders.Add(key, new Shaders.ArrayManipulation.StrideCopy(stride: 4, index: 2));
            }

            Shader shader = shaders[key];

            shader.Execute(Shader.DefaultStream, src_j, dst, length);
        }

        /// <summary>純K成分</summary>
        public static void PureK(uint length, CudaArray<float> src_k, CudaArray<float> dst) {
            string key = "quaternion_purek";
            
            if (!shaders.ContainsKey(key)) {
                shaders.Add(key, new Shaders.ArrayManipulation.StrideCopy(stride: 4, index: 3));
            }

            Shader shader = shaders[key];

            shader.Execute(Shader.DefaultStream, src_k, dst, length);
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