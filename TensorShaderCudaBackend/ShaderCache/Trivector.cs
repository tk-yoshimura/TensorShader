using System;
using System.Collections.Generic;

namespace TensorShaderCudaBackend {

    /// <summary>3次元ベクトル</summary>
    public static class Trivector {
        private readonly static Dictionary<string, Shader> shaders = new Dictionary<string, Shader>();

        private static Shader UnaryArithmetric(string name, string func) {
            if (!shaders.ContainsKey(name)) {
                shaders.Add(name, new Shaders.Trivector.Arithmetric.UnaryArithmetric(name, func));
            }

            return shaders[name];
        }

        private static Shader BinaryArithmetric(string name, string func) {
            if (!shaders.ContainsKey(name)) {
                shaders.Add(name, new Shaders.Trivector.Arithmetric.BinaryArithmetric(name, func));
            }

            return shaders[name];
        }
        
        /// <summary>回転積</summary>
        public static void Mul(uint vectorlength, CudaArray<float> src_vector, CudaArray<float> src_quaternion, CudaArray<float> dst_vector) {
            string key = "trivector_quaternion_mul";
            
            if (!shaders.ContainsKey(key)) {
                shaders.Add(key, new Shaders.Trivector.Arithmetric.QuaternionMul());
            }

            Shader shader = shaders[key];

            shader.Execute(Shader.DefaultStream, src_vector, src_quaternion, dst_vector, vectorlength);
        }

        /// <summary>回転積四元数勾配</summary>
        public static void MulQGrad(uint vectorlength, CudaArray<float> src_vector_value, CudaArray<float> src_vector_grad, CudaArray<float> src_quaternion, CudaArray<float> dst_quaternion) {
            string key = "trivector_quaternion_mulqgrad";
            
            if (!shaders.ContainsKey(key)) {
                shaders.Add(key, new Shaders.Trivector.Arithmetric.QuaternionMulQGrad());
            }

            Shader shader = shaders[key];

            shader.Execute(Shader.DefaultStream, src_vector_value, src_vector_grad, src_quaternion, dst_quaternion, vectorlength);
        }

        /// <summary>回転積ベクトル勾配</summary>
        public static void MulVGrad(uint vectorlength, CudaArray<float> src_vector, CudaArray<float> src_quaternion, CudaArray<float> dst_vector) {
            string key = "trivector_quaternion_mulvgrad";
            
            if (!shaders.ContainsKey(key)) {
                shaders.Add(key, new Shaders.Trivector.Arithmetric.QuaternionMulVGrad());
            }

            Shader shader = shaders[key];

            shader.Execute(Shader.DefaultStream, src_vector, src_quaternion, dst_vector, vectorlength);
        }

        /// <summary>外積</summary>
        public static void Cross(uint length, CudaArray<float> src1, CudaArray<float> src2, CudaArray<float> dst) {
            Shader shader = BinaryArithmetric(
                "trivector_cross_ew", 
                "#y.x = #x1.y * #x2.z - #x1.z * #x2.y;" + 
                "#y.y = #x1.z * #x2.x - #x1.x * #x2.z;" + 
                "#y.z = #x1.x * #x2.y - #x1.y * #x2.x;"
            );
            shader.Execute(Shader.DefaultStream, src1, src2, dst, length);
        }

        /// <summary>3次元ベクトル減衰</summary>
        public static void Decay(uint length, CudaArray<float> src, CudaArray<float> dst) {
            Shader shader = UnaryArithmetric(
                "trivector_decay_ew", 
                "float norm = #x.x * #x.x + #x.y * #x.y + #x.z * #x.z;" +
                "float s = norm / (norm + 1.0);" + 
                "#y.x = #x.x * s;" +
                "#y.y = #x.y * s;" + 
                "#y.z = #x.z * s;"
            );
            shader.Execute(Shader.DefaultStream, src, dst, length);
        }

        /// <summary>3次元ベクトル減衰勾配</summary>
        public static void DecayGrad(uint length, CudaArray<float> src1, CudaArray<float> src2, CudaArray<float> dst) {
            Shader shader = BinaryArithmetric(
                "trivector_decaygrad_ew", 
                "float x12x = #x1.x * #x2.x, x12y = #x1.y * #x2.y, x12z = #x1.z * #x2.z;" +
                "float sx2x = #x2.x * #x2.x, sx2y = #x2.y * #x2.y, sx2z = #x2.z * #x2.z;" +
                "float norm = sx2x + sx2y + sx2z, norm_p1 = norm + 1;" +
                "float norm_norm_p1 = norm_p1 * norm, inv_squa_norm_p1 = 1.0 / (norm_p1 * norm_p1);" +
                "#y.x = (#x1.x * (2.0 * sx2x + norm_norm_p1) + 2.0 * #x2.x * (x12y + x12z)) * inv_squa_norm_p1;" + 
                "#y.y = (#x1.y * (2.0 * sx2y + norm_norm_p1) + 2.0 * #x2.y * (x12z + x12x)) * inv_squa_norm_p1;" +
                "#y.z = (#x1.z * (2.0 * sx2z + norm_norm_p1) + 2.0 * #x2.z * (x12x + x12y)) * inv_squa_norm_p1;"
            );
            shader.Execute(Shader.DefaultStream, src1, src2, dst, length);
        }

        /// <summary>3次元ベクトルSquash</summary>
        public static void Squash(uint length, CudaArray<float> src, CudaArray<float> dst) {
            Shader shader = UnaryArithmetric(
                "trivector_squash_ew", 
                "float norm = #x.x * #x.x + #x.y * #x.y + #x.z * #x.z;" +
                "float s = 1.0 / (sqrtf(norm) + 1.0);" + 
                "#y.x = #x.x * s;" +
                "#y.y = #x.y * s;" +
                "#y.z = #x.z * s;"
            );
            shader.Execute(Shader.DefaultStream, src, dst, length);
        }

        /// <summary>3次元ベクトルSquash勾配</summary>
        public static void SquashGrad(uint length, CudaArray<float> src1, CudaArray<float> src2, CudaArray<float> dst) {
            Shader shader = BinaryArithmetric(
                "trivector_squashgrad_ew", 
                "float x12x = #x1.x * #x2.x, x12y = #x1.y * #x2.y, x12z = #x1.z * #x2.z;" +
                "float sx2x = #x2.x * #x2.x, sx2y = #x2.y * #x2.y, sx2z = #x2.z * #x2.z;" +
                "float length = sqrtf(sx2x + sx2y + sx2z), length_p1 = length + 1;" +
                "float length_length_p1 = length_p1 * length, inv_length_squa_length_p1 = 1.0 / (length * length_p1 * length_p1);" +
                "#y.x = (#x1.x * (length_length_p1 - sx2x) - #x2.x * (x12y + x12z)) * inv_length_squa_length_p1;" + 
                "#y.y = (#x1.y * (length_length_p1 - sx2y) - #x2.y * (x12z + x12x)) * inv_length_squa_length_p1;" +
                "#y.z = (#x1.z * (length_length_p1 - sx2z) - #x2.z * (x12x + x12y)) * inv_length_squa_length_p1;"
            );
            shader.Execute(Shader.DefaultStream, src1, src2, dst, length);
        }

        /// <summary>3次元ベクトル正規化</summary>
        public static void Normalize(uint length, CudaArray<float> src, CudaArray<float> dst) {
            Shader shader = UnaryArithmetric(
                "trivector_normalize_ew", 
                "float norm = #x.x * #x.x + #x.y * #x.y + #x.z * #x.z;" +
                "float s = 1.0 / fmaxf(sqrtf(norm), 1e-5);" + 
                "#y.x = #x.x * s;" +
                "#y.y = #x.y * s;" +
                "#y.z = #x.z * s;"
            );
            shader.Execute(Shader.DefaultStream, src, dst, length);
        }

        /// <summary>3次元ベクトル正規化勾配</summary>
        public static void NormalizeGrad(uint length, CudaArray<float> src1, CudaArray<float> src2, CudaArray<float> dst) {
            Shader shader = BinaryArithmetric(
                "trivector_normalizegrad_ew", 
                "float x12x = #x1.x * #x2.x, x12y = #x1.y * #x2.y, x12z = #x1.z * #x2.z;" +
                "float sx2x = #x2.x * #x2.x, sx2y = #x2.y * #x2.y, sx2z = #x2.z * #x2.z;" +
                "float length = fmaxf(sqrtf(sx2x + sx2y + sx2z), 1e-5);" +
                "float inv_cube_length = 1.0 / (length * length * length);" +
                "#y.x = (#x1.x * (sx2y + sx2z) - #x2.x * (x12y + x12z)) * inv_cube_length;" + 
                "#y.y = (#x1.y * (sx2z + sx2x) - #x2.y * (x12z + x12x)) * inv_cube_length;" + 
                "#y.z = (#x1.z * (sx2x + sx2y) - #x2.z * (x12x + x12y)) * inv_cube_length;"
            );
            shader.Execute(Shader.DefaultStream, src1, src2, dst, length);
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


        public static void Deconvolution1D(uint inchannels, uint outchannels, uint inwidth,
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


        public static void Deconvolution2D(uint inchannels, uint outchannels, uint inwidth, uint inheight,
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


        public static void Deconvolution3D(uint inchannels, uint outchannels, uint inwidth, uint inheight, uint indepth,
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