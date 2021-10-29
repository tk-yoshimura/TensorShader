﻿using System;
using System.Collections.Generic;

namespace TensorShaderCudaBackend {

    /// <summary>四元数</summary>
    public static class Quaternion {
        private readonly static Dictionary<string, Shader> shaders = new();

        private static Shader UnaryArithmetric(string name, string func) {
            if (!shaders.ContainsKey(name)) {
                shaders.Add(name, new Shaders.Quaternion.Arithmetric.UnaryArithmetric(name, func));
            }

            return shaders[name];
        }

        private static Shader BinaryArithmetric(string name, string func) {
            if (!shaders.ContainsKey(name)) {
                shaders.Add(name, new Shaders.Quaternion.Arithmetric.BinaryArithmetric(name, func));
            }

            return shaders[name];
        }

        /// <summary>四元数積</summary>
        public static void Mul(uint length, CudaArray<float> src1, CudaArray<float> src2, CudaArray<float> dst, Stream stream = null) {
            Shader shader = BinaryArithmetric(
                "quaternion_mul_ew",
                "#y.x = #x1.x * #x2.x - #x1.y * #x2.y - #x1.z * #x2.z - #x1.w * #x2.w;" +
                "#y.y = #x1.x * #x2.y + #x1.y * #x2.x + #x1.z * #x2.w - #x1.w * #x2.z;" +
                "#y.z = #x1.x * #x2.z - #x1.y * #x2.w + #x1.z * #x2.x + #x1.w * #x2.y;" +
                "#y.w = #x1.x * #x2.w + #x1.y * #x2.z - #x1.z * #x2.y + #x1.w * #x2.x;"
            );

            if (stream is null) {
                stream = Shader.DefaultStream;
            }

            shader.Execute(stream, src1, src2, dst, length);
        }

        /// <summary>四元数積勾配</summary>
        public static void MulGrad(uint length, CudaArray<float> src1, CudaArray<float> src2, CudaArray<float> dst, Stream stream = null) {
            Shader shader = BinaryArithmetric(
                "quaternion_mulgrad_ew",
                "#y.x = #x1.x * #x2.x + #x1.y * #x2.y + #x1.z * #x2.z + #x1.w * #x2.w;" +
                "#y.y = #x1.x * #x2.y - #x1.y * #x2.x + #x1.z * #x2.w - #x1.w * #x2.z;" +
                "#y.z = #x1.x * #x2.z - #x1.y * #x2.w - #x1.z * #x2.x + #x1.w * #x2.y;" +
                "#y.w = #x1.x * #x2.w + #x1.y * #x2.z - #x1.z * #x2.y - #x1.w * #x2.x;"
            );

            if (stream is null) {
                stream = Shader.DefaultStream;
            }

            shader.Execute(stream, src1, src2, dst, length);
        }

        /// <summary>四元数積転置勾配</summary>
        public static void MulTransposeGrad(uint length, CudaArray<float> src1, CudaArray<float> src2, CudaArray<float> dst, Stream stream = null) {
            Shader shader = BinaryArithmetric(
                "quaternion_multransposegrad_ew",
                "#y.x = #x1.x * #x2.x + #x1.y * #x2.y + #x1.z * #x2.z + #x1.w * #x2.w;" +
                "#y.y = #x1.x * #x2.y - #x1.y * #x2.x - #x1.z * #x2.w + #x1.w * #x2.z;" +
                "#y.z = #x1.x * #x2.z + #x1.y * #x2.w - #x1.z * #x2.x - #x1.w * #x2.y;" +
                "#y.w = #x1.x * #x2.w - #x1.y * #x2.z + #x1.z * #x2.y - #x1.w * #x2.x;"
            );

            if (stream is null) {
                stream = Shader.DefaultStream;
            }

            shader.Execute(stream, src1, src2, dst, length);
        }

        /// <summary>RRelu</summary>
        public static void RRelu(uint length, CudaArray<float> src, CudaArray<float> dst, Stream stream = null) {
            Shader shader = UnaryArithmetric(
                "quaternion_rrelu_ew",
                "#y.x = fmaxf(0.0, #x.x);" +
                "#y.y = #x.y;" +
                "#y.z = #x.z;" +
                "#y.w = #x.w;"
            );

            if (stream is null) {
                stream = Shader.DefaultStream;
            }

            shader.Execute(stream, src, dst, length);
        }

        /// <summary>RRelu勾配</summary>
        public static void RReluGrad(uint length, CudaArray<float> src1, CudaArray<float> src2, CudaArray<float> dst, Stream stream = null) {
            Shader shader = BinaryArithmetric(
                "quaternion_rrelugrad_ew",
                "#y.x = #x2.x >= 0.0 ? #x1.x : 0.0;" +
                "#y.y = #x1.y;" +
                "#y.z = #x1.z;" +
                "#y.w = #x1.w;"
            );

            if (stream is null) {
                stream = Shader.DefaultStream;
            }

            shader.Execute(stream, src1, src2, dst, length);
        }

        /// <summary>四元数共役</summary>
        public static void Conjugate(uint length, CudaArray<float> src, CudaArray<float> dst, Stream stream = null) {
            Shader shader = UnaryArithmetric(
                "quaternion_conj_ew",
                "#y.x = #x.x;" +
                "#y.y = -#x.y;" +
                "#y.z = -#x.z;" +
                "#y.w = -#x.w;"
            );

            if (stream is null) {
                stream = Shader.DefaultStream;
            }

            shader.Execute(stream, src, dst, length);
        }

        /// <summary>四元数2乗</summary>
        public static void Square(uint length, CudaArray<float> src, CudaArray<float> dst, Stream stream = null) {
            Shader shader = UnaryArithmetric(
                "quaternion_square_ew",
                "#y.x = #x.x * #x.x - #x.y * #x.y - #x.z * #x.z - #x.w * #x.w;" +
                "#y.y = ldexpf(#x.x * #x.y, 1);" +
                "#y.z = ldexpf(#x.x * #x.z, 1);" +
                "#y.w = ldexpf(#x.x * #x.w, 1);"
            );

            if (stream is null) {
                stream = Shader.DefaultStream;
            }

            shader.Execute(stream, src, dst, length);
        }

        /// <summary>四元数減衰</summary>
        public static void Decay(uint length, CudaArray<float> src, CudaArray<float> dst, Stream stream = null) {
            Shader shader = UnaryArithmetric(
                "quaternion_decay_ew",
                "float norm = #x.x * #x.x + #x.y * #x.y + #x.z * #x.z + #x.w * #x.w;" +
                "float s = norm / (norm + 1.0);" +
                "#y.x = #x.x * s;" +
                "#y.y = #x.y * s;" +
                "#y.z = #x.z * s;" +
                "#y.w = #x.w * s;"
            );

            if (stream is null) {
                stream = Shader.DefaultStream;
            }

            shader.Execute(stream, src, dst, length);
        }

        /// <summary>四元数減衰勾配</summary>
        public static void DecayGrad(uint length, CudaArray<float> src1, CudaArray<float> src2, CudaArray<float> dst, Stream stream = null) {
            Shader shader = BinaryArithmetric(
                "quaternion_decaygrad_ew",
                "float x12x = #x1.x * #x2.x, x12y = #x1.y * #x2.y, x12z = #x1.z * #x2.z, x12w = #x1.w * #x2.w;" +
                "float sx2x = #x2.x * #x2.x, sx2y = #x2.y * #x2.y, sx2z = #x2.z * #x2.z, sx2w = #x2.w * #x2.w;" +
                "float norm = sx2x + sx2y + sx2z + sx2w, norm_p1 = norm + 1;" +
                "float norm_norm_p1 = norm_p1 * norm, inv_squa_norm_p1 = 1.0 / (norm_p1 * norm_p1);" +
                "#y.x = (#x1.x * (ldexpf(sx2x, 1) + norm_norm_p1) + ldexpf(#x2.x * (x12y + x12z + x12w), 1)) * inv_squa_norm_p1;" +
                "#y.y = (#x1.y * (ldexpf(sx2y, 1) + norm_norm_p1) + ldexpf(#x2.y * (x12z + x12w + x12x), 1)) * inv_squa_norm_p1;" +
                "#y.z = (#x1.z * (ldexpf(sx2z, 1) + norm_norm_p1) + ldexpf(#x2.z * (x12w + x12x + x12y), 1)) * inv_squa_norm_p1;" +
                "#y.w = (#x1.w * (ldexpf(sx2w, 1) + norm_norm_p1) + ldexpf(#x2.w * (x12x + x12y + x12z), 1)) * inv_squa_norm_p1;"
            );

            if (stream is null) {
                stream = Shader.DefaultStream;
            }

            shader.Execute(stream, src1, src2, dst, length);
        }

        /// <summary>四元数Squash</summary>
        public static void Squash(uint length, CudaArray<float> src, CudaArray<float> dst, Stream stream = null) {
            Shader shader = UnaryArithmetric(
                "quaternion_squash_ew",
                "float norm = #x.x * #x.x + #x.y * #x.y + #x.z * #x.z + #x.w * #x.w;" +
                "float s = 1.0 / (sqrtf(norm) + 1.0);" +
                "#y.x = #x.x * s;" +
                "#y.y = #x.y * s;" +
                "#y.z = #x.z * s;" +
                "#y.w = #x.w * s;"
            );

            if (stream is null) {
                stream = Shader.DefaultStream;
            }

            shader.Execute(stream, src, dst, length);
        }

        /// <summary>四元数Squash勾配</summary>
        public static void SquashGrad(uint length, CudaArray<float> src1, CudaArray<float> src2, CudaArray<float> dst, Stream stream = null) {
            Shader shader = BinaryArithmetric(
                "quaternion_squashgrad_ew",
                "float x12x = #x1.x * #x2.x, x12y = #x1.y * #x2.y, x12z = #x1.z * #x2.z, x12w = #x1.w * #x2.w;" +
                "float sx2x = #x2.x * #x2.x, sx2y = #x2.y * #x2.y, sx2z = #x2.z * #x2.z, sx2w = #x2.w * #x2.w;" +
                "float length = sqrtf(sx2x + sx2y + sx2z + sx2w), length_p1 = length + 1;" +
                "float length_length_p1 = length_p1 * length, inv_length_squa_length_p1 = 1.0 / (length * length_p1 * length_p1);" +
                "#y.x = (#x1.x * (length_length_p1 - sx2x) - #x2.x * (x12y + x12z + x12w)) * inv_length_squa_length_p1;" +
                "#y.y = (#x1.y * (length_length_p1 - sx2y) - #x2.y * (x12z + x12w + x12x)) * inv_length_squa_length_p1;" +
                "#y.z = (#x1.z * (length_length_p1 - sx2z) - #x2.z * (x12w + x12x + x12y)) * inv_length_squa_length_p1;" +
                "#y.w = (#x1.w * (length_length_p1 - sx2w) - #x2.w * (x12x + x12y + x12z)) * inv_length_squa_length_p1;"
            );

            if (stream is null) {
                stream = Shader.DefaultStream;
            }

            shader.Execute(stream, src1, src2, dst, length);
        }

        /// <summary>四元数正規化</summary>
        public static void Normalize(uint length, CudaArray<float> src, CudaArray<float> dst, Stream stream = null) {
            Shader shader = UnaryArithmetric(
                "quaternion_normalize_ew",
                "float norm = #x.x * #x.x + #x.y * #x.y + #x.z * #x.z + #x.w * #x.w;" +
                "float s = 1.0 / fmaxf(sqrtf(norm), 1e-5);" +
                "#y.x = #x.x * s;" +
                "#y.y = #x.y * s;" +
                "#y.z = #x.z * s;" +
                "#y.w = #x.w * s;"
            );

            if (stream is null) {
                stream = Shader.DefaultStream;
            }

            shader.Execute(stream, src, dst, length);
        }

        /// <summary>四元数正規化勾配</summary>
        public static void NormalizeGrad(uint length, CudaArray<float> src1, CudaArray<float> src2, CudaArray<float> dst, Stream stream = null) {
            Shader shader = BinaryArithmetric(
                "quaternion_normalizegrad_ew",
                "float x12x = #x1.x * #x2.x, x12y = #x1.y * #x2.y, x12z = #x1.z * #x2.z, x12w = #x1.w * #x2.w;" +
                "float sx2x = #x2.x * #x2.x, sx2y = #x2.y * #x2.y, sx2z = #x2.z * #x2.z, sx2w = #x2.w * #x2.w;" +
                "float length = fmaxf(sqrtf(sx2x + sx2y + sx2z + sx2w), 1e-5);" +
                "float inv_cube_length = 1.0 / (length * length * length);" +
                "#y.x = (#x1.x * (sx2y + sx2z + sx2w) - #x2.x * (x12y + x12z + x12w)) * inv_cube_length;" +
                "#y.y = (#x1.y * (sx2z + sx2w + sx2x) - #x2.y * (x12z + x12w + x12x)) * inv_cube_length;" +
                "#y.z = (#x1.z * (sx2w + sx2x + sx2y) - #x2.z * (x12w + x12x + x12y)) * inv_cube_length;" +
                "#y.w = (#x1.w * (sx2x + sx2y + sx2z) - #x2.w * (x12x + x12y + x12z)) * inv_cube_length;"
            );

            if (stream is null) {
                stream = Shader.DefaultStream;
            }

            shader.Execute(stream, src1, src2, dst, length);
        }

        /// <summary>実数から四元数へキャスト</summary>
        public static void Cast(uint length, CudaArray<float> src_r, CudaArray<float> src_i, CudaArray<float> src_j, CudaArray<float> src_k, CudaArray<float> dst, Stream stream = null) {
            string key = "quaternion_cast";

            if (!shaders.ContainsKey(key)) {
                shaders.Add(key, new Shaders.ArrayManipulation.Gather(arrays: 4));
            }

            Shader shader = shaders[key];

            if (stream is null) {
                stream = Shader.DefaultStream;
            }

            shader.Execute(stream, src_r, src_i, src_j, src_k, dst, length);
        }

        /// <summary>R成分</summary>
        public static void R(uint length, CudaArray<float> src, CudaArray<float> dst_r, Stream stream = null) {
            string key = "quaternion_r";

            if (!shaders.ContainsKey(key)) {
                shaders.Add(key, new Shaders.ArrayManipulation.Scatter(arrays: 4, index: 0));
            }

            Shader shader = shaders[key];

            if (stream is null) {
                stream = Shader.DefaultStream;
            }

            shader.Execute(stream, src, dst_r, length);
        }

        /// <summary>I成分</summary>
        public static void I(uint length, CudaArray<float> src, CudaArray<float> dst_i, Stream stream = null) {
            string key = "quaternion_i";

            if (!shaders.ContainsKey(key)) {
                shaders.Add(key, new Shaders.ArrayManipulation.Scatter(arrays: 4, index: 1));
            }

            Shader shader = shaders[key];

            if (stream is null) {
                stream = Shader.DefaultStream;
            }

            shader.Execute(stream, src, dst_i, length);
        }

        /// <summary>J成分</summary>
        public static void J(uint length, CudaArray<float> src, CudaArray<float> dst_j, Stream stream = null) {
            string key = "quaternion_j";

            if (!shaders.ContainsKey(key)) {
                shaders.Add(key, new Shaders.ArrayManipulation.Scatter(arrays: 4, index: 2));
            }

            Shader shader = shaders[key];

            if (stream is null) {
                stream = Shader.DefaultStream;
            }

            shader.Execute(stream, src, dst_j, length);
        }

        /// <summary>K成分</summary>
        public static void K(uint length, CudaArray<float> src, CudaArray<float> dst_k, Stream stream = null) {
            string key = "quaternion_k";

            if (!shaders.ContainsKey(key)) {
                shaders.Add(key, new Shaders.ArrayManipulation.Scatter(arrays: 4, index: 3));
            }

            Shader shader = shaders[key];

            if (stream is null) {
                stream = Shader.DefaultStream;
            }

            shader.Execute(stream, src, dst_k, length);
        }

        /// <summary>純R成分</summary>
        public static void PureR(uint length, CudaArray<float> src_r, CudaArray<float> dst, Stream stream = null) {
            string key = "quaternion_purer";

            if (!shaders.ContainsKey(key)) {
                shaders.Add(key, new Shaders.ArrayManipulation.StrideCopy(stride: 4, index: 0));
            }

            Shader shader = shaders[key];

            if (stream is null) {
                stream = Shader.DefaultStream;
            }

            shader.Execute(stream, src_r, dst, length);
        }

        /// <summary>純I成分</summary>
        public static void PureI(uint length, CudaArray<float> src_i, CudaArray<float> dst, Stream stream = null) {
            string key = "quaternion_purei";

            if (!shaders.ContainsKey(key)) {
                shaders.Add(key, new Shaders.ArrayManipulation.StrideCopy(stride: 4, index: 1));
            }

            Shader shader = shaders[key];

            if (stream is null) {
                stream = Shader.DefaultStream;
            }

            shader.Execute(stream, src_i, dst, length);
        }

        /// <summary>純J成分</summary>
        public static void PureJ(uint length, CudaArray<float> src_j, CudaArray<float> dst, Stream stream = null) {
            string key = "quaternion_purej";

            if (!shaders.ContainsKey(key)) {
                shaders.Add(key, new Shaders.ArrayManipulation.StrideCopy(stride: 4, index: 2));
            }

            Shader shader = shaders[key];

            if (stream is null) {
                stream = Shader.DefaultStream;
            }

            shader.Execute(stream, src_j, dst, length);
        }

        /// <summary>純K成分</summary>
        public static void PureK(uint length, CudaArray<float> src_k, CudaArray<float> dst, Stream stream = null) {
            string key = "quaternion_purek";

            if (!shaders.ContainsKey(key)) {
                shaders.Add(key, new Shaders.ArrayManipulation.StrideCopy(stride: 4, index: 3));
            }

            Shader shader = shaders[key];

            if (stream is null) {
                stream = Shader.DefaultStream;
            }

            shader.Execute(stream, src_k, dst, length);
        }

        /// <summary>全結合</summary>
        public static void Dense(uint inchannels, uint outchannels,
                                 uint batch, bool gradmode,
                                 CudaArray<float> inmap, CudaArray<float> kernel, CudaArray<float> outmap,
                                 Stream stream = null) {

            string key = $"quaternion_dense " +
                         $"{nameof(inchannels)}={inchannels} {nameof(outchannels)}={outchannels} " +
                         $"{nameof(gradmode)}={gradmode} " + 
                         $"precision={Environment.Precision}";

            if (!shaders.ContainsKey(key)) {
                if (Environment.Precision == Environment.PrecisionMode.Float) {
                    shaders.Add(key, new Shaders.Quaternion.Convolution.FloatPrecision.Dense(inchannels, outchannels, gradmode));
                }
                else if (Environment.Precision == Environment.PrecisionMode.FloatFloat) {
                    shaders.Add(key, new Shaders.Quaternion.Convolution.FloatFloatPrecision.Dense(inchannels, outchannels, gradmode));
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
                                          uint batch, bool gradmode,
                                          CudaArray<float> inmap, CudaArray<float> kernel, CudaArray<float> outmap,
                                          Stream stream = null) {

            string key = $"quaternion_transpose_dense " +
                         $"{nameof(inchannels)}={inchannels} {nameof(outchannels)}={outchannels} " +
                         $"{nameof(gradmode)}={gradmode} " + 
                         $"precision={Environment.Precision}";

            if (!shaders.ContainsKey(key)) {
                if (Environment.Precision == Environment.PrecisionMode.Float) {
                    shaders.Add(key, new Shaders.Quaternion.Convolution.FloatPrecision.TransposeDense(inchannels, outchannels, gradmode));
                }
                else if (Environment.Precision == Environment.PrecisionMode.FloatFloat) {
                    shaders.Add(key, new Shaders.Quaternion.Convolution.FloatFloatPrecision.TransposeDense(inchannels, outchannels, gradmode));
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
                                              uint batch, bool transpose,
                                              CudaArray<float> inmap, CudaArray<float> outmap, CudaArray<float> kernel,
                                              Stream stream = null) {

            string key = $"quaternion_kernelproduct_dense " +
                         $"{nameof(inchannels)}={inchannels} {nameof(outchannels)}={outchannels} " +
                         $"{nameof(transpose)}={transpose} " + 
                         $"precision={Environment.Precision}";

            if (!shaders.ContainsKey(key)) {
                if (Environment.Precision == Environment.PrecisionMode.Float) {
                    shaders.Add(key, new Shaders.Quaternion.Convolution.FloatPrecision.KernelProductDense(inchannels, outchannels, transpose));
                }
                else if (Environment.Precision == Environment.PrecisionMode.FloatFloat) {
                    shaders.Add(key, new Shaders.Quaternion.Convolution.FloatFloatPrecision.KernelProductDense(inchannels, outchannels, transpose));
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
                                         uint batch, uint kwidth, bool gradmode,
                                         CudaArray<float> inmap, CudaArray<float> kernel, CudaArray<float> outmap,
                                         Stream stream = null) {

            string key = $"quaternion_convolution_1d " +
                         $"{nameof(inchannels)}={inchannels} {nameof(outchannels)}={outchannels} " +
                         $"{nameof(kwidth)}={kwidth} " +
                         $"{nameof(gradmode)}={gradmode} " + 
                         $"precision={Environment.Precision}";

            if (!shaders.ContainsKey(key)) {
                if (Environment.Precision == Environment.PrecisionMode.Float) {
                    shaders.Add(key, new Shaders.Quaternion.Convolution.FloatPrecision.Convolution1D(inchannels, outchannels, kwidth, gradmode));
                }
                else if (Environment.Precision == Environment.PrecisionMode.FloatFloat) {
                    shaders.Add(key, new Shaders.Quaternion.Convolution.FloatFloatPrecision.Convolution1D(inchannels, outchannels, kwidth, gradmode));
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
                                           uint batch, uint kwidth, bool gradmode,
                                           CudaArray<float> inmap, CudaArray<float> kernel, CudaArray<float> outmap,
                                           Stream stream = null) {

            string key = $"quaternion_deconvolution_1d " +
                         $"{nameof(inchannels)}={inchannels} {nameof(outchannels)}={outchannels} " +
                         $"{nameof(kwidth)}={kwidth} " +
                         $"{nameof(gradmode)}={gradmode} " + 
                         $"precision={Environment.Precision}";

            if (!shaders.ContainsKey(key)) {
                if (Environment.Precision == Environment.PrecisionMode.Float) {
                    shaders.Add(key, new Shaders.Quaternion.Convolution.FloatPrecision.Deconvolution1D(inchannels, outchannels, kwidth, gradmode));
                }
                else if (Environment.Precision == Environment.PrecisionMode.FloatFloat) {
                    shaders.Add(key, new Shaders.Quaternion.Convolution.FloatFloatPrecision.Deconvolution1D(inchannels, outchannels, kwidth, gradmode));
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
                                           uint batch, uint kwidth, bool transpose,
                                           CudaArray<float> inmap, CudaArray<float> outmap, CudaArray<float> kernel,
                                           Stream stream = null) {

            string key = $"quaternion_kernelproduct_1d " +
                         $"{nameof(inchannels)}={inchannels} {nameof(outchannels)}={outchannels} " +
                         $"{nameof(kwidth)}={kwidth} " +
                         $"{nameof(transpose)}={transpose} " + 
                         $"precision={Environment.Precision}";

            if (!shaders.ContainsKey(key)) {
                if (Environment.Precision == Environment.PrecisionMode.Float) {
                    shaders.Add(key, new Shaders.Quaternion.Convolution.FloatPrecision.KernelProduct1D(inchannels, outchannels, kwidth, transpose));
                }
                else if (Environment.Precision == Environment.PrecisionMode.FloatFloat) {
                    shaders.Add(key, new Shaders.Quaternion.Convolution.FloatFloatPrecision.KernelProduct1D(inchannels, outchannels, kwidth, transpose));
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
                                         uint batch, uint kwidth, uint kheight, bool gradmode,
                                         CudaArray<float> inmap, CudaArray<float> kernel, CudaArray<float> outmap,
                                         Stream stream = null) {

            string key = $"quaternion_convolution_2d " +
                         $"{nameof(inchannels)}={inchannels} {nameof(outchannels)}={outchannels} " +
                         $"{nameof(kwidth)}={kwidth} {nameof(kheight)}={kheight} " +
                         $"{nameof(gradmode)}={gradmode} " + 
                         $"precision={Environment.Precision}";

            if (!shaders.ContainsKey(key)) {
                if (Environment.Precision == Environment.PrecisionMode.Float) {
                    shaders.Add(key, new Shaders.Quaternion.Convolution.FloatPrecision.Convolution2D(inchannels, outchannels, kwidth, kheight, gradmode));
                }
                else if (Environment.Precision == Environment.PrecisionMode.FloatFloat) {
                    shaders.Add(key, new Shaders.Quaternion.Convolution.FloatFloatPrecision.Convolution2D(inchannels, outchannels, kwidth, kheight, gradmode));
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
                                           uint batch, uint kwidth, uint kheight, bool gradmode,
                                           CudaArray<float> inmap, CudaArray<float> kernel, CudaArray<float> outmap,
                                           Stream stream = null) {

            string key = $"quaternion_deconvolution_2d " +
                         $"{nameof(inchannels)}={inchannels} {nameof(outchannels)}={outchannels} " +
                         $"{nameof(kwidth)}={kwidth} {nameof(kheight)}={kheight} " +
                         $"{nameof(gradmode)}={gradmode} " + 
                         $"precision={Environment.Precision}";

            if (!shaders.ContainsKey(key)) {
                if (Environment.Precision == Environment.PrecisionMode.Float) {
                    shaders.Add(key, new Shaders.Quaternion.Convolution.FloatPrecision.Deconvolution2D(inchannels, outchannels, kwidth, kheight, gradmode));
                }
                else if (Environment.Precision == Environment.PrecisionMode.FloatFloat) {
                    shaders.Add(key, new Shaders.Quaternion.Convolution.FloatFloatPrecision.Deconvolution2D(inchannels, outchannels, kwidth, kheight, gradmode));
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
                                           uint batch, uint kwidth, uint kheight, bool transpose,
                                           CudaArray<float> inmap, CudaArray<float> outmap, CudaArray<float> kernel,
                                           Stream stream = null) {

            string key = $"quaternion_kernelproduct_2d " +
                         $"{nameof(inchannels)}={inchannels} {nameof(outchannels)}={outchannels} " +
                         $"{nameof(kwidth)}={kwidth} {nameof(kheight)}={kheight} " +
                         $"{nameof(transpose)}={transpose} " + 
                         $"precision={Environment.Precision}";

            if (!shaders.ContainsKey(key)) {
                if (Environment.Precision == Environment.PrecisionMode.Float) {
                    shaders.Add(key, new Shaders.Quaternion.Convolution.FloatPrecision.KernelProduct2D(inchannels, outchannels, kwidth, kheight, transpose));
                }
                else if (Environment.Precision == Environment.PrecisionMode.FloatFloat) {
                    shaders.Add(key, new Shaders.Quaternion.Convolution.FloatFloatPrecision.KernelProduct2D(inchannels, outchannels, kwidth, kheight, transpose));
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
                                         uint batch, uint kwidth, uint kheight, uint kdepth, bool gradmode,
                                         CudaArray<float> inmap, CudaArray<float> kernel, CudaArray<float> outmap,
                                         Stream stream = null) {

            string key = $"quaternion_convolution_3d " +
                         $"{nameof(inchannels)}={inchannels} {nameof(outchannels)}={outchannels} " +
                         $"{nameof(kwidth)}={kwidth} {nameof(kheight)}={kheight} {nameof(kdepth)}={kdepth} " +
                         $"{nameof(gradmode)}={gradmode} " + 
                         $"precision={Environment.Precision}";

            if (!shaders.ContainsKey(key)) {
                if (Environment.Precision == Environment.PrecisionMode.Float) {
                    shaders.Add(key, new Shaders.Quaternion.Convolution.FloatPrecision.Convolution3D(inchannels, outchannels, kwidth, kheight, kdepth, gradmode));
                }
                else if (Environment.Precision == Environment.PrecisionMode.FloatFloat) {
                    shaders.Add(key, new Shaders.Quaternion.Convolution.FloatFloatPrecision.Convolution3D(inchannels, outchannels, kwidth, kheight, kdepth, gradmode));
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
                                           uint batch, uint kwidth, uint kheight, uint kdepth, bool gradmode,
                                           CudaArray<float> inmap, CudaArray<float> kernel, CudaArray<float> outmap,
                                           Stream stream = null) {

            string key = $"quaternion_deconvolution_3d " +
                         $"{nameof(inchannels)}={inchannels} {nameof(outchannels)}={outchannels} " +
                         $"{nameof(kwidth)}={kwidth} {nameof(kheight)}={kheight} {nameof(kdepth)}={kdepth} " +
                         $"{nameof(gradmode)}={gradmode} " + 
                         $"precision={Environment.Precision}";

            if (!shaders.ContainsKey(key)) {
                if (Environment.Precision == Environment.PrecisionMode.Float) {
                    shaders.Add(key, new Shaders.Quaternion.Convolution.FloatPrecision.Deconvolution3D(inchannels, outchannels, kwidth, kheight, kdepth, gradmode));
                }
                else if (Environment.Precision == Environment.PrecisionMode.FloatFloat) {
                    shaders.Add(key, new Shaders.Quaternion.Convolution.FloatFloatPrecision.Deconvolution3D(inchannels, outchannels, kwidth, kheight, kdepth, gradmode));
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
                                           uint batch, uint kwidth, uint kheight, uint kdepth, bool transpose,
                                           CudaArray<float> inmap, CudaArray<float> outmap, CudaArray<float> kernel,
                                           Stream stream = null) {

            string key = $"quaternion_kernelproduct_3d " +
                         $"{nameof(inchannels)}={inchannels} {nameof(outchannels)}={outchannels} " +
                         $"{nameof(kwidth)}={kwidth} {nameof(kheight)}={kheight} {nameof(kdepth)}={kdepth} " +
                         $"{nameof(transpose)}={transpose} " + 
                         $"precision={Environment.Precision}";

            if (!shaders.ContainsKey(key)) {
                if (Environment.Precision == Environment.PrecisionMode.Float) {
                    shaders.Add(key, new Shaders.Quaternion.Convolution.FloatPrecision.KernelProduct3D(inchannels, outchannels, kwidth, kheight, kdepth, transpose));
                }
                else if (Environment.Precision == Environment.PrecisionMode.FloatFloat) {
                    shaders.Add(key, new Shaders.Quaternion.Convolution.FloatFloatPrecision.KernelProduct3D(inchannels, outchannels, kwidth, kheight, kdepth, transpose));
                }
            }

            Shader shader = shaders[key];

            if (stream is null) {
                stream = Shader.DefaultStream;
            }

            shader.Execute(stream, inmap, outmap, kernel, inwidth, inheight, indepth, batch);
        }
    }
}