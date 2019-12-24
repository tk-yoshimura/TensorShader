using System.Collections.Generic;

namespace TensorShaderCudaBackend {

    /// <summary>要素独立演算</summary>
    public static class Elementwise {
        private readonly static Dictionary<string, Shader> shaders = new Dictionary<string, Shader>();

        private static Shader UnaryArithmetric(string name, string func) {
            if (!shaders.ContainsKey(name)) {
                shaders.Add(name, new Shaders.Elementwise.UnaryArithmetric(name, func));
            }

            return shaders[name];
        }

        private static Shader BinaryArithmetric(string name, string func) {
            if (!shaders.ContainsKey(name)) {
                shaders.Add(name, new Shaders.Elementwise.BinaryArithmetric(name, func));
            }

            return shaders[name];
        }

        private static Shader BinaryConstantArithmetric(string name, string func) {
            if (!shaders.ContainsKey(name)) {
                shaders.Add(name, new Shaders.Elementwise.BinaryConstantArithmetric(name, func));
            }

            return shaders[name];
        }

        private static Shader BinaryFactorArithmetric(string name, string func) {
            if (!shaders.ContainsKey(name)) {
                shaders.Add(name, new Shaders.Elementwise.BinaryFactorArithmetric(name, func));
            }

            return shaders[name];
        }

        private static Shader TrinaryArithmetric(string name, string func) {
            if (!shaders.ContainsKey(name)) {
                shaders.Add(name, new Shaders.Elementwise.TrinaryArithmetric(name, func));
            }

            return shaders[name];
        }

        private static Shader TrinaryBiConstantArithmetric(string name, string func) {
            if (!shaders.ContainsKey(name)) {
                shaders.Add(name, new Shaders.Elementwise.TrinaryBiConstantArithmetric(name, func));
            }

            return shaders[name];
        }

        private static Shader TrinaryBiFactorArithmetric(string name, string func) {
            if (!shaders.ContainsKey(name)) {
                shaders.Add(name, new Shaders.Elementwise.TrinaryBiFactorArithmetric(name, func));
            }

            return shaders[name];
        }

        private static Shader TrinaryUniConstantArithmetric(string name, string func) {
            if (!shaders.ContainsKey(name)) {
                shaders.Add(name, new Shaders.Elementwise.TrinaryUniConstantArithmetric(name, func));
            }

            return shaders[name];
        }

        /// <summary>加算</summary>
        public static void Add(uint length, CudaArray<float> src1, CudaArray<float> src2, CudaArray<float> dst) {
            Shader shader = BinaryArithmetric("add_ew", "#y = #x1 + #x2;");
            shader.Execute(Shader.DefaultStream, src1, src2, dst, length);
        }

        /// <summary>減算</summary>
        public static void Sub(uint length, CudaArray<float> src1, CudaArray<float> src2, CudaArray<float> dst) {
            Shader shader = BinaryArithmetric("sub_ew", "#y = #x1 - #x2;");
            shader.Execute(Shader.DefaultStream, src1, src2, dst, length);
        }

        /// <summary>乗算</summary>
        public static void Mul(uint length, CudaArray<float> src1, CudaArray<float> src2, CudaArray<float> dst) {
            Shader shader = BinaryArithmetric("mul_ew", "#y = #x1 * #x2;");
            shader.Execute(Shader.DefaultStream, src1, src2, dst, length);
        }

        /// <summary>除算</summary>
        public static void Div(uint length, CudaArray<float> src1, CudaArray<float> src2, CudaArray<float> dst) {
            Shader shader = BinaryArithmetric("div_ew", "#y = #x1 / #x2;");
            shader.Execute(Shader.DefaultStream, src1, src2, dst, length);
        }

        /// <summary>べき乗</summary>
        public static void Pow(uint length, CudaArray<float> src1, CudaArray<float> src2, CudaArray<float> dst) {
            Shader shader = BinaryArithmetric("pow_ew", "#y = powf(#x1, #x2);");
            shader.Execute(Shader.DefaultStream, src1, src2, dst, length);
        }

        /// <summary>符号付きべき乗</summary>
        public static void SignedPow(uint length, CudaArray<float> src1, CudaArray<float> src2, CudaArray<float> dst) {
            Shader shader = BinaryArithmetric("signedpow_ew", "#y = ((#x > 0.0 ? 1.0 : 0.0) - (#x < 0.0 ? 1.0 : 0.0)) * powf(fabs(#x1), #x2);");
            shader.Execute(Shader.DefaultStream, src1, src2, dst, length);
        }

        /// <summary>ArcTan2</summary>
        public static void ArcTan2(uint length, CudaArray<float> src1, CudaArray<float> src2, CudaArray<float> dst) {
            Shader shader = BinaryArithmetric("atan2_ew", "#y = atan2f(#x1, #x2);");
            shader.Execute(Shader.DefaultStream, src1, src2, dst, length);
        }

        /// <summary>加算</summary>
        public static void AddConstant(uint length, float c, CudaArray<float> src, CudaArray<float> dst) {
            Shader shader = BinaryConstantArithmetric("add_const_ew", "#y = c + #x;");
            shader.Execute(Shader.DefaultStream, c, src, dst, length);
        }

        /// <summary>減算</summary>
        public static void SubConstant(uint length, float c, CudaArray<float> src, CudaArray<float> dst) {
            Shader shader = BinaryConstantArithmetric("sub_const_ew", "#y = c - #x;");
            shader.Execute(Shader.DefaultStream, c, src, dst, length);
        }

        /// <summary>乗算</summary>
        public static void MulConstant(uint length, float c, CudaArray<float> src, CudaArray<float> dst) {
            Shader shader = BinaryConstantArithmetric("mul_const_ew", "#y = c * #x;");
            shader.Execute(Shader.DefaultStream, c, src, dst, length);
        }

        /// <summary>除算</summary>
        public static void DivConstant(uint length, float c, CudaArray<float> src, CudaArray<float> dst) {
            Shader shader = BinaryConstantArithmetric("div_const_ew", "#y = c / #x;");
            shader.Execute(Shader.DefaultStream, c, src, dst, length);
        }

        /// <summary>べき乗</summary>
        public static void PowConstant(uint length, float c, CudaArray<float> src, CudaArray<float> dst) {
            Shader shader = BinaryConstantArithmetric("pow_const_ew", "#y = powf(#x, c);");
            shader.Execute(Shader.DefaultStream, c, src, dst, length);
        }

        /// <summary>符号付きべき乗</summary>
        public static void SignedPowConstant(uint length, float c, CudaArray<float> src, CudaArray<float> dst) {
            Shader shader = BinaryConstantArithmetric("signedpow_const_ew", "#y = ((#x > 0.0 ? 1.0 : 0.0) - (#x < 0.0 ? 1.0 : 0.0)) * powf(fabs(#x), c);");
            shader.Execute(Shader.DefaultStream, c, src, dst, length);
        }

        /// <summary>加算</summary>
        public static void AddFactor(uint length, CudaArray<float> c, CudaArray<float> src, CudaArray<float> dst) {
            Shader shader = BinaryFactorArithmetric("add_factor_ew", "#y = c + #x;");
            shader.Execute(Shader.DefaultStream, c, src, dst, length);
        }

        /// <summary>減算</summary>
        public static void SubFactor(uint length, CudaArray<float> c, CudaArray<float> src, CudaArray<float> dst) {
            Shader shader = BinaryFactorArithmetric("sub_factor_ew", "#y = c - #x;");
            shader.Execute(Shader.DefaultStream, c, src, dst, length);
        }

        /// <summary>乗算</summary>
        public static void MulFactor(uint length, CudaArray<float> c, CudaArray<float> src, CudaArray<float> dst) {
            Shader shader = BinaryFactorArithmetric("mul_factor_ew", "#y = c * #x;");
            shader.Execute(Shader.DefaultStream, c, src, dst, length);
        }

        /// <summary>除算</summary>
        public static void DivFactor(uint length, CudaArray<float> c, CudaArray<float> src, CudaArray<float> dst) {
            Shader shader = BinaryFactorArithmetric("div_factor_ew", "#y = c / #x;");
            shader.Execute(Shader.DefaultStream, c, src, dst, length);
        }

        /// <summary>べき乗</summary>
        public static void PowFactor(uint length, CudaArray<float> c, CudaArray<float> src, CudaArray<float> dst) {
            Shader shader = BinaryFactorArithmetric("pow_factor_ew", "#y = powf(#x, c);");
            shader.Execute(Shader.DefaultStream, c, src, dst, length);
        }

        /// <summary>符号付きべき乗</summary>
        public static void SignedPowFactor(uint length, CudaArray<float> c, CudaArray<float> src, CudaArray<float> dst) {
            Shader shader = BinaryFactorArithmetric("signedpow_factor_ew", "#y = ((#x > 0.0 ? 1.0 : 0.0) - (#x < 0.0 ? 1.0 : 0.0)) * powf(fabs(#x), c);");
            shader.Execute(Shader.DefaultStream, c, src, dst, length);
        }

        /// <summary>最大値</summary>
        public static void Maximum(uint length, CudaArray<float> src1, CudaArray<float> src2, CudaArray<float> dst) {
            Shader shader = BinaryArithmetric("maximum_ew", "#y = fmaxf(#x1, #x2);");
            shader.Execute(Shader.DefaultStream, src1, src2, dst, length);
        }

        /// <summary>最小値</summary>
        public static void Minimum(uint length, CudaArray<float> src1, CudaArray<float> src2, CudaArray<float> dst) {
            Shader shader = BinaryArithmetric("minimum_ew", "#y = fminf(#x1, #x2);");
            shader.Execute(Shader.DefaultStream, src1, src2, dst, length);
        }

        /// <summary>Clamp</summary>
        public static void Clamp(uint length, CudaArray<float> srcval, CudaArray<float> srcmin, CudaArray<float> srcmax, CudaArray<float> dst) {
            Shader shader = TrinaryArithmetric("clamp_ew", "#y = fminf(fmaxf(#x1, #x2), #x3);");
            shader.Execute(Shader.DefaultStream, srcval, srcmin, srcmax, dst, length);
        }

        /// <summary>最大値</summary>
        public static void MaximumConstant(uint length, float c, CudaArray<float> src, CudaArray<float> dst) {
            Shader shader = BinaryConstantArithmetric("maximum_const_ew", "#y = fmaxf(c, #x);");
            shader.Execute(Shader.DefaultStream, c, src, dst, length);
        }

        /// <summary>最小値</summary>
        public static void MinimumConstant(uint length, float c, CudaArray<float> src, CudaArray<float> dst) {
            Shader shader = BinaryConstantArithmetric("minimum_const_ew", "#y = fminf(c, #x);");
            shader.Execute(Shader.DefaultStream, c, src, dst, length);
        }

        /// <summary>Clamp</summary>
        public static void ClampConstant(uint length, float cmin, float cmax, CudaArray<float> src, CudaArray<float> dst) {
            Shader shader = TrinaryBiConstantArithmetric("clamp_const_ew", "#y = fminf(fmaxf(#x, c1), c2);");
            shader.Execute(Shader.DefaultStream, cmin, cmax, src, dst, length);
        }

        /// <summary>最大値</summary>
        public static void MaximumFactor(uint length, CudaArray<float> c, CudaArray<float> src, CudaArray<float> dst) {
            Shader shader = BinaryFactorArithmetric("maximum_factor_ew", "#y = fmaxf(c, #x);");
            shader.Execute(Shader.DefaultStream, c, src, dst, length);
        }

        /// <summary>最小値</summary>
        public static void MinimumFactor(uint length, CudaArray<float> c, CudaArray<float> src, CudaArray<float> dst) {
            Shader shader = BinaryFactorArithmetric("minimum_factor_ew", "#y = fminf(c, #x);");
            shader.Execute(Shader.DefaultStream, c, src, dst, length);
        }

        /// <summary>Clamp</summary>
        public static void ClampFactor(uint length, CudaArray<float> cmin, CudaArray<float> cmax, CudaArray<float> src, CudaArray<float> dst) {
            Shader shader = TrinaryBiFactorArithmetric("clamp_factor_ew", "#y = fminf(fmaxf(#x, c1), c2);");
            shader.Execute(Shader.DefaultStream, cmin, cmax, src, dst, length);
        }

        /// <summary>比較</summary>
        public static void GreaterThan(uint length, CudaArray<float> src1, CudaArray<float> src2, CudaArray<float> dst) {
            Shader shader = BinaryArithmetric("cmpgt_ew", "#y = #x1 > #x2 ? 1.0 : 0.0;");
            shader.Execute(Shader.DefaultStream, src1, src2, dst, length);
        }

        /// <summary>比較</summary>
        public static void GreaterThanOrEqual(uint length, CudaArray<float> src1, CudaArray<float> src2, CudaArray<float> dst) {
            Shader shader = BinaryArithmetric("cmpge_ew", "#y = #x1 >= #x2 ? 1.0 : 0.0;");
            shader.Execute(Shader.DefaultStream, src1, src2, dst, length);
        }

        /// <summary>比較</summary>
        public static void LessThan(uint length, CudaArray<float> src1, CudaArray<float> src2, CudaArray<float> dst) {
            Shader shader = BinaryArithmetric("cmplt_ew", "#y = #x1 < #x2 ? 1.0 : 0.0;");
            shader.Execute(Shader.DefaultStream, src1, src2, dst, length);
        }

        /// <summary>比較</summary>
        public static void LessThanOrEqual(uint length, CudaArray<float> src1, CudaArray<float> src2, CudaArray<float> dst) {
            Shader shader = BinaryArithmetric("cmple_ew", "#y = #x1 <= #x2 ? 1.0 : 0.0;");
            shader.Execute(Shader.DefaultStream, src1, src2, dst, length);
        }

        /// <summary>比較</summary>
        public static void Equal(uint length, CudaArray<float> src1, CudaArray<float> src2, CudaArray<float> dst) {
            Shader shader = BinaryArithmetric("cmpeq_ew", "#y = #x1 == #x2 ? 1.0 : 0.0;");
            shader.Execute(Shader.DefaultStream, src1, src2, dst, length);
        }

        /// <summary>比較</summary>
        public static void NotEqual(uint length, CudaArray<float> src1, CudaArray<float> src2, CudaArray<float> dst) {
            Shader shader = BinaryArithmetric("cmpneq_ew", "#y = #x1 != #x2 ? 1.0 : 0.0;");
            shader.Execute(Shader.DefaultStream, src1, src2, dst, length);
        }

        /// <summary>比較</summary>
        public static void GreaterThanConstant(uint length, float c, CudaArray<float> src, CudaArray<float> dst) {
            Shader shader = BinaryConstantArithmetric("cmpgt_const_ew", "#y = c > #x ? 1.0 : 0.0;");
            shader.Execute(Shader.DefaultStream, c, src, dst, length);
        }

        /// <summary>比較</summary>
        public static void GreaterThanOrEqualConstant(uint length, float c, CudaArray<float> src, CudaArray<float> dst) {
            Shader shader = BinaryConstantArithmetric("cmpge_const_ew", "#y = c >= #x ? 1.0 : 0.0;");
            shader.Execute(Shader.DefaultStream, c, src, dst, length);
        }

        /// <summary>比較</summary>
        public static void LessThanConstant(uint length, float c, CudaArray<float> src, CudaArray<float> dst) {
            Shader shader = BinaryConstantArithmetric("cmplt_const_ew", "#y = c < #x ? 1.0 : 0.0;");
            shader.Execute(Shader.DefaultStream, c, src, dst, length);
        }

        /// <summary>比較</summary>
        public static void LessThanOrEqualConstant(uint length, float c, CudaArray<float> src, CudaArray<float> dst) {
            Shader shader = BinaryConstantArithmetric("cmple_const_ew", "#y = c <= #x ? 1.0 : 0.0;");
            shader.Execute(Shader.DefaultStream, c, src, dst, length);
        }

        /// <summary>比較</summary>
        public static void EqualConstant(uint length, float c, CudaArray<float> src, CudaArray<float> dst) {
            Shader shader = BinaryConstantArithmetric("cmpeq_const_ew", "#y = c == #x ? 1.0 : 0.0;");
            shader.Execute(Shader.DefaultStream, c, src, dst, length);
        }

        /// <summary>比較</summary>
        public static void NotEqualConstant(uint length, float c, CudaArray<float> src, CudaArray<float> dst) {
            Shader shader = BinaryConstantArithmetric("cmpneq_const_ew", "#y = c != #x ? 1.0 : 0.0;");
            shader.Execute(Shader.DefaultStream, c, src, dst, length);
        }

        /// <summary>比較</summary>
        public static void GreaterThanFactor(uint length, CudaArray<float> c, CudaArray<float> src, CudaArray<float> dst) {
            Shader shader = BinaryFactorArithmetric("cmpgt_factor_ew", "#y = c > #x ? 1.0 : 0.0;");
            shader.Execute(Shader.DefaultStream, c, src, dst, length);
        }

        /// <summary>比較</summary>
        public static void GreaterThanOrEqualFactor(uint length, CudaArray<float> c, CudaArray<float> src, CudaArray<float> dst) {
            Shader shader = BinaryFactorArithmetric("cmpge_factor_ew", "#y = c >= #x ? 1.0 : 0.0;");
            shader.Execute(Shader.DefaultStream, c, src, dst, length);
        }

        /// <summary>比較</summary>
        public static void LessThanFactor(uint length, CudaArray<float> c, CudaArray<float> src, CudaArray<float> dst) {
            Shader shader = BinaryFactorArithmetric("cmplt_factor_ew", "#y = c < #x ? 1.0 : 0.0;");
            shader.Execute(Shader.DefaultStream, c, src, dst, length);
        }

        /// <summary>比較</summary>
        public static void LessThanOrEqualFactor(uint length, CudaArray<float> c, CudaArray<float> src, CudaArray<float> dst) {
            Shader shader = BinaryFactorArithmetric("cmple_factor_ew", "#y = c <= #x ? 1.0 : 0.0;");
            shader.Execute(Shader.DefaultStream, c, src, dst, length);
        }

        /// <summary>比較</summary>
        public static void EqualFactor(uint length, CudaArray<float> c, CudaArray<float> src, CudaArray<float> dst) {
            Shader shader = BinaryFactorArithmetric("cmpeq_factor_ew", "#y = c == #x ? 1.0 : 0.0;");
            shader.Execute(Shader.DefaultStream, c, src, dst, length);
        }

        /// <summary>比較</summary>
        public static void NotEqualFactor(uint length, CudaArray<float> c, CudaArray<float> src, CudaArray<float> dst) {
            Shader shader = BinaryFactorArithmetric("cmpneq_factor_ew", "#y = c != #x ? 1.0 : 0.0;");
            shader.Execute(Shader.DefaultStream, c, src, dst, length);
        }

        /// <summary>NOT</summary>
        public static void LogicalNot(uint length, CudaArray<float> src, CudaArray<float> dst) {
            Shader shader = UnaryArithmetric("logicalnot_ew", "#y = 1.0 - #x;");
            shader.Execute(Shader.DefaultStream, src, dst, length);
        }

        /// <summary>AND</summary>
        public static void LogicalAnd(uint length, CudaArray<float> src1, CudaArray<float> src2, CudaArray<float> dst) {
            Minimum(length, src1, src2, dst);
        }

        /// <summary>OR</summary>
        public static void LogicalOr(uint length, CudaArray<float> src1, CudaArray<float> src2, CudaArray<float> dst) {
            Maximum(length, src1, src2, dst);
        }

        /// <summary>XOR</summary>
        public static void LogicalXor(uint length, CudaArray<float> src1, CudaArray<float> src2, CudaArray<float> dst) {
            Shader shader = BinaryArithmetric("logicalxor_ew", "#y = fmaxf(#x1, #x2) - fminf(#x1, #x2);");
            shader.Execute(Shader.DefaultStream, src1, src2, dst, length);
        }

        /// <summary>絶対値</summary>
        public static void Abs(uint length, CudaArray<float> src, CudaArray<float> dst) {
            Shader shader = UnaryArithmetric("abs_ew", "#y = fabs(#x);");
            shader.Execute(Shader.DefaultStream, src, dst, length);
        }

        /// <summary>符号関数</summary>
        public static void Sign(uint length, CudaArray<float> src, CudaArray<float> dst) {
            Shader shader = UnaryArithmetric("sign_ew", "#y = ((#x > 0.0 ? 1.0 : 0.0) - (#x < 0 ? 1.0 : 0.0));");
            shader.Execute(Shader.DefaultStream, src, dst, length);
        }

        /// <summary>ステップ関数</summary>
        public static void Step(uint length, CudaArray<float> src, CudaArray<float> dst) {
            Shader shader = UnaryArithmetric("step_ew", "#y = #x >= 0.0 ? 1.0 : 0.0;");
            shader.Execute(Shader.DefaultStream, src, dst, length);
        }

        /// <summary>符号反転</summary>
        public static void Neg(uint length, CudaArray<float> src, CudaArray<float> dst) {
            Shader shader = UnaryArithmetric("neg_ew", "#y = -#x;");
            shader.Execute(Shader.DefaultStream, src, dst, length);
        }

        /// <summary>逆数</summary>
        public static void Rcp(uint length, CudaArray<float> src, CudaArray<float> dst) {
            Shader shader = UnaryArithmetric("rcp_ew", "#y = 1.0 / #x;");
            shader.Execute(Shader.DefaultStream, src, dst, length);
        }

        /// <summary>2乗</summary>
        public static void Square(uint length, CudaArray<float> src, CudaArray<float> dst) {
            Shader shader = UnaryArithmetric("square_ew", "#y = #x * #x;");
            shader.Execute(Shader.DefaultStream, src, dst, length);
        }

        /// <summary>3乗</summary>
        public static void Cube(uint length, CudaArray<float> src, CudaArray<float> dst) {
            Shader shader = UnaryArithmetric("cube_ew", "#y = #x * #x * #x;");
            shader.Execute(Shader.DefaultStream, src, dst, length);
        }

        /// <summary>平方根</summary>
        public static void Sqrt(uint length, CudaArray<float> src, CudaArray<float> dst) {
            Shader shader = UnaryArithmetric("sqrt_ew", "#y = sqrtf(#x);");
            shader.Execute(Shader.DefaultStream, src, dst, length);
        }

        /// <summary>平方根逆数</summary>
        public static void Rsqrt(uint length, CudaArray<float> src, CudaArray<float> dst) {
            Shader shader = UnaryArithmetric("rsqrt_ew", "#y = rsqrtf(#x);");
            shader.Execute(Shader.DefaultStream, src, dst, length);
        }

        /// <summary>符号付き平方根</summary>
        public static void SignedSqrt(uint length, CudaArray<float> src, CudaArray<float> dst) {
            Shader shader = UnaryArithmetric("signedsqrt_ew", "#y = ((#x > 0.0 ? 1.0 : 0.0) - (#x < 0 ? 1.0 : 0.0)) * sqrtf(fabs(#x));");
            shader.Execute(Shader.DefaultStream, src, dst, length);
        }

        /// <summary>立方根</summary>
        public static void Cbrt(uint length, CudaArray<float> src, CudaArray<float> dst) {
            Shader shader = UnaryArithmetric("cbrt_ew", "#y = cbrtf(#x);");
            shader.Execute(Shader.DefaultStream, src, dst, length);
        }

        /// <summary>立方根逆数</summary>
        public static void Rcbrt(uint length, CudaArray<float> src, CudaArray<float> dst) {
            Shader shader = UnaryArithmetric("rcbrt_ew", "#y = 1.0 / cbrtf(#x);");
            shader.Execute(Shader.DefaultStream, src, dst, length);
        }

        /// <summary>Sin</summary>
        public static void Sin(uint length, CudaArray<float> src, CudaArray<float> dst) {
            Shader shader = UnaryArithmetric("sin_ew", "#y = sinf(#x);");
            shader.Execute(Shader.DefaultStream, src, dst, length);
        }

        /// <summary>Cos</summary>
        public static void Cos(uint length, CudaArray<float> src, CudaArray<float> dst) {
            Shader shader = UnaryArithmetric("cos_ew", "#y = cosf(#x);");
            shader.Execute(Shader.DefaultStream, src, dst, length);
        }

        /// <summary>Tan</summary>
        public static void Tan(uint length, CudaArray<float> src, CudaArray<float> dst) {
            Shader shader = UnaryArithmetric("tan_ew", "#y = tanf(#x);");
            shader.Execute(Shader.DefaultStream, src, dst, length);
        }

        /// <summary>Sinh</summary>
        public static void Sinh(uint length, CudaArray<float> src, CudaArray<float> dst) {
            Shader shader = UnaryArithmetric("sinh_ew", "#y = sinhf(#x);");
            shader.Execute(Shader.DefaultStream, src, dst, length);
        }

        /// <summary>Cosh</summary>
        public static void Cosh(uint length, CudaArray<float> src, CudaArray<float> dst) {
            Shader shader = UnaryArithmetric("cosh_ew", "#y = coshf(#x);");
            shader.Execute(Shader.DefaultStream, src, dst, length);
        }

        /// <summary>Tanh</summary>
        public static void Tanh(uint length, CudaArray<float> src, CudaArray<float> dst) {
            Shader shader = UnaryArithmetric("tanh_ew", "#y = tanhf(#x);");
            shader.Execute(Shader.DefaultStream, src, dst, length);
        }

        /// <summary>ArcSin</summary>
        public static void ArcSin(uint length, CudaArray<float> src, CudaArray<float> dst) {
            Shader shader = UnaryArithmetric("asin_ew", "#y = asinf(#x);");
            shader.Execute(Shader.DefaultStream, src, dst, length);
        }

        /// <summary>ArcCos</summary>
        public static void ArcCos(uint length, CudaArray<float> src, CudaArray<float> dst) {
            Shader shader = UnaryArithmetric("acos_ew", "#y = acosf(#x);");
            shader.Execute(Shader.DefaultStream, src, dst, length);
        }

        /// <summary>ArcTan</summary>
        public static void ArcTan(uint length, CudaArray<float> src, CudaArray<float> dst) {
            Shader shader = UnaryArithmetric("atan_ew", "#y = atanf(#x);");
            shader.Execute(Shader.DefaultStream, src, dst, length);
        }

        /// <summary>Exp</summary>
        public static void Exp(uint length, CudaArray<float> src, CudaArray<float> dst) {
            Shader shader = UnaryArithmetric("exp_ew", "#y = expf(#x);");
            shader.Execute(Shader.DefaultStream, src, dst, length);
        }

        /// <summary>Exp2</summary>
        public static void Exp2(uint length, CudaArray<float> src, CudaArray<float> dst) {
            Shader shader = UnaryArithmetric("exp2_ew", "#y = exp2f(#x);");
            shader.Execute(Shader.DefaultStream, src, dst, length);
        }

        /// <summary>Log</summary>
        public static void Log(uint length, CudaArray<float> src, CudaArray<float> dst) {
            Shader shader = UnaryArithmetric("log_ew", "#y = logf(#x);");
            shader.Execute(Shader.DefaultStream, src, dst, length);
        }

        /// <summary>Log2</summary>
        public static void Log2(uint length, CudaArray<float> src, CudaArray<float> dst) {
            Shader shader = UnaryArithmetric("log2_ew", "#y = log2f(#x);");
            shader.Execute(Shader.DefaultStream, src, dst, length);
        }

        /// <summary>Log10</summary>
        public static void Log10(uint length, CudaArray<float> src, CudaArray<float> dst) {
            Shader shader = UnaryArithmetric("log10_ew", "#y = log10f(#x);");
            shader.Execute(Shader.DefaultStream, src, dst, length);
        }

        /// <summary>Floor</summary>
        public static void Floor(uint length, CudaArray<float> src, CudaArray<float> dst) {
            Shader shader = UnaryArithmetric("floor_ew", "#y = floorf(#x);");
            shader.Execute(Shader.DefaultStream, src, dst, length);
        }

        /// <summary>Ceil</summary>
        public static void Ceil(uint length, CudaArray<float> src, CudaArray<float> dst) {
            Shader shader = UnaryArithmetric("ceil_ew", "#y = ceilf(#x);");
            shader.Execute(Shader.DefaultStream, src, dst, length);
        }

        /// <summary>Round</summary>
        public static void Round(uint length, CudaArray<float> src, CudaArray<float> dst) {
            Shader shader = UnaryArithmetric("round_ew", "#y = roundf(#x);");
            shader.Execute(Shader.DefaultStream, src, dst, length);
        }

        /// <summary>Sigmoid</summary>
        public static void Sigmoid(uint length, CudaArray<float> src, CudaArray<float> dst) {
            Shader shader = UnaryArithmetric("sigmoid_ew", "#y = 1.0 / (1.0 + expf(-#x));");
            shader.Execute(Shader.DefaultStream, src, dst, length);
        }

        /// <summary>SoftPlus</summary>
        public static void SoftPlus(uint length, CudaArray<float> src, CudaArray<float> dst) {
            Shader shader = UnaryArithmetric("softplus_ew", "#y = logf(1.0 + expf(#x));");
            shader.Execute(Shader.DefaultStream, src, dst, length);
        }

        /// <summary>LogCosh</summary>
        public static void LogCosh(uint length, CudaArray<float> src, CudaArray<float> dst) {
            Shader shader = UnaryArithmetric("logcosh_ew", "#y = logf(coshf(#x));");
            shader.Execute(Shader.DefaultStream, src, dst, length);
        }

        /// <summary>非数判定</summary>
        public static void IsNan(uint length, CudaArray<float> src, CudaArray<float> dst) {
            Shader shader = UnaryArithmetric("isnan_ew", "#y = #x != #x;");
            shader.Execute(Shader.DefaultStream, src, dst, length);
        }

        /// <summary>NanAsZero</summary>
        public static void NanAsZero(uint length, CudaArray<float> src, CudaArray<float> dst) {
            Shader shader = UnaryArithmetric("nanaszero_ew", "#y = #x == #x ? #x : 0.0;");
            shader.Execute(Shader.DefaultStream, src, dst, length);
        }

        /// <summary>Relu</summary>
        public static void Relu(uint length, CudaArray<float> src, CudaArray<float> dst) {
            Shader shader = UnaryArithmetric("relu_ew", "#y = fmaxf(0.0, #x);");
            shader.Execute(Shader.DefaultStream, src, dst, length);
        }

        /// <summary>Relu勾配</summary>
        public static void ReluGrad(uint length, CudaArray<float> src1, CudaArray<float> src2, CudaArray<float> dst) {
            Shader shader = BinaryArithmetric("relugrad_ew", "#y = #x2 >= 0.0 ? #x1 : 0.0;");
            shader.Execute(Shader.DefaultStream, src1, src2, dst, length);
        }

        /// <summary>LeakyRelu</summary>
        public static void LeakyRelu(uint length, float slope, CudaArray<float> src, CudaArray<float> dst) {
            Shader shader = BinaryConstantArithmetric("leakyrelu_ew", "#y = #x - c * fminf(#x, 0.0);");
            shader.Execute(Shader.DefaultStream, 1 - slope, src, dst, length);
        }

        /// <summary>LeakyRelu勾配</summary>
        public static void LeakyReluGrad(uint length, float slope, CudaArray<float> src1, CudaArray<float> src2, CudaArray<float> dst) {
            Shader shader = TrinaryUniConstantArithmetric("leakyrelugrad_ew", "#y = #x2 >= 0.0 ? #x1 : (c * #x1);");
            shader.Execute(Shader.DefaultStream, slope, src1, src2, dst, length);
        }

        /// <summary>Elu</summary>
        public static void Elu(uint length, float slope, CudaArray<float> src, CudaArray<float> dst) {
            Shader shader = BinaryConstantArithmetric("elu_ew", "#y = #x >= 0.0 ? #x : (c * (expf(#x) - 1.0));");
            shader.Execute(Shader.DefaultStream, slope, src, dst, length);
        }

        /// <summary>Elu勾配</summary>
        public static void EluGrad(uint length, float slope, CudaArray<float> src1, CudaArray<float> src2, CudaArray<float> dst) {
            Shader shader = TrinaryUniConstantArithmetric("elugrad_ew", "#y = #x2 >= 0.0 ? #x1 : (c * #x1 * expf(#x2));");
            shader.Execute(Shader.DefaultStream, slope, src1, src2, dst, length);
        }

        /// <summary>テンソル和</summary>
        public static void Sum(uint length, CudaArray<float>[] src, CudaArray<float> dst) {
            int n = src.Length;

            if (n == 0) {
                dst.Zeroset();
                return;
            }
            if (n == 1) {
                src[0].CopyTo(dst, length);
                return;
            }

            Add(length, src[0], src[1], dst);

            for (int i = 2, j = n / 2 * 2; i < j; i += 2) {
                Shader shader = BinaryArithmetric("sum2_ew", "#y += #x1 + #x2;");
                shader.Execute(Shader.DefaultStream, src[i], src[i + 1], dst, length);
            }

            if ((n & 1) == 1) {
                Shader shader = UnaryArithmetric("sum1_ew", "#y += #x;");
                shader.Execute(Shader.DefaultStream, src[n - 1], dst, length);
            }
        }

        /// <summary>線形補間</summary>
        public static void Lerp(uint length, CudaArray<float> srccondition, CudaArray<float> src1, CudaArray<float> src2, CudaArray<float> dst) {
            Shader shader = TrinaryArithmetric("lerp_ew", "#y = #x1 * #x2 + (1.0 - #x1) * #x3;");
            shader.Execute(Shader.DefaultStream, srccondition, src1, src2, dst, length);
        }
    }
}
