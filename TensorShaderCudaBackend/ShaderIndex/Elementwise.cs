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
        public static void Add(uint length, CudaArray<float> src1, CudaArray<float> src2, CudaArray<float> dst, Stream stream = null) {
            Shader shader = BinaryArithmetric("add_ew", "#y = #x1 + #x2;");
            
            if(stream == null) { 
                stream = Shader.DefaultStream;
            }
            
            shader.Execute(stream, src1, src2, dst, length);
        }

        /// <summary>減算</summary>
        public static void Sub(uint length, CudaArray<float> src1, CudaArray<float> src2, CudaArray<float> dst, Stream stream = null) {
            Shader shader = BinaryArithmetric("sub_ew", "#y = #x1 - #x2;");
            
            if(stream == null) { 
                stream = Shader.DefaultStream;
            }
            
            shader.Execute(stream, src1, src2, dst, length);
        }

        /// <summary>乗算</summary>
        public static void Mul(uint length, CudaArray<float> src1, CudaArray<float> src2, CudaArray<float> dst, Stream stream = null) {
            Shader shader = BinaryArithmetric("mul_ew", "#y = #x1 * #x2;");
            
            if(stream == null) { 
                stream = Shader.DefaultStream;
            }
            
            shader.Execute(stream, src1, src2, dst, length);
        }

        /// <summary>除算</summary>
        public static void Div(uint length, CudaArray<float> src1, CudaArray<float> src2, CudaArray<float> dst, Stream stream = null) {
            Shader shader = BinaryArithmetric("div_ew", "#y = #x1 / #x2;");
            
            if(stream == null) { 
                stream = Shader.DefaultStream;
            }
            
            shader.Execute(stream, src1, src2, dst, length);
        }

        /// <summary>べき乗</summary>
        public static void Pow(uint length, CudaArray<float> src1, CudaArray<float> src2, CudaArray<float> dst, Stream stream = null) {
            Shader shader = BinaryArithmetric("pow_ew", "#y = powf(#x1, #x2);");
            
            if(stream == null) { 
                stream = Shader.DefaultStream;
            }
            
            shader.Execute(stream, src1, src2, dst, length);
        }

        /// <summary>符号付きべき乗</summary>
        public static void SignedPow(uint length, CudaArray<float> src1, CudaArray<float> src2, CudaArray<float> dst, Stream stream = null) {
            Shader shader = BinaryArithmetric("signedpow_ew", "#y = ((#x > 0.0 ? 1.0 : 0.0) - (#x < 0.0 ? 1.0 : 0.0)) * powf(fabs(#x1), #x2);");
            
            if(stream == null) { 
                stream = Shader.DefaultStream;
            }
            
            shader.Execute(stream, src1, src2, dst, length);
        }

        /// <summary>ArcTan2</summary>
        public static void ArcTan2(uint length, CudaArray<float> src1, CudaArray<float> src2, CudaArray<float> dst, Stream stream = null) {
            Shader shader = BinaryArithmetric("atan2_ew", "#y = atan2f(#x1, #x2);");
            
            if(stream == null) { 
                stream = Shader.DefaultStream;
            }
            
            shader.Execute(stream, src1, src2, dst, length);
        }

        /// <summary>加算</summary>
        public static void AddConstant(uint length, float c, CudaArray<float> src, CudaArray<float> dst, Stream stream = null) {
            Shader shader = BinaryConstantArithmetric("add_const_ew", "#y = c + #x;");
            
            if(stream == null) { 
                stream = Shader.DefaultStream;
            }
            
            shader.Execute(stream, c, src, dst, length);
        }

        /// <summary>減算</summary>
        public static void SubConstant(uint length, float c, CudaArray<float> src, CudaArray<float> dst, Stream stream = null) {
            Shader shader = BinaryConstantArithmetric("sub_const_ew", "#y = c - #x;");
            
            if(stream == null) { 
                stream = Shader.DefaultStream;
            }
            
            shader.Execute(stream, c, src, dst, length);
        }

        /// <summary>乗算</summary>
        public static void MulConstant(uint length, float c, CudaArray<float> src, CudaArray<float> dst, Stream stream = null) {
            Shader shader = BinaryConstantArithmetric("mul_const_ew", "#y = c * #x;");
            
            if(stream == null) { 
                stream = Shader.DefaultStream;
            }
            
            shader.Execute(stream, c, src, dst, length);
        }

        /// <summary>除算</summary>
        public static void DivConstant(uint length, float c, CudaArray<float> src, CudaArray<float> dst, Stream stream = null) {
            Shader shader = BinaryConstantArithmetric("div_const_ew", "#y = c / #x;");
            
            if(stream == null) { 
                stream = Shader.DefaultStream;
            }
            
            shader.Execute(stream, c, src, dst, length);
        }

        /// <summary>べき乗</summary>
        public static void PowConstant(uint length, float c, CudaArray<float> src, CudaArray<float> dst, Stream stream = null) {
            Shader shader = BinaryConstantArithmetric("pow_const_ew", "#y = powf(#x, c);");
            
            if(stream == null) { 
                stream = Shader.DefaultStream;
            }
            
            shader.Execute(stream, c, src, dst, length);
        }

        /// <summary>符号付きべき乗</summary>
        public static void SignedPowConstant(uint length, float c, CudaArray<float> src, CudaArray<float> dst, Stream stream = null) {
            Shader shader = BinaryConstantArithmetric("signedpow_const_ew", "#y = ((#x > 0.0 ? 1.0 : 0.0) - (#x < 0.0 ? 1.0 : 0.0)) * powf(fabs(#x), c);");
            
            if(stream == null) { 
                stream = Shader.DefaultStream;
            }
            
            shader.Execute(stream, c, src, dst, length);
        }

        /// <summary>加算</summary>
        public static void AddFactor(uint length, CudaArray<float> c, CudaArray<float> src, CudaArray<float> dst, Stream stream = null) {
            Shader shader = BinaryFactorArithmetric("add_factor_ew", "#y = c + #x;");
            
            if(stream == null) { 
                stream = Shader.DefaultStream;
            }
            
            shader.Execute(stream, c, src, dst, length);
        }

        /// <summary>減算</summary>
        public static void SubFactor(uint length, CudaArray<float> c, CudaArray<float> src, CudaArray<float> dst, Stream stream = null) {
            Shader shader = BinaryFactorArithmetric("sub_factor_ew", "#y = c - #x;");
            
            if(stream == null) { 
                stream = Shader.DefaultStream;
            }
            
            shader.Execute(stream, c, src, dst, length);
        }

        /// <summary>乗算</summary>
        public static void MulFactor(uint length, CudaArray<float> c, CudaArray<float> src, CudaArray<float> dst, Stream stream = null) {
            Shader shader = BinaryFactorArithmetric("mul_factor_ew", "#y = c * #x;");
            
            if(stream == null) { 
                stream = Shader.DefaultStream;
            }
            
            shader.Execute(stream, c, src, dst, length);
        }

        /// <summary>除算</summary>
        public static void DivFactor(uint length, CudaArray<float> c, CudaArray<float> src, CudaArray<float> dst, Stream stream = null) {
            Shader shader = BinaryFactorArithmetric("div_factor_ew", "#y = c / #x;");
            
            if(stream == null) { 
                stream = Shader.DefaultStream;
            }
            
            shader.Execute(stream, c, src, dst, length);
        }

        /// <summary>べき乗</summary>
        public static void PowFactor(uint length, CudaArray<float> c, CudaArray<float> src, CudaArray<float> dst, Stream stream = null) {
            Shader shader = BinaryFactorArithmetric("pow_factor_ew", "#y = powf(#x, c);");
            
            if(stream == null) { 
                stream = Shader.DefaultStream;
            }
            
            shader.Execute(stream, c, src, dst, length);
        }

        /// <summary>符号付きべき乗</summary>
        public static void SignedPowFactor(uint length, CudaArray<float> c, CudaArray<float> src, CudaArray<float> dst, Stream stream = null) {
            Shader shader = BinaryFactorArithmetric("signedpow_factor_ew", "#y = ((#x > 0.0 ? 1.0 : 0.0) - (#x < 0.0 ? 1.0 : 0.0)) * powf(fabs(#x), c);");
            
            if(stream == null) { 
                stream = Shader.DefaultStream;
            }
            
            shader.Execute(stream, c, src, dst, length);
        }

        /// <summary>最大値</summary>
        public static void Maximum(uint length, CudaArray<float> src1, CudaArray<float> src2, CudaArray<float> dst, Stream stream = null) {
            Shader shader = BinaryArithmetric("maximum_ew", "#y = fmaxf(#x1, #x2);");
            
            if(stream == null) { 
                stream = Shader.DefaultStream;
            }
            
            shader.Execute(stream, src1, src2, dst, length);
        }

        /// <summary>最小値</summary>
        public static void Minimum(uint length, CudaArray<float> src1, CudaArray<float> src2, CudaArray<float> dst, Stream stream = null) {
            Shader shader = BinaryArithmetric("minimum_ew", "#y = fminf(#x1, #x2);");
            
            if(stream == null) { 
                stream = Shader.DefaultStream;
            }
            
            shader.Execute(stream, src1, src2, dst, length);
        }

        /// <summary>Clamp</summary>
        public static void Clamp(uint length, CudaArray<float> srcval, CudaArray<float> srcmin, CudaArray<float> srcmax, CudaArray<float> dst, Stream stream = null) {
            Shader shader = TrinaryArithmetric("clamp_ew", "#y = fminf(fmaxf(#x1, #x2), #x3);");
            
            if(stream == null) { 
                stream = Shader.DefaultStream;
            }
            
            shader.Execute(stream, srcval, srcmin, srcmax, dst, length);
        }

        /// <summary>最大値</summary>
        public static void MaximumConstant(uint length, float c, CudaArray<float> src, CudaArray<float> dst, Stream stream = null) {
            Shader shader = BinaryConstantArithmetric("maximum_const_ew", "#y = fmaxf(c, #x);");
            
            if(stream == null) { 
                stream = Shader.DefaultStream;
            }
            
            shader.Execute(stream, c, src, dst, length);
        }

        /// <summary>最小値</summary>
        public static void MinimumConstant(uint length, float c, CudaArray<float> src, CudaArray<float> dst, Stream stream = null) {
            Shader shader = BinaryConstantArithmetric("minimum_const_ew", "#y = fminf(c, #x);");
            
            if(stream == null) { 
                stream = Shader.DefaultStream;
            }
            
            shader.Execute(stream, c, src, dst, length);
        }

        /// <summary>Clamp</summary>
        public static void ClampConstant(uint length, float cmin, float cmax, CudaArray<float> src, CudaArray<float> dst, Stream stream = null) {
            Shader shader = TrinaryBiConstantArithmetric("clamp_const_ew", "#y = fminf(fmaxf(#x, c1), c2);");
            
            if(stream == null) { 
                stream = Shader.DefaultStream;
            }
            
            shader.Execute(stream, cmin, cmax, src, dst, length);
        }

        /// <summary>最大値</summary>
        public static void MaximumFactor(uint length, CudaArray<float> c, CudaArray<float> src, CudaArray<float> dst, Stream stream = null) {
            Shader shader = BinaryFactorArithmetric("maximum_factor_ew", "#y = fmaxf(c, #x);");
            
            if(stream == null) { 
                stream = Shader.DefaultStream;
            }
            
            shader.Execute(stream, c, src, dst, length);
        }

        /// <summary>最小値</summary>
        public static void MinimumFactor(uint length, CudaArray<float> c, CudaArray<float> src, CudaArray<float> dst, Stream stream = null) {
            Shader shader = BinaryFactorArithmetric("minimum_factor_ew", "#y = fminf(c, #x);");
            
            if(stream == null) { 
                stream = Shader.DefaultStream;
            }
            
            shader.Execute(stream, c, src, dst, length);
        }

        /// <summary>Clamp</summary>
        public static void ClampFactor(uint length, CudaArray<float> cmin, CudaArray<float> cmax, CudaArray<float> src, CudaArray<float> dst, Stream stream = null) {
            Shader shader = TrinaryBiFactorArithmetric("clamp_factor_ew", "#y = fminf(fmaxf(#x, c1), c2);");
            
            if(stream == null) { 
                stream = Shader.DefaultStream;
            }
            
            shader.Execute(stream, cmin, cmax, src, dst, length);
        }

        /// <summary>比較</summary>
        public static void GreaterThan(uint length, CudaArray<float> src1, CudaArray<float> src2, CudaArray<float> dst, Stream stream = null) {
            Shader shader = BinaryArithmetric("cmpgt_ew", "#y = #x1 > #x2 ? 1.0 : 0.0;");
            
            if(stream == null) { 
                stream = Shader.DefaultStream;
            }
            
            shader.Execute(stream, src1, src2, dst, length);
        }

        /// <summary>比較</summary>
        public static void GreaterThanOrEqual(uint length, CudaArray<float> src1, CudaArray<float> src2, CudaArray<float> dst, Stream stream = null) {
            Shader shader = BinaryArithmetric("cmpge_ew", "#y = #x1 >= #x2 ? 1.0 : 0.0;");
            
            if(stream == null) { 
                stream = Shader.DefaultStream;
            }
            
            shader.Execute(stream, src1, src2, dst, length);
        }

        /// <summary>比較</summary>
        public static void LessThan(uint length, CudaArray<float> src1, CudaArray<float> src2, CudaArray<float> dst, Stream stream = null) {
            Shader shader = BinaryArithmetric("cmplt_ew", "#y = #x1 < #x2 ? 1.0 : 0.0;");
            
            if(stream == null) { 
                stream = Shader.DefaultStream;
            }
            
            shader.Execute(stream, src1, src2, dst, length);
        }

        /// <summary>比較</summary>
        public static void LessThanOrEqual(uint length, CudaArray<float> src1, CudaArray<float> src2, CudaArray<float> dst, Stream stream = null) {
            Shader shader = BinaryArithmetric("cmple_ew", "#y = #x1 <= #x2 ? 1.0 : 0.0;");
            
            if(stream == null) { 
                stream = Shader.DefaultStream;
            }
            
            shader.Execute(stream, src1, src2, dst, length);
        }

        /// <summary>比較</summary>
        public static void Equal(uint length, CudaArray<float> src1, CudaArray<float> src2, CudaArray<float> dst, Stream stream = null) {
            Shader shader = BinaryArithmetric("cmpeq_ew", "#y = #x1 == #x2 ? 1.0 : 0.0;");
            
            if(stream == null) { 
                stream = Shader.DefaultStream;
            }
            
            shader.Execute(stream, src1, src2, dst, length);
        }

        /// <summary>比較</summary>
        public static void NotEqual(uint length, CudaArray<float> src1, CudaArray<float> src2, CudaArray<float> dst, Stream stream = null) {
            Shader shader = BinaryArithmetric("cmpneq_ew", "#y = #x1 != #x2 ? 1.0 : 0.0;");
            
            if(stream == null) { 
                stream = Shader.DefaultStream;
            }
            
            shader.Execute(stream, src1, src2, dst, length);
        }

        /// <summary>比較</summary>
        public static void GreaterThanConstant(uint length, float c, CudaArray<float> src, CudaArray<float> dst, Stream stream = null) {
            Shader shader = BinaryConstantArithmetric("cmpgt_const_ew", "#y = c > #x ? 1.0 : 0.0;");
            
            if(stream == null) { 
                stream = Shader.DefaultStream;
            }
            
            shader.Execute(stream, c, src, dst, length);
        }

        /// <summary>比較</summary>
        public static void GreaterThanOrEqualConstant(uint length, float c, CudaArray<float> src, CudaArray<float> dst, Stream stream = null) {
            Shader shader = BinaryConstantArithmetric("cmpge_const_ew", "#y = c >= #x ? 1.0 : 0.0;");
            
            if(stream == null) { 
                stream = Shader.DefaultStream;
            }
            
            shader.Execute(stream, c, src, dst, length);
        }

        /// <summary>比較</summary>
        public static void LessThanConstant(uint length, float c, CudaArray<float> src, CudaArray<float> dst, Stream stream = null) {
            Shader shader = BinaryConstantArithmetric("cmplt_const_ew", "#y = c < #x ? 1.0 : 0.0;");
            
            if(stream == null) { 
                stream = Shader.DefaultStream;
            }
            
            shader.Execute(stream, c, src, dst, length);
        }

        /// <summary>比較</summary>
        public static void LessThanOrEqualConstant(uint length, float c, CudaArray<float> src, CudaArray<float> dst, Stream stream = null) {
            Shader shader = BinaryConstantArithmetric("cmple_const_ew", "#y = c <= #x ? 1.0 : 0.0;");
            
            if(stream == null) { 
                stream = Shader.DefaultStream;
            }
            
            shader.Execute(stream, c, src, dst, length);
        }

        /// <summary>比較</summary>
        public static void EqualConstant(uint length, float c, CudaArray<float> src, CudaArray<float> dst, Stream stream = null) {
            Shader shader = BinaryConstantArithmetric("cmpeq_const_ew", "#y = c == #x ? 1.0 : 0.0;");
            
            if(stream == null) { 
                stream = Shader.DefaultStream;
            }
            
            shader.Execute(stream, c, src, dst, length);
        }

        /// <summary>比較</summary>
        public static void NotEqualConstant(uint length, float c, CudaArray<float> src, CudaArray<float> dst, Stream stream = null) {
            Shader shader = BinaryConstantArithmetric("cmpneq_const_ew", "#y = c != #x ? 1.0 : 0.0;");
            
            if(stream == null) { 
                stream = Shader.DefaultStream;
            }
            
            shader.Execute(stream, c, src, dst, length);
        }

        /// <summary>比較</summary>
        public static void GreaterThanFactor(uint length, CudaArray<float> c, CudaArray<float> src, CudaArray<float> dst, Stream stream = null) {
            Shader shader = BinaryFactorArithmetric("cmpgt_factor_ew", "#y = c > #x ? 1.0 : 0.0;");
            
            if(stream == null) { 
                stream = Shader.DefaultStream;
            }
            
            shader.Execute(stream, c, src, dst, length);
        }

        /// <summary>比較</summary>
        public static void GreaterThanOrEqualFactor(uint length, CudaArray<float> c, CudaArray<float> src, CudaArray<float> dst, Stream stream = null) {
            Shader shader = BinaryFactorArithmetric("cmpge_factor_ew", "#y = c >= #x ? 1.0 : 0.0;");
            
            if(stream == null) { 
                stream = Shader.DefaultStream;
            }
            
            shader.Execute(stream, c, src, dst, length);
        }

        /// <summary>比較</summary>
        public static void LessThanFactor(uint length, CudaArray<float> c, CudaArray<float> src, CudaArray<float> dst, Stream stream = null) {
            Shader shader = BinaryFactorArithmetric("cmplt_factor_ew", "#y = c < #x ? 1.0 : 0.0;");
            
            if(stream == null) { 
                stream = Shader.DefaultStream;
            }
            
            shader.Execute(stream, c, src, dst, length);
        }

        /// <summary>比較</summary>
        public static void LessThanOrEqualFactor(uint length, CudaArray<float> c, CudaArray<float> src, CudaArray<float> dst, Stream stream = null) {
            Shader shader = BinaryFactorArithmetric("cmple_factor_ew", "#y = c <= #x ? 1.0 : 0.0;");
            
            if(stream == null) { 
                stream = Shader.DefaultStream;
            }
            
            shader.Execute(stream, c, src, dst, length);
        }

        /// <summary>比較</summary>
        public static void EqualFactor(uint length, CudaArray<float> c, CudaArray<float> src, CudaArray<float> dst, Stream stream = null) {
            Shader shader = BinaryFactorArithmetric("cmpeq_factor_ew", "#y = c == #x ? 1.0 : 0.0;");
            
            if(stream == null) { 
                stream = Shader.DefaultStream;
            }
            
            shader.Execute(stream, c, src, dst, length);
        }

        /// <summary>比較</summary>
        public static void NotEqualFactor(uint length, CudaArray<float> c, CudaArray<float> src, CudaArray<float> dst, Stream stream = null) {
            Shader shader = BinaryFactorArithmetric("cmpneq_factor_ew", "#y = c != #x ? 1.0 : 0.0;");
            
            if(stream == null) { 
                stream = Shader.DefaultStream;
            }
            
            shader.Execute(stream, c, src, dst, length);
        }

        /// <summary>NOT</summary>
        public static void LogicalNot(uint length, CudaArray<float> src, CudaArray<float> dst, Stream stream = null) {
            Shader shader = UnaryArithmetric("logicalnot_ew", "#y = 1.0 - #x;");
            
            if(stream == null) { 
                stream = Shader.DefaultStream;
            }
            
            shader.Execute(stream, src, dst, length);
        }

        /// <summary>AND</summary>
        public static void LogicalAnd(uint length, CudaArray<float> src1, CudaArray<float> src2, CudaArray<float> dst, Stream stream = null) {
            Minimum(length, src1, src2, dst, stream);
        }

        /// <summary>OR</summary>
        public static void LogicalOr(uint length, CudaArray<float> src1, CudaArray<float> src2, CudaArray<float> dst, Stream stream = null) {
            Maximum(length, src1, src2, dst, stream);
        }

        /// <summary>XOR</summary>
        public static void LogicalXor(uint length, CudaArray<float> src1, CudaArray<float> src2, CudaArray<float> dst, Stream stream = null) {
            Shader shader = BinaryArithmetric("logicalxor_ew", "#y = fmaxf(#x1, #x2) - fminf(#x1, #x2);");
            
            if(stream == null) { 
                stream = Shader.DefaultStream;
            }
            
            shader.Execute(stream, src1, src2, dst, length);
        }

        /// <summary>絶対値</summary>
        public static void Abs(uint length, CudaArray<float> src, CudaArray<float> dst, Stream stream = null) {
            Shader shader = UnaryArithmetric("abs_ew", "#y = fabs(#x);");
            
            if(stream == null) { 
                stream = Shader.DefaultStream;
            }
            
            shader.Execute(stream, src, dst, length);
        }

        /// <summary>符号関数</summary>
        public static void Sign(uint length, CudaArray<float> src, CudaArray<float> dst, Stream stream = null) {
            Shader shader = UnaryArithmetric("sign_ew", "#y = ((#x > 0.0 ? 1.0 : 0.0) - (#x < 0 ? 1.0 : 0.0));");
            
            if(stream == null) { 
                stream = Shader.DefaultStream;
            }
            
            shader.Execute(stream, src, dst, length);
        }

        /// <summary>ステップ関数</summary>
        public static void Step(uint length, CudaArray<float> src, CudaArray<float> dst, Stream stream = null) {
            Shader shader = UnaryArithmetric("step_ew", "#y = #x >= 0.0 ? 1.0 : 0.0;");
            
            if(stream == null) { 
                stream = Shader.DefaultStream;
            }
            
            shader.Execute(stream, src, dst, length);
        }

        /// <summary>符号反転</summary>
        public static void Neg(uint length, CudaArray<float> src, CudaArray<float> dst, Stream stream = null) {
            Shader shader = UnaryArithmetric("neg_ew", "#y = -#x;");
            
            if(stream == null) { 
                stream = Shader.DefaultStream;
            }
            
            shader.Execute(stream, src, dst, length);
        }

        /// <summary>逆数</summary>
        public static void Rcp(uint length, CudaArray<float> src, CudaArray<float> dst, Stream stream = null) {
            Shader shader = UnaryArithmetric("rcp_ew", "#y = 1.0 / #x;");
            
            if(stream == null) { 
                stream = Shader.DefaultStream;
            }
            
            shader.Execute(stream, src, dst, length);
        }

        /// <summary>2乗</summary>
        public static void Square(uint length, CudaArray<float> src, CudaArray<float> dst, Stream stream = null) {
            Shader shader = UnaryArithmetric("square_ew", "#y = #x * #x;");
            
            if(stream == null) { 
                stream = Shader.DefaultStream;
            }
            
            shader.Execute(stream, src, dst, length);
        }

        /// <summary>3乗</summary>
        public static void Cube(uint length, CudaArray<float> src, CudaArray<float> dst, Stream stream = null) {
            Shader shader = UnaryArithmetric("cube_ew", "#y = #x * #x * #x;");
            
            if(stream == null) { 
                stream = Shader.DefaultStream;
            }
            
            shader.Execute(stream, src, dst, length);
        }

        /// <summary>平方根</summary>
        public static void Sqrt(uint length, CudaArray<float> src, CudaArray<float> dst, Stream stream = null) {
            Shader shader = UnaryArithmetric("sqrt_ew", "#y = sqrtf(#x);");
            
            if(stream == null) { 
                stream = Shader.DefaultStream;
            }
            
            shader.Execute(stream, src, dst, length);
        }

        /// <summary>平方根逆数</summary>
        public static void Rsqrt(uint length, CudaArray<float> src, CudaArray<float> dst, Stream stream = null) {
            Shader shader = UnaryArithmetric("rsqrt_ew", "#y = rsqrtf(#x);");
            
            if(stream == null) { 
                stream = Shader.DefaultStream;
            }
            
            shader.Execute(stream, src, dst, length);
        }

        /// <summary>符号付き平方根</summary>
        public static void SignedSqrt(uint length, CudaArray<float> src, CudaArray<float> dst, Stream stream = null) {
            Shader shader = UnaryArithmetric("signedsqrt_ew", "#y = ((#x > 0.0 ? 1.0 : 0.0) - (#x < 0 ? 1.0 : 0.0)) * sqrtf(fabs(#x));");
            
            if(stream == null) { 
                stream = Shader.DefaultStream;
            }
            
            shader.Execute(stream, src, dst, length);
        }

        /// <summary>立方根</summary>
        public static void Cbrt(uint length, CudaArray<float> src, CudaArray<float> dst, Stream stream = null) {
            Shader shader = UnaryArithmetric("cbrt_ew", "#y = cbrtf(#x);");
            
            if(stream == null) { 
                stream = Shader.DefaultStream;
            }
            
            shader.Execute(stream, src, dst, length);
        }

        /// <summary>立方根逆数</summary>
        public static void Rcbrt(uint length, CudaArray<float> src, CudaArray<float> dst, Stream stream = null) {
            Shader shader = UnaryArithmetric("rcbrt_ew", "#y = 1.0 / cbrtf(#x);");
            
            if(stream == null) { 
                stream = Shader.DefaultStream;
            }
            
            shader.Execute(stream, src, dst, length);
        }

        /// <summary>Sin</summary>
        public static void Sin(uint length, CudaArray<float> src, CudaArray<float> dst, Stream stream = null) {
            Shader shader = UnaryArithmetric("sin_ew", "#y = sinf(#x);");
            
            if(stream == null) { 
                stream = Shader.DefaultStream;
            }
            
            shader.Execute(stream, src, dst, length);
        }

        /// <summary>Cos</summary>
        public static void Cos(uint length, CudaArray<float> src, CudaArray<float> dst, Stream stream = null) {
            Shader shader = UnaryArithmetric("cos_ew", "#y = cosf(#x);");
            
            if(stream == null) { 
                stream = Shader.DefaultStream;
            }
            
            shader.Execute(stream, src, dst, length);
        }

        /// <summary>Tan</summary>
        public static void Tan(uint length, CudaArray<float> src, CudaArray<float> dst, Stream stream = null) {
            Shader shader = UnaryArithmetric("tan_ew", "#y = tanf(#x);");
            
            if(stream == null) { 
                stream = Shader.DefaultStream;
            }
            
            shader.Execute(stream, src, dst, length);
        }

        /// <summary>Sinh</summary>
        public static void Sinh(uint length, CudaArray<float> src, CudaArray<float> dst, Stream stream = null) {
            Shader shader = UnaryArithmetric("sinh_ew", "#y = sinhf(#x);");
            
            if(stream == null) { 
                stream = Shader.DefaultStream;
            }
            
            shader.Execute(stream, src, dst, length);
        }

        /// <summary>Cosh</summary>
        public static void Cosh(uint length, CudaArray<float> src, CudaArray<float> dst, Stream stream = null) {
            Shader shader = UnaryArithmetric("cosh_ew", "#y = coshf(#x);");
            
            if(stream == null) { 
                stream = Shader.DefaultStream;
            }
            
            shader.Execute(stream, src, dst, length);
        }

        /// <summary>Tanh</summary>
        public static void Tanh(uint length, CudaArray<float> src, CudaArray<float> dst, Stream stream = null) {
            Shader shader = UnaryArithmetric("tanh_ew", "#y = tanhf(#x);");
            
            if(stream == null) { 
                stream = Shader.DefaultStream;
            }
            
            shader.Execute(stream, src, dst, length);
        }

        /// <summary>ArcSin</summary>
        public static void ArcSin(uint length, CudaArray<float> src, CudaArray<float> dst, Stream stream = null) {
            Shader shader = UnaryArithmetric("asin_ew", "#y = asinf(#x);");
            
            if(stream == null) { 
                stream = Shader.DefaultStream;
            }
            
            shader.Execute(stream, src, dst, length);
        }

        /// <summary>ArcCos</summary>
        public static void ArcCos(uint length, CudaArray<float> src, CudaArray<float> dst, Stream stream = null) {
            Shader shader = UnaryArithmetric("acos_ew", "#y = acosf(#x);");
            
            if(stream == null) { 
                stream = Shader.DefaultStream;
            }
            
            shader.Execute(stream, src, dst, length);
        }

        /// <summary>ArcTan</summary>
        public static void ArcTan(uint length, CudaArray<float> src, CudaArray<float> dst, Stream stream = null) {
            Shader shader = UnaryArithmetric("atan_ew", "#y = atanf(#x);");
            
            if(stream == null) { 
                stream = Shader.DefaultStream;
            }
            
            shader.Execute(stream, src, dst, length);
        }

        /// <summary>Exp</summary>
        public static void Exp(uint length, CudaArray<float> src, CudaArray<float> dst, Stream stream = null) {
            Shader shader = UnaryArithmetric("exp_ew", "#y = expf(#x);");
            
            if(stream == null) { 
                stream = Shader.DefaultStream;
            }
            
            shader.Execute(stream, src, dst, length);
        }

        /// <summary>Exp2</summary>
        public static void Exp2(uint length, CudaArray<float> src, CudaArray<float> dst, Stream stream = null) {
            Shader shader = UnaryArithmetric("exp2_ew", "#y = exp2f(#x);");
            
            if(stream == null) { 
                stream = Shader.DefaultStream;
            }
            
            shader.Execute(stream, src, dst, length);
        }

        /// <summary>Log</summary>
        public static void Log(uint length, CudaArray<float> src, CudaArray<float> dst, Stream stream = null) {
            Shader shader = UnaryArithmetric("log_ew", "#y = logf(#x);");
            
            if(stream == null) { 
                stream = Shader.DefaultStream;
            }
            
            shader.Execute(stream, src, dst, length);
        }

        /// <summary>Log2</summary>
        public static void Log2(uint length, CudaArray<float> src, CudaArray<float> dst, Stream stream = null) {
            Shader shader = UnaryArithmetric("log2_ew", "#y = log2f(#x);");
            
            if(stream == null) { 
                stream = Shader.DefaultStream;
            }
            
            shader.Execute(stream, src, dst, length);
        }

        /// <summary>Log10</summary>
        public static void Log10(uint length, CudaArray<float> src, CudaArray<float> dst, Stream stream = null) {
            Shader shader = UnaryArithmetric("log10_ew", "#y = log10f(#x);");
            
            if(stream == null) { 
                stream = Shader.DefaultStream;
            }
            
            shader.Execute(stream, src, dst, length);
        }

        /// <summary>Floor</summary>
        public static void Floor(uint length, CudaArray<float> src, CudaArray<float> dst, Stream stream = null) {
            Shader shader = UnaryArithmetric("floor_ew", "#y = floorf(#x);");
            
            if(stream == null) { 
                stream = Shader.DefaultStream;
            }
            
            shader.Execute(stream, src, dst, length);
        }

        /// <summary>Ceil</summary>
        public static void Ceil(uint length, CudaArray<float> src, CudaArray<float> dst, Stream stream = null) {
            Shader shader = UnaryArithmetric("ceil_ew", "#y = ceilf(#x);");
            
            if(stream == null) { 
                stream = Shader.DefaultStream;
            }
            
            shader.Execute(stream, src, dst, length);
        }

        /// <summary>Round</summary>
        public static void Round(uint length, CudaArray<float> src, CudaArray<float> dst, Stream stream = null) {
            Shader shader = UnaryArithmetric("round_ew", "#y = roundf(#x);");
            
            if(stream == null) { 
                stream = Shader.DefaultStream;
            }
            
            shader.Execute(stream, src, dst, length);
        }

        /// <summary>Sigmoid</summary>
        public static void Sigmoid(uint length, CudaArray<float> src, CudaArray<float> dst, Stream stream = null) {
            Shader shader = UnaryArithmetric("sigmoid_ew", "#y = 1.0 / (1.0 + expf(-#x));");
            
            if(stream == null) { 
                stream = Shader.DefaultStream;
            }
            
            shader.Execute(stream, src, dst, length);
        }

        /// <summary>SoftPlus</summary>
        public static void SoftPlus(uint length, CudaArray<float> src, CudaArray<float> dst, Stream stream = null) {
            Shader shader = UnaryArithmetric("softplus_ew", "#y = logf(1.0 + expf(#x));");
            
            if(stream == null) { 
                stream = Shader.DefaultStream;
            }
            
            shader.Execute(stream, src, dst, length);
        }

        /// <summary>LogCosh</summary>
        public static void LogCosh(uint length, CudaArray<float> src, CudaArray<float> dst, Stream stream = null) {
            Shader shader = UnaryArithmetric("logcosh_ew", "#y = logf(coshf(#x));");
            
            if(stream == null) { 
                stream = Shader.DefaultStream;
            }
            
            shader.Execute(stream, src, dst, length);
        }

        /// <summary>非数判定</summary>
        public static void IsNan(uint length, CudaArray<float> src, CudaArray<float> dst, Stream stream = null) {
            Shader shader = UnaryArithmetric("isnan_ew", "#y = #x != #x;");
            
            if(stream == null) { 
                stream = Shader.DefaultStream;
            }
            
            shader.Execute(stream, src, dst, length);
        }

        /// <summary>NanAsZero</summary>
        public static void NanAsZero(uint length, CudaArray<float> src, CudaArray<float> dst, Stream stream = null) {
            Shader shader = UnaryArithmetric("nanaszero_ew", "#y = #x == #x ? #x : 0.0;");
            
            if(stream == null) { 
                stream = Shader.DefaultStream;
            }
            
            shader.Execute(stream, src, dst, length);
        }

        /// <summary>Relu</summary>
        public static void Relu(uint length, CudaArray<float> src, CudaArray<float> dst, Stream stream = null) {
            Shader shader = UnaryArithmetric("relu_ew", "#y = fmaxf(0.0, #x);");
            
            if(stream == null) { 
                stream = Shader.DefaultStream;
            }
            
            shader.Execute(stream, src, dst, length);
        }

        /// <summary>Relu勾配</summary>
        public static void ReluGrad(uint length, CudaArray<float> src1, CudaArray<float> src2, CudaArray<float> dst, Stream stream = null) {
            Shader shader = BinaryArithmetric("relugrad_ew", "#y = #x2 >= 0.0 ? #x1 : 0.0;");
            
            if(stream == null) { 
                stream = Shader.DefaultStream;
            }
            
            shader.Execute(stream, src1, src2, dst, length);
        }

        /// <summary>LeakyRelu</summary>
        public static void LeakyRelu(uint length, float slope, CudaArray<float> src, CudaArray<float> dst, Stream stream = null) {
            Shader shader = BinaryConstantArithmetric("leakyrelu_ew", "#y = #x - c * fminf(#x, 0.0);");
            
            if(stream == null) { 
                stream = Shader.DefaultStream;
            }
            
            shader.Execute(stream, 1 - slope, src, dst, length);
        }

        /// <summary>LeakyRelu勾配</summary>
        public static void LeakyReluGrad(uint length, float slope, CudaArray<float> src1, CudaArray<float> src2, CudaArray<float> dst, Stream stream = null) {
            Shader shader = TrinaryUniConstantArithmetric("leakyrelugrad_ew", "#y = #x2 >= 0.0 ? #x1 : (c * #x1);");
            
            if(stream == null) { 
                stream = Shader.DefaultStream;
            }
            
            shader.Execute(stream, slope, src1, src2, dst, length);
        }

        /// <summary>Elu</summary>
        public static void Elu(uint length, float slope, CudaArray<float> src, CudaArray<float> dst, Stream stream = null) {
            Shader shader = BinaryConstantArithmetric("elu_ew", "#y = #x >= 0.0 ? #x : (c * (expf(#x) - 1.0));");
            
            if(stream == null) { 
                stream = Shader.DefaultStream;
            }
            
            shader.Execute(stream, slope, src, dst, length);
        }

        /// <summary>Elu勾配</summary>
        public static void EluGrad(uint length, float slope, CudaArray<float> src1, CudaArray<float> src2, CudaArray<float> dst, Stream stream = null) {
            Shader shader = TrinaryUniConstantArithmetric("elugrad_ew", "#y = #x2 >= 0.0 ? #x1 : (c * #x1 * expf(#x2));");
            
            if(stream == null) { 
                stream = Shader.DefaultStream;
            }
            
            shader.Execute(stream, slope, src1, src2, dst, length);
        }
               
        /// <summary>線形補間</summary>
        public static void Lerp(uint length, CudaArray<float> srccondition, CudaArray<float> src1, CudaArray<float> src2, CudaArray<float> dst, Stream stream = null) {
            Shader shader = TrinaryArithmetric("lerp_ew", "#y = #x1 * #x2 + (1.0 - #x1) * #x3;");

            if(stream == null) { 
                stream = Shader.DefaultStream;
            }
            
            shader.Execute(stream, srccondition, src1, src2, dst, length);
        }

        /// <summary>テンソル和</summary>
        public static void Sum(uint length, CudaArray<float>[] src, CudaArray<float> dst, Stream stream = null) {
            int n = src.Length;

            if (n == 0) {
                dst.Zeroset();
                return;
            }
            if (n == 1) {
                src[0].CopyTo(dst, length);
                return;
            }
                        
            if(stream == null) { 
                stream = Shader.DefaultStream;
            }

            Add(length, src[0], src[1], dst, stream);

            for (int i = 2, j = n / 2 * 2; i < j; i += 2) {
                Shader shader = BinaryArithmetric("sum2_ew", "#y += #x1 + #x2;");
                shader.Execute(stream, src[i], src[i + 1], dst, length);
            }

            if ((n & 1) == 1) {
                Shader shader = UnaryArithmetric("sum1_ew", "#y += #x;");
                shader.Execute(stream, src[n - 1], dst, length);
            }
        }
    }
}
