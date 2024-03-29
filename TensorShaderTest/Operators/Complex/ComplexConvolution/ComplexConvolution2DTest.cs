using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;
using TensorShader;
using TensorShader.Operators.ComplexConvolution;
using TensorShaderCudaBackend.API;

namespace TensorShaderTest.Operators.Complex {
    [TestClass]
    public class ComplexConvolution2DTest {
        [TestMethod]
        public void ExecuteFPTest() {
            TensorShaderCudaBackend.Environment.Precision = TensorShaderCudaBackend.Environment.PrecisionMode.Float;
            TensorShaderCudaBackend.Environment.CudnnEnabled = false;

            Random random = new(1234);

            float max_err = 0;

            foreach (int batch in new int[] { 1, 2 }) {
                foreach (int inchannels in new int[] { 2, 4, 10, 20, 32, 34 }) {
                    foreach (int outchannels in new int[] { 2, 4, 10, 20, 32, 34 }) {
                        foreach (int kheight in new int[] { 1, 3, 5 }) {
                            foreach (int kwidth in new int[] { 1, 3, 5 }) {
                                foreach (int inwidth in new int[] { kwidth, kwidth * 2, 8, 9, 13, 17, 25 }) {
                                    foreach (int inheight in new int[] { kheight, kheight * 2, 8, 9, 13, 17, 25 }) {
                                        int outwidth = inwidth - kwidth + 1, outheight = inheight - kheight + 1;

                                        float[] xval = (new float[inwidth * inheight * inchannels * batch]).Select((_, idx) => (float)random.NextDouble() * 1e-2f).ToArray();
                                        float[] wval = (new float[kwidth * kheight * inchannels * outchannels / 2]).Select((_, idx) => (float)random.NextDouble() * 1e-2f).ToArray();

                                        System.Numerics.Complex[] xcval = (new System.Numerics.Complex[xval.Length / 2])
                                            .Select((_, idx) => new System.Numerics.Complex(xval[idx * 2], xval[idx * 2 + 1])).ToArray();

                                        System.Numerics.Complex[] wcval = (new System.Numerics.Complex[wval.Length / 2])
                                            .Select((_, idx) => new System.Numerics.Complex(wval[idx * 2], wval[idx * 2 + 1])).ToArray();

                                        ComplexMap2D x = new(inchannels / 2, inwidth, inheight, batch, xcval);
                                        ComplexFilter2D w = new(inchannels / 2, outchannels / 2, kwidth, kheight, wcval);

                                        ComplexMap2D y = Reference(x, w, kwidth, kheight);

                                        OverflowCheckedTensor x_tensor = new(Shape.Map2D(inchannels, inwidth, inheight, batch), xval);
                                        OverflowCheckedTensor w_tensor = new(Shape.Kernel2D(inchannels, outchannels / 2, kwidth, kheight), wval);

                                        OverflowCheckedTensor y_tensor = new(Shape.Map2D(outchannels, outwidth, outheight, batch));

                                        ComplexConvolution2D ope = new(inwidth, inheight, inchannels, outchannels, kwidth, kheight, gradmode: false, batch);

                                        ope.Execute(x_tensor, w_tensor, y_tensor);

                                        float[] y_expect = y.ToArray();
                                        float[] y_actual = y_tensor.State.Value;

                                        CollectionAssert.AreEqual(xval, x_tensor.State.Value);
                                        CollectionAssert.AreEqual(wval, w_tensor.State.Value);

                                        AssertError.Tolerance(y_expect, y_actual, 1e-6f, 1e-4f, ref max_err, $"mismatch value {inchannels},{outchannels},{kwidth},{kheight},{inwidth},{inheight},{batch}");

                                        Console.WriteLine($"pass: {inchannels},{outchannels},{kwidth},{kheight},{inwidth},{inheight},{batch}");
                                    }
                                }
                            }
                        }
                    }
                }
            }

            Console.WriteLine($"maxerr:{max_err}");
        }

        [TestMethod]
        public void ExecuteFFPTest() {
            TensorShaderCudaBackend.Environment.CudnnEnabled = false;
            TensorShaderCudaBackend.Environment.Precision = TensorShaderCudaBackend.Environment.PrecisionMode.FloatFloat;

            Random random = new(1234);

            float max_err = 0;

            foreach (int batch in new int[] { 1, 2 }) {
                foreach (int inchannels in new int[] { 2, 4, 10, 20, 32, 34 }) {
                    foreach (int outchannels in new int[] { 2, 4, 10, 20, 32, 34 }) {
                        foreach (int kheight in new int[] { 1, 3, 5 }) {
                            foreach (int kwidth in new int[] { 1, 3, 5 }) {
                                foreach (int inwidth in new int[] { kwidth, kwidth * 2, 8, 9, 13, 17, 25 }) {
                                    foreach (int inheight in new int[] { kheight, kheight * 2, 8, 9, 13, 17, 25 }) {
                                        int outwidth = inwidth - kwidth + 1, outheight = inheight - kheight + 1;

                                        float[] xval = (new float[inwidth * inheight * inchannels * batch]).Select((_, idx) => (float)random.NextDouble() * 1e-2f).ToArray();
                                        float[] wval = (new float[kwidth * kheight * inchannels * outchannels / 2]).Select((_, idx) => (float)random.NextDouble() * 1e-2f).ToArray();

                                        System.Numerics.Complex[] xcval = (new System.Numerics.Complex[xval.Length / 2])
                                            .Select((_, idx) => new System.Numerics.Complex(xval[idx * 2], xval[idx * 2 + 1])).ToArray();

                                        System.Numerics.Complex[] wcval = (new System.Numerics.Complex[wval.Length / 2])
                                            .Select((_, idx) => new System.Numerics.Complex(wval[idx * 2], wval[idx * 2 + 1])).ToArray();

                                        ComplexMap2D x = new(inchannels / 2, inwidth, inheight, batch, xcval);
                                        ComplexFilter2D w = new(inchannels / 2, outchannels / 2, kwidth, kheight, wcval);

                                        ComplexMap2D y = Reference(x, w, kwidth, kheight);

                                        OverflowCheckedTensor x_tensor = new(Shape.Map2D(inchannels, inwidth, inheight, batch), xval);
                                        OverflowCheckedTensor w_tensor = new(Shape.Kernel2D(inchannels, outchannels / 2, kwidth, kheight), wval);

                                        OverflowCheckedTensor y_tensor = new(Shape.Map2D(outchannels, outwidth, outheight, batch));

                                        ComplexConvolution2D ope = new(inwidth, inheight, inchannels, outchannels, kwidth, kheight, gradmode: false, batch);

                                        ope.Execute(x_tensor, w_tensor, y_tensor);

                                        float[] y_expect = y.ToArray();
                                        float[] y_actual = y_tensor.State.Value;

                                        CollectionAssert.AreEqual(xval, x_tensor.State.Value);
                                        CollectionAssert.AreEqual(wval, w_tensor.State.Value);

                                        AssertError.Tolerance(y_expect, y_actual, 1e-7f, 1e-5f, ref max_err, $"mismatch value {inchannels},{outchannels},{kwidth},{kheight},{inwidth},{inheight},{batch}");

                                        Console.WriteLine($"pass: {inchannels},{outchannels},{kwidth},{kheight},{inwidth},{inheight},{batch}");
                                    }
                                }
                            }
                        }
                    }
                }
            }

            Console.WriteLine($"maxerr:{max_err}");
        }

        [TestMethod]
        public void LargeMapTest() {
            TensorShaderCudaBackend.Environment.CudnnEnabled = false;
            TensorShaderCudaBackend.Environment.Precision = TensorShaderCudaBackend.Environment.PrecisionMode.FloatFloat;

            Random random = new(1234);

            float max_err = 0;

            int batch = 3;
            int inchannels = 98, outchannels = 100;
            int kwidth = 5, kheight = 3;
            int inwidth = 250, inheight = 196;
            int outwidth = inwidth - kwidth + 1, outheight = inheight - kheight + 1;

            float[] xval = (new float[inwidth * inheight * inchannels * batch]).Select((_, idx) => (float)random.NextDouble() * 1e-2f).ToArray();
            float[] wval = (new float[kwidth * kheight * inchannels * outchannels / 2]).Select((_, idx) => (float)random.NextDouble() * 1e-2f).ToArray();

            System.Numerics.Complex[] xcval = (new System.Numerics.Complex[xval.Length / 2])
                .Select((_, idx) => new System.Numerics.Complex(xval[idx * 2], xval[idx * 2 + 1])).ToArray();

            System.Numerics.Complex[] wcval = (new System.Numerics.Complex[wval.Length / 2])
                .Select((_, idx) => new System.Numerics.Complex(wval[idx * 2], wval[idx * 2 + 1])).ToArray();

            ComplexMap2D x = new(inchannels / 2, inwidth, inheight, batch, xcval);
            ComplexFilter2D w = new(inchannels / 2, outchannels / 2, kwidth, kheight, wcval);

            ComplexMap2D y = Reference(x, w, kwidth, kheight);

            OverflowCheckedTensor x_tensor = new(Shape.Map2D(inchannels, inwidth, inheight, batch), xval);
            OverflowCheckedTensor w_tensor = new(Shape.Kernel2D(inchannels, outchannels / 2, kwidth, kheight), wval);

            OverflowCheckedTensor y_tensor = new(Shape.Map2D(outchannels, outwidth, outheight, batch));

            ComplexConvolution2D ope = new(inwidth, inheight, inchannels, outchannels, kwidth, kheight, gradmode: false, batch);

            ope.Execute(x_tensor, w_tensor, y_tensor);

            float[] y_expect = y.ToArray();
            float[] y_actual = y_tensor.State.Value;

            CollectionAssert.AreEqual(xval, x_tensor.State.Value);
            CollectionAssert.AreEqual(wval, w_tensor.State.Value);

            AssertError.Tolerance(y_expect, y_actual, 1e-7f, 1e-5f, ref max_err, $"mismatch value {inchannels},{outchannels},{kwidth},{kheight},{inwidth},{inheight},{batch}");

            Console.WriteLine($"pass: {inchannels},{outchannels},{kwidth},{kheight},{inwidth},{inheight},{batch}");

            Console.WriteLine($"maxerr:{max_err}");
        }

        [TestMethod]
        public void SpeedFPTest() {
            TensorShaderCudaBackend.Environment.Precision = TensorShaderCudaBackend.Environment.PrecisionMode.Float;
            TensorShaderCudaBackend.Environment.CudnnEnabled = false;

            int inwidth = 512, inheight = 512, inchannels = 32, outchannels = 32, ksize = 3;
            int outwidth = inwidth - ksize + 1, outheight = inheight - ksize + 1;

            OverflowCheckedTensor x_tensor = new(Shape.Map2D(inchannels, inwidth, inheight));
            OverflowCheckedTensor w_tensor = new(Shape.Kernel2D(inchannels, outchannels / 2, ksize, ksize));

            OverflowCheckedTensor y_tensor = new(Shape.Map2D(outchannels, outwidth, outheight));

            ComplexConvolution2D ope = new(inwidth, inheight, inchannels, outchannels, ksize, ksize);

            Cuda.Profiler.Initialize("../../../../profiler.nvsetting", "../../nvprofiles/complex_convolution_2d_fp.nvvp");
            Cuda.Profiler.Start();

            ope.Execute(x_tensor, w_tensor, y_tensor);

            Cuda.Profiler.Stop();
        }

        [TestMethod]
        public void SpeedFFPTest() {
            TensorShaderCudaBackend.Environment.CudnnEnabled = false;
            TensorShaderCudaBackend.Environment.Precision = TensorShaderCudaBackend.Environment.PrecisionMode.FloatFloat;

            int inwidth = 512, inheight = 512, inchannels = 32, outchannels = 32, ksize = 3;
            int outwidth = inwidth - ksize + 1, outheight = inheight - ksize + 1;

            OverflowCheckedTensor x_tensor = new(Shape.Map2D(inchannels, inwidth, inheight));
            OverflowCheckedTensor w_tensor = new(Shape.Kernel2D(inchannels, outchannels / 2, ksize, ksize));

            OverflowCheckedTensor y_tensor = new(Shape.Map2D(outchannels, outwidth, outheight));

            ComplexConvolution2D ope = new(inwidth, inheight, inchannels, outchannels, ksize, ksize);

            Cuda.Profiler.Initialize("../../../../profiler.nvsetting", "../../nvprofiles/complex_convolution_2d_ffp.nvvp");
            Cuda.Profiler.Start();

            ope.Execute(x_tensor, w_tensor, y_tensor);

            Cuda.Profiler.Stop();
        }

        public static ComplexMap2D Reference(ComplexMap2D x, ComplexFilter2D w, int kwidth, int kheight) {
            int inchannels = x.Channels, outchannels = w.OutChannels, batch = x.Batch;
            int inw = x.Width, inh = x.Height, outw = inw - kwidth + 1, outh = inh - kheight + 1;

            ComplexMap2D y = new(outchannels, outw, outh, batch);

            for (int kx, ky = 0; ky < kheight; ky++) {
                for (kx = 0; kx < kwidth; kx++) {
                    for (int th = 0; th < batch; th++) {
                        for (int ox, oy = 0; oy < outh; oy++) {
                            for (ox = 0; ox < outw; ox++) {
                                for (int outch = 0; outch < outchannels; outch++) {
                                    System.Numerics.Complex sum = y[outch, ox, oy, th];

                                    for (int inch = 0; inch < inchannels; inch++) {
                                        sum += x[inch, kx + ox, ky + oy, th] * w[inch, outch, kx, ky];
                                    }

                                    y[outch, ox, oy, th] = sum;
                                }
                            }
                        }
                    }
                }
            }

            return y;
        }

        [TestMethod]
        public void ReferenceTest() {
            int inchannels = 6, outchannels = 8, kwidth = 3, kheight = 5, inwidth = 13, inheight = 17, batch = 3;
            int outwidth = inwidth - kwidth + 1, outheight = inheight - kheight + 1;

            float[] xval = (new float[batch * inwidth * inheight * inchannels]).Select((_, idx) => idx * 1e-3f).ToArray();
            float[] wval = (new float[kwidth * kheight * outchannels * inchannels / 2]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

            System.Numerics.Complex[] xcval = (new System.Numerics.Complex[xval.Length / 2])
                .Select((_, idx) => new System.Numerics.Complex(xval[idx * 2], xval[idx * 2 + 1])).ToArray();

            System.Numerics.Complex[] wcval = (new System.Numerics.Complex[wval.Length / 2])
                .Select((_, idx) => new System.Numerics.Complex(wval[idx * 2], wval[idx * 2 + 1])).ToArray();

            ComplexMap2D x = new(inchannels / 2, inwidth, inheight, batch, xcval);
            ComplexFilter2D w = new(inchannels / 2, outchannels / 2, kwidth, kheight, wcval);

            ComplexMap2D y = Reference(x, w, kwidth, kheight);

            float[] y_expect = {
                -1.080000000e-03f,  1.771005000e+00f,  -8.100000000e-04f,  1.682175000e+00f,  -5.400000000e-04f,  1.593345000e+00f,
                -2.700000000e-04f,  1.504515000e+00f,  -8.100000000e-04f,  1.872795000e+00f,  -5.400000000e-04f,  1.780725000e+00f,
                -2.700000000e-04f,  1.688655000e+00f,  0.000000000e+00f,  1.596585000e+00f,  -5.400000000e-04f,  1.974585000e+00f,
                -2.700000000e-04f,  1.879275000e+00f,  1.110223025e-16f,  1.783965000e+00f,  2.700000000e-04f,  1.688655000e+00f,
                -2.700000000e-04f,  2.076375000e+00f,  4.440892099e-16f,  1.977825000e+00f,  2.700000000e-04f,  1.879275000e+00f,
                5.400000000e-04f,  1.780725000e+00f,  -6.661338148e-16f,  2.178165000e+00f,  2.700000000e-04f,  2.076375000e+00f,
                5.400000000e-04f,  1.974585000e+00f,  8.100000000e-04f,  1.872795000e+00f,  2.700000000e-04f,  2.279955000e+00f,
                5.400000000e-04f,  2.174925000e+00f,  8.100000000e-04f,  2.069895000e+00f,  1.080000000e-03f,  1.964865000e+00f,
                5.400000000e-04f,  2.381745000e+00f,  8.100000000e-04f,  2.273475000e+00f,  1.080000000e-03f,  2.165205000e+00f,
                1.350000000e-03f,  2.056935000e+00f,  8.100000000e-04f,  2.483535000e+00f,  1.080000000e-03f,  2.372025000e+00f,
                1.350000000e-03f,  2.260515000e+00f,  1.620000000e-03f,  2.149005000e+00f,  1.080000000e-03f,  2.585325000e+00f,
                1.350000000e-03f,  2.470575000e+00f,  1.620000000e-03f,  2.355825000e+00f,  1.890000000e-03f,  2.241075000e+00f,
                1.350000000e-03f,  2.687115000e+00f,  1.620000000e-03f,  2.569125000e+00f,  1.890000000e-03f,  2.451135000e+00f,
                2.160000000e-03f,  2.333145000e+00f,  1.620000000e-03f,  2.788905000e+00f,  1.890000000e-03f,  2.667675000e+00f,
                2.160000000e-03f,  2.546445000e+00f,  2.430000000e-03f,  2.425215000e+00f,  2.430000000e-03f,  3.094275000e+00f,
                2.700000000e-03f,  2.963325000e+00f,  2.970000000e-03f,  2.832375000e+00f,  3.240000000e-03f,  2.701425000e+00f,
                2.700000000e-03f,  3.196065000e+00f,  2.970000000e-03f,  3.061875000e+00f,  3.240000000e-03f,  2.927685000e+00f,
                3.510000000e-03f,  2.793495000e+00f,  2.970000000e-03f,  3.297855000e+00f,  3.240000000e-03f,  3.160425000e+00f,
                3.510000000e-03f,  3.022995000e+00f,  3.780000000e-03f,  2.885565000e+00f,  3.240000000e-03f,  3.399645000e+00f,
                3.510000000e-03f,  3.258975000e+00f,  3.780000000e-03f,  3.118305000e+00f,  4.050000000e-03f,  2.977635000e+00f,
                3.510000000e-03f,  3.501435000e+00f,  3.780000000e-03f,  3.357525000e+00f,  4.050000000e-03f,  3.213615000e+00f,
                4.320000000e-03f,  3.069705000e+00f,  3.780000000e-03f,  3.603225000e+00f,  4.050000000e-03f,  3.456075000e+00f,
                4.320000000e-03f,  3.308925000e+00f,  4.590000000e-03f,  3.161775000e+00f,  4.050000000e-03f,  3.705015000e+00f,
                4.320000000e-03f,  3.554625000e+00f,  4.590000000e-03f,  3.404235000e+00f,  4.860000000e-03f,  3.253845000e+00f,
                4.320000000e-03f,  3.806805000e+00f,  4.590000000e-03f,  3.653175000e+00f,  4.860000000e-03f,  3.499545000e+00f,
                5.130000000e-03f,  3.345915000e+00f,  4.590000000e-03f,  3.908595000e+00f,  4.860000000e-03f,  3.751725000e+00f,
                5.130000000e-03f,  3.594855000e+00f,  5.400000000e-03f,  3.437985000e+00f,  4.860000000e-03f,  4.010385000e+00f,
                5.130000000e-03f,  3.850275000e+00f,  5.400000000e-03f,  3.690165000e+00f,  5.670000000e-03f,  3.530055000e+00f,
                5.130000000e-03f,  4.112175000e+00f,  5.400000000e-03f,  3.948825000e+00f,  5.670000000e-03f,  3.785475000e+00f,
                5.940000000e-03f,  3.622125000e+00f,  5.940000000e-03f,  4.417545000e+00f,  6.210000000e-03f,  4.244475000e+00f,
                6.480000000e-03f,  4.071405000e+00f,  6.750000000e-03f,  3.898335000e+00f,  6.210000000e-03f,  4.519335000e+00f,
                6.480000000e-03f,  4.343025000e+00f,  6.750000000e-03f,  4.166715000e+00f,  7.020000000e-03f,  3.990405000e+00f,
                6.480000000e-03f,  4.621125000e+00f,  6.750000000e-03f,  4.441575000e+00f,  7.020000000e-03f,  4.262025000e+00f,
                7.290000000e-03f,  4.082475000e+00f,  6.750000000e-03f,  4.722915000e+00f,  7.020000000e-03f,  4.540125000e+00f,
                7.290000000e-03f,  4.357335000e+00f,  7.560000000e-03f,  4.174545000e+00f,  7.020000000e-03f,  4.824705000e+00f,
                7.290000000e-03f,  4.638675000e+00f,  7.560000000e-03f,  4.452645000e+00f,  7.830000000e-03f,  4.266615000e+00f,
                7.290000000e-03f,  4.926495000e+00f,  7.560000000e-03f,  4.737225000e+00f,  7.830000000e-03f,  4.547955000e+00f,
                8.100000000e-03f,  4.358685000e+00f,  7.560000000e-03f,  5.028285000e+00f,  7.830000000e-03f,  4.835775000e+00f,
                8.100000000e-03f,  4.643265000e+00f,  8.370000000e-03f,  4.450755000e+00f,  7.830000000e-03f,  5.130075000e+00f,
                8.100000000e-03f,  4.934325000e+00f,  8.370000000e-03f,  4.738575000e+00f,  8.640000000e-03f,  4.542825000e+00f,
                8.100000000e-03f,  5.231865000e+00f,  8.370000000e-03f,  5.032875000e+00f,  8.640000000e-03f,  4.833885000e+00f,
                8.910000000e-03f,  4.634895000e+00f,  8.370000000e-03f,  5.333655000e+00f,  8.640000000e-03f,  5.131425000e+00f,
                8.910000000e-03f,  4.929195000e+00f,  9.180000000e-03f,  4.726965000e+00f,  8.640000000e-03f,  5.435445000e+00f,
                8.910000000e-03f,  5.229975000e+00f,  9.180000000e-03f,  5.024505000e+00f,  9.450000000e-03f,  4.819035000e+00f,
                9.450000000e-03f,  5.740815000e+00f,  9.720000000e-03f,  5.525625000e+00f,  9.990000000e-03f,  5.310435000e+00f,
                1.026000000e-02f,  5.095245000e+00f,  9.720000000e-03f,  5.842605000e+00f,  9.990000000e-03f,  5.624175000e+00f,
                1.026000000e-02f,  5.405745000e+00f,  1.053000000e-02f,  5.187315000e+00f,  9.990000000e-03f,  5.944395000e+00f,
                1.026000000e-02f,  5.722725000e+00f,  1.053000000e-02f,  5.501055000e+00f,  1.080000000e-02f,  5.279385000e+00f,
                1.026000000e-02f,  6.046185000e+00f,  1.053000000e-02f,  5.821275000e+00f,  1.080000000e-02f,  5.596365000e+00f,
                1.107000000e-02f,  5.371455000e+00f,  1.053000000e-02f,  6.147975000e+00f,  1.080000000e-02f,  5.919825000e+00f,
                1.107000000e-02f,  5.691675000e+00f,  1.134000000e-02f,  5.463525000e+00f,  1.080000000e-02f,  6.249765000e+00f,
                1.107000000e-02f,  6.018375000e+00f,  1.134000000e-02f,  5.786985000e+00f,  1.161000000e-02f,  5.555595000e+00f,
                1.107000000e-02f,  6.351555000e+00f,  1.134000000e-02f,  6.116925000e+00f,  1.161000000e-02f,  5.882295000e+00f,
                1.188000000e-02f,  5.647665000e+00f,  1.134000000e-02f,  6.453345000e+00f,  1.161000000e-02f,  6.215475000e+00f,
                1.188000000e-02f,  5.977605000e+00f,  1.215000000e-02f,  5.739735000e+00f,  1.161000000e-02f,  6.555135000e+00f,
                1.188000000e-02f,  6.314025000e+00f,  1.215000000e-02f,  6.072915000e+00f,  1.242000000e-02f,  5.831805000e+00f,
                1.188000000e-02f,  6.656925000e+00f,  1.215000000e-02f,  6.412575000e+00f,  1.242000000e-02f,  6.168225000e+00f,
                1.269000000e-02f,  5.923875000e+00f,  1.215000000e-02f,  6.758715000e+00f,  1.242000000e-02f,  6.511125000e+00f,
                1.269000000e-02f,  6.263535000e+00f,  1.296000000e-02f,  6.015945000e+00f,  1.296000000e-02f,  7.064085000e+00f,
                1.323000000e-02f,  6.806775000e+00f,  1.350000000e-02f,  6.549465000e+00f,  1.377000000e-02f,  6.292155000e+00f,
                1.323000000e-02f,  7.165875000e+00f,  1.350000000e-02f,  6.905325000e+00f,  1.377000000e-02f,  6.644775000e+00f,
                1.404000000e-02f,  6.384225000e+00f,  1.350000000e-02f,  7.267665000e+00f,  1.377000000e-02f,  7.003875000e+00f,
                1.404000000e-02f,  6.740085000e+00f,  1.431000000e-02f,  6.476295000e+00f,  1.377000000e-02f,  7.369455000e+00f,
                1.404000000e-02f,  7.102425000e+00f,  1.431000000e-02f,  6.835395000e+00f,  1.458000000e-02f,  6.568365000e+00f,
                1.404000000e-02f,  7.471245000e+00f,  1.431000000e-02f,  7.200975000e+00f,  1.458000000e-02f,  6.930705000e+00f,
                1.485000000e-02f,  6.660435000e+00f,  1.431000000e-02f,  7.573035000e+00f,  1.458000000e-02f,  7.299525000e+00f,
                1.485000000e-02f,  7.026015000e+00f,  1.512000000e-02f,  6.752505000e+00f,  1.458000000e-02f,  7.674825000e+00f,
                1.485000000e-02f,  7.398075000e+00f,  1.512000000e-02f,  7.121325000e+00f,  1.539000000e-02f,  6.844575000e+00f,
                1.485000000e-02f,  7.776615000e+00f,  1.512000000e-02f,  7.496625000e+00f,  1.539000000e-02f,  7.216635000e+00f,
                1.566000000e-02f,  6.936645000e+00f,  1.512000000e-02f,  7.878405000e+00f,  1.539000000e-02f,  7.595175000e+00f,
                1.566000000e-02f,  7.311945000e+00f,  1.593000000e-02f,  7.028715000e+00f,  1.539000000e-02f,  7.980195000e+00f,
                1.566000000e-02f,  7.693725000e+00f,  1.593000000e-02f,  7.407255000e+00f,  1.620000000e-02f,  7.120785000e+00f,
                1.566000000e-02f,  8.081985000e+00f,  1.593000000e-02f,  7.792275000e+00f,  1.620000000e-02f,  7.502565000e+00f,
                1.647000000e-02f,  7.212855000e+00f,  1.647000000e-02f,  8.387355000e+00f,  1.674000000e-02f,  8.087925000e+00f,
                1.701000000e-02f,  7.788495000e+00f,  1.728000000e-02f,  7.489065000e+00f,  1.674000000e-02f,  8.489145000e+00f,
                1.701000000e-02f,  8.186475000e+00f,  1.728000000e-02f,  7.883805000e+00f,  1.755000000e-02f,  7.581135000e+00f,
                1.701000000e-02f,  8.590935000e+00f,  1.728000000e-02f,  8.285025000e+00f,  1.755000000e-02f,  7.979115000e+00f,
                1.782000000e-02f,  7.673205000e+00f,  1.728000000e-02f,  8.692725000e+00f,  1.755000000e-02f,  8.383575000e+00f,
                1.782000000e-02f,  8.074425000e+00f,  1.809000000e-02f,  7.765275000e+00f,  1.755000000e-02f,  8.794515000e+00f,
                1.782000000e-02f,  8.482125000e+00f,  1.809000000e-02f,  8.169735000e+00f,  1.836000000e-02f,  7.857345000e+00f,
                1.782000000e-02f,  8.896305000e+00f,  1.809000000e-02f,  8.580675000e+00f,  1.836000000e-02f,  8.265045000e+00f,
                1.863000000e-02f,  7.949415000e+00f,  1.809000000e-02f,  8.998095000e+00f,  1.836000000e-02f,  8.679225000e+00f,
                1.863000000e-02f,  8.360355000e+00f,  1.890000000e-02f,  8.041485000e+00f,  1.836000000e-02f,  9.099885000e+00f,
                1.863000000e-02f,  8.777775000e+00f,  1.890000000e-02f,  8.455665000e+00f,  1.917000000e-02f,  8.133555000e+00f,
                1.863000000e-02f,  9.201675000e+00f,  1.890000000e-02f,  8.876325000e+00f,  1.917000000e-02f,  8.550975000e+00f,
                1.944000000e-02f,  8.225625000e+00f,  1.890000000e-02f,  9.303465000e+00f,  1.917000000e-02f,  8.974875000e+00f,
                1.944000000e-02f,  8.646285000e+00f,  1.971000000e-02f,  8.317695000e+00f,  1.917000000e-02f,  9.405255000e+00f,
                1.944000000e-02f,  9.073425000e+00f,  1.971000000e-02f,  8.741595000e+00f,  1.998000000e-02f,  8.409765000e+00f,
                1.998000000e-02f,  9.710625000e+00f,  2.025000000e-02f,  9.369075000e+00f,  2.052000000e-02f,  9.027525000e+00f,
                2.079000000e-02f,  8.685975000e+00f,  2.025000000e-02f,  9.812415000e+00f,  2.052000000e-02f,  9.467625000e+00f,
                2.079000000e-02f,  9.122835000e+00f,  2.106000000e-02f,  8.778045000e+00f,  2.052000000e-02f,  9.914205000e+00f,
                2.079000000e-02f,  9.566175000e+00f,  2.106000000e-02f,  9.218145000e+00f,  2.133000000e-02f,  8.870115000e+00f,
                2.079000000e-02f,  1.001599500e+01f,  2.106000000e-02f,  9.664725000e+00f,  2.133000000e-02f,  9.313455000e+00f,
                2.160000000e-02f,  8.962185000e+00f,  2.106000000e-02f,  1.011778500e+01f,  2.133000000e-02f,  9.763275000e+00f,
                2.160000000e-02f,  9.408765000e+00f,  2.187000000e-02f,  9.054255000e+00f,  2.133000000e-02f,  1.021957500e+01f,
                2.160000000e-02f,  9.861825000e+00f,  2.187000000e-02f,  9.504075000e+00f,  2.214000000e-02f,  9.146325000e+00f,
                2.160000000e-02f,  1.032136500e+01f,  2.187000000e-02f,  9.960375000e+00f,  2.214000000e-02f,  9.599385000e+00f,
                2.241000000e-02f,  9.238395000e+00f,  2.187000000e-02f,  1.042315500e+01f,  2.214000000e-02f,  1.005892500e+01f,
                2.241000000e-02f,  9.694695000e+00f,  2.268000000e-02f,  9.330465000e+00f,  2.214000000e-02f,  1.052494500e+01f,
                2.241000000e-02f,  1.015747500e+01f,  2.268000000e-02f,  9.790005000e+00f,  2.295000000e-02f,  9.422535000e+00f,
                2.241000000e-02f,  1.062673500e+01f,  2.268000000e-02f,  1.025602500e+01f,  2.295000000e-02f,  9.885315000e+00f,
                2.322000000e-02f,  9.514605000e+00f,  2.268000000e-02f,  1.072852500e+01f,  2.295000000e-02f,  1.035457500e+01f,
                2.322000000e-02f,  9.980625000e+00f,  2.349000000e-02f,  9.606675000e+00f,  2.349000000e-02f,  1.103389500e+01f,
                2.376000000e-02f,  1.065022500e+01f,  2.403000000e-02f,  1.026655500e+01f,  2.430000000e-02f,  9.882885000e+00f,
                2.376000000e-02f,  1.113568500e+01f,  2.403000000e-02f,  1.074877500e+01f,  2.430000000e-02f,  1.036186500e+01f,
                2.457000000e-02f,  9.974955000e+00f,  2.403000000e-02f,  1.123747500e+01f,  2.430000000e-02f,  1.084732500e+01f,
                2.457000000e-02f,  1.045717500e+01f,  2.484000000e-02f,  1.006702500e+01f,  2.430000000e-02f,  1.133926500e+01f,
                2.457000000e-02f,  1.094587500e+01f,  2.484000000e-02f,  1.055248500e+01f,  2.511000000e-02f,  1.015909500e+01f,
                2.457000000e-02f,  1.144105500e+01f,  2.484000000e-02f,  1.104442500e+01f,  2.511000000e-02f,  1.064779500e+01f,
                2.538000000e-02f,  1.025116500e+01f,  2.484000000e-02f,  1.154284500e+01f,  2.511000000e-02f,  1.114297500e+01f,
                2.538000000e-02f,  1.074310500e+01f,  2.565000000e-02f,  1.034323500e+01f,  2.511000000e-02f,  1.164463500e+01f,
                2.538000000e-02f,  1.124152500e+01f,  2.565000000e-02f,  1.083841500e+01f,  2.592000000e-02f,  1.043530500e+01f,
                2.538000000e-02f,  1.174642500e+01f,  2.565000000e-02f,  1.134007500e+01f,  2.592000000e-02f,  1.093372500e+01f,
                2.619000000e-02f,  1.052737500e+01f,  2.565000000e-02f,  1.184821500e+01f,  2.592000000e-02f,  1.143862500e+01f,
                2.619000000e-02f,  1.102903500e+01f,  2.646000000e-02f,  1.061944500e+01f,  2.592000000e-02f,  1.195000500e+01f,
                2.619000000e-02f,  1.153717500e+01f,  2.646000000e-02f,  1.112434500e+01f,  2.673000000e-02f,  1.071151500e+01f,
                2.619000000e-02f,  1.205179500e+01f,  2.646000000e-02f,  1.163572500e+01f,  2.673000000e-02f,  1.121965500e+01f,
                2.700000000e-02f,  1.080358500e+01f,  2.700000000e-02f,  1.235716500e+01f,  2.727000000e-02f,  1.193137500e+01f,
                2.754000000e-02f,  1.150558500e+01f,  2.781000000e-02f,  1.107979500e+01f,  2.727000000e-02f,  1.245895500e+01f,
                2.754000000e-02f,  1.202992500e+01f,  2.781000000e-02f,  1.160089500e+01f,  2.808000000e-02f,  1.117186500e+01f,
                2.754000000e-02f,  1.256074500e+01f,  2.781000000e-02f,  1.212847500e+01f,  2.808000000e-02f,  1.169620500e+01f,
                2.835000000e-02f,  1.126393500e+01f,  2.781000000e-02f,  1.266253500e+01f,  2.808000000e-02f,  1.222702500e+01f,
                2.835000000e-02f,  1.179151500e+01f,  2.862000000e-02f,  1.135600500e+01f,  2.808000000e-02f,  1.276432500e+01f,
                2.835000000e-02f,  1.232557500e+01f,  2.862000000e-02f,  1.188682500e+01f,  2.889000000e-02f,  1.144807500e+01f,
                2.835000000e-02f,  1.286611500e+01f,  2.862000000e-02f,  1.242412500e+01f,  2.889000000e-02f,  1.198213500e+01f,
                2.916000000e-02f,  1.154014500e+01f,  2.862000000e-02f,  1.296790500e+01f,  2.889000000e-02f,  1.252267500e+01f,
                2.916000000e-02f,  1.207744500e+01f,  2.943000000e-02f,  1.163221500e+01f,  2.889000000e-02f,  1.306969500e+01f,
                2.916000000e-02f,  1.262122500e+01f,  2.943000000e-02f,  1.217275500e+01f,  2.970000000e-02f,  1.172428500e+01f,
                2.916000000e-02f,  1.317148500e+01f,  2.943000000e-02f,  1.271977500e+01f,  2.970000000e-02f,  1.226806500e+01f,
                2.997000000e-02f,  1.181635500e+01f,  2.943000000e-02f,  1.327327500e+01f,  2.970000000e-02f,  1.281832500e+01f,
                2.997000000e-02f,  1.236337500e+01f,  3.024000000e-02f,  1.190842500e+01f,  2.970000000e-02f,  1.337506500e+01f,
                2.997000000e-02f,  1.291687500e+01f,  3.024000000e-02f,  1.245868500e+01f,  3.051000000e-02f,  1.200049500e+01f,
                3.051000000e-02f,  1.368043500e+01f,  3.078000000e-02f,  1.321252500e+01f,  3.105000000e-02f,  1.274461500e+01f,
                3.132000000e-02f,  1.227670500e+01f,  3.078000000e-02f,  1.378222500e+01f,  3.105000000e-02f,  1.331107500e+01f,
                3.132000000e-02f,  1.283992500e+01f,  3.159000000e-02f,  1.236877500e+01f,  3.105000000e-02f,  1.388401500e+01f,
                3.132000000e-02f,  1.340962500e+01f,  3.159000000e-02f,  1.293523500e+01f,  3.186000000e-02f,  1.246084500e+01f,
                3.132000000e-02f,  1.398580500e+01f,  3.159000000e-02f,  1.350817500e+01f,  3.186000000e-02f,  1.303054500e+01f,
                3.213000000e-02f,  1.255291500e+01f,  3.159000000e-02f,  1.408759500e+01f,  3.186000000e-02f,  1.360672500e+01f,
                3.213000000e-02f,  1.312585500e+01f,  3.240000000e-02f,  1.264498500e+01f,  3.186000000e-02f,  1.418938500e+01f,
                3.213000000e-02f,  1.370527500e+01f,  3.240000000e-02f,  1.322116500e+01f,  3.267000000e-02f,  1.273705500e+01f,
                3.213000000e-02f,  1.429117500e+01f,  3.240000000e-02f,  1.380382500e+01f,  3.267000000e-02f,  1.331647500e+01f,
                3.294000000e-02f,  1.282912500e+01f,  3.240000000e-02f,  1.439296500e+01f,  3.267000000e-02f,  1.390237500e+01f,
                3.294000000e-02f,  1.341178500e+01f,  3.321000000e-02f,  1.292119500e+01f,  3.267000000e-02f,  1.449475500e+01f,
                3.294000000e-02f,  1.400092500e+01f,  3.321000000e-02f,  1.350709500e+01f,  3.348000000e-02f,  1.301326500e+01f,
                3.294000000e-02f,  1.459654500e+01f,  3.321000000e-02f,  1.409947500e+01f,  3.348000000e-02f,  1.360240500e+01f,
                3.375000000e-02f,  1.310533500e+01f,  3.321000000e-02f,  1.469833500e+01f,  3.348000000e-02f,  1.419802500e+01f,
                3.375000000e-02f,  1.369771500e+01f,  3.402000000e-02f,  1.319740500e+01f,  3.402000000e-02f,  1.500370500e+01f,
                3.429000000e-02f,  1.449367500e+01f,  3.456000000e-02f,  1.398364500e+01f,  3.483000000e-02f,  1.347361500e+01f,
                3.429000000e-02f,  1.510549500e+01f,  3.456000000e-02f,  1.459222500e+01f,  3.483000000e-02f,  1.407895500e+01f,
                3.510000000e-02f,  1.356568500e+01f,  3.456000000e-02f,  1.520728500e+01f,  3.483000000e-02f,  1.469077500e+01f,
                3.510000000e-02f,  1.417426500e+01f,  3.537000000e-02f,  1.365775500e+01f,  3.483000000e-02f,  1.530907500e+01f,
                3.510000000e-02f,  1.478932500e+01f,  3.537000000e-02f,  1.426957500e+01f,  3.564000000e-02f,  1.374982500e+01f,
                3.510000000e-02f,  1.541086500e+01f,  3.537000000e-02f,  1.488787500e+01f,  3.564000000e-02f,  1.436488500e+01f,
                3.591000000e-02f,  1.384189500e+01f,  3.537000000e-02f,  1.551265500e+01f,  3.564000000e-02f,  1.498642500e+01f,
                3.591000000e-02f,  1.446019500e+01f,  3.618000000e-02f,  1.393396500e+01f,  3.564000000e-02f,  1.561444500e+01f,
                3.591000000e-02f,  1.508497500e+01f,  3.618000000e-02f,  1.455550500e+01f,  3.645000000e-02f,  1.402603500e+01f,
                3.591000000e-02f,  1.571623500e+01f,  3.618000000e-02f,  1.518352500e+01f,  3.645000000e-02f,  1.465081500e+01f,
                3.672000000e-02f,  1.411810500e+01f,  3.618000000e-02f,  1.581802500e+01f,  3.645000000e-02f,  1.528207500e+01f,
                3.672000000e-02f,  1.474612500e+01f,  3.699000000e-02f,  1.421017500e+01f,  3.645000000e-02f,  1.591981500e+01f,
                3.672000000e-02f,  1.538062500e+01f,  3.699000000e-02f,  1.484143500e+01f,  3.726000000e-02f,  1.430224500e+01f,
                3.672000000e-02f,  1.602160500e+01f,  3.699000000e-02f,  1.547917500e+01f,  3.726000000e-02f,  1.493674500e+01f,
                3.753000000e-02f,  1.439431500e+01f,  3.753000000e-02f,  1.632697500e+01f,  3.780000000e-02f,  1.577482500e+01f,
                3.807000000e-02f,  1.522267500e+01f,  3.834000000e-02f,  1.467052500e+01f,  3.780000000e-02f,  1.642876500e+01f,
                3.807000000e-02f,  1.587337500e+01f,  3.834000000e-02f,  1.531798500e+01f,  3.861000000e-02f,  1.476259500e+01f,
                3.807000000e-02f,  1.653055500e+01f,  3.834000000e-02f,  1.597192500e+01f,  3.861000000e-02f,  1.541329500e+01f,
                3.888000000e-02f,  1.485466500e+01f,  3.834000000e-02f,  1.663234500e+01f,  3.861000000e-02f,  1.607047500e+01f,
                3.888000000e-02f,  1.550860500e+01f,  3.915000000e-02f,  1.494673500e+01f,  3.861000000e-02f,  1.673413500e+01f,
                3.888000000e-02f,  1.616902500e+01f,  3.915000000e-02f,  1.560391500e+01f,  3.942000000e-02f,  1.503880500e+01f,
                3.888000000e-02f,  1.683592500e+01f,  3.915000000e-02f,  1.626757500e+01f,  3.942000000e-02f,  1.569922500e+01f,
                3.969000000e-02f,  1.513087500e+01f,  3.915000000e-02f,  1.693771500e+01f,  3.942000000e-02f,  1.636612500e+01f,
                3.969000000e-02f,  1.579453500e+01f,  3.996000000e-02f,  1.522294500e+01f,  3.942000000e-02f,  1.703950500e+01f,
                3.969000000e-02f,  1.646467500e+01f,  3.996000000e-02f,  1.588984500e+01f,  4.023000000e-02f,  1.531501500e+01f,
                3.969000000e-02f,  1.714129500e+01f,  3.996000000e-02f,  1.656322500e+01f,  4.023000000e-02f,  1.598515500e+01f,
                4.050000000e-02f,  1.540708500e+01f,  3.996000000e-02f,  1.724308500e+01f,  4.023000000e-02f,  1.666177500e+01f,
                4.050000000e-02f,  1.608046500e+01f,  4.077000000e-02f,  1.549915500e+01f,  4.023000000e-02f,  1.734487500e+01f,
                4.050000000e-02f,  1.676032500e+01f,  4.077000000e-02f,  1.617577500e+01f,  4.104000000e-02f,  1.559122500e+01f,
                4.104000000e-02f,  1.765024500e+01f,  4.131000000e-02f,  1.705597500e+01f,  4.158000000e-02f,  1.646170500e+01f,
                4.185000000e-02f,  1.586743500e+01f,  4.131000000e-02f,  1.775203500e+01f,  4.158000000e-02f,  1.715452500e+01f,
                4.185000000e-02f,  1.655701500e+01f,  4.212000000e-02f,  1.595950500e+01f,  4.158000000e-02f,  1.785382500e+01f,
                4.185000000e-02f,  1.725307500e+01f,  4.212000000e-02f,  1.665232500e+01f,  4.239000000e-02f,  1.605157500e+01f,
                4.185000000e-02f,  1.795561500e+01f,  4.212000000e-02f,  1.735162500e+01f,  4.239000000e-02f,  1.674763500e+01f,
                4.266000000e-02f,  1.614364500e+01f,  4.212000000e-02f,  1.805740500e+01f,  4.239000000e-02f,  1.745017500e+01f,
                4.266000000e-02f,  1.684294500e+01f,  4.293000000e-02f,  1.623571500e+01f,  4.239000000e-02f,  1.815919500e+01f,
                4.266000000e-02f,  1.754872500e+01f,  4.293000000e-02f,  1.693825500e+01f,  4.320000000e-02f,  1.632778500e+01f,
                4.266000000e-02f,  1.826098500e+01f,  4.293000000e-02f,  1.764727500e+01f,  4.320000000e-02f,  1.703356500e+01f,
                4.347000000e-02f,  1.641985500e+01f,  4.293000000e-02f,  1.836277500e+01f,  4.320000000e-02f,  1.774582500e+01f,
                4.347000000e-02f,  1.712887500e+01f,  4.374000000e-02f,  1.651192500e+01f,  4.320000000e-02f,  1.846456500e+01f,
                4.347000000e-02f,  1.784437500e+01f,  4.374000000e-02f,  1.722418500e+01f,  4.401000000e-02f,  1.660399500e+01f,
                4.347000000e-02f,  1.856635500e+01f,  4.374000000e-02f,  1.794292500e+01f,  4.401000000e-02f,  1.731949500e+01f,
                4.428000000e-02f,  1.669606500e+01f,  4.374000000e-02f,  1.866814500e+01f,  4.401000000e-02f,  1.804147500e+01f,
                4.428000000e-02f,  1.741480500e+01f,  4.455000000e-02f,  1.678813500e+01f,  5.859000000e-02f,  2.426659500e+01f,
                5.886000000e-02f,  2.346172500e+01f,  5.913000000e-02f,  2.265685500e+01f,  5.940000000e-02f,  2.185198500e+01f,
                5.886000000e-02f,  2.436838500e+01f,  5.913000000e-02f,  2.356027500e+01f,  5.940000000e-02f,  2.275216500e+01f,
                5.967000000e-02f,  2.194405500e+01f,  5.913000000e-02f,  2.447017500e+01f,  5.940000000e-02f,  2.365882500e+01f,
                5.967000000e-02f,  2.284747500e+01f,  5.994000000e-02f,  2.203612500e+01f,  5.940000000e-02f,  2.457196500e+01f,
                5.967000000e-02f,  2.375737500e+01f,  5.994000000e-02f,  2.294278500e+01f,  6.021000000e-02f,  2.212819500e+01f,
                5.967000000e-02f,  2.467375500e+01f,  5.994000000e-02f,  2.385592500e+01f,  6.021000000e-02f,  2.303809500e+01f,
                6.048000000e-02f,  2.222026500e+01f,  5.994000000e-02f,  2.477554500e+01f,  6.021000000e-02f,  2.395447500e+01f,
                6.048000000e-02f,  2.313340500e+01f,  6.075000000e-02f,  2.231233500e+01f,  6.021000000e-02f,  2.487733500e+01f,
                6.048000000e-02f,  2.405302500e+01f,  6.075000000e-02f,  2.322871500e+01f,  6.102000000e-02f,  2.240440500e+01f,
                6.048000000e-02f,  2.497912500e+01f,  6.075000000e-02f,  2.415157500e+01f,  6.102000000e-02f,  2.332402500e+01f,
                6.129000000e-02f,  2.249647500e+01f,  6.075000000e-02f,  2.508091500e+01f,  6.102000000e-02f,  2.425012500e+01f,
                6.129000000e-02f,  2.341933500e+01f,  6.156000000e-02f,  2.258854500e+01f,  6.102000000e-02f,  2.518270500e+01f,
                6.129000000e-02f,  2.434867500e+01f,  6.156000000e-02f,  2.351464500e+01f,  6.183000000e-02f,  2.268061500e+01f,
                6.129000000e-02f,  2.528449500e+01f,  6.156000000e-02f,  2.444722500e+01f,  6.183000000e-02f,  2.360995500e+01f,
                6.210000000e-02f,  2.277268500e+01f,  6.210000000e-02f,  2.558986500e+01f,  6.237000000e-02f,  2.474287500e+01f,
                6.264000000e-02f,  2.389588500e+01f,  6.291000000e-02f,  2.304889500e+01f,  6.237000000e-02f,  2.569165500e+01f,
                6.264000000e-02f,  2.484142500e+01f,  6.291000000e-02f,  2.399119500e+01f,  6.318000000e-02f,  2.314096500e+01f,
                6.264000000e-02f,  2.579344500e+01f,  6.291000000e-02f,  2.493997500e+01f,  6.318000000e-02f,  2.408650500e+01f,
                6.345000000e-02f,  2.323303500e+01f,  6.291000000e-02f,  2.589523500e+01f,  6.318000000e-02f,  2.503852500e+01f,
                6.345000000e-02f,  2.418181500e+01f,  6.372000000e-02f,  2.332510500e+01f,  6.318000000e-02f,  2.599702500e+01f,
                6.345000000e-02f,  2.513707500e+01f,  6.372000000e-02f,  2.427712500e+01f,  6.399000000e-02f,  2.341717500e+01f,
                6.345000000e-02f,  2.609881500e+01f,  6.372000000e-02f,  2.523562500e+01f,  6.399000000e-02f,  2.437243500e+01f,
                6.426000000e-02f,  2.350924500e+01f,  6.372000000e-02f,  2.620060500e+01f,  6.399000000e-02f,  2.533417500e+01f,
                6.426000000e-02f,  2.446774500e+01f,  6.453000000e-02f,  2.360131500e+01f,  6.399000000e-02f,  2.630239500e+01f,
                6.426000000e-02f,  2.543272500e+01f,  6.453000000e-02f,  2.456305500e+01f,  6.480000000e-02f,  2.369338500e+01f,
                6.426000000e-02f,  2.640418500e+01f,  6.453000000e-02f,  2.553127500e+01f,  6.480000000e-02f,  2.465836500e+01f,
                6.507000000e-02f,  2.378545500e+01f,  6.453000000e-02f,  2.650597500e+01f,  6.480000000e-02f,  2.562982500e+01f,
                6.507000000e-02f,  2.475367500e+01f,  6.534000000e-02f,  2.387752500e+01f,  6.480000000e-02f,  2.660776500e+01f,
                6.507000000e-02f,  2.572837500e+01f,  6.534000000e-02f,  2.484898500e+01f,  6.561000000e-02f,  2.396959500e+01f,
                6.561000000e-02f,  2.691313500e+01f,  6.588000000e-02f,  2.602402500e+01f,  6.615000000e-02f,  2.513491500e+01f,
                6.642000000e-02f,  2.424580500e+01f,  6.588000000e-02f,  2.701492500e+01f,  6.615000000e-02f,  2.612257500e+01f,
                6.642000000e-02f,  2.523022500e+01f,  6.669000000e-02f,  2.433787500e+01f,  6.615000000e-02f,  2.711671500e+01f,
                6.642000000e-02f,  2.622112500e+01f,  6.669000000e-02f,  2.532553500e+01f,  6.696000000e-02f,  2.442994500e+01f,
                6.642000000e-02f,  2.721850500e+01f,  6.669000000e-02f,  2.631967500e+01f,  6.696000000e-02f,  2.542084500e+01f,
                6.723000000e-02f,  2.452201500e+01f,  6.669000000e-02f,  2.732029500e+01f,  6.696000000e-02f,  2.641822500e+01f,
                6.723000000e-02f,  2.551615500e+01f,  6.750000000e-02f,  2.461408500e+01f,  6.696000000e-02f,  2.742208500e+01f,
                6.723000000e-02f,  2.651677500e+01f,  6.750000000e-02f,  2.561146500e+01f,  6.777000000e-02f,  2.470615500e+01f,
                6.723000000e-02f,  2.752387500e+01f,  6.750000000e-02f,  2.661532500e+01f,  6.777000000e-02f,  2.570677500e+01f,
                6.804000000e-02f,  2.479822500e+01f,  6.750000000e-02f,  2.762566500e+01f,  6.777000000e-02f,  2.671387500e+01f,
                6.804000000e-02f,  2.580208500e+01f,  6.831000000e-02f,  2.489029500e+01f,  6.777000000e-02f,  2.772745500e+01f,
                6.804000000e-02f,  2.681242500e+01f,  6.831000000e-02f,  2.589739500e+01f,  6.858000000e-02f,  2.498236500e+01f,
                6.804000000e-02f,  2.782924500e+01f,  6.831000000e-02f,  2.691097500e+01f,  6.858000000e-02f,  2.599270500e+01f,
                6.885000000e-02f,  2.507443500e+01f,  6.831000000e-02f,  2.793103500e+01f,  6.858000000e-02f,  2.700952500e+01f,
                6.885000000e-02f,  2.608801500e+01f,  6.912000000e-02f,  2.516650500e+01f,  6.912000000e-02f,  2.823640500e+01f,
                6.939000000e-02f,  2.730517500e+01f,  6.966000000e-02f,  2.637394500e+01f,  6.993000000e-02f,  2.544271500e+01f,
                6.939000000e-02f,  2.833819500e+01f,  6.966000000e-02f,  2.740372500e+01f,  6.993000000e-02f,  2.646925500e+01f,
                7.020000000e-02f,  2.553478500e+01f,  6.966000000e-02f,  2.843998500e+01f,  6.993000000e-02f,  2.750227500e+01f,
                7.020000000e-02f,  2.656456500e+01f,  7.047000000e-02f,  2.562685500e+01f,  6.993000000e-02f,  2.854177500e+01f,
                7.020000000e-02f,  2.760082500e+01f,  7.047000000e-02f,  2.665987500e+01f,  7.074000000e-02f,  2.571892500e+01f,
                7.020000000e-02f,  2.864356500e+01f,  7.047000000e-02f,  2.769937500e+01f,  7.074000000e-02f,  2.675518500e+01f,
                7.101000000e-02f,  2.581099500e+01f,  7.047000000e-02f,  2.874535500e+01f,  7.074000000e-02f,  2.779792500e+01f,
                7.101000000e-02f,  2.685049500e+01f,  7.128000000e-02f,  2.590306500e+01f,  7.074000000e-02f,  2.884714500e+01f,
                7.101000000e-02f,  2.789647500e+01f,  7.128000000e-02f,  2.694580500e+01f,  7.155000000e-02f,  2.599513500e+01f,
                7.101000000e-02f,  2.894893500e+01f,  7.128000000e-02f,  2.799502500e+01f,  7.155000000e-02f,  2.704111500e+01f,
                7.182000000e-02f,  2.608720500e+01f,  7.128000000e-02f,  2.905072500e+01f,  7.155000000e-02f,  2.809357500e+01f,
                7.182000000e-02f,  2.713642500e+01f,  7.209000000e-02f,  2.617927500e+01f,  7.155000000e-02f,  2.915251500e+01f,
                7.182000000e-02f,  2.819212500e+01f,  7.209000000e-02f,  2.723173500e+01f,  7.236000000e-02f,  2.627134500e+01f,
                7.182000000e-02f,  2.925430500e+01f,  7.209000000e-02f,  2.829067500e+01f,  7.236000000e-02f,  2.732704500e+01f,
                7.263000000e-02f,  2.636341500e+01f,  7.263000000e-02f,  2.955967500e+01f,  7.290000000e-02f,  2.858632500e+01f,
                7.317000000e-02f,  2.761297500e+01f,  7.344000000e-02f,  2.663962500e+01f,  7.290000000e-02f,  2.966146500e+01f,
                7.317000000e-02f,  2.868487500e+01f,  7.344000000e-02f,  2.770828500e+01f,  7.371000000e-02f,  2.673169500e+01f,
                7.317000000e-02f,  2.976325500e+01f,  7.344000000e-02f,  2.878342500e+01f,  7.371000000e-02f,  2.780359500e+01f,
                7.398000000e-02f,  2.682376500e+01f,  7.344000000e-02f,  2.986504500e+01f,  7.371000000e-02f,  2.888197500e+01f,
                7.398000000e-02f,  2.789890500e+01f,  7.425000000e-02f,  2.691583500e+01f,  7.371000000e-02f,  2.996683500e+01f,
                7.398000000e-02f,  2.898052500e+01f,  7.425000000e-02f,  2.799421500e+01f,  7.452000000e-02f,  2.700790500e+01f,
                7.398000000e-02f,  3.006862500e+01f,  7.425000000e-02f,  2.907907500e+01f,  7.452000000e-02f,  2.808952500e+01f,
                7.479000000e-02f,  2.709997500e+01f,  7.425000000e-02f,  3.017041500e+01f,  7.452000000e-02f,  2.917762500e+01f,
                7.479000000e-02f,  2.818483500e+01f,  7.506000000e-02f,  2.719204500e+01f,  7.452000000e-02f,  3.027220500e+01f,
                7.479000000e-02f,  2.927617500e+01f,  7.506000000e-02f,  2.828014500e+01f,  7.533000000e-02f,  2.728411500e+01f,
                7.479000000e-02f,  3.037399500e+01f,  7.506000000e-02f,  2.937472500e+01f,  7.533000000e-02f,  2.837545500e+01f,
                7.560000000e-02f,  2.737618500e+01f,  7.506000000e-02f,  3.047578500e+01f,  7.533000000e-02f,  2.947327500e+01f,
                7.560000000e-02f,  2.847076500e+01f,  7.587000000e-02f,  2.746825500e+01f,  7.533000000e-02f,  3.057757500e+01f,
                7.560000000e-02f,  2.957182500e+01f,  7.587000000e-02f,  2.856607500e+01f,  7.614000000e-02f,  2.756032500e+01f,
                7.614000000e-02f,  3.088294500e+01f,  7.641000000e-02f,  2.986747500e+01f,  7.668000000e-02f,  2.885200500e+01f,
                7.695000000e-02f,  2.783653500e+01f,  7.641000000e-02f,  3.098473500e+01f,  7.668000000e-02f,  2.996602500e+01f,
                7.695000000e-02f,  2.894731500e+01f,  7.722000000e-02f,  2.792860500e+01f,  7.668000000e-02f,  3.108652500e+01f,
                7.695000000e-02f,  3.006457500e+01f,  7.722000000e-02f,  2.904262500e+01f,  7.749000000e-02f,  2.802067500e+01f,
                7.695000000e-02f,  3.118831500e+01f,  7.722000000e-02f,  3.016312500e+01f,  7.749000000e-02f,  2.913793500e+01f,
                7.776000000e-02f,  2.811274500e+01f,  7.722000000e-02f,  3.129010500e+01f,  7.749000000e-02f,  3.026167500e+01f,
                7.776000000e-02f,  2.923324500e+01f,  7.803000000e-02f,  2.820481500e+01f,  7.749000000e-02f,  3.139189500e+01f,
                7.776000000e-02f,  3.036022500e+01f,  7.803000000e-02f,  2.932855500e+01f,  7.830000000e-02f,  2.829688500e+01f,
                7.776000000e-02f,  3.149368500e+01f,  7.803000000e-02f,  3.045877500e+01f,  7.830000000e-02f,  2.942386500e+01f,
                7.857000000e-02f,  2.838895500e+01f,  7.803000000e-02f,  3.159547500e+01f,  7.830000000e-02f,  3.055732500e+01f,
                7.857000000e-02f,  2.951917500e+01f,  7.884000000e-02f,  2.848102500e+01f,  7.830000000e-02f,  3.169726500e+01f,
                7.857000000e-02f,  3.065587500e+01f,  7.884000000e-02f,  2.961448500e+01f,  7.911000000e-02f,  2.857309500e+01f,
                7.857000000e-02f,  3.179905500e+01f,  7.884000000e-02f,  3.075442500e+01f,  7.911000000e-02f,  2.970979500e+01f,
                7.938000000e-02f,  2.866516500e+01f,  7.884000000e-02f,  3.190084500e+01f,  7.911000000e-02f,  3.085297500e+01f,
                7.938000000e-02f,  2.980510500e+01f,  7.965000000e-02f,  2.875723500e+01f,  7.965000000e-02f,  3.220621500e+01f,
                7.992000000e-02f,  3.114862500e+01f,  8.019000000e-02f,  3.009103500e+01f,  8.046000000e-02f,  2.903344500e+01f,
                7.992000000e-02f,  3.230800500e+01f,  8.019000000e-02f,  3.124717500e+01f,  8.046000000e-02f,  3.018634500e+01f,
                8.073000000e-02f,  2.912551500e+01f,  8.019000000e-02f,  3.240979500e+01f,  8.046000000e-02f,  3.134572500e+01f,
                8.073000000e-02f,  3.028165500e+01f,  8.100000000e-02f,  2.921758500e+01f,  8.046000000e-02f,  3.251158500e+01f,
                8.073000000e-02f,  3.144427500e+01f,  8.100000000e-02f,  3.037696500e+01f,  8.127000000e-02f,  2.930965500e+01f,
                8.073000000e-02f,  3.261337500e+01f,  8.100000000e-02f,  3.154282500e+01f,  8.127000000e-02f,  3.047227500e+01f,
                8.154000000e-02f,  2.940172500e+01f,  8.100000000e-02f,  3.271516500e+01f,  8.127000000e-02f,  3.164137500e+01f,
                8.154000000e-02f,  3.056758500e+01f,  8.181000000e-02f,  2.949379500e+01f,  8.127000000e-02f,  3.281695500e+01f,
                8.154000000e-02f,  3.173992500e+01f,  8.181000000e-02f,  3.066289500e+01f,  8.208000000e-02f,  2.958586500e+01f,
                8.154000000e-02f,  3.291874500e+01f,  8.181000000e-02f,  3.183847500e+01f,  8.208000000e-02f,  3.075820500e+01f,
                8.235000000e-02f,  2.967793500e+01f,  8.181000000e-02f,  3.302053500e+01f,  8.208000000e-02f,  3.193702500e+01f,
                8.235000000e-02f,  3.085351500e+01f,  8.262000000e-02f,  2.977000500e+01f,  8.208000000e-02f,  3.312232500e+01f,
                8.235000000e-02f,  3.203557500e+01f,  8.262000000e-02f,  3.094882500e+01f,  8.289000000e-02f,  2.986207500e+01f,
                8.235000000e-02f,  3.322411500e+01f,  8.262000000e-02f,  3.213412500e+01f,  8.289000000e-02f,  3.104413500e+01f,
                8.316000000e-02f,  2.995414500e+01f,  8.316000000e-02f,  3.352948500e+01f,  8.343000000e-02f,  3.242977500e+01f,
                8.370000000e-02f,  3.133006500e+01f,  8.397000000e-02f,  3.023035500e+01f,  8.343000000e-02f,  3.363127500e+01f,
                8.370000000e-02f,  3.252832500e+01f,  8.397000000e-02f,  3.142537500e+01f,  8.424000000e-02f,  3.032242500e+01f,
                8.370000000e-02f,  3.373306500e+01f,  8.397000000e-02f,  3.262687500e+01f,  8.424000000e-02f,  3.152068500e+01f,
                8.451000000e-02f,  3.041449500e+01f,  8.397000000e-02f,  3.383485500e+01f,  8.424000000e-02f,  3.272542500e+01f,
                8.451000000e-02f,  3.161599500e+01f,  8.478000000e-02f,  3.050656500e+01f,  8.424000000e-02f,  3.393664500e+01f,
                8.451000000e-02f,  3.282397500e+01f,  8.478000000e-02f,  3.171130500e+01f,  8.505000000e-02f,  3.059863500e+01f,
                8.451000000e-02f,  3.403843500e+01f,  8.478000000e-02f,  3.292252500e+01f,  8.505000000e-02f,  3.180661500e+01f,
                8.532000000e-02f,  3.069070500e+01f,  8.478000000e-02f,  3.414022500e+01f,  8.505000000e-02f,  3.302107500e+01f,
                8.532000000e-02f,  3.190192500e+01f,  8.559000000e-02f,  3.078277500e+01f,  8.505000000e-02f,  3.424201500e+01f,
                8.532000000e-02f,  3.311962500e+01f,  8.559000000e-02f,  3.199723500e+01f,  8.586000000e-02f,  3.087484500e+01f,
                8.532000000e-02f,  3.434380500e+01f,  8.559000000e-02f,  3.321817500e+01f,  8.586000000e-02f,  3.209254500e+01f,
                8.613000000e-02f,  3.096691500e+01f,  8.559000000e-02f,  3.444559500e+01f,  8.586000000e-02f,  3.331672500e+01f,
                8.613000000e-02f,  3.218785500e+01f,  8.640000000e-02f,  3.105898500e+01f,  8.586000000e-02f,  3.454738500e+01f,
                8.613000000e-02f,  3.341527500e+01f,  8.640000000e-02f,  3.228316500e+01f,  8.667000000e-02f,  3.115105500e+01f,
                8.667000000e-02f,  3.485275500e+01f,  8.694000000e-02f,  3.371092500e+01f,  8.721000000e-02f,  3.256909500e+01f,
                8.748000000e-02f,  3.142726500e+01f,  8.694000000e-02f,  3.495454500e+01f,  8.721000000e-02f,  3.380947500e+01f,
                8.748000000e-02f,  3.266440500e+01f,  8.775000000e-02f,  3.151933500e+01f,  8.721000000e-02f,  3.505633500e+01f,
                8.748000000e-02f,  3.390802500e+01f,  8.775000000e-02f,  3.275971500e+01f,  8.802000000e-02f,  3.161140500e+01f,
                8.748000000e-02f,  3.515812500e+01f,  8.775000000e-02f,  3.400657500e+01f,  8.802000000e-02f,  3.285502500e+01f,
                8.829000000e-02f,  3.170347500e+01f,  8.775000000e-02f,  3.525991500e+01f,  8.802000000e-02f,  3.410512500e+01f,
                8.829000000e-02f,  3.295033500e+01f,  8.856000000e-02f,  3.179554500e+01f,  8.802000000e-02f,  3.536170500e+01f,
                8.829000000e-02f,  3.420367500e+01f,  8.856000000e-02f,  3.304564500e+01f,  8.883000000e-02f,  3.188761500e+01f,
                8.829000000e-02f,  3.546349500e+01f,  8.856000000e-02f,  3.430222500e+01f,  8.883000000e-02f,  3.314095500e+01f,
                8.910000000e-02f,  3.197968500e+01f,  8.856000000e-02f,  3.556528500e+01f,  8.883000000e-02f,  3.440077500e+01f,
                8.910000000e-02f,  3.323626500e+01f,  8.937000000e-02f,  3.207175500e+01f,  8.883000000e-02f,  3.566707500e+01f,
                8.910000000e-02f,  3.449932500e+01f,  8.937000000e-02f,  3.333157500e+01f,  8.964000000e-02f,  3.216382500e+01f,
                8.910000000e-02f,  3.576886500e+01f,  8.937000000e-02f,  3.459787500e+01f,  8.964000000e-02f,  3.342688500e+01f,
                8.991000000e-02f,  3.225589500e+01f,  8.937000000e-02f,  3.587065500e+01f,  8.964000000e-02f,  3.469642500e+01f,
                8.991000000e-02f,  3.352219500e+01f,  9.018000000e-02f,  3.234796500e+01f,  9.018000000e-02f,  3.617602500e+01f,
                9.045000000e-02f,  3.499207500e+01f,  9.072000000e-02f,  3.380812500e+01f,  9.099000000e-02f,  3.262417500e+01f,
                9.045000000e-02f,  3.627781500e+01f,  9.072000000e-02f,  3.509062500e+01f,  9.099000000e-02f,  3.390343500e+01f,
                9.126000000e-02f,  3.271624500e+01f,  9.072000000e-02f,  3.637960500e+01f,  9.099000000e-02f,  3.518917500e+01f,
                9.126000000e-02f,  3.399874500e+01f,  9.153000000e-02f,  3.280831500e+01f,  9.099000000e-02f,  3.648139500e+01f,
                9.126000000e-02f,  3.528772500e+01f,  9.153000000e-02f,  3.409405500e+01f,  9.180000000e-02f,  3.290038500e+01f,
                9.126000000e-02f,  3.658318500e+01f,  9.153000000e-02f,  3.538627500e+01f,  9.180000000e-02f,  3.418936500e+01f,
                9.207000000e-02f,  3.299245500e+01f,  9.153000000e-02f,  3.668497500e+01f,  9.180000000e-02f,  3.548482500e+01f,
                9.207000000e-02f,  3.428467500e+01f,  9.234000000e-02f,  3.308452500e+01f,  9.180000000e-02f,  3.678676500e+01f,
                9.207000000e-02f,  3.558337500e+01f,  9.234000000e-02f,  3.437998500e+01f,  9.261000000e-02f,  3.317659500e+01f,
                9.207000000e-02f,  3.688855500e+01f,  9.234000000e-02f,  3.568192500e+01f,  9.261000000e-02f,  3.447529500e+01f,
                9.288000000e-02f,  3.326866500e+01f,  9.234000000e-02f,  3.699034500e+01f,  9.261000000e-02f,  3.578047500e+01f,
                9.288000000e-02f,  3.457060500e+01f,  9.315000000e-02f,  3.336073500e+01f,  9.261000000e-02f,  3.709213500e+01f,
                9.288000000e-02f,  3.587902500e+01f,  9.315000000e-02f,  3.466591500e+01f,  9.342000000e-02f,  3.345280500e+01f,
                9.288000000e-02f,  3.719392500e+01f,  9.315000000e-02f,  3.597757500e+01f,  9.342000000e-02f,  3.476122500e+01f,
                9.369000000e-02f,  3.354487500e+01f,  9.369000000e-02f,  3.749929500e+01f,  9.396000000e-02f,  3.627322500e+01f,
                9.423000000e-02f,  3.504715500e+01f,  9.450000000e-02f,  3.382108500e+01f,  9.396000000e-02f,  3.760108500e+01f,
                9.423000000e-02f,  3.637177500e+01f,  9.450000000e-02f,  3.514246500e+01f,  9.477000000e-02f,  3.391315500e+01f,
                9.423000000e-02f,  3.770287500e+01f,  9.450000000e-02f,  3.647032500e+01f,  9.477000000e-02f,  3.523777500e+01f,
                9.504000000e-02f,  3.400522500e+01f,  9.450000000e-02f,  3.780466500e+01f,  9.477000000e-02f,  3.656887500e+01f,
                9.504000000e-02f,  3.533308500e+01f,  9.531000000e-02f,  3.409729500e+01f,  9.477000000e-02f,  3.790645500e+01f,
                9.504000000e-02f,  3.666742500e+01f,  9.531000000e-02f,  3.542839500e+01f,  9.558000000e-02f,  3.418936500e+01f,
                9.504000000e-02f,  3.800824500e+01f,  9.531000000e-02f,  3.676597500e+01f,  9.558000000e-02f,  3.552370500e+01f,
                9.585000000e-02f,  3.428143500e+01f,  9.531000000e-02f,  3.811003500e+01f,  9.558000000e-02f,  3.686452500e+01f,
                9.585000000e-02f,  3.561901500e+01f,  9.612000000e-02f,  3.437350500e+01f,  9.558000000e-02f,  3.821182500e+01f,
                9.585000000e-02f,  3.696307500e+01f,  9.612000000e-02f,  3.571432500e+01f,  9.639000000e-02f,  3.446557500e+01f,
                9.585000000e-02f,  3.831361500e+01f,  9.612000000e-02f,  3.706162500e+01f,  9.639000000e-02f,  3.580963500e+01f,
                9.666000000e-02f,  3.455764500e+01f,  9.612000000e-02f,  3.841540500e+01f,  9.639000000e-02f,  3.716017500e+01f,
                9.666000000e-02f,  3.590494500e+01f,  9.693000000e-02f,  3.464971500e+01f,  9.639000000e-02f,  3.851719500e+01f,
                9.666000000e-02f,  3.725872500e+01f,  9.693000000e-02f,  3.600025500e+01f,  9.720000000e-02f,  3.474178500e+01f,
                9.720000000e-02f,  3.882256500e+01f,  9.747000000e-02f,  3.755437500e+01f,  9.774000000e-02f,  3.628618500e+01f,
                9.801000000e-02f,  3.501799500e+01f,  9.747000000e-02f,  3.892435500e+01f,  9.774000000e-02f,  3.765292500e+01f,
                9.801000000e-02f,  3.638149500e+01f,  9.828000000e-02f,  3.511006500e+01f,  9.774000000e-02f,  3.902614500e+01f,
                9.801000000e-02f,  3.775147500e+01f,  9.828000000e-02f,  3.647680500e+01f,  9.855000000e-02f,  3.520213500e+01f,
                9.801000000e-02f,  3.912793500e+01f,  9.828000000e-02f,  3.785002500e+01f,  9.855000000e-02f,  3.657211500e+01f,
                9.882000000e-02f,  3.529420500e+01f,  9.828000000e-02f,  3.922972500e+01f,  9.855000000e-02f,  3.794857500e+01f,
                9.882000000e-02f,  3.666742500e+01f,  9.909000000e-02f,  3.538627500e+01f,  9.855000000e-02f,  3.933151500e+01f,
                9.882000000e-02f,  3.804712500e+01f,  9.909000000e-02f,  3.676273500e+01f,  9.936000000e-02f,  3.547834500e+01f,
                9.882000000e-02f,  3.943330500e+01f,  9.909000000e-02f,  3.814567500e+01f,  9.936000000e-02f,  3.685804500e+01f,
                9.963000000e-02f,  3.557041500e+01f,  9.909000000e-02f,  3.953509500e+01f,  9.936000000e-02f,  3.824422500e+01f,
                9.963000000e-02f,  3.695335500e+01f,  9.990000000e-02f,  3.566248500e+01f,  9.936000000e-02f,  3.963688500e+01f,
                9.963000000e-02f,  3.834277500e+01f,  9.990000000e-02f,  3.704866500e+01f,  1.001700000e-01f,  3.575455500e+01f,
                9.963000000e-02f,  3.973867500e+01f,  9.990000000e-02f,  3.844132500e+01f,  1.001700000e-01f,  3.714397500e+01f,
                1.004400000e-01f,  3.584662500e+01f,  9.990000000e-02f,  3.984046500e+01f,  1.001700000e-01f,  3.853987500e+01f,
                1.004400000e-01f,  3.723928500e+01f,  1.007100000e-01f,  3.593869500e+01f,  1.007100000e-01f,  4.014583500e+01f,
                1.009800000e-01f,  3.883552500e+01f,  1.012500000e-01f,  3.752521500e+01f,  1.015200000e-01f,  3.621490500e+01f,
                1.009800000e-01f,  4.024762500e+01f,  1.012500000e-01f,  3.893407500e+01f,  1.015200000e-01f,  3.762052500e+01f,
                1.017900000e-01f,  3.630697500e+01f,  1.012500000e-01f,  4.034941500e+01f,  1.015200000e-01f,  3.903262500e+01f,
                1.017900000e-01f,  3.771583500e+01f,  1.020600000e-01f,  3.639904500e+01f,  1.015200000e-01f,  4.045120500e+01f,
                1.017900000e-01f,  3.913117500e+01f,  1.020600000e-01f,  3.781114500e+01f,  1.023300000e-01f,  3.649111500e+01f,
                1.017900000e-01f,  4.055299500e+01f,  1.020600000e-01f,  3.922972500e+01f,  1.023300000e-01f,  3.790645500e+01f,
                1.026000000e-01f,  3.658318500e+01f,  1.020600000e-01f,  4.065478500e+01f,  1.023300000e-01f,  3.932827500e+01f,
                1.026000000e-01f,  3.800176500e+01f,  1.028700000e-01f,  3.667525500e+01f,  1.023300000e-01f,  4.075657500e+01f,
                1.026000000e-01f,  3.942682500e+01f,  1.028700000e-01f,  3.809707500e+01f,  1.031400000e-01f,  3.676732500e+01f,
                1.026000000e-01f,  4.085836500e+01f,  1.028700000e-01f,  3.952537500e+01f,  1.031400000e-01f,  3.819238500e+01f,
                1.034100000e-01f,  3.685939500e+01f,  1.028700000e-01f,  4.096015500e+01f,  1.031400000e-01f,  3.962392500e+01f,
                1.034100000e-01f,  3.828769500e+01f,  1.036800000e-01f,  3.695146500e+01f,  1.031400000e-01f,  4.106194500e+01f,
                1.034100000e-01f,  3.972247500e+01f,  1.036800000e-01f,  3.838300500e+01f,  1.039500000e-01f,  3.704353500e+01f,
                1.034100000e-01f,  4.116373500e+01f,  1.036800000e-01f,  3.982102500e+01f,  1.039500000e-01f,  3.847831500e+01f,
                1.042200000e-01f,  3.713560500e+01f,  1.182600000e-01f,  4.676218500e+01f,  1.185300000e-01f,  4.524127500e+01f,
                1.188000000e-01f,  4.372036500e+01f,  1.190700000e-01f,  4.219945500e+01f,  1.185300000e-01f,  4.686397500e+01f,
                1.188000000e-01f,  4.533982500e+01f,  1.190700000e-01f,  4.381567500e+01f,  1.193400000e-01f,  4.229152500e+01f,
                1.188000000e-01f,  4.696576500e+01f,  1.190700000e-01f,  4.543837500e+01f,  1.193400000e-01f,  4.391098500e+01f,
                1.196100000e-01f,  4.238359500e+01f,  1.190700000e-01f,  4.706755500e+01f,  1.193400000e-01f,  4.553692500e+01f,
                1.196100000e-01f,  4.400629500e+01f,  1.198800000e-01f,  4.247566500e+01f,  1.193400000e-01f,  4.716934500e+01f,
                1.196100000e-01f,  4.563547500e+01f,  1.198800000e-01f,  4.410160500e+01f,  1.201500000e-01f,  4.256773500e+01f,
                1.196100000e-01f,  4.727113500e+01f,  1.198800000e-01f,  4.573402500e+01f,  1.201500000e-01f,  4.419691500e+01f,
                1.204200000e-01f,  4.265980500e+01f,  1.198800000e-01f,  4.737292500e+01f,  1.201500000e-01f,  4.583257500e+01f,
                1.204200000e-01f,  4.429222500e+01f,  1.206900000e-01f,  4.275187500e+01f,  1.201500000e-01f,  4.747471500e+01f,
                1.204200000e-01f,  4.593112500e+01f,  1.206900000e-01f,  4.438753500e+01f,  1.209600000e-01f,  4.284394500e+01f,
                1.204200000e-01f,  4.757650500e+01f,  1.206900000e-01f,  4.602967500e+01f,  1.209600000e-01f,  4.448284500e+01f,
                1.212300000e-01f,  4.293601500e+01f,  1.206900000e-01f,  4.767829500e+01f,  1.209600000e-01f,  4.612822500e+01f,
                1.212300000e-01f,  4.457815500e+01f,  1.215000000e-01f,  4.302808500e+01f,  1.209600000e-01f,  4.778008500e+01f,
                1.212300000e-01f,  4.622677500e+01f,  1.215000000e-01f,  4.467346500e+01f,  1.217700000e-01f,  4.312015500e+01f,
                1.217700000e-01f,  4.808545500e+01f,  1.220400000e-01f,  4.652242500e+01f,  1.223100000e-01f,  4.495939500e+01f,
                1.225800000e-01f,  4.339636500e+01f,  1.220400000e-01f,  4.818724500e+01f,  1.223100000e-01f,  4.662097500e+01f,
                1.225800000e-01f,  4.505470500e+01f,  1.228500000e-01f,  4.348843500e+01f,  1.223100000e-01f,  4.828903500e+01f,
                1.225800000e-01f,  4.671952500e+01f,  1.228500000e-01f,  4.515001500e+01f,  1.231200000e-01f,  4.358050500e+01f,
                1.225800000e-01f,  4.839082500e+01f,  1.228500000e-01f,  4.681807500e+01f,  1.231200000e-01f,  4.524532500e+01f,
                1.233900000e-01f,  4.367257500e+01f,  1.228500000e-01f,  4.849261500e+01f,  1.231200000e-01f,  4.691662500e+01f,
                1.233900000e-01f,  4.534063500e+01f,  1.236600000e-01f,  4.376464500e+01f,  1.231200000e-01f,  4.859440500e+01f,
                1.233900000e-01f,  4.701517500e+01f,  1.236600000e-01f,  4.543594500e+01f,  1.239300000e-01f,  4.385671500e+01f,
                1.233900000e-01f,  4.869619500e+01f,  1.236600000e-01f,  4.711372500e+01f,  1.239300000e-01f,  4.553125500e+01f,
                1.242000000e-01f,  4.394878500e+01f,  1.236600000e-01f,  4.879798500e+01f,  1.239300000e-01f,  4.721227500e+01f,
                1.242000000e-01f,  4.562656500e+01f,  1.244700000e-01f,  4.404085500e+01f,  1.239300000e-01f,  4.889977500e+01f,
                1.242000000e-01f,  4.731082500e+01f,  1.244700000e-01f,  4.572187500e+01f,  1.247400000e-01f,  4.413292500e+01f,
                1.242000000e-01f,  4.900156500e+01f,  1.244700000e-01f,  4.740937500e+01f,  1.247400000e-01f,  4.581718500e+01f,
                1.250100000e-01f,  4.422499500e+01f,  1.244700000e-01f,  4.910335500e+01f,  1.247400000e-01f,  4.750792500e+01f,
                1.250100000e-01f,  4.591249500e+01f,  1.252800000e-01f,  4.431706500e+01f,  1.252800000e-01f,  4.940872500e+01f,
                1.255500000e-01f,  4.780357500e+01f,  1.258200000e-01f,  4.619842500e+01f,  1.260900000e-01f,  4.459327500e+01f,
                1.255500000e-01f,  4.951051500e+01f,  1.258200000e-01f,  4.790212500e+01f,  1.260900000e-01f,  4.629373500e+01f,
                1.263600000e-01f,  4.468534500e+01f,  1.258200000e-01f,  4.961230500e+01f,  1.260900000e-01f,  4.800067500e+01f,
                1.263600000e-01f,  4.638904500e+01f,  1.266300000e-01f,  4.477741500e+01f,  1.260900000e-01f,  4.971409500e+01f,
                1.263600000e-01f,  4.809922500e+01f,  1.266300000e-01f,  4.648435500e+01f,  1.269000000e-01f,  4.486948500e+01f,
                1.263600000e-01f,  4.981588500e+01f,  1.266300000e-01f,  4.819777500e+01f,  1.269000000e-01f,  4.657966500e+01f,
                1.271700000e-01f,  4.496155500e+01f,  1.266300000e-01f,  4.991767500e+01f,  1.269000000e-01f,  4.829632500e+01f,
                1.271700000e-01f,  4.667497500e+01f,  1.274400000e-01f,  4.505362500e+01f,  1.269000000e-01f,  5.001946500e+01f,
                1.271700000e-01f,  4.839487500e+01f,  1.274400000e-01f,  4.677028500e+01f,  1.277100000e-01f,  4.514569500e+01f,
                1.271700000e-01f,  5.012125500e+01f,  1.274400000e-01f,  4.849342500e+01f,  1.277100000e-01f,  4.686559500e+01f,
                1.279800000e-01f,  4.523776500e+01f,  1.274400000e-01f,  5.022304500e+01f,  1.277100000e-01f,  4.859197500e+01f,
                1.279800000e-01f,  4.696090500e+01f,  1.282500000e-01f,  4.532983500e+01f,  1.277100000e-01f,  5.032483500e+01f,
                1.279800000e-01f,  4.869052500e+01f,  1.282500000e-01f,  4.705621500e+01f,  1.285200000e-01f,  4.542190500e+01f,
                1.279800000e-01f,  5.042662500e+01f,  1.282500000e-01f,  4.878907500e+01f,  1.285200000e-01f,  4.715152500e+01f,
                1.287900000e-01f,  4.551397500e+01f,  1.287900000e-01f,  5.073199500e+01f,  1.290600000e-01f,  4.908472500e+01f,
                1.293300000e-01f,  4.743745500e+01f,  1.296000000e-01f,  4.579018500e+01f,  1.290600000e-01f,  5.083378500e+01f,
                1.293300000e-01f,  4.918327500e+01f,  1.296000000e-01f,  4.753276500e+01f,  1.298700000e-01f,  4.588225500e+01f,
                1.293300000e-01f,  5.093557500e+01f,  1.296000000e-01f,  4.928182500e+01f,  1.298700000e-01f,  4.762807500e+01f,
                1.301400000e-01f,  4.597432500e+01f,  1.296000000e-01f,  5.103736500e+01f,  1.298700000e-01f,  4.938037500e+01f,
                1.301400000e-01f,  4.772338500e+01f,  1.304100000e-01f,  4.606639500e+01f,  1.298700000e-01f,  5.113915500e+01f,
                1.301400000e-01f,  4.947892500e+01f,  1.304100000e-01f,  4.781869500e+01f,  1.306800000e-01f,  4.615846500e+01f,
                1.301400000e-01f,  5.124094500e+01f,  1.304100000e-01f,  4.957747500e+01f,  1.306800000e-01f,  4.791400500e+01f,
                1.309500000e-01f,  4.625053500e+01f,  1.304100000e-01f,  5.134273500e+01f,  1.306800000e-01f,  4.967602500e+01f,
                1.309500000e-01f,  4.800931500e+01f,  1.312200000e-01f,  4.634260500e+01f,  1.306800000e-01f,  5.144452500e+01f,
                1.309500000e-01f,  4.977457500e+01f,  1.312200000e-01f,  4.810462500e+01f,  1.314900000e-01f,  4.643467500e+01f,
                1.309500000e-01f,  5.154631500e+01f,  1.312200000e-01f,  4.987312500e+01f,  1.314900000e-01f,  4.819993500e+01f,
                1.317600000e-01f,  4.652674500e+01f,  1.312200000e-01f,  5.164810500e+01f,  1.314900000e-01f,  4.997167500e+01f,
                1.317600000e-01f,  4.829524500e+01f,  1.320300000e-01f,  4.661881500e+01f,  1.314900000e-01f,  5.174989500e+01f,
                1.317600000e-01f,  5.007022500e+01f,  1.320300000e-01f,  4.839055500e+01f,  1.323000000e-01f,  4.671088500e+01f,
                1.323000000e-01f,  5.205526500e+01f,  1.325700000e-01f,  5.036587500e+01f,  1.328400000e-01f,  4.867648500e+01f,
                1.331100000e-01f,  4.698709500e+01f,  1.325700000e-01f,  5.215705500e+01f,  1.328400000e-01f,  5.046442500e+01f,
                1.331100000e-01f,  4.877179500e+01f,  1.333800000e-01f,  4.707916500e+01f,  1.328400000e-01f,  5.225884500e+01f,
                1.331100000e-01f,  5.056297500e+01f,  1.333800000e-01f,  4.886710500e+01f,  1.336500000e-01f,  4.717123500e+01f,
                1.331100000e-01f,  5.236063500e+01f,  1.333800000e-01f,  5.066152500e+01f,  1.336500000e-01f,  4.896241500e+01f,
                1.339200000e-01f,  4.726330500e+01f,  1.333800000e-01f,  5.246242500e+01f,  1.336500000e-01f,  5.076007500e+01f,
                1.339200000e-01f,  4.905772500e+01f,  1.341900000e-01f,  4.735537500e+01f,  1.336500000e-01f,  5.256421500e+01f,
                1.339200000e-01f,  5.085862500e+01f,  1.341900000e-01f,  4.915303500e+01f,  1.344600000e-01f,  4.744744500e+01f,
                1.339200000e-01f,  5.266600500e+01f,  1.341900000e-01f,  5.095717500e+01f,  1.344600000e-01f,  4.924834500e+01f,
                1.347300000e-01f,  4.753951500e+01f,  1.341900000e-01f,  5.276779500e+01f,  1.344600000e-01f,  5.105572500e+01f,
                1.347300000e-01f,  4.934365500e+01f,  1.350000000e-01f,  4.763158500e+01f,  1.344600000e-01f,  5.286958500e+01f,
                1.347300000e-01f,  5.115427500e+01f,  1.350000000e-01f,  4.943896500e+01f,  1.352700000e-01f,  4.772365500e+01f,
                1.347300000e-01f,  5.297137500e+01f,  1.350000000e-01f,  5.125282500e+01f,  1.352700000e-01f,  4.953427500e+01f,
                1.355400000e-01f,  4.781572500e+01f,  1.350000000e-01f,  5.307316500e+01f,  1.352700000e-01f,  5.135137500e+01f,
                1.355400000e-01f,  4.962958500e+01f,  1.358100000e-01f,  4.790779500e+01f,  1.358100000e-01f,  5.337853500e+01f,
                1.360800000e-01f,  5.164702500e+01f,  1.363500000e-01f,  4.991551500e+01f,  1.366200000e-01f,  4.818400500e+01f,
                1.360800000e-01f,  5.348032500e+01f,  1.363500000e-01f,  5.174557500e+01f,  1.366200000e-01f,  5.001082500e+01f,
                1.368900000e-01f,  4.827607500e+01f,  1.363500000e-01f,  5.358211500e+01f,  1.366200000e-01f,  5.184412500e+01f,
                1.368900000e-01f,  5.010613500e+01f,  1.371600000e-01f,  4.836814500e+01f,  1.366200000e-01f,  5.368390500e+01f,
                1.368900000e-01f,  5.194267500e+01f,  1.371600000e-01f,  5.020144500e+01f,  1.374300000e-01f,  4.846021500e+01f,
                1.368900000e-01f,  5.378569500e+01f,  1.371600000e-01f,  5.204122500e+01f,  1.374300000e-01f,  5.029675500e+01f,
                1.377000000e-01f,  4.855228500e+01f,  1.371600000e-01f,  5.388748500e+01f,  1.374300000e-01f,  5.213977500e+01f,
                1.377000000e-01f,  5.039206500e+01f,  1.379700000e-01f,  4.864435500e+01f,  1.374300000e-01f,  5.398927500e+01f,
                1.377000000e-01f,  5.223832500e+01f,  1.379700000e-01f,  5.048737500e+01f,  1.382400000e-01f,  4.873642500e+01f,
                1.377000000e-01f,  5.409106500e+01f,  1.379700000e-01f,  5.233687500e+01f,  1.382400000e-01f,  5.058268500e+01f,
                1.385100000e-01f,  4.882849500e+01f,  1.379700000e-01f,  5.419285500e+01f,  1.382400000e-01f,  5.243542500e+01f,
                1.385100000e-01f,  5.067799500e+01f,  1.387800000e-01f,  4.892056500e+01f,  1.382400000e-01f,  5.429464500e+01f,
                1.385100000e-01f,  5.253397500e+01f,  1.387800000e-01f,  5.077330500e+01f,  1.390500000e-01f,  4.901263500e+01f,
                1.385100000e-01f,  5.439643500e+01f,  1.387800000e-01f,  5.263252500e+01f,  1.390500000e-01f,  5.086861500e+01f,
                1.393200000e-01f,  4.910470500e+01f,  1.393200000e-01f,  5.470180500e+01f,  1.395900000e-01f,  5.292817500e+01f,
                1.398600000e-01f,  5.115454500e+01f,  1.401300000e-01f,  4.938091500e+01f,  1.395900000e-01f,  5.480359500e+01f,
                1.398600000e-01f,  5.302672500e+01f,  1.401300000e-01f,  5.124985500e+01f,  1.404000000e-01f,  4.947298500e+01f,
                1.398600000e-01f,  5.490538500e+01f,  1.401300000e-01f,  5.312527500e+01f,  1.404000000e-01f,  5.134516500e+01f,
                1.406700000e-01f,  4.956505500e+01f,  1.401300000e-01f,  5.500717500e+01f,  1.404000000e-01f,  5.322382500e+01f,
                1.406700000e-01f,  5.144047500e+01f,  1.409400000e-01f,  4.965712500e+01f,  1.404000000e-01f,  5.510896500e+01f,
                1.406700000e-01f,  5.332237500e+01f,  1.409400000e-01f,  5.153578500e+01f,  1.412100000e-01f,  4.974919500e+01f,
                1.406700000e-01f,  5.521075500e+01f,  1.409400000e-01f,  5.342092500e+01f,  1.412100000e-01f,  5.163109500e+01f,
                1.414800000e-01f,  4.984126500e+01f,  1.409400000e-01f,  5.531254500e+01f,  1.412100000e-01f,  5.351947500e+01f,
                1.414800000e-01f,  5.172640500e+01f,  1.417500000e-01f,  4.993333500e+01f,  1.412100000e-01f,  5.541433500e+01f,
                1.414800000e-01f,  5.361802500e+01f,  1.417500000e-01f,  5.182171500e+01f,  1.420200000e-01f,  5.002540500e+01f,
                1.414800000e-01f,  5.551612500e+01f,  1.417500000e-01f,  5.371657500e+01f,  1.420200000e-01f,  5.191702500e+01f,
                1.422900000e-01f,  5.011747500e+01f,  1.417500000e-01f,  5.561791500e+01f,  1.420200000e-01f,  5.381512500e+01f,
                1.422900000e-01f,  5.201233500e+01f,  1.425600000e-01f,  5.020954500e+01f,  1.420200000e-01f,  5.571970500e+01f,
                1.422900000e-01f,  5.391367500e+01f,  1.425600000e-01f,  5.210764500e+01f,  1.428300000e-01f,  5.030161500e+01f,
                1.428300000e-01f,  5.602507500e+01f,  1.431000000e-01f,  5.420932500e+01f,  1.433700000e-01f,  5.239357500e+01f,
                1.436400000e-01f,  5.057782500e+01f,  1.431000000e-01f,  5.612686500e+01f,  1.433700000e-01f,  5.430787500e+01f,
                1.436400000e-01f,  5.248888500e+01f,  1.439100000e-01f,  5.066989500e+01f,  1.433700000e-01f,  5.622865500e+01f,
                1.436400000e-01f,  5.440642500e+01f,  1.439100000e-01f,  5.258419500e+01f,  1.441800000e-01f,  5.076196500e+01f,
                1.436400000e-01f,  5.633044500e+01f,  1.439100000e-01f,  5.450497500e+01f,  1.441800000e-01f,  5.267950500e+01f,
                1.444500000e-01f,  5.085403500e+01f,  1.439100000e-01f,  5.643223500e+01f,  1.441800000e-01f,  5.460352500e+01f,
                1.444500000e-01f,  5.277481500e+01f,  1.447200000e-01f,  5.094610500e+01f,  1.441800000e-01f,  5.653402500e+01f,
                1.444500000e-01f,  5.470207500e+01f,  1.447200000e-01f,  5.287012500e+01f,  1.449900000e-01f,  5.103817500e+01f,
                1.444500000e-01f,  5.663581500e+01f,  1.447200000e-01f,  5.480062500e+01f,  1.449900000e-01f,  5.296543500e+01f,
                1.452600000e-01f,  5.113024500e+01f,  1.447200000e-01f,  5.673760500e+01f,  1.449900000e-01f,  5.489917500e+01f,
                1.452600000e-01f,  5.306074500e+01f,  1.455300000e-01f,  5.122231500e+01f,  1.449900000e-01f,  5.683939500e+01f,
                1.452600000e-01f,  5.499772500e+01f,  1.455300000e-01f,  5.315605500e+01f,  1.458000000e-01f,  5.131438500e+01f,
                1.452600000e-01f,  5.694118500e+01f,  1.455300000e-01f,  5.509627500e+01f,  1.458000000e-01f,  5.325136500e+01f,
                1.460700000e-01f,  5.140645500e+01f,  1.455300000e-01f,  5.704297500e+01f,  1.458000000e-01f,  5.519482500e+01f,
                1.460700000e-01f,  5.334667500e+01f,  1.463400000e-01f,  5.149852500e+01f,  1.463400000e-01f,  5.734834500e+01f,
                1.466100000e-01f,  5.549047500e+01f,  1.468800000e-01f,  5.363260500e+01f,  1.471500000e-01f,  5.177473500e+01f,
                1.466100000e-01f,  5.745013500e+01f,  1.468800000e-01f,  5.558902500e+01f,  1.471500000e-01f,  5.372791500e+01f,
                1.474200000e-01f,  5.186680500e+01f,  1.468800000e-01f,  5.755192500e+01f,  1.471500000e-01f,  5.568757500e+01f,
                1.474200000e-01f,  5.382322500e+01f,  1.476900000e-01f,  5.195887500e+01f,  1.471500000e-01f,  5.765371500e+01f,
                1.474200000e-01f,  5.578612500e+01f,  1.476900000e-01f,  5.391853500e+01f,  1.479600000e-01f,  5.205094500e+01f,
                1.474200000e-01f,  5.775550500e+01f,  1.476900000e-01f,  5.588467500e+01f,  1.479600000e-01f,  5.401384500e+01f,
                1.482300000e-01f,  5.214301500e+01f,  1.476900000e-01f,  5.785729500e+01f,  1.479600000e-01f,  5.598322500e+01f,
                1.482300000e-01f,  5.410915500e+01f,  1.485000000e-01f,  5.223508500e+01f,  1.479600000e-01f,  5.795908500e+01f,
                1.482300000e-01f,  5.608177500e+01f,  1.485000000e-01f,  5.420446500e+01f,  1.487700000e-01f,  5.232715500e+01f,
                1.482300000e-01f,  5.806087500e+01f,  1.485000000e-01f,  5.618032500e+01f,  1.487700000e-01f,  5.429977500e+01f,
                1.490400000e-01f,  5.241922500e+01f,  1.485000000e-01f,  5.816266500e+01f,  1.487700000e-01f,  5.627887500e+01f,
                1.490400000e-01f,  5.439508500e+01f,  1.493100000e-01f,  5.251129500e+01f,  1.487700000e-01f,  5.826445500e+01f,
                1.490400000e-01f,  5.637742500e+01f,  1.493100000e-01f,  5.449039500e+01f,  1.495800000e-01f,  5.260336500e+01f,
                1.490400000e-01f,  5.836624500e+01f,  1.493100000e-01f,  5.647597500e+01f,  1.495800000e-01f,  5.458570500e+01f,
                1.498500000e-01f,  5.269543500e+01f,  1.498500000e-01f,  5.867161500e+01f,  1.501200000e-01f,  5.677162500e+01f,
                1.503900000e-01f,  5.487163500e+01f,  1.506600000e-01f,  5.297164500e+01f,  1.501200000e-01f,  5.877340500e+01f,
                1.503900000e-01f,  5.687017500e+01f,  1.506600000e-01f,  5.496694500e+01f,  1.509300000e-01f,  5.306371500e+01f,
                1.503900000e-01f,  5.887519500e+01f,  1.506600000e-01f,  5.696872500e+01f,  1.509300000e-01f,  5.506225500e+01f,
                1.512000000e-01f,  5.315578500e+01f,  1.506600000e-01f,  5.897698500e+01f,  1.509300000e-01f,  5.706727500e+01f,
                1.512000000e-01f,  5.515756500e+01f,  1.514700000e-01f,  5.324785500e+01f,  1.509300000e-01f,  5.907877500e+01f,
                1.512000000e-01f,  5.716582500e+01f,  1.514700000e-01f,  5.525287500e+01f,  1.517400000e-01f,  5.333992500e+01f,
                1.512000000e-01f,  5.918056500e+01f,  1.514700000e-01f,  5.726437500e+01f,  1.517400000e-01f,  5.534818500e+01f,
                1.520100000e-01f,  5.343199500e+01f,  1.514700000e-01f,  5.928235500e+01f,  1.517400000e-01f,  5.736292500e+01f,
                1.520100000e-01f,  5.544349500e+01f,  1.522800000e-01f,  5.352406500e+01f,  1.517400000e-01f,  5.938414500e+01f,
                1.520100000e-01f,  5.746147500e+01f,  1.522800000e-01f,  5.553880500e+01f,  1.525500000e-01f,  5.361613500e+01f,
                1.520100000e-01f,  5.948593500e+01f,  1.522800000e-01f,  5.756002500e+01f,  1.525500000e-01f,  5.563411500e+01f,
                1.528200000e-01f,  5.370820500e+01f,  1.522800000e-01f,  5.958772500e+01f,  1.525500000e-01f,  5.765857500e+01f,
                1.528200000e-01f,  5.572942500e+01f,  1.530900000e-01f,  5.380027500e+01f,  1.525500000e-01f,  5.968951500e+01f,
                1.528200000e-01f,  5.775712500e+01f,  1.530900000e-01f,  5.582473500e+01f,  1.533600000e-01f,  5.389234500e+01f,
                1.533600000e-01f,  5.999488500e+01f,  1.536300000e-01f,  5.805277500e+01f,  1.539000000e-01f,  5.611066500e+01f,
                1.541700000e-01f,  5.416855500e+01f,  1.536300000e-01f,  6.009667500e+01f,  1.539000000e-01f,  5.815132500e+01f,
                1.541700000e-01f,  5.620597500e+01f,  1.544400000e-01f,  5.426062500e+01f,  1.539000000e-01f,  6.019846500e+01f,
                1.541700000e-01f,  5.824987500e+01f,  1.544400000e-01f,  5.630128500e+01f,  1.547100000e-01f,  5.435269500e+01f,
                1.541700000e-01f,  6.030025500e+01f,  1.544400000e-01f,  5.834842500e+01f,  1.547100000e-01f,  5.639659500e+01f,
                1.549800000e-01f,  5.444476500e+01f,  1.544400000e-01f,  6.040204500e+01f,  1.547100000e-01f,  5.844697500e+01f,
                1.549800000e-01f,  5.649190500e+01f,  1.552500000e-01f,  5.453683500e+01f,  1.547100000e-01f,  6.050383500e+01f,
                1.549800000e-01f,  5.854552500e+01f,  1.552500000e-01f,  5.658721500e+01f,  1.555200000e-01f,  5.462890500e+01f,
                1.549800000e-01f,  6.060562500e+01f,  1.552500000e-01f,  5.864407500e+01f,  1.555200000e-01f,  5.668252500e+01f,
                1.557900000e-01f,  5.472097500e+01f,  1.552500000e-01f,  6.070741500e+01f,  1.555200000e-01f,  5.874262500e+01f,
                1.557900000e-01f,  5.677783500e+01f,  1.560600000e-01f,  5.481304500e+01f,  1.555200000e-01f,  6.080920500e+01f,
                1.557900000e-01f,  5.884117500e+01f,  1.560600000e-01f,  5.687314500e+01f,  1.563300000e-01f,  5.490511500e+01f,
                1.557900000e-01f,  6.091099500e+01f,  1.560600000e-01f,  5.893972500e+01f,  1.563300000e-01f,  5.696845500e+01f,
                1.566000000e-01f,  5.499718500e+01f,  1.560600000e-01f,  6.101278500e+01f,  1.563300000e-01f,  5.903827500e+01f,
                1.566000000e-01f,  5.706376500e+01f,  1.568700000e-01f,  5.508925500e+01f,  1.568700000e-01f,  6.131815500e+01f,
                1.571400000e-01f,  5.933392500e+01f,  1.574100000e-01f,  5.734969500e+01f,  1.576800000e-01f,  5.536546500e+01f,
                1.571400000e-01f,  6.141994500e+01f,  1.574100000e-01f,  5.943247500e+01f,  1.576800000e-01f,  5.744500500e+01f,
                1.579500000e-01f,  5.545753500e+01f,  1.574100000e-01f,  6.152173500e+01f,  1.576800000e-01f,  5.953102500e+01f,
                1.579500000e-01f,  5.754031500e+01f,  1.582200000e-01f,  5.554960500e+01f,  1.576800000e-01f,  6.162352500e+01f,
                1.579500000e-01f,  5.962957500e+01f,  1.582200000e-01f,  5.763562500e+01f,  1.584900000e-01f,  5.564167500e+01f,
                1.579500000e-01f,  6.172531500e+01f,  1.582200000e-01f,  5.972812500e+01f,  1.584900000e-01f,  5.773093500e+01f,
                1.587600000e-01f,  5.573374500e+01f,  1.582200000e-01f,  6.182710500e+01f,  1.584900000e-01f,  5.982667500e+01f,
                1.587600000e-01f,  5.782624500e+01f,  1.590300000e-01f,  5.582581500e+01f,  1.584900000e-01f,  6.192889500e+01f,
                1.587600000e-01f,  5.992522500e+01f,  1.590300000e-01f,  5.792155500e+01f,  1.593000000e-01f,  5.591788500e+01f,
                1.587600000e-01f,  6.203068500e+01f,  1.590300000e-01f,  6.002377500e+01f,  1.593000000e-01f,  5.801686500e+01f,
                1.595700000e-01f,  5.600995500e+01f,  1.590300000e-01f,  6.213247500e+01f,  1.593000000e-01f,  6.012232500e+01f,
                1.595700000e-01f,  5.811217500e+01f,  1.598400000e-01f,  5.610202500e+01f,  1.593000000e-01f,  6.223426500e+01f,
                1.595700000e-01f,  6.022087500e+01f,  1.598400000e-01f,  5.820748500e+01f,  1.601100000e-01f,  5.619409500e+01f,
                1.595700000e-01f,  6.233605500e+01f,  1.598400000e-01f,  6.031942500e+01f,  1.601100000e-01f,  5.830279500e+01f,
                1.603800000e-01f,  5.628616500e+01f,  1.603800000e-01f,  6.264142500e+01f,  1.606500000e-01f,  6.061507500e+01f,
                1.609200000e-01f,  5.858872500e+01f,  1.611900000e-01f,  5.656237500e+01f,  1.606500000e-01f,  6.274321500e+01f,
                1.609200000e-01f,  6.071362500e+01f,  1.611900000e-01f,  5.868403500e+01f,  1.614600000e-01f,  5.665444500e+01f,
                1.609200000e-01f,  6.284500500e+01f,  1.611900000e-01f,  6.081217500e+01f,  1.614600000e-01f,  5.877934500e+01f,
                1.617300000e-01f,  5.674651500e+01f,  1.611900000e-01f,  6.294679500e+01f,  1.614600000e-01f,  6.091072500e+01f,
                1.617300000e-01f,  5.887465500e+01f,  1.620000000e-01f,  5.683858500e+01f,  1.614600000e-01f,  6.304858500e+01f,
                1.617300000e-01f,  6.100927500e+01f,  1.620000000e-01f,  5.896996500e+01f,  1.622700000e-01f,  5.693065500e+01f,
                1.617300000e-01f,  6.315037500e+01f,  1.620000000e-01f,  6.110782500e+01f,  1.622700000e-01f,  5.906527500e+01f,
                1.625400000e-01f,  5.702272500e+01f,  1.620000000e-01f,  6.325216500e+01f,  1.622700000e-01f,  6.120637500e+01f,
                1.625400000e-01f,  5.916058500e+01f,  1.628100000e-01f,  5.711479500e+01f,  1.622700000e-01f,  6.335395500e+01f,
                1.625400000e-01f,  6.130492500e+01f,  1.628100000e-01f,  5.925589500e+01f,  1.630800000e-01f,  5.720686500e+01f,
                1.625400000e-01f,  6.345574500e+01f,  1.628100000e-01f,  6.140347500e+01f,  1.630800000e-01f,  5.935120500e+01f,
                1.633500000e-01f,  5.729893500e+01f,  1.628100000e-01f,  6.355753500e+01f,  1.630800000e-01f,  6.150202500e+01f,
                1.633500000e-01f,  5.944651500e+01f,  1.636200000e-01f,  5.739100500e+01f,  1.630800000e-01f,  6.365932500e+01f,
                1.633500000e-01f,  6.160057500e+01f,  1.636200000e-01f,  5.954182500e+01f,  1.638900000e-01f,  5.748307500e+01f,
            };

            float[] y_actual = y.ToArray();

            AssertError.Tolerance(y_expect, y_actual, 1e-7f, 1e-5f, $"mismatch value {inchannels},{outchannels},{kwidth},{kheight},{inwidth},{inheight},{batch}");
        }
    }
}
