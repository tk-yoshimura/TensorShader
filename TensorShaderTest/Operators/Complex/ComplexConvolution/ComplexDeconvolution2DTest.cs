using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;
using TensorShader;
using TensorShader.Operators.ComplexConvolution;
using TensorShaderCudaBackend.API;

namespace TensorShaderTest.Operators.Complex {
    [TestClass]
    public class ComplexDeconvolution2DTest {
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

                                        float[] yval = (new float[outwidth * outheight * outchannels * batch]).Select((_, idx) => (float)random.NextDouble() * 1e-2f).ToArray();
                                        float[] wval = (new float[kwidth * kheight * inchannels * outchannels / 2]).Select((_, idx) => (float)random.NextDouble() * 1e-2f).ToArray();

                                        System.Numerics.Complex[] ycval = (new System.Numerics.Complex[yval.Length / 2])
                                            .Select((_, idx) => new System.Numerics.Complex(yval[idx * 2], yval[idx * 2 + 1])).ToArray();

                                        System.Numerics.Complex[] wcval = (new System.Numerics.Complex[wval.Length / 2])
                                            .Select((_, idx) => new System.Numerics.Complex(wval[idx * 2], wval[idx * 2 + 1])).ToArray();

                                        ComplexMap2D y = new(outchannels / 2, outwidth, outheight, batch, ycval);
                                        ComplexFilter2D w = new(inchannels / 2, outchannels / 2, kwidth, kheight, wcval);

                                        ComplexMap2D x = Reference(y, w, inwidth, inheight, kwidth, kheight);

                                        OverflowCheckedTensor y_tensor = new(Shape.Map2D(outchannels, outwidth, outheight, batch), yval);
                                        OverflowCheckedTensor w_tensor = new(Shape.Kernel2D(inchannels, outchannels / 2, kwidth, kheight), wval);

                                        OverflowCheckedTensor x_tensor = new(Shape.Map2D(inchannels, inwidth, inheight, batch));

                                        ComplexDeconvolution2D ope = new(outwidth, outheight, outchannels, inchannels, kwidth, kheight, gradmode: false, batch);

                                        ope.Execute(y_tensor, w_tensor, x_tensor);

                                        float[] x_expect = x.ToArray();
                                        float[] x_actual = x_tensor.State.Value;

                                        CollectionAssert.AreEqual(yval, y_tensor.State.Value);
                                        CollectionAssert.AreEqual(wval, w_tensor.State.Value);

                                        AssertError.Tolerance(x_expect, x_actual, 1e-6f, 1e-4f, ref max_err, $"mismatch value {inchannels},{outchannels},{kwidth},{kheight},{inwidth},{inheight},{batch}");

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

                                        float[] yval = (new float[outwidth * outheight * outchannels * batch]).Select((_, idx) => (float)random.NextDouble() * 1e-2f).ToArray();
                                        float[] wval = (new float[kwidth * kheight * inchannels * outchannels / 2]).Select((_, idx) => (float)random.NextDouble() * 1e-2f).ToArray();

                                        System.Numerics.Complex[] ycval = (new System.Numerics.Complex[yval.Length / 2])
                                            .Select((_, idx) => new System.Numerics.Complex(yval[idx * 2], yval[idx * 2 + 1])).ToArray();

                                        System.Numerics.Complex[] wcval = (new System.Numerics.Complex[wval.Length / 2])
                                            .Select((_, idx) => new System.Numerics.Complex(wval[idx * 2], wval[idx * 2 + 1])).ToArray();

                                        ComplexMap2D y = new(outchannels / 2, outwidth, outheight, batch, ycval);
                                        ComplexFilter2D w = new(inchannels / 2, outchannels / 2, kwidth, kheight, wcval);

                                        ComplexMap2D x = Reference(y, w, inwidth, inheight, kwidth, kheight);

                                        OverflowCheckedTensor y_tensor = new(Shape.Map2D(outchannels, outwidth, outheight, batch), yval);
                                        OverflowCheckedTensor w_tensor = new(Shape.Kernel2D(inchannels, outchannels / 2, kwidth, kheight), wval);

                                        OverflowCheckedTensor x_tensor = new(Shape.Map2D(inchannels, inwidth, inheight, batch));

                                        ComplexDeconvolution2D ope = new(outwidth, outheight, outchannels, inchannels, kwidth, kheight, gradmode: false, batch);

                                        ope.Execute(y_tensor, w_tensor, x_tensor);

                                        float[] x_expect = x.ToArray();
                                        float[] x_actual = x_tensor.State.Value;

                                        CollectionAssert.AreEqual(yval, y_tensor.State.Value);
                                        CollectionAssert.AreEqual(wval, w_tensor.State.Value);

                                        AssertError.Tolerance(x_expect, x_actual, 1e-7f, 1e-5f, ref max_err, $"mismatch value {inchannels},{outchannels},{kwidth},{kheight},{inwidth},{inheight},{batch}");

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

            float[] yval = (new float[outwidth * outheight * outchannels * batch]).Select((_, idx) => (float)random.NextDouble() * 1e-2f).ToArray();
            float[] wval = (new float[kwidth * kheight * inchannels * outchannels / 2]).Select((_, idx) => (float)random.NextDouble() * 1e-2f).ToArray();

            System.Numerics.Complex[] ycval = (new System.Numerics.Complex[yval.Length / 2])
                .Select((_, idx) => new System.Numerics.Complex(yval[idx * 2], yval[idx * 2 + 1])).ToArray();

            System.Numerics.Complex[] wcval = (new System.Numerics.Complex[wval.Length / 2])
                .Select((_, idx) => new System.Numerics.Complex(wval[idx * 2], wval[idx * 2 + 1])).ToArray();

            ComplexMap2D y = new(outchannels / 2, outwidth, outheight, batch, ycval);
            ComplexFilter2D w = new(inchannels / 2, outchannels / 2, kwidth, kheight, wcval);

            ComplexMap2D x = Reference(y, w, inwidth, inheight, kwidth, kheight);

            OverflowCheckedTensor y_tensor = new(Shape.Map2D(outchannels, outwidth, outheight, batch), yval);
            OverflowCheckedTensor w_tensor = new(Shape.Kernel2D(inchannels, outchannels / 2, kwidth, kheight), wval);

            OverflowCheckedTensor x_tensor = new(Shape.Map2D(inchannels, inwidth, inheight, batch));

            ComplexDeconvolution2D ope = new(outwidth, outheight, outchannels, inchannels, kwidth, kheight, gradmode: false, batch);

            ope.Execute(y_tensor, w_tensor, x_tensor);

            float[] x_expect = x.ToArray();
            float[] x_actual = x_tensor.State.Value;

            CollectionAssert.AreEqual(yval, y_tensor.State.Value);
            CollectionAssert.AreEqual(wval, w_tensor.State.Value);

            AssertError.Tolerance(x_expect, x_actual, 1e-7f, 1e-5f, ref max_err, $"mismatch value {inchannels},{outchannels},{kwidth},{kheight},{inwidth},{inheight},{batch}");

            Console.WriteLine($"pass: {inchannels},{outchannels},{kwidth},{kheight},{inwidth},{inheight},{batch}");

            Console.WriteLine($"maxerr:{max_err}");
        }

        [TestMethod]
        public void SpeedFPTest() {
            TensorShaderCudaBackend.Environment.Precision = TensorShaderCudaBackend.Environment.PrecisionMode.Float;
            TensorShaderCudaBackend.Environment.CudnnEnabled = false;

            int inwidth = 512, inheight = 512, inchannels = 32, outchannels = 32, ksize = 3;
            int outwidth = inwidth - ksize + 1, outheight = inheight - ksize + 1;

            OverflowCheckedTensor y_tensor = new(Shape.Map2D(outchannels, outwidth, outheight));
            OverflowCheckedTensor w_tensor = new(Shape.Kernel2D(inchannels, outchannels / 2, ksize, ksize));

            OverflowCheckedTensor x_tensor = new(Shape.Map2D(inchannels, inwidth, inheight));

            ComplexDeconvolution2D ope = new(outwidth, outheight, outchannels, inchannels, ksize, ksize);

            ope.Execute(y_tensor, w_tensor, x_tensor);

            Cuda.Profiler.Initialize("../../../../profiler.nvsetting", "../../nvprofiles/complex_deconvolution_2d_fp.nvvp");
            Cuda.Profiler.Start();

            ope.Execute(y_tensor, w_tensor, x_tensor);

            Cuda.Profiler.Stop();
        }

        [TestMethod]
        public void SpeedFFPTest() {
            TensorShaderCudaBackend.Environment.CudnnEnabled = false;
            TensorShaderCudaBackend.Environment.Precision = TensorShaderCudaBackend.Environment.PrecisionMode.FloatFloat;

            int inwidth = 512, inheight = 512, inchannels = 32, outchannels = 32, ksize = 3;
            int outwidth = inwidth - ksize + 1, outheight = inheight - ksize + 1;

            OverflowCheckedTensor y_tensor = new(Shape.Map2D(outchannels, outwidth, outheight));
            OverflowCheckedTensor w_tensor = new(Shape.Kernel2D(inchannels, outchannels / 2, ksize, ksize));

            OverflowCheckedTensor x_tensor = new(Shape.Map2D(inchannels, inwidth, inheight));

            ComplexDeconvolution2D ope = new(outwidth, outheight, outchannels, inchannels, ksize, ksize);

            ope.Execute(y_tensor, w_tensor, x_tensor);

            Cuda.Profiler.Initialize("../../../../profiler.nvsetting", "../../nvprofiles/complex_deconvolution_2d_ffp.nvvp");
            Cuda.Profiler.Start();

            ope.Execute(y_tensor, w_tensor, x_tensor);

            Cuda.Profiler.Stop();
        }

        public static ComplexMap2D Reference(ComplexMap2D y, ComplexFilter2D w, int inw, int inh, int kwidth, int kheight) {
            int inchannels = w.InChannels, outchannels = w.OutChannels, batch = y.Batch;
            int outw = inw - kwidth + 1, outh = inh - kheight + 1;

            if (y.Width != outw || y.Height != outh) {
                throw new ArgumentException("mismatch shape");
            }

            ComplexMap2D x = new(inchannels, inw, inh, batch);

            for (int kx, ky = 0; ky < kheight; ky++) {
                for (kx = 0; kx < kwidth; kx++) {
                    for (int th = 0; th < batch; th++) {
                        for (int ox, oy = 0; oy < outh; oy++) {
                            for (ox = 0; ox < outw; ox++) {
                                for (int outch = 0; outch < outchannels; outch++) {
                                    System.Numerics.Complex v = y[outch, ox, oy, th];

                                    for (int inch = 0; inch < inchannels; inch++) {
                                        x[inch, kx + ox, ky + oy, th] += v * w[inch, outch, kx, ky];
                                    }
                                }
                            }
                        }
                    }
                }
            }

            return x;
        }

        [TestMethod]
        public void ReferenceTest() {
            int inchannels = 6, outchannels = 8, kwidth = 3, kheight = 5, inwidth = 13, inheight = 17, batch = 3;
            int outwidth = inwidth - kwidth + 1, outheight = inheight - kheight + 1;

            float[] yval = (new float[outwidth * outheight * outchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
            float[] wval = (new float[kwidth * kheight * outchannels * inchannels / 2]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

            System.Numerics.Complex[] ycval = (new System.Numerics.Complex[yval.Length / 2])
                .Select((_, idx) => new System.Numerics.Complex(yval[idx * 2], yval[idx * 2 + 1])).ToArray();

            System.Numerics.Complex[] wcval = (new System.Numerics.Complex[wval.Length / 2])
                .Select((_, idx) => new System.Numerics.Complex(wval[idx * 2], wval[idx * 2 + 1])).ToArray();

            ComplexMap2D y = new(outchannels / 2, outwidth, outheight, batch, ycval);
            ComplexFilter2D w = new(inchannels / 2, outchannels / 2, kwidth, kheight, wcval);

            ComplexMap2D x = Reference(y, w, inwidth, inheight, kwidth, kheight);

            float[] x_expect = {
                -1.384000000e-03f,  9.668000000e-03f,  -1.376000000e-03f,  9.612000000e-03f,  -1.368000000e-03f,  9.556000000e-03f,
                -2.640000000e-03f,  4.103200000e-02f,  -2.624000000e-03f,  4.079200000e-02f,  -2.608000000e-03f,  4.055200000e-02f,
                -3.768000000e-03f,  9.255600000e-02f,  -3.744000000e-03f,  9.200400000e-02f,  -3.720000000e-03f,  9.145200000e-02f,
                -3.672000000e-03f,  1.550520000e-01f,  -3.648000000e-03f,  1.541160000e-01f,  -3.624000000e-03f,  1.531800000e-01f,
                -3.576000000e-03f,  2.175480000e-01f,  -3.552000000e-03f,  2.162280000e-01f,  -3.528000000e-03f,  2.149080000e-01f,
                -3.480000000e-03f,  2.800440000e-01f,  -3.456000000e-03f,  2.783400000e-01f,  -3.432000000e-03f,  2.766360000e-01f,
                -3.384000000e-03f,  3.425400000e-01f,  -3.360000000e-03f,  3.404520000e-01f,  -3.336000000e-03f,  3.383640000e-01f,
                -3.288000000e-03f,  4.050360000e-01f,  -3.264000000e-03f,  4.025640000e-01f,  -3.240000000e-03f,  4.000920000e-01f,
                -3.192000000e-03f,  4.675320000e-01f,  -3.168000000e-03f,  4.646760000e-01f,  -3.144000000e-03f,  4.618200000e-01f,
                -3.096000000e-03f,  5.300280000e-01f,  -3.072000000e-03f,  5.267880000e-01f,  -3.048000000e-03f,  5.235480000e-01f,
                -3.000000000e-03f,  5.925240000e-01f,  -2.976000000e-03f,  5.889000000e-01f,  -2.952000000e-03f,  5.852760000e-01f,
                -1.872000000e-03f,  3.993040000e-01f,  -1.856000000e-03f,  3.967600000e-01f,  -1.840000000e-03f,  3.942160000e-01f,
                -8.720000000e-04f,  2.012840000e-01f,  -8.640000000e-04f,  1.999480000e-01f,  -8.560000000e-04f,  1.986120000e-01f,
                -2.128000000e-03f,  2.633680000e-01f,  -2.112000000e-03f,  2.618480000e-01f,  -2.096000000e-03f,  2.603280000e-01f,
                -4.000000000e-03f,  5.486240000e-01f,  -3.968000000e-03f,  5.453280000e-01f,  -3.936000000e-03f,  5.420320000e-01f,
                -5.616000000e-03f,  8.526960000e-01f,  -5.568000000e-03f,  8.473680000e-01f,  -5.520000000e-03f,  8.420400000e-01f,
                -5.424000000e-03f,  9.638640000e-01f,  -5.376000000e-03f,  9.577680000e-01f,  -5.328000000e-03f,  9.516720000e-01f,
                -5.232000000e-03f,  1.075032000e+00f,  -5.184000000e-03f,  1.068168000e+00f,  -5.136000000e-03f,  1.061304000e+00f,
                -5.040000000e-03f,  1.186200000e+00f,  -4.992000000e-03f,  1.178568000e+00f,  -4.944000000e-03f,  1.170936000e+00f,
                -4.848000000e-03f,  1.297368000e+00f,  -4.800000000e-03f,  1.288968000e+00f,  -4.752000000e-03f,  1.280568000e+00f,
                -4.656000000e-03f,  1.408536000e+00f,  -4.608000000e-03f,  1.399368000e+00f,  -4.560000000e-03f,  1.390200000e+00f,
                -4.464000000e-03f,  1.519704000e+00f,  -4.416000000e-03f,  1.509768000e+00f,  -4.368000000e-03f,  1.499832000e+00f,
                -4.272000000e-03f,  1.630872000e+00f,  -4.224000000e-03f,  1.620168000e+00f,  -4.176000000e-03f,  1.609464000e+00f,
                -4.080000000e-03f,  1.742040000e+00f,  -4.032000000e-03f,  1.730568000e+00f,  -3.984000000e-03f,  1.719096000e+00f,
                -2.464000000e-03f,  1.148432000e+00f,  -2.432000000e-03f,  1.140528000e+00f,  -2.400000000e-03f,  1.132624000e+00f,
                -1.104000000e-03f,  5.667280000e-01f,  -1.088000000e-03f,  5.626480000e-01f,  -1.072000000e-03f,  5.585680000e-01f,
                -2.232000000e-03f,  7.104120000e-01f,  -2.208000000e-03f,  7.060200000e-01f,  -2.184000000e-03f,  7.016280000e-01f,
                -4.080000000e-03f,  1.421400000e+00f,  -4.032000000e-03f,  1.412232000e+00f,  -3.984000000e-03f,  1.403064000e+00f,
                -5.544000000e-03f,  2.128356000e+00f,  -5.472000000e-03f,  2.114028000e+00f,  -5.400000000e-03f,  2.099700000e+00f,
                -5.256000000e-03f,  2.274372000e+00f,  -5.184000000e-03f,  2.258892000e+00f,  -5.112000000e-03f,  2.243412000e+00f,
                -4.968000000e-03f,  2.420388000e+00f,  -4.896000000e-03f,  2.403756000e+00f,  -4.824000000e-03f,  2.387124000e+00f,
                -4.680000000e-03f,  2.566404000e+00f,  -4.608000000e-03f,  2.548620000e+00f,  -4.536000000e-03f,  2.530836000e+00f,
                -4.392000000e-03f,  2.712420000e+00f,  -4.320000000e-03f,  2.693484000e+00f,  -4.248000000e-03f,  2.674548000e+00f,
                -4.104000000e-03f,  2.858436000e+00f,  -4.032000000e-03f,  2.838348000e+00f,  -3.960000000e-03f,  2.818260000e+00f,
                -3.816000000e-03f,  3.004452000e+00f,  -3.744000000e-03f,  2.983212000e+00f,  -3.672000000e-03f,  2.961972000e+00f,
                -3.528000000e-03f,  3.150468000e+00f,  -3.456000000e-03f,  3.128076000e+00f,  -3.384000000e-03f,  3.105684000e+00f,
                -3.240000000e-03f,  3.296484000e+00f,  -3.168000000e-03f,  3.272940000e+00f,  -3.096000000e-03f,  3.249396000e+00f,
                -1.776000000e-03f,  2.146008000e+00f,  -1.728000000e-03f,  2.129928000e+00f,  -1.680000000e-03f,  2.113848000e+00f,
                -6.960000000e-04f,  1.045644000e+00f,  -6.720000000e-04f,  1.037412000e+00f,  -6.480000000e-04f,  1.029180000e+00f,
                -1.696000000e-03f,  1.300112000e+00f,  -1.664000000e-03f,  1.291440000e+00f,  -1.632000000e-03f,  1.282768000e+00f,
                -2.880000000e-03f,  2.557984000e+00f,  -2.816000000e-03f,  2.540128000e+00f,  -2.752000000e-03f,  2.522272000e+00f,
                -3.552000000e-03f,  3.767472000e+00f,  -3.456000000e-03f,  3.739920000e+00f,  -3.360000000e-03f,  3.712368000e+00f,
                -3.168000000e-03f,  3.934512000e+00f,  -3.072000000e-03f,  3.905424000e+00f,  -2.976000000e-03f,  3.876336000e+00f,
                -2.784000000e-03f,  4.101552000e+00f,  -2.688000000e-03f,  4.070928000e+00f,  -2.592000000e-03f,  4.040304000e+00f,
                -2.400000000e-03f,  4.268592000e+00f,  -2.304000000e-03f,  4.236432000e+00f,  -2.208000000e-03f,  4.204272000e+00f,
                -2.016000000e-03f,  4.435632000e+00f,  -1.920000000e-03f,  4.401936000e+00f,  -1.824000000e-03f,  4.368240000e+00f,
                -1.632000000e-03f,  4.602672000e+00f,  -1.536000000e-03f,  4.567440000e+00f,  -1.440000000e-03f,  4.532208000e+00f,
                -1.248000000e-03f,  4.769712000e+00f,  -1.152000000e-03f,  4.732944000e+00f,  -1.056000000e-03f,  4.696176000e+00f,
                -8.640000000e-04f,  4.936752000e+00f,  -7.680000000e-04f,  4.898448000e+00f,  -6.720000000e-04f,  4.860144000e+00f,
                -4.800000000e-04f,  5.103792000e+00f,  -3.840000000e-04f,  5.063952000e+00f,  -2.880000000e-04f,  5.024112000e+00f,
                1.920000000e-04f,  3.290656000e+00f,  2.560000000e-04f,  3.263584000e+00f,  3.200000000e-04f,  3.236512000e+00f,
                3.520000000e-04f,  1.587344000e+00f,  3.840000000e-04f,  1.573552000e+00f,  4.160000000e-04f,  1.559760000e+00f,
                -5.200000000e-04f,  1.981780000e+00f,  -4.800000000e-04f,  1.967420000e+00f,  -4.400000000e-04f,  1.953060000e+00f,
                -4.000000000e-04f,  3.857000000e+00f,  -3.200000000e-04f,  3.827640000e+00f,  -2.400000000e-04f,  3.798280000e+00f,
                3.600000000e-04f,  5.617980000e+00f,  4.800000000e-04f,  5.572980000e+00f,  6.000000000e-04f,  5.527980000e+00f,
                8.400000000e-04f,  5.792220000e+00f,  9.600000000e-04f,  5.745300000e+00f,  1.080000000e-03f,  5.698380000e+00f,
                1.320000000e-03f,  5.966460000e+00f,  1.440000000e-03f,  5.917620000e+00f,  1.560000000e-03f,  5.868780000e+00f,
                1.800000000e-03f,  6.140700000e+00f,  1.920000000e-03f,  6.089940000e+00f,  2.040000000e-03f,  6.039180000e+00f,
                2.280000000e-03f,  6.314940000e+00f,  2.400000000e-03f,  6.262260000e+00f,  2.520000000e-03f,  6.209580000e+00f,
                2.760000000e-03f,  6.489180000e+00f,  2.880000000e-03f,  6.434580000e+00f,  3.000000000e-03f,  6.379980000e+00f,
                3.240000000e-03f,  6.663420000e+00f,  3.360000000e-03f,  6.606900000e+00f,  3.480000000e-03f,  6.550380000e+00f,
                3.720000000e-03f,  6.837660000e+00f,  3.840000000e-03f,  6.779220000e+00f,  3.960000000e-03f,  6.720780000e+00f,
                4.200000000e-03f,  7.011900000e+00f,  4.320000000e-03f,  6.951540000e+00f,  4.440000000e-03f,  6.891180000e+00f,
                3.440000000e-03f,  4.481000000e+00f,  3.520000000e-03f,  4.440120000e+00f,  3.600000000e-03f,  4.399240000e+00f,
                2.040000000e-03f,  2.141140000e+00f,  2.080000000e-03f,  2.120380000e+00f,  2.120000000e-03f,  2.099620000e+00f,
                1.240000000e-03f,  2.705140000e+00f,  1.280000000e-03f,  2.683740000e+00f,  1.320000000e-03f,  2.662340000e+00f,
                3.120000000e-03f,  5.219240000e+00f,  3.200000000e-03f,  5.175800000e+00f,  3.280000000e-03f,  5.132360000e+00f,
                5.640000000e-03f,  7.534620000e+00f,  5.760000000e-03f,  7.468500000e+00f,  5.880000000e-03f,  7.402380000e+00f,
                6.120000000e-03f,  7.708860000e+00f,  6.240000000e-03f,  7.640820000e+00f,  6.360000000e-03f,  7.572780000e+00f,
                6.600000000e-03f,  7.883100000e+00f,  6.720000000e-03f,  7.813140000e+00f,  6.840000000e-03f,  7.743180000e+00f,
                7.080000000e-03f,  8.057340000e+00f,  7.200000000e-03f,  7.985460000e+00f,  7.320000000e-03f,  7.913580000e+00f,
                7.560000000e-03f,  8.231580000e+00f,  7.680000000e-03f,  8.157780000e+00f,  7.800000000e-03f,  8.083980000e+00f,
                8.040000000e-03f,  8.405820000e+00f,  8.160000000e-03f,  8.330100000e+00f,  8.280000000e-03f,  8.254380000e+00f,
                8.520000000e-03f,  8.580060000e+00f,  8.640000000e-03f,  8.502420000e+00f,  8.760000000e-03f,  8.424780000e+00f,
                9.000000000e-03f,  8.754300000e+00f,  9.120000000e-03f,  8.674740000e+00f,  9.240000000e-03f,  8.595180000e+00f,
                9.480000000e-03f,  8.928540000e+00f,  9.600000000e-03f,  8.847060000e+00f,  9.720000000e-03f,  8.765580000e+00f,
                6.960000000e-03f,  5.674280000e+00f,  7.040000000e-03f,  5.619320000e+00f,  7.120000000e-03f,  5.564360000e+00f,
                3.800000000e-03f,  2.695540000e+00f,  3.840000000e-03f,  2.667740000e+00f,  3.880000000e-03f,  2.639940000e+00f,
                3.000000000e-03f,  3.428500000e+00f,  3.040000000e-03f,  3.400060000e+00f,  3.080000000e-03f,  3.371620000e+00f,
                6.640000000e-03f,  6.581480000e+00f,  6.720000000e-03f,  6.523960000e+00f,  6.800000000e-03f,  6.466440000e+00f,
                1.092000000e-02f,  9.451260000e+00f,  1.104000000e-02f,  9.364020000e+00f,  1.116000000e-02f,  9.276780000e+00f,
                1.140000000e-02f,  9.625500000e+00f,  1.152000000e-02f,  9.536340000e+00f,  1.164000000e-02f,  9.447180000e+00f,
                1.188000000e-02f,  9.799740000e+00f,  1.200000000e-02f,  9.708660000e+00f,  1.212000000e-02f,  9.617580000e+00f,
                1.236000000e-02f,  9.973980000e+00f,  1.248000000e-02f,  9.880980000e+00f,  1.260000000e-02f,  9.787980000e+00f,
                1.284000000e-02f,  1.014822000e+01f,  1.296000000e-02f,  1.005330000e+01f,  1.308000000e-02f,  9.958380000e+00f,
                1.332000000e-02f,  1.032246000e+01f,  1.344000000e-02f,  1.022562000e+01f,  1.356000000e-02f,  1.012878000e+01f,
                1.380000000e-02f,  1.049670000e+01f,  1.392000000e-02f,  1.039794000e+01f,  1.404000000e-02f,  1.029918000e+01f,
                1.428000000e-02f,  1.067094000e+01f,  1.440000000e-02f,  1.057026000e+01f,  1.452000000e-02f,  1.046958000e+01f,
                1.476000000e-02f,  1.084518000e+01f,  1.488000000e-02f,  1.074258000e+01f,  1.500000000e-02f,  1.063998000e+01f,
                1.048000000e-02f,  6.867560000e+00f,  1.056000000e-02f,  6.798520000e+00f,  1.064000000e-02f,  6.729480000e+00f,
                5.560000000e-03f,  3.249940000e+00f,  5.600000000e-03f,  3.215100000e+00f,  5.640000000e-03f,  3.180260000e+00f,
                4.760000000e-03f,  4.151860000e+00f,  4.800000000e-03f,  4.116380000e+00f,  4.840000000e-03f,  4.080900000e+00f,
                1.016000000e-02f,  7.943720000e+00f,  1.024000000e-02f,  7.872120000e+00f,  1.032000000e-02f,  7.800520000e+00f,
                1.620000000e-02f,  1.136790000e+01f,  1.632000000e-02f,  1.125954000e+01f,  1.644000000e-02f,  1.115118000e+01f,
                1.668000000e-02f,  1.154214000e+01f,  1.680000000e-02f,  1.143186000e+01f,  1.692000000e-02f,  1.132158000e+01f,
                1.716000000e-02f,  1.171638000e+01f,  1.728000000e-02f,  1.160418000e+01f,  1.740000000e-02f,  1.149198000e+01f,
                1.764000000e-02f,  1.189062000e+01f,  1.776000000e-02f,  1.177650000e+01f,  1.788000000e-02f,  1.166238000e+01f,
                1.812000000e-02f,  1.206486000e+01f,  1.824000000e-02f,  1.194882000e+01f,  1.836000000e-02f,  1.183278000e+01f,
                1.860000000e-02f,  1.223910000e+01f,  1.872000000e-02f,  1.212114000e+01f,  1.884000000e-02f,  1.200318000e+01f,
                1.908000000e-02f,  1.241334000e+01f,  1.920000000e-02f,  1.229346000e+01f,  1.932000000e-02f,  1.217358000e+01f,
                1.956000000e-02f,  1.258758000e+01f,  1.968000000e-02f,  1.246578000e+01f,  1.980000000e-02f,  1.234398000e+01f,
                2.004000000e-02f,  1.276182000e+01f,  2.016000000e-02f,  1.263810000e+01f,  2.028000000e-02f,  1.251438000e+01f,
                1.400000000e-02f,  8.060840000e+00f,  1.408000000e-02f,  7.977720000e+00f,  1.416000000e-02f,  7.894600000e+00f,
                7.320000000e-03f,  3.804340000e+00f,  7.360000000e-03f,  3.762460000e+00f,  7.400000000e-03f,  3.720580000e+00f,
                6.520000000e-03f,  4.875220000e+00f,  6.560000000e-03f,  4.832700000e+00f,  6.600000000e-03f,  4.790180000e+00f,
                1.368000000e-02f,  9.305960000e+00f,  1.376000000e-02f,  9.220280000e+00f,  1.384000000e-02f,  9.134600000e+00f,
                2.148000000e-02f,  1.328454000e+01f,  2.160000000e-02f,  1.315506000e+01f,  2.172000000e-02f,  1.302558000e+01f,
                2.196000000e-02f,  1.345878000e+01f,  2.208000000e-02f,  1.332738000e+01f,  2.220000000e-02f,  1.319598000e+01f,
                2.244000000e-02f,  1.363302000e+01f,  2.256000000e-02f,  1.349970000e+01f,  2.268000000e-02f,  1.336638000e+01f,
                2.292000000e-02f,  1.380726000e+01f,  2.304000000e-02f,  1.367202000e+01f,  2.316000000e-02f,  1.353678000e+01f,
                2.340000000e-02f,  1.398150000e+01f,  2.352000000e-02f,  1.384434000e+01f,  2.364000000e-02f,  1.370718000e+01f,
                2.388000000e-02f,  1.415574000e+01f,  2.400000000e-02f,  1.401666000e+01f,  2.412000000e-02f,  1.387758000e+01f,
                2.436000000e-02f,  1.432998000e+01f,  2.448000000e-02f,  1.418898000e+01f,  2.460000000e-02f,  1.404798000e+01f,
                2.484000000e-02f,  1.450422000e+01f,  2.496000000e-02f,  1.436130000e+01f,  2.508000000e-02f,  1.421838000e+01f,
                2.532000000e-02f,  1.467846000e+01f,  2.544000000e-02f,  1.453362000e+01f,  2.556000000e-02f,  1.438878000e+01f,
                1.752000000e-02f,  9.254120000e+00f,  1.760000000e-02f,  9.156920000e+00f,  1.768000000e-02f,  9.059720000e+00f,
                9.080000000e-03f,  4.358740000e+00f,  9.120000000e-03f,  4.309820000e+00f,  9.160000000e-03f,  4.260900000e+00f,
                8.280000000e-03f,  5.598580000e+00f,  8.320000000e-03f,  5.549020000e+00f,  8.360000000e-03f,  5.499460000e+00f,
                1.720000000e-02f,  1.066820000e+01f,  1.728000000e-02f,  1.056844000e+01f,  1.736000000e-02f,  1.046868000e+01f,
                2.676000000e-02f,  1.520118000e+01f,  2.688000000e-02f,  1.505058000e+01f,  2.700000000e-02f,  1.489998000e+01f,
                2.724000000e-02f,  1.537542000e+01f,  2.736000000e-02f,  1.522290000e+01f,  2.748000000e-02f,  1.507038000e+01f,
                2.772000000e-02f,  1.554966000e+01f,  2.784000000e-02f,  1.539522000e+01f,  2.796000000e-02f,  1.524078000e+01f,
                2.820000000e-02f,  1.572390000e+01f,  2.832000000e-02f,  1.556754000e+01f,  2.844000000e-02f,  1.541118000e+01f,
                2.868000000e-02f,  1.589814000e+01f,  2.880000000e-02f,  1.573986000e+01f,  2.892000000e-02f,  1.558158000e+01f,
                2.916000000e-02f,  1.607238000e+01f,  2.928000000e-02f,  1.591218000e+01f,  2.940000000e-02f,  1.575198000e+01f,
                2.964000000e-02f,  1.624662000e+01f,  2.976000000e-02f,  1.608450000e+01f,  2.988000000e-02f,  1.592238000e+01f,
                3.012000000e-02f,  1.642086000e+01f,  3.024000000e-02f,  1.625682000e+01f,  3.036000000e-02f,  1.609278000e+01f,
                3.060000000e-02f,  1.659510000e+01f,  3.072000000e-02f,  1.642914000e+01f,  3.084000000e-02f,  1.626318000e+01f,
                2.104000000e-02f,  1.044740000e+01f,  2.112000000e-02f,  1.033612000e+01f,  2.120000000e-02f,  1.022484000e+01f,
                1.084000000e-02f,  4.913140000e+00f,  1.088000000e-02f,  4.857180000e+00f,  1.092000000e-02f,  4.801220000e+00f,
                1.004000000e-02f,  6.321940000e+00f,  1.008000000e-02f,  6.265340000e+00f,  1.012000000e-02f,  6.208740000e+00f,
                2.072000000e-02f,  1.203044000e+01f,  2.080000000e-02f,  1.191660000e+01f,  2.088000000e-02f,  1.180276000e+01f,
                3.204000000e-02f,  1.711782000e+01f,  3.216000000e-02f,  1.694610000e+01f,  3.228000000e-02f,  1.677438000e+01f,
                3.252000000e-02f,  1.729206000e+01f,  3.264000000e-02f,  1.711842000e+01f,  3.276000000e-02f,  1.694478000e+01f,
                3.300000000e-02f,  1.746630000e+01f,  3.312000000e-02f,  1.729074000e+01f,  3.324000000e-02f,  1.711518000e+01f,
                3.348000000e-02f,  1.764054000e+01f,  3.360000000e-02f,  1.746306000e+01f,  3.372000000e-02f,  1.728558000e+01f,
                3.396000000e-02f,  1.781478000e+01f,  3.408000000e-02f,  1.763538000e+01f,  3.420000000e-02f,  1.745598000e+01f,
                3.444000000e-02f,  1.798902000e+01f,  3.456000000e-02f,  1.780770000e+01f,  3.468000000e-02f,  1.762638000e+01f,
                3.492000000e-02f,  1.816326000e+01f,  3.504000000e-02f,  1.798002000e+01f,  3.516000000e-02f,  1.779678000e+01f,
                3.540000000e-02f,  1.833750000e+01f,  3.552000000e-02f,  1.815234000e+01f,  3.564000000e-02f,  1.796718000e+01f,
                3.588000000e-02f,  1.851174000e+01f,  3.600000000e-02f,  1.832466000e+01f,  3.612000000e-02f,  1.813758000e+01f,
                2.456000000e-02f,  1.164068000e+01f,  2.464000000e-02f,  1.151532000e+01f,  2.472000000e-02f,  1.138996000e+01f,
                1.260000000e-02f,  5.467540000e+00f,  1.264000000e-02f,  5.404540000e+00f,  1.268000000e-02f,  5.341540000e+00f,
                1.180000000e-02f,  7.045300000e+00f,  1.184000000e-02f,  6.981660000e+00f,  1.188000000e-02f,  6.918020000e+00f,
                2.424000000e-02f,  1.339268000e+01f,  2.432000000e-02f,  1.326476000e+01f,  2.440000000e-02f,  1.313684000e+01f,
                3.732000000e-02f,  1.903446000e+01f,  3.744000000e-02f,  1.884162000e+01f,  3.756000000e-02f,  1.864878000e+01f,
                3.780000000e-02f,  1.920870000e+01f,  3.792000000e-02f,  1.901394000e+01f,  3.804000000e-02f,  1.881918000e+01f,
                3.828000000e-02f,  1.938294000e+01f,  3.840000000e-02f,  1.918626000e+01f,  3.852000000e-02f,  1.898958000e+01f,
                3.876000000e-02f,  1.955718000e+01f,  3.888000000e-02f,  1.935858000e+01f,  3.900000000e-02f,  1.915998000e+01f,
                3.924000000e-02f,  1.973142000e+01f,  3.936000000e-02f,  1.953090000e+01f,  3.948000000e-02f,  1.933038000e+01f,
                3.972000000e-02f,  1.990566000e+01f,  3.984000000e-02f,  1.970322000e+01f,  3.996000000e-02f,  1.950078000e+01f,
                4.020000000e-02f,  2.007990000e+01f,  4.032000000e-02f,  1.987554000e+01f,  4.044000000e-02f,  1.967118000e+01f,
                4.068000000e-02f,  2.025414000e+01f,  4.080000000e-02f,  2.004786000e+01f,  4.092000000e-02f,  1.984158000e+01f,
                4.116000000e-02f,  2.042838000e+01f,  4.128000000e-02f,  2.022018000e+01f,  4.140000000e-02f,  2.001198000e+01f,
                2.808000000e-02f,  1.283396000e+01f,  2.816000000e-02f,  1.269452000e+01f,  2.824000000e-02f,  1.255508000e+01f,
                1.436000000e-02f,  6.021940000e+00f,  1.440000000e-02f,  5.951900000e+00f,  1.444000000e-02f,  5.881860000e+00f,
                1.356000000e-02f,  7.768660000e+00f,  1.360000000e-02f,  7.697980000e+00f,  1.364000000e-02f,  7.627300000e+00f,
                2.776000000e-02f,  1.475492000e+01f,  2.784000000e-02f,  1.461292000e+01f,  2.792000000e-02f,  1.447092000e+01f,
                4.260000000e-02f,  2.095110000e+01f,  4.272000000e-02f,  2.073714000e+01f,  4.284000000e-02f,  2.052318000e+01f,
                4.308000000e-02f,  2.112534000e+01f,  4.320000000e-02f,  2.090946000e+01f,  4.332000000e-02f,  2.069358000e+01f,
                4.356000000e-02f,  2.129958000e+01f,  4.368000000e-02f,  2.108178000e+01f,  4.380000000e-02f,  2.086398000e+01f,
                4.404000000e-02f,  2.147382000e+01f,  4.416000000e-02f,  2.125410000e+01f,  4.428000000e-02f,  2.103438000e+01f,
                4.452000000e-02f,  2.164806000e+01f,  4.464000000e-02f,  2.142642000e+01f,  4.476000000e-02f,  2.120478000e+01f,
                4.500000000e-02f,  2.182230000e+01f,  4.512000000e-02f,  2.159874000e+01f,  4.524000000e-02f,  2.137518000e+01f,
                4.548000000e-02f,  2.199654000e+01f,  4.560000000e-02f,  2.177106000e+01f,  4.572000000e-02f,  2.154558000e+01f,
                4.596000000e-02f,  2.217078000e+01f,  4.608000000e-02f,  2.194338000e+01f,  4.620000000e-02f,  2.171598000e+01f,
                4.644000000e-02f,  2.234502000e+01f,  4.656000000e-02f,  2.211570000e+01f,  4.668000000e-02f,  2.188638000e+01f,
                3.160000000e-02f,  1.402724000e+01f,  3.168000000e-02f,  1.387372000e+01f,  3.176000000e-02f,  1.372020000e+01f,
                1.612000000e-02f,  6.576340000e+00f,  1.616000000e-02f,  6.499260000e+00f,  1.620000000e-02f,  6.422180000e+00f,
                1.212800000e-02f,  5.283728000e+00f,  1.216000000e-02f,  5.224368000e+00f,  1.219200000e-02f,  5.165008000e+00f,
                2.476800000e-02f,  9.898528000e+00f,  2.483200000e-02f,  9.779296000e+00f,  2.489600000e-02f,  9.660064000e+00f,
                3.792000000e-02f,  1.383825600e+01f,  3.801600000e-02f,  1.365864000e+01f,  3.811200000e-02f,  1.347902400e+01f,
                3.830400000e-02f,  1.395000000e+01f,  3.840000000e-02f,  1.376884800e+01f,  3.849600000e-02f,  1.358769600e+01f,
                3.868800000e-02f,  1.406174400e+01f,  3.878400000e-02f,  1.387905600e+01f,  3.888000000e-02f,  1.369636800e+01f,
                3.907200000e-02f,  1.417348800e+01f,  3.916800000e-02f,  1.398926400e+01f,  3.926400000e-02f,  1.380504000e+01f,
                3.945600000e-02f,  1.428523200e+01f,  3.955200000e-02f,  1.409947200e+01f,  3.964800000e-02f,  1.391371200e+01f,
                3.984000000e-02f,  1.439697600e+01f,  3.993600000e-02f,  1.420968000e+01f,  4.003200000e-02f,  1.402238400e+01f,
                4.022400000e-02f,  1.450872000e+01f,  4.032000000e-02f,  1.431988800e+01f,  4.041600000e-02f,  1.413105600e+01f,
                4.060800000e-02f,  1.462046400e+01f,  4.070400000e-02f,  1.443009600e+01f,  4.080000000e-02f,  1.423972800e+01f,
                4.099200000e-02f,  1.473220800e+01f,  4.108800000e-02f,  1.454030400e+01f,  4.118400000e-02f,  1.434840000e+01f,
                2.784000000e-02f,  9.082912000e+00f,  2.790400000e-02f,  8.954464000e+00f,  2.796800000e-02f,  8.826016000e+00f,
                1.417600000e-02f,  4.170128000e+00f,  1.420800000e-02f,  4.105648000e+00f,  1.424000000e-02f,  4.041168000e+00f,
                1.005600000e-02f,  3.213708000e+00f,  1.008000000e-02f,  3.167076000e+00f,  1.010400000e-02f,  3.120444000e+00f,
                2.049600000e-02f,  5.893464000e+00f,  2.054400000e-02f,  5.799816000e+00f,  2.059200000e-02f,  5.706168000e+00f,
                3.132000000e-02f,  8.034660000e+00f,  3.139200000e-02f,  7.893612000e+00f,  3.146400000e-02f,  7.752564000e+00f,
                3.160800000e-02f,  8.097732000e+00f,  3.168000000e-02f,  7.955532000e+00f,  3.175200000e-02f,  7.813332000e+00f,
                3.189600000e-02f,  8.160804000e+00f,  3.196800000e-02f,  8.017452000e+00f,  3.204000000e-02f,  7.874100000e+00f,
                3.218400000e-02f,  8.223876000e+00f,  3.225600000e-02f,  8.079372000e+00f,  3.232800000e-02f,  7.934868000e+00f,
                3.247200000e-02f,  8.286948000e+00f,  3.254400000e-02f,  8.141292000e+00f,  3.261600000e-02f,  7.995636000e+00f,
                3.276000000e-02f,  8.350020000e+00f,  3.283200000e-02f,  8.203212000e+00f,  3.290400000e-02f,  8.056404000e+00f,
                3.304800000e-02f,  8.413092000e+00f,  3.312000000e-02f,  8.265132000e+00f,  3.319200000e-02f,  8.117172000e+00f,
                3.333600000e-02f,  8.476164000e+00f,  3.340800000e-02f,  8.327052000e+00f,  3.348000000e-02f,  8.177940000e+00f,
                3.362400000e-02f,  8.539236000e+00f,  3.369600000e-02f,  8.388972000e+00f,  3.376800000e-02f,  8.238708000e+00f,
                2.280000000e-02f,  5.106648000e+00f,  2.284800000e-02f,  5.006088000e+00f,  2.289600000e-02f,  4.905528000e+00f,
                1.159200000e-02f,  2.258700000e+00f,  1.161600000e-02f,  2.208228000e+00f,  1.164000000e-02f,  2.157756000e+00f,
                7.344000000e-03f,  1.609288000e+00f,  7.360000000e-03f,  1.576792000e+00f,  7.376000000e-03f,  1.544296000e+00f,
                1.494400000e-02f,  2.841104000e+00f,  1.497600000e-02f,  2.775856000e+00f,  1.500800000e-02f,  2.710608000e+00f,
                2.280000000e-02f,  3.692376000e+00f,  2.284800000e-02f,  3.594120000e+00f,  2.289600000e-02f,  3.495864000e+00f,
                2.299200000e-02f,  3.720600000e+00f,  2.304000000e-02f,  3.621576000e+00f,  2.308800000e-02f,  3.522552000e+00f,
                2.318400000e-02f,  3.748824000e+00f,  2.323200000e-02f,  3.649032000e+00f,  2.328000000e-02f,  3.549240000e+00f,
                2.337600000e-02f,  3.777048000e+00f,  2.342400000e-02f,  3.676488000e+00f,  2.347200000e-02f,  3.575928000e+00f,
                2.356800000e-02f,  3.805272000e+00f,  2.361600000e-02f,  3.703944000e+00f,  2.366400000e-02f,  3.602616000e+00f,
                2.376000000e-02f,  3.833496000e+00f,  2.380800000e-02f,  3.731400000e+00f,  2.385600000e-02f,  3.629304000e+00f,
                2.395200000e-02f,  3.861720000e+00f,  2.400000000e-02f,  3.758856000e+00f,  2.404800000e-02f,  3.655992000e+00f,
                2.414400000e-02f,  3.889944000e+00f,  2.419200000e-02f,  3.786312000e+00f,  2.424000000e-02f,  3.682680000e+00f,
                2.433600000e-02f,  3.918168000e+00f,  2.438400000e-02f,  3.813768000e+00f,  2.443200000e-02f,  3.709368000e+00f,
                1.648000000e-02f,  2.199824000e+00f,  1.651200000e-02f,  2.129968000e+00f,  1.654400000e-02f,  2.060112000e+00f,
                8.368000000e-03f,  8.927440000e-01f,  8.384000000e-03f,  8.576880000e-01f,  8.400000000e-03f,  8.226320000e-01f,
                3.992000000e-03f,  5.211560000e-01f,  4.000000000e-03f,  5.042040000e-01f,  4.008000000e-03f,  4.872520000e-01f,
                8.112000000e-03f,  8.428240000e-01f,  8.128000000e-03f,  8.087920000e-01f,  8.144000000e-03f,  7.747600000e-01f,
                1.236000000e-02f,  9.634680000e-01f,  1.238400000e-02f,  9.122280000e-01f,  1.240800000e-02f,  8.609880000e-01f,
                1.245600000e-02f,  9.706680000e-01f,  1.248000000e-02f,  9.190440000e-01f,  1.250400000e-02f,  8.674200000e-01f,
                1.255200000e-02f,  9.778680000e-01f,  1.257600000e-02f,  9.258600000e-01f,  1.260000000e-02f,  8.738520000e-01f,
                1.264800000e-02f,  9.850680000e-01f,  1.267200000e-02f,  9.326760000e-01f,  1.269600000e-02f,  8.802840000e-01f,
                1.274400000e-02f,  9.922680000e-01f,  1.276800000e-02f,  9.394920000e-01f,  1.279200000e-02f,  8.867160000e-01f,
                1.284000000e-02f,  9.994680000e-01f,  1.286400000e-02f,  9.463080000e-01f,  1.288800000e-02f,  8.931480000e-01f,
                1.293600000e-02f,  1.006668000e+00f,  1.296000000e-02f,  9.531240000e-01f,  1.298400000e-02f,  8.995800000e-01f,
                1.303200000e-02f,  1.013868000e+00f,  1.305600000e-02f,  9.599400000e-01f,  1.308000000e-02f,  9.060120000e-01f,
                1.312800000e-02f,  1.021068000e+00f,  1.315200000e-02f,  9.667560000e-01f,  1.317600000e-02f,  9.124440000e-01f,
                8.880000000e-03f,  4.638160000e-01f,  8.896000000e-03f,  4.274800000e-01f,  8.912000000e-03f,  3.911440000e-01f,
                4.504000000e-03f,  1.229480000e-01f,  4.512000000e-03f,  1.047160000e-01f,  4.520000000e-03f,  8.648400000e-02f,
                3.192000000e-03f,  3.208292000e+00f,  3.200000000e-03f,  3.189932000e+00f,  3.208000000e-03f,  3.171572000e+00f,
                6.512000000e-03f,  6.218632000e+00f,  6.528000000e-03f,  6.181784000e+00f,  6.544000000e-03f,  6.144936000e+00f,
                9.960000000e-03f,  9.029484000e+00f,  9.984000000e-03f,  8.974020000e+00f,  1.000800000e-02f,  8.918556000e+00f,
                1.005600000e-02f,  9.091980000e+00f,  1.008000000e-02f,  9.036132000e+00f,  1.010400000e-02f,  8.980284000e+00f,
                1.015200000e-02f,  9.154476000e+00f,  1.017600000e-02f,  9.098244000e+00f,  1.020000000e-02f,  9.042012000e+00f,
                1.024800000e-02f,  9.216972000e+00f,  1.027200000e-02f,  9.160356000e+00f,  1.029600000e-02f,  9.103740000e+00f,
                1.034400000e-02f,  9.279468000e+00f,  1.036800000e-02f,  9.222468000e+00f,  1.039200000e-02f,  9.165468000e+00f,
                1.044000000e-02f,  9.341964000e+00f,  1.046400000e-02f,  9.284580000e+00f,  1.048800000e-02f,  9.227196000e+00f,
                1.053600000e-02f,  9.404460000e+00f,  1.056000000e-02f,  9.346692000e+00f,  1.058400000e-02f,  9.288924000e+00f,
                1.063200000e-02f,  9.466956000e+00f,  1.065600000e-02f,  9.408804000e+00f,  1.068000000e-02f,  9.350652000e+00f,
                1.072800000e-02f,  9.529452000e+00f,  1.075200000e-02f,  9.470916000e+00f,  1.077600000e-02f,  9.412380000e+00f,
                7.280000000e-03f,  6.137608000e+00f,  7.296000000e-03f,  6.098456000e+00f,  7.312000000e-03f,  6.059304000e+00f,
                3.704000000e-03f,  2.960612000e+00f,  3.712000000e-03f,  2.940972000e+00f,  3.720000000e-03f,  2.921332000e+00f,
                7.024000000e-03f,  6.001672000e+00f,  7.040000000e-03f,  5.963544000e+00f,  7.056000000e-03f,  5.925416000e+00f,
                1.430400000e-02f,  1.158593600e+01f,  1.433600000e-02f,  1.150942400e+01f,  1.436800000e-02f,  1.143291200e+01f,
                2.184000000e-02f,  1.674972000e+01f,  2.188800000e-02f,  1.663456800e+01f,  2.193600000e-02f,  1.651941600e+01f,
                2.203200000e-02f,  1.686088800e+01f,  2.208000000e-02f,  1.674496800e+01f,  2.212800000e-02f,  1.662904800e+01f,
                2.222400000e-02f,  1.697205600e+01f,  2.227200000e-02f,  1.685536800e+01f,  2.232000000e-02f,  1.673868000e+01f,
                2.241600000e-02f,  1.708322400e+01f,  2.246400000e-02f,  1.696576800e+01f,  2.251200000e-02f,  1.684831200e+01f,
                2.260800000e-02f,  1.719439200e+01f,  2.265600000e-02f,  1.707616800e+01f,  2.270400000e-02f,  1.695794400e+01f,
                2.280000000e-02f,  1.730556000e+01f,  2.284800000e-02f,  1.718656800e+01f,  2.289600000e-02f,  1.706757600e+01f,
                2.299200000e-02f,  1.741672800e+01f,  2.304000000e-02f,  1.729696800e+01f,  2.308800000e-02f,  1.717720800e+01f,
                2.318400000e-02f,  1.752789600e+01f,  2.323200000e-02f,  1.740736800e+01f,  2.328000000e-02f,  1.728684000e+01f,
                2.337600000e-02f,  1.763906400e+01f,  2.342400000e-02f,  1.751776800e+01f,  2.347200000e-02f,  1.739647200e+01f,
                1.584000000e-02f,  1.130715200e+01f,  1.587200000e-02f,  1.122603200e+01f,  1.590400000e-02f,  1.114491200e+01f,
                8.048000000e-03f,  5.426440000e+00f,  8.064000000e-03f,  5.385752000e+00f,  8.080000000e-03f,  5.345064000e+00f,
                1.149600000e-02f,  8.329452000e+00f,  1.152000000e-02f,  8.270148000e+00f,  1.154400000e-02f,  8.210844000e+00f,
                2.337600000e-02f,  1.600053600e+01f,  2.342400000e-02f,  1.588154400e+01f,  2.347200000e-02f,  1.576255200e+01f,
                3.564000000e-02f,  2.300864400e+01f,  3.571200000e-02f,  2.282958000e+01f,  3.578400000e-02f,  2.265051600e+01f,
                3.592800000e-02f,  2.315466000e+01f,  3.600000000e-02f,  2.297444400e+01f,  3.607200000e-02f,  2.279422800e+01f,
                3.621600000e-02f,  2.330067600e+01f,  3.628800000e-02f,  2.311930800e+01f,  3.636000000e-02f,  2.293794000e+01f,
                3.650400000e-02f,  2.344669200e+01f,  3.657600000e-02f,  2.326417200e+01f,  3.664800000e-02f,  2.308165200e+01f,
                3.679200000e-02f,  2.359270800e+01f,  3.686400000e-02f,  2.340903600e+01f,  3.693600000e-02f,  2.322536400e+01f,
                3.708000000e-02f,  2.373872400e+01f,  3.715200000e-02f,  2.355390000e+01f,  3.722400000e-02f,  2.336907600e+01f,
                3.736800000e-02f,  2.388474000e+01f,  3.744000000e-02f,  2.369876400e+01f,  3.751200000e-02f,  2.351278800e+01f,
                3.765600000e-02f,  2.403075600e+01f,  3.772800000e-02f,  2.384362800e+01f,  3.780000000e-02f,  2.365650000e+01f,
                3.794400000e-02f,  2.417677200e+01f,  3.801600000e-02f,  2.398849200e+01f,  3.808800000e-02f,  2.380021200e+01f,
                2.568000000e-02f,  1.540725600e+01f,  2.572800000e-02f,  1.528135200e+01f,  2.577600000e-02f,  1.515544800e+01f,
                1.303200000e-02f,  7.346796000e+00f,  1.305600000e-02f,  7.283652000e+00f,  1.308000000e-02f,  7.220508000e+00f,
                1.660800000e-02f,  1.014094400e+01f,  1.664000000e-02f,  1.005905600e+01f,  1.667200000e-02f,  9.977168000e+00f,
                3.372800000e-02f,  1.936105600e+01f,  3.379200000e-02f,  1.919676800e+01f,  3.385600000e-02f,  1.903248000e+01f,
                5.136000000e-02f,  2.765419200e+01f,  5.145600000e-02f,  2.740699200e+01f,  5.155200000e-02f,  2.715979200e+01f,
                5.174400000e-02f,  2.782123200e+01f,  5.184000000e-02f,  2.757249600e+01f,  5.193600000e-02f,  2.732376000e+01f,
                5.212800000e-02f,  2.798827200e+01f,  5.222400000e-02f,  2.773800000e+01f,  5.232000000e-02f,  2.748772800e+01f,
                5.251200000e-02f,  2.815531200e+01f,  5.260800000e-02f,  2.790350400e+01f,  5.270400000e-02f,  2.765169600e+01f,
                5.289600000e-02f,  2.832235200e+01f,  5.299200000e-02f,  2.806900800e+01f,  5.308800000e-02f,  2.781566400e+01f,
                5.328000000e-02f,  2.848939200e+01f,  5.337600000e-02f,  2.823451200e+01f,  5.347200000e-02f,  2.797963200e+01f,
                5.366400000e-02f,  2.865643200e+01f,  5.376000000e-02f,  2.840001600e+01f,  5.385600000e-02f,  2.814360000e+01f,
                5.404800000e-02f,  2.882347200e+01f,  5.414400000e-02f,  2.856552000e+01f,  5.424000000e-02f,  2.830756800e+01f,
                5.443200000e-02f,  2.899051200e+01f,  5.452800000e-02f,  2.873102400e+01f,  5.462400000e-02f,  2.847153600e+01f,
                3.680000000e-02f,  1.833654400e+01f,  3.686400000e-02f,  1.816304000e+01f,  3.692800000e-02f,  1.798953600e+01f,
                1.865600000e-02f,  8.670992000e+00f,  1.868800000e-02f,  8.583984000e+00f,  1.872000000e-02f,  8.496976000e+00f,
                2.236000000e-02f,  1.138546000e+01f,  2.240000000e-02f,  1.127958000e+01f,  2.244000000e-02f,  1.117370000e+01f,
                4.536000000e-02f,  2.156612000e+01f,  4.544000000e-02f,  2.135372000e+01f,  4.552000000e-02f,  2.114132000e+01f,
                6.900000000e-02f,  3.053430000e+01f,  6.912000000e-02f,  3.021474000e+01f,  6.924000000e-02f,  2.989518000e+01f,
                6.948000000e-02f,  3.070854000e+01f,  6.960000000e-02f,  3.038706000e+01f,  6.972000000e-02f,  3.006558000e+01f,
                6.996000000e-02f,  3.088278000e+01f,  7.008000000e-02f,  3.055938000e+01f,  7.020000000e-02f,  3.023598000e+01f,
                7.044000000e-02f,  3.105702000e+01f,  7.056000000e-02f,  3.073170000e+01f,  7.068000000e-02f,  3.040638000e+01f,
                7.092000000e-02f,  3.123126000e+01f,  7.104000000e-02f,  3.090402000e+01f,  7.116000000e-02f,  3.057678000e+01f,
                7.140000000e-02f,  3.140550000e+01f,  7.152000000e-02f,  3.107634000e+01f,  7.164000000e-02f,  3.074718000e+01f,
                7.188000000e-02f,  3.157974000e+01f,  7.200000000e-02f,  3.124866000e+01f,  7.212000000e-02f,  3.091758000e+01f,
                7.236000000e-02f,  3.175398000e+01f,  7.248000000e-02f,  3.142098000e+01f,  7.260000000e-02f,  3.108798000e+01f,
                7.284000000e-02f,  3.192822000e+01f,  7.296000000e-02f,  3.159330000e+01f,  7.308000000e-02f,  3.125838000e+01f,
                4.920000000e-02f,  1.999364000e+01f,  4.928000000e-02f,  1.976972000e+01f,  4.936000000e-02f,  1.954580000e+01f,
                2.492000000e-02f,  9.348340000e+00f,  2.496000000e-02f,  9.236060000e+00f,  2.500000000e-02f,  9.123780000e+00f,
                2.412000000e-02f,  1.210882000e+01f,  2.416000000e-02f,  1.199590000e+01f,  2.420000000e-02f,  1.188298000e+01f,
                4.888000000e-02f,  2.292836000e+01f,  4.896000000e-02f,  2.270188000e+01f,  4.904000000e-02f,  2.247540000e+01f,
                7.428000000e-02f,  3.245094000e+01f,  7.440000000e-02f,  3.211026000e+01f,  7.452000000e-02f,  3.176958000e+01f,
                7.476000000e-02f,  3.262518000e+01f,  7.488000000e-02f,  3.228258000e+01f,  7.500000000e-02f,  3.193998000e+01f,
                7.524000000e-02f,  3.279942000e+01f,  7.536000000e-02f,  3.245490000e+01f,  7.548000000e-02f,  3.211038000e+01f,
                7.572000000e-02f,  3.297366000e+01f,  7.584000000e-02f,  3.262722000e+01f,  7.596000000e-02f,  3.228078000e+01f,
                7.620000000e-02f,  3.314790000e+01f,  7.632000000e-02f,  3.279954000e+01f,  7.644000000e-02f,  3.245118000e+01f,
                7.668000000e-02f,  3.332214000e+01f,  7.680000000e-02f,  3.297186000e+01f,  7.692000000e-02f,  3.262158000e+01f,
                7.716000000e-02f,  3.349638000e+01f,  7.728000000e-02f,  3.314418000e+01f,  7.740000000e-02f,  3.279198000e+01f,
                7.764000000e-02f,  3.367062000e+01f,  7.776000000e-02f,  3.331650000e+01f,  7.788000000e-02f,  3.296238000e+01f,
                7.812000000e-02f,  3.384486000e+01f,  7.824000000e-02f,  3.348882000e+01f,  7.836000000e-02f,  3.313278000e+01f,
                5.272000000e-02f,  2.118692000e+01f,  5.280000000e-02f,  2.094892000e+01f,  5.288000000e-02f,  2.071092000e+01f,
                2.668000000e-02f,  9.902740000e+00f,  2.672000000e-02f,  9.783420000e+00f,  2.676000000e-02f,  9.664100000e+00f,
                2.588000000e-02f,  1.283218000e+01f,  2.592000000e-02f,  1.271222000e+01f,  2.596000000e-02f,  1.259226000e+01f,
                5.240000000e-02f,  2.429060000e+01f,  5.248000000e-02f,  2.405004000e+01f,  5.256000000e-02f,  2.380948000e+01f,
                7.956000000e-02f,  3.436758000e+01f,  7.968000000e-02f,  3.400578000e+01f,  7.980000000e-02f,  3.364398000e+01f,
                8.004000000e-02f,  3.454182000e+01f,  8.016000000e-02f,  3.417810000e+01f,  8.028000000e-02f,  3.381438000e+01f,
                8.052000000e-02f,  3.471606000e+01f,  8.064000000e-02f,  3.435042000e+01f,  8.076000000e-02f,  3.398478000e+01f,
                8.100000000e-02f,  3.489030000e+01f,  8.112000000e-02f,  3.452274000e+01f,  8.124000000e-02f,  3.415518000e+01f,
                8.148000000e-02f,  3.506454000e+01f,  8.160000000e-02f,  3.469506000e+01f,  8.172000000e-02f,  3.432558000e+01f,
                8.196000000e-02f,  3.523878000e+01f,  8.208000000e-02f,  3.486738000e+01f,  8.220000000e-02f,  3.449598000e+01f,
                8.244000000e-02f,  3.541302000e+01f,  8.256000000e-02f,  3.503970000e+01f,  8.268000000e-02f,  3.466638000e+01f,
                8.292000000e-02f,  3.558726000e+01f,  8.304000000e-02f,  3.521202000e+01f,  8.316000000e-02f,  3.483678000e+01f,
                8.340000000e-02f,  3.576150000e+01f,  8.352000000e-02f,  3.538434000e+01f,  8.364000000e-02f,  3.500718000e+01f,
                5.624000000e-02f,  2.238020000e+01f,  5.632000000e-02f,  2.212812000e+01f,  5.640000000e-02f,  2.187604000e+01f,
                2.844000000e-02f,  1.045714000e+01f,  2.848000000e-02f,  1.033078000e+01f,  2.852000000e-02f,  1.020442000e+01f,
                2.764000000e-02f,  1.355554000e+01f,  2.768000000e-02f,  1.342854000e+01f,  2.772000000e-02f,  1.330154000e+01f,
                5.592000000e-02f,  2.565284000e+01f,  5.600000000e-02f,  2.539820000e+01f,  5.608000000e-02f,  2.514356000e+01f,
                8.484000000e-02f,  3.628422000e+01f,  8.496000000e-02f,  3.590130000e+01f,  8.508000000e-02f,  3.551838000e+01f,
                8.532000000e-02f,  3.645846000e+01f,  8.544000000e-02f,  3.607362000e+01f,  8.556000000e-02f,  3.568878000e+01f,
                8.580000000e-02f,  3.663270000e+01f,  8.592000000e-02f,  3.624594000e+01f,  8.604000000e-02f,  3.585918000e+01f,
                8.628000000e-02f,  3.680694000e+01f,  8.640000000e-02f,  3.641826000e+01f,  8.652000000e-02f,  3.602958000e+01f,
                8.676000000e-02f,  3.698118000e+01f,  8.688000000e-02f,  3.659058000e+01f,  8.700000000e-02f,  3.619998000e+01f,
                8.724000000e-02f,  3.715542000e+01f,  8.736000000e-02f,  3.676290000e+01f,  8.748000000e-02f,  3.637038000e+01f,
                8.772000000e-02f,  3.732966000e+01f,  8.784000000e-02f,  3.693522000e+01f,  8.796000000e-02f,  3.654078000e+01f,
                8.820000000e-02f,  3.750390000e+01f,  8.832000000e-02f,  3.710754000e+01f,  8.844000000e-02f,  3.671118000e+01f,
                8.868000000e-02f,  3.767814000e+01f,  8.880000000e-02f,  3.727986000e+01f,  8.892000000e-02f,  3.688158000e+01f,
                5.976000000e-02f,  2.357348000e+01f,  5.984000000e-02f,  2.330732000e+01f,  5.992000000e-02f,  2.304116000e+01f,
                3.020000000e-02f,  1.101154000e+01f,  3.024000000e-02f,  1.087814000e+01f,  3.028000000e-02f,  1.074474000e+01f,
                2.940000000e-02f,  1.427890000e+01f,  2.944000000e-02f,  1.414486000e+01f,  2.948000000e-02f,  1.401082000e+01f,
                5.944000000e-02f,  2.701508000e+01f,  5.952000000e-02f,  2.674636000e+01f,  5.960000000e-02f,  2.647764000e+01f,
                9.012000000e-02f,  3.820086000e+01f,  9.024000000e-02f,  3.779682000e+01f,  9.036000000e-02f,  3.739278000e+01f,
                9.060000000e-02f,  3.837510000e+01f,  9.072000000e-02f,  3.796914000e+01f,  9.084000000e-02f,  3.756318000e+01f,
                9.108000000e-02f,  3.854934000e+01f,  9.120000000e-02f,  3.814146000e+01f,  9.132000000e-02f,  3.773358000e+01f,
                9.156000000e-02f,  3.872358000e+01f,  9.168000000e-02f,  3.831378000e+01f,  9.180000000e-02f,  3.790398000e+01f,
                9.204000000e-02f,  3.889782000e+01f,  9.216000000e-02f,  3.848610000e+01f,  9.228000000e-02f,  3.807438000e+01f,
                9.252000000e-02f,  3.907206000e+01f,  9.264000000e-02f,  3.865842000e+01f,  9.276000000e-02f,  3.824478000e+01f,
                9.300000000e-02f,  3.924630000e+01f,  9.312000000e-02f,  3.883074000e+01f,  9.324000000e-02f,  3.841518000e+01f,
                9.348000000e-02f,  3.942054000e+01f,  9.360000000e-02f,  3.900306000e+01f,  9.372000000e-02f,  3.858558000e+01f,
                9.396000000e-02f,  3.959478000e+01f,  9.408000000e-02f,  3.917538000e+01f,  9.420000000e-02f,  3.875598000e+01f,
                6.328000000e-02f,  2.476676000e+01f,  6.336000000e-02f,  2.448652000e+01f,  6.344000000e-02f,  2.420628000e+01f,
                3.196000000e-02f,  1.156594000e+01f,  3.200000000e-02f,  1.142550000e+01f,  3.204000000e-02f,  1.128506000e+01f,
                3.116000000e-02f,  1.500226000e+01f,  3.120000000e-02f,  1.486118000e+01f,  3.124000000e-02f,  1.472010000e+01f,
                6.296000000e-02f,  2.837732000e+01f,  6.304000000e-02f,  2.809452000e+01f,  6.312000000e-02f,  2.781172000e+01f,
                9.540000000e-02f,  4.011750000e+01f,  9.552000000e-02f,  3.969234000e+01f,  9.564000000e-02f,  3.926718000e+01f,
                9.588000000e-02f,  4.029174000e+01f,  9.600000000e-02f,  3.986466000e+01f,  9.612000000e-02f,  3.943758000e+01f,
                9.636000000e-02f,  4.046598000e+01f,  9.648000000e-02f,  4.003698000e+01f,  9.660000000e-02f,  3.960798000e+01f,
                9.684000000e-02f,  4.064022000e+01f,  9.696000000e-02f,  4.020930000e+01f,  9.708000000e-02f,  3.977838000e+01f,
                9.732000000e-02f,  4.081446000e+01f,  9.744000000e-02f,  4.038162000e+01f,  9.756000000e-02f,  3.994878000e+01f,
                9.780000000e-02f,  4.098870000e+01f,  9.792000000e-02f,  4.055394000e+01f,  9.804000000e-02f,  4.011918000e+01f,
                9.828000000e-02f,  4.116294000e+01f,  9.840000000e-02f,  4.072626000e+01f,  9.852000000e-02f,  4.028958000e+01f,
                9.876000000e-02f,  4.133718000e+01f,  9.888000000e-02f,  4.089858000e+01f,  9.900000000e-02f,  4.045998000e+01f,
                9.924000000e-02f,  4.151142000e+01f,  9.936000000e-02f,  4.107090000e+01f,  9.948000000e-02f,  4.063038000e+01f,
                6.680000000e-02f,  2.596004000e+01f,  6.688000000e-02f,  2.566572000e+01f,  6.696000000e-02f,  2.537140000e+01f,
                3.372000000e-02f,  1.212034000e+01f,  3.376000000e-02f,  1.197286000e+01f,  3.380000000e-02f,  1.182538000e+01f,
                3.292000000e-02f,  1.572562000e+01f,  3.296000000e-02f,  1.557750000e+01f,  3.300000000e-02f,  1.542938000e+01f,
                6.648000000e-02f,  2.973956000e+01f,  6.656000000e-02f,  2.944268000e+01f,  6.664000000e-02f,  2.914580000e+01f,
                1.006800000e-01f,  4.203414000e+01f,  1.008000000e-01f,  4.158786000e+01f,  1.009200000e-01f,  4.114158000e+01f,
                1.011600000e-01f,  4.220838000e+01f,  1.012800000e-01f,  4.176018000e+01f,  1.014000000e-01f,  4.131198000e+01f,
                1.016400000e-01f,  4.238262000e+01f,  1.017600000e-01f,  4.193250000e+01f,  1.018800000e-01f,  4.148238000e+01f,
                1.021200000e-01f,  4.255686000e+01f,  1.022400000e-01f,  4.210482000e+01f,  1.023600000e-01f,  4.165278000e+01f,
                1.026000000e-01f,  4.273110000e+01f,  1.027200000e-01f,  4.227714000e+01f,  1.028400000e-01f,  4.182318000e+01f,
                1.030800000e-01f,  4.290534000e+01f,  1.032000000e-01f,  4.244946000e+01f,  1.033200000e-01f,  4.199358000e+01f,
                1.035600000e-01f,  4.307958000e+01f,  1.036800000e-01f,  4.262178000e+01f,  1.038000000e-01f,  4.216398000e+01f,
                1.040400000e-01f,  4.325382000e+01f,  1.041600000e-01f,  4.279410000e+01f,  1.042800000e-01f,  4.233438000e+01f,
                1.045200000e-01f,  4.342806000e+01f,  1.046400000e-01f,  4.296642000e+01f,  1.047600000e-01f,  4.250478000e+01f,
                7.032000000e-02f,  2.715332000e+01f,  7.040000000e-02f,  2.684492000e+01f,  7.048000000e-02f,  2.653652000e+01f,
                3.548000000e-02f,  1.267474000e+01f,  3.552000000e-02f,  1.252022000e+01f,  3.556000000e-02f,  1.236570000e+01f,
                3.468000000e-02f,  1.644898000e+01f,  3.472000000e-02f,  1.629382000e+01f,  3.476000000e-02f,  1.613866000e+01f,
                7.000000000e-02f,  3.110180000e+01f,  7.008000000e-02f,  3.079084000e+01f,  7.016000000e-02f,  3.047988000e+01f,
                1.059600000e-01f,  4.395078000e+01f,  1.060800000e-01f,  4.348338000e+01f,  1.062000000e-01f,  4.301598000e+01f,
                1.064400000e-01f,  4.412502000e+01f,  1.065600000e-01f,  4.365570000e+01f,  1.066800000e-01f,  4.318638000e+01f,
                1.069200000e-01f,  4.429926000e+01f,  1.070400000e-01f,  4.382802000e+01f,  1.071600000e-01f,  4.335678000e+01f,
                1.074000000e-01f,  4.447350000e+01f,  1.075200000e-01f,  4.400034000e+01f,  1.076400000e-01f,  4.352718000e+01f,
                1.078800000e-01f,  4.464774000e+01f,  1.080000000e-01f,  4.417266000e+01f,  1.081200000e-01f,  4.369758000e+01f,
                1.083600000e-01f,  4.482198000e+01f,  1.084800000e-01f,  4.434498000e+01f,  1.086000000e-01f,  4.386798000e+01f,
                1.088400000e-01f,  4.499622000e+01f,  1.089600000e-01f,  4.451730000e+01f,  1.090800000e-01f,  4.403838000e+01f,
                1.093200000e-01f,  4.517046000e+01f,  1.094400000e-01f,  4.468962000e+01f,  1.095600000e-01f,  4.420878000e+01f,
                1.098000000e-01f,  4.534470000e+01f,  1.099200000e-01f,  4.486194000e+01f,  1.100400000e-01f,  4.437918000e+01f,
                7.384000000e-02f,  2.834660000e+01f,  7.392000000e-02f,  2.802412000e+01f,  7.400000000e-02f,  2.770164000e+01f,
                3.724000000e-02f,  1.322914000e+01f,  3.728000000e-02f,  1.306758000e+01f,  3.732000000e-02f,  1.290602000e+01f,
                3.644000000e-02f,  1.717234000e+01f,  3.648000000e-02f,  1.701014000e+01f,  3.652000000e-02f,  1.684794000e+01f,
                7.352000000e-02f,  3.246404000e+01f,  7.360000000e-02f,  3.213900000e+01f,  7.368000000e-02f,  3.181396000e+01f,
                1.112400000e-01f,  4.586742000e+01f,  1.113600000e-01f,  4.537890000e+01f,  1.114800000e-01f,  4.489038000e+01f,
                1.117200000e-01f,  4.604166000e+01f,  1.118400000e-01f,  4.555122000e+01f,  1.119600000e-01f,  4.506078000e+01f,
                1.122000000e-01f,  4.621590000e+01f,  1.123200000e-01f,  4.572354000e+01f,  1.124400000e-01f,  4.523118000e+01f,
                1.126800000e-01f,  4.639014000e+01f,  1.128000000e-01f,  4.589586000e+01f,  1.129200000e-01f,  4.540158000e+01f,
                1.131600000e-01f,  4.656438000e+01f,  1.132800000e-01f,  4.606818000e+01f,  1.134000000e-01f,  4.557198000e+01f,
                1.136400000e-01f,  4.673862000e+01f,  1.137600000e-01f,  4.624050000e+01f,  1.138800000e-01f,  4.574238000e+01f,
                1.141200000e-01f,  4.691286000e+01f,  1.142400000e-01f,  4.641282000e+01f,  1.143600000e-01f,  4.591278000e+01f,
                1.146000000e-01f,  4.708710000e+01f,  1.147200000e-01f,  4.658514000e+01f,  1.148400000e-01f,  4.608318000e+01f,
                1.150800000e-01f,  4.726134000e+01f,  1.152000000e-01f,  4.675746000e+01f,  1.153200000e-01f,  4.625358000e+01f,
                7.736000000e-02f,  2.953988000e+01f,  7.744000000e-02f,  2.920332000e+01f,  7.752000000e-02f,  2.886676000e+01f,
                3.900000000e-02f,  1.378354000e+01f,  3.904000000e-02f,  1.361494000e+01f,  3.908000000e-02f,  1.344634000e+01f,
                3.043200000e-02f,  1.148878400e+01f,  3.046400000e-02f,  1.135620800e+01f,  3.049600000e-02f,  1.122363200e+01f,
                6.137600000e-02f,  2.143004800e+01f,  6.144000000e-02f,  2.116438400e+01f,  6.150400000e-02f,  2.089872000e+01f,
                9.283200000e-02f,  2.981764800e+01f,  9.292800000e-02f,  2.941838400e+01f,  9.302400000e-02f,  2.901912000e+01f,
                9.321600000e-02f,  2.992939200e+01f,  9.331200000e-02f,  2.952859200e+01f,  9.340800000e-02f,  2.912779200e+01f,
                9.360000000e-02f,  3.004113600e+01f,  9.369600000e-02f,  2.963880000e+01f,  9.379200000e-02f,  2.923646400e+01f,
                9.398400000e-02f,  3.015288000e+01f,  9.408000000e-02f,  2.974900800e+01f,  9.417600000e-02f,  2.934513600e+01f,
                9.436800000e-02f,  3.026462400e+01f,  9.446400000e-02f,  2.985921600e+01f,  9.456000000e-02f,  2.945380800e+01f,
                9.475200000e-02f,  3.037636800e+01f,  9.484800000e-02f,  2.996942400e+01f,  9.494400000e-02f,  2.956248000e+01f,
                9.513600000e-02f,  3.048811200e+01f,  9.523200000e-02f,  3.007963200e+01f,  9.532800000e-02f,  2.967115200e+01f,
                9.552000000e-02f,  3.059985600e+01f,  9.561600000e-02f,  3.018984000e+01f,  9.571200000e-02f,  2.977982400e+01f,
                9.590400000e-02f,  3.071160000e+01f,  9.600000000e-02f,  3.030004800e+01f,  9.609600000e-02f,  2.988849600e+01f,
                6.444800000e-02f,  1.885724800e+01f,  6.451200000e-02f,  1.858236800e+01f,  6.457600000e-02f,  1.830748800e+01f,
                3.248000000e-02f,  8.618000000e+00f,  3.251200000e-02f,  8.480304000e+00f,  3.254400000e-02f,  8.342608000e+00f,
                2.378400000e-02f,  6.879084000e+00f,  2.380800000e-02f,  6.777540000e+00f,  2.383200000e-02f,  6.675996000e+00f,
                4.795200000e-02f,  1.256527200e+01f,  4.800000000e-02f,  1.236180000e+01f,  4.804800000e-02f,  1.215832800e+01f,
                7.250400000e-02f,  1.705395600e+01f,  7.257600000e-02f,  1.674817200e+01f,  7.264800000e-02f,  1.644238800e+01f,
                7.279200000e-02f,  1.711702800e+01f,  7.286400000e-02f,  1.681009200e+01f,  7.293600000e-02f,  1.650315600e+01f,
                7.308000000e-02f,  1.718010000e+01f,  7.315200000e-02f,  1.687201200e+01f,  7.322400000e-02f,  1.656392400e+01f,
                7.336800000e-02f,  1.724317200e+01f,  7.344000000e-02f,  1.693393200e+01f,  7.351200000e-02f,  1.662469200e+01f,
                7.365600000e-02f,  1.730624400e+01f,  7.372800000e-02f,  1.699585200e+01f,  7.380000000e-02f,  1.668546000e+01f,
                7.394400000e-02f,  1.736931600e+01f,  7.401600000e-02f,  1.705777200e+01f,  7.408800000e-02f,  1.674622800e+01f,
                7.423200000e-02f,  1.743238800e+01f,  7.430400000e-02f,  1.711969200e+01f,  7.437600000e-02f,  1.680699600e+01f,
                7.452000000e-02f,  1.749546000e+01f,  7.459200000e-02f,  1.718161200e+01f,  7.466400000e-02f,  1.686776400e+01f,
                7.480800000e-02f,  1.755853200e+01f,  7.488000000e-02f,  1.724353200e+01f,  7.495200000e-02f,  1.692853200e+01f,
                5.025600000e-02f,  1.046056800e+01f,  5.030400000e-02f,  1.025018400e+01f,  5.035200000e-02f,  1.003980000e+01f,
                2.532000000e-02f,  4.606188000e+00f,  2.534400000e-02f,  4.500804000e+00f,  2.536800000e-02f,  4.395420000e+00f,
                1.649600000e-02f,  3.393928000e+00f,  1.651200000e-02f,  3.324824000e+00f,  1.652800000e-02f,  3.255720000e+00f,
                3.324800000e-02f,  5.971088000e+00f,  3.328000000e-02f,  5.832624000e+00f,  3.331200000e-02f,  5.694160000e+00f,
                5.025600000e-02f,  7.728408000e+00f,  5.030400000e-02f,  7.520328000e+00f,  5.035200000e-02f,  7.312248000e+00f,
                5.044800000e-02f,  7.756632000e+00f,  5.049600000e-02f,  7.547784000e+00f,  5.054400000e-02f,  7.338936000e+00f,
                5.064000000e-02f,  7.784856000e+00f,  5.068800000e-02f,  7.575240000e+00f,  5.073600000e-02f,  7.365624000e+00f,
                5.083200000e-02f,  7.813080000e+00f,  5.088000000e-02f,  7.602696000e+00f,  5.092800000e-02f,  7.392312000e+00f,
                5.102400000e-02f,  7.841304000e+00f,  5.107200000e-02f,  7.630152000e+00f,  5.112000000e-02f,  7.419000000e+00f,
                5.121600000e-02f,  7.869528000e+00f,  5.126400000e-02f,  7.657608000e+00f,  5.131200000e-02f,  7.445688000e+00f,
                5.140800000e-02f,  7.897752000e+00f,  5.145600000e-02f,  7.685064000e+00f,  5.150400000e-02f,  7.472376000e+00f,
                5.160000000e-02f,  7.925976000e+00f,  5.164800000e-02f,  7.712520000e+00f,  5.169600000e-02f,  7.499064000e+00f,
                5.179200000e-02f,  7.954200000e+00f,  5.184000000e-02f,  7.739976000e+00f,  5.188800000e-02f,  7.525752000e+00f,
                3.478400000e-02f,  4.451216000e+00f,  3.481600000e-02f,  4.308144000e+00f,  3.484800000e-02f,  4.165072000e+00f,
                1.752000000e-02f,  1.798792000e+00f,  1.753600000e-02f,  1.727128000e+00f,  1.755200000e-02f,  1.655464000e+00f,
                8.568000000e-03f,  1.084004000e+00f,  8.576000000e-03f,  1.048748000e+00f,  8.584000000e-03f,  1.013492000e+00f,
                1.726400000e-02f,  1.748872000e+00f,  1.728000000e-02f,  1.678232000e+00f,  1.729600000e-02f,  1.607592000e+00f,
                2.608800000e-02f,  1.993068000e+00f,  2.611200000e-02f,  1.886916000e+00f,  2.613600000e-02f,  1.780764000e+00f,
                2.618400000e-02f,  2.000268000e+00f,  2.620800000e-02f,  1.893732000e+00f,  2.623200000e-02f,  1.787196000e+00f,
                2.628000000e-02f,  2.007468000e+00f,  2.630400000e-02f,  1.900548000e+00f,  2.632800000e-02f,  1.793628000e+00f,
                2.637600000e-02f,  2.014668000e+00f,  2.640000000e-02f,  1.907364000e+00f,  2.642400000e-02f,  1.800060000e+00f,
                2.647200000e-02f,  2.021868000e+00f,  2.649600000e-02f,  1.914180000e+00f,  2.652000000e-02f,  1.806492000e+00f,
                2.656800000e-02f,  2.029068000e+00f,  2.659200000e-02f,  1.920996000e+00f,  2.661600000e-02f,  1.812924000e+00f,
                2.666400000e-02f,  2.036268000e+00f,  2.668800000e-02f,  1.927812000e+00f,  2.671200000e-02f,  1.819356000e+00f,
                2.676000000e-02f,  2.043468000e+00f,  2.678400000e-02f,  1.934628000e+00f,  2.680800000e-02f,  1.825788000e+00f,
                2.685600000e-02f,  2.050668000e+00f,  2.688000000e-02f,  1.941444000e+00f,  2.690400000e-02f,  1.832220000e+00f,
                1.803200000e-02f,  9.305680000e-01f,  1.804800000e-02f,  8.576240000e-01f,  1.806400000e-02f,  7.846800000e-01f,
                9.080000000e-03f,  2.465000000e-01f,  9.088000000e-03f,  2.099640000e-01f,  9.096000000e-03f,  1.734280000e-01f,
                7.768000000e-03f,  6.406916000e+00f,  7.776000000e-03f,  6.370252000e+00f,  7.784000000e-03f,  6.333588000e+00f,
                1.566400000e-02f,  1.239623200e+01f,  1.568000000e-02f,  1.232277600e+01f,  1.569600000e-02f,  1.224932000e+01f,
                2.368800000e-02f,  1.796641200e+01f,  2.371200000e-02f,  1.785603600e+01f,  2.373600000e-02f,  1.774566000e+01f,
                2.378400000e-02f,  1.802890800e+01f,  2.380800000e-02f,  1.791814800e+01f,  2.383200000e-02f,  1.780738800e+01f,
                2.388000000e-02f,  1.809140400e+01f,  2.390400000e-02f,  1.798026000e+01f,  2.392800000e-02f,  1.786911600e+01f,
                2.397600000e-02f,  1.815390000e+01f,  2.400000000e-02f,  1.804237200e+01f,  2.402400000e-02f,  1.793084400e+01f,
                2.407200000e-02f,  1.821639600e+01f,  2.409600000e-02f,  1.810448400e+01f,  2.412000000e-02f,  1.799257200e+01f,
                2.416800000e-02f,  1.827889200e+01f,  2.419200000e-02f,  1.816659600e+01f,  2.421600000e-02f,  1.805430000e+01f,
                2.426400000e-02f,  1.834138800e+01f,  2.428800000e-02f,  1.822870800e+01f,  2.431200000e-02f,  1.811602800e+01f,
                2.436000000e-02f,  1.840388400e+01f,  2.438400000e-02f,  1.829082000e+01f,  2.440800000e-02f,  1.817775600e+01f,
                2.445600000e-02f,  1.846638000e+01f,  2.448000000e-02f,  1.835293200e+01f,  2.450400000e-02f,  1.823948400e+01f,
                1.643200000e-02f,  1.187591200e+01f,  1.644800000e-02f,  1.180015200e+01f,  1.646400000e-02f,  1.172439200e+01f,
                8.280000000e-03f,  5.719940000e+00f,  8.288000000e-03f,  5.681996000e+00f,  8.296000000e-03f,  5.644052000e+00f,
                1.617600000e-02f,  1.173997600e+01f,  1.619200000e-02f,  1.166524000e+01f,  1.620800000e-02f,  1.159050400e+01f,
                3.260800000e-02f,  2.262324800e+01f,  3.264000000e-02f,  2.247352000e+01f,  3.267200000e-02f,  2.232379200e+01f,
                4.929600000e-02f,  3.264674400e+01f,  4.934400000e-02f,  3.242176800e+01f,  4.939200000e-02f,  3.219679200e+01f,
                4.948800000e-02f,  3.275791200e+01f,  4.953600000e-02f,  3.253216800e+01f,  4.958400000e-02f,  3.230642400e+01f,
                4.968000000e-02f,  3.286908000e+01f,  4.972800000e-02f,  3.264256800e+01f,  4.977600000e-02f,  3.241605600e+01f,
                4.987200000e-02f,  3.298024800e+01f,  4.992000000e-02f,  3.275296800e+01f,  4.996800000e-02f,  3.252568800e+01f,
                5.006400000e-02f,  3.309141600e+01f,  5.011200000e-02f,  3.286336800e+01f,  5.016000000e-02f,  3.263532000e+01f,
                5.025600000e-02f,  3.320258400e+01f,  5.030400000e-02f,  3.297376800e+01f,  5.035200000e-02f,  3.274495200e+01f,
                5.044800000e-02f,  3.331375200e+01f,  5.049600000e-02f,  3.308416800e+01f,  5.054400000e-02f,  3.285458400e+01f,
                5.064000000e-02f,  3.342492000e+01f,  5.068800000e-02f,  3.319456800e+01f,  5.073600000e-02f,  3.296421600e+01f,
                5.083200000e-02f,  3.353608800e+01f,  5.088000000e-02f,  3.330496800e+01f,  5.092800000e-02f,  3.307384800e+01f,
                3.414400000e-02f,  2.146587200e+01f,  3.417600000e-02f,  2.131153600e+01f,  3.420800000e-02f,  2.115720000e+01f,
                1.720000000e-02f,  1.028615200e+01f,  1.721600000e-02f,  1.020885600e+01f,  1.723200000e-02f,  1.013156000e+01f,
                2.522400000e-02f,  1.594849200e+01f,  2.524800000e-02f,  1.583427600e+01f,  2.527200000e-02f,  1.572006000e+01f,
                5.083200000e-02f,  3.057967200e+01f,  5.088000000e-02f,  3.035085600e+01f,  5.092800000e-02f,  3.012204000e+01f,
                7.682400000e-02f,  4.388893200e+01f,  7.689600000e-02f,  4.354513200e+01f,  7.696800000e-02f,  4.320133200e+01f,
                7.711200000e-02f,  4.403494800e+01f,  7.718400000e-02f,  4.368999600e+01f,  7.725600000e-02f,  4.334504400e+01f,
                7.740000000e-02f,  4.418096400e+01f,  7.747200000e-02f,  4.383486000e+01f,  7.754400000e-02f,  4.348875600e+01f,
                7.768800000e-02f,  4.432698000e+01f,  7.776000000e-02f,  4.397972400e+01f,  7.783200000e-02f,  4.363246800e+01f,
                7.797600000e-02f,  4.447299600e+01f,  7.804800000e-02f,  4.412458800e+01f,  7.812000000e-02f,  4.377618000e+01f,
                7.826400000e-02f,  4.461901200e+01f,  7.833600000e-02f,  4.426945200e+01f,  7.840800000e-02f,  4.391989200e+01f,
                7.855200000e-02f,  4.476502800e+01f,  7.862400000e-02f,  4.441431600e+01f,  7.869600000e-02f,  4.406360400e+01f,
                7.884000000e-02f,  4.491104400e+01f,  7.891200000e-02f,  4.455918000e+01f,  7.898400000e-02f,  4.420731600e+01f,
                7.912800000e-02f,  4.505706000e+01f,  7.920000000e-02f,  4.470404400e+01f,  7.927200000e-02f,  4.435102800e+01f,
                5.313600000e-02f,  2.866850400e+01f,  5.318400000e-02f,  2.843277600e+01f,  5.323200000e-02f,  2.819704800e+01f,
                2.676000000e-02f,  1.364794800e+01f,  2.678400000e-02f,  1.352989200e+01f,  2.680800000e-02f,  1.341183600e+01f,
                3.491200000e-02f,  1.898177600e+01f,  3.494400000e-02f,  1.882667200e+01f,  3.497600000e-02f,  1.867156800e+01f,
                7.033600000e-02f,  3.616412800e+01f,  7.040000000e-02f,  3.585340800e+01f,  7.046400000e-02f,  3.554268800e+01f,
                1.062720000e-01f,  5.154091200e+01f,  1.063680000e-01f,  5.107406400e+01f,  1.064640000e-01f,  5.060721600e+01f,
                1.066560000e-01f,  5.170795200e+01f,  1.067520000e-01f,  5.123956800e+01f,  1.068480000e-01f,  5.077118400e+01f,
                1.070400000e-01f,  5.187499200e+01f,  1.071360000e-01f,  5.140507200e+01f,  1.072320000e-01f,  5.093515200e+01f,
                1.074240000e-01f,  5.204203200e+01f,  1.075200000e-01f,  5.157057600e+01f,  1.076160000e-01f,  5.109912000e+01f,
                1.078080000e-01f,  5.220907200e+01f,  1.079040000e-01f,  5.173608000e+01f,  1.080000000e-01f,  5.126308800e+01f,
                1.081920000e-01f,  5.237611200e+01f,  1.082880000e-01f,  5.190158400e+01f,  1.083840000e-01f,  5.142705600e+01f,
                1.085760000e-01f,  5.254315200e+01f,  1.086720000e-01f,  5.206708800e+01f,  1.087680000e-01f,  5.159102400e+01f,
                1.089600000e-01f,  5.271019200e+01f,  1.090560000e-01f,  5.223259200e+01f,  1.091520000e-01f,  5.175499200e+01f,
                1.093440000e-01f,  5.287723200e+01f,  1.094400000e-01f,  5.239809600e+01f,  1.095360000e-01f,  5.191896000e+01f,
                7.340800000e-02f,  3.338243200e+01f,  7.347200000e-02f,  3.306249600e+01f,  7.353600000e-02f,  3.274256000e+01f,
                3.696000000e-02f,  1.575464000e+01f,  3.699200000e-02f,  1.559441600e+01f,  3.702400000e-02f,  1.543419200e+01f,
                4.524000000e-02f,  2.078914000e+01f,  4.528000000e-02f,  2.059174000e+01f,  4.532000000e-02f,  2.039434000e+01f,
                9.112000000e-02f,  3.927524000e+01f,  9.120000000e-02f,  3.887980000e+01f,  9.128000000e-02f,  3.848436000e+01f,
                1.376400000e-01f,  5.545062000e+01f,  1.377600000e-01f,  5.485650000e+01f,  1.378800000e-01f,  5.426238000e+01f,
                1.381200000e-01f,  5.562486000e+01f,  1.382400000e-01f,  5.502882000e+01f,  1.383600000e-01f,  5.443278000e+01f,
                1.386000000e-01f,  5.579910000e+01f,  1.387200000e-01f,  5.520114000e+01f,  1.388400000e-01f,  5.460318000e+01f,
                1.390800000e-01f,  5.597334000e+01f,  1.392000000e-01f,  5.537346000e+01f,  1.393200000e-01f,  5.477358000e+01f,
                1.395600000e-01f,  5.614758000e+01f,  1.396800000e-01f,  5.554578000e+01f,  1.398000000e-01f,  5.494398000e+01f,
                1.400400000e-01f,  5.632182000e+01f,  1.401600000e-01f,  5.571810000e+01f,  1.402800000e-01f,  5.511438000e+01f,
                1.405200000e-01f,  5.649606000e+01f,  1.406400000e-01f,  5.589042000e+01f,  1.407600000e-01f,  5.528478000e+01f,
                1.410000000e-01f,  5.667030000e+01f,  1.411200000e-01f,  5.606274000e+01f,  1.412400000e-01f,  5.545518000e+01f,
                1.414800000e-01f,  5.684454000e+01f,  1.416000000e-01f,  5.623506000e+01f,  1.417200000e-01f,  5.562558000e+01f,
                9.496000000e-02f,  3.550628000e+01f,  9.504000000e-02f,  3.509932000e+01f,  9.512000000e-02f,  3.469236000e+01f,
                4.780000000e-02f,  1.655554000e+01f,  4.784000000e-02f,  1.635174000e+01f,  4.788000000e-02f,  1.614794000e+01f,
                4.700000000e-02f,  2.151250000e+01f,  4.704000000e-02f,  2.130806000e+01f,  4.708000000e-02f,  2.110362000e+01f,
                9.464000000e-02f,  4.063748000e+01f,  9.472000000e-02f,  4.022796000e+01f,  9.480000000e-02f,  3.981844000e+01f,
                1.429200000e-01f,  5.736726000e+01f,  1.430400000e-01f,  5.675202000e+01f,  1.431600000e-01f,  5.613678000e+01f,
                1.434000000e-01f,  5.754150000e+01f,  1.435200000e-01f,  5.692434000e+01f,  1.436400000e-01f,  5.630718000e+01f,
                1.438800000e-01f,  5.771574000e+01f,  1.440000000e-01f,  5.709666000e+01f,  1.441200000e-01f,  5.647758000e+01f,
                1.443600000e-01f,  5.788998000e+01f,  1.444800000e-01f,  5.726898000e+01f,  1.446000000e-01f,  5.664798000e+01f,
                1.448400000e-01f,  5.806422000e+01f,  1.449600000e-01f,  5.744130000e+01f,  1.450800000e-01f,  5.681838000e+01f,
                1.453200000e-01f,  5.823846000e+01f,  1.454400000e-01f,  5.761362000e+01f,  1.455600000e-01f,  5.698878000e+01f,
                1.458000000e-01f,  5.841270000e+01f,  1.459200000e-01f,  5.778594000e+01f,  1.460400000e-01f,  5.715918000e+01f,
                1.462800000e-01f,  5.858694000e+01f,  1.464000000e-01f,  5.795826000e+01f,  1.465200000e-01f,  5.732958000e+01f,
                1.467600000e-01f,  5.876118000e+01f,  1.468800000e-01f,  5.813058000e+01f,  1.470000000e-01f,  5.749998000e+01f,
                9.848000000e-02f,  3.669956000e+01f,  9.856000000e-02f,  3.627852000e+01f,  9.864000000e-02f,  3.585748000e+01f,
                4.956000000e-02f,  1.710994000e+01f,  4.960000000e-02f,  1.689910000e+01f,  4.964000000e-02f,  1.668826000e+01f,
                4.876000000e-02f,  2.223586000e+01f,  4.880000000e-02f,  2.202438000e+01f,  4.884000000e-02f,  2.181290000e+01f,
                9.816000000e-02f,  4.199972000e+01f,  9.824000000e-02f,  4.157612000e+01f,  9.832000000e-02f,  4.115252000e+01f,
                1.482000000e-01f,  5.928390000e+01f,  1.483200000e-01f,  5.864754000e+01f,  1.484400000e-01f,  5.801118000e+01f,
                1.486800000e-01f,  5.945814000e+01f,  1.488000000e-01f,  5.881986000e+01f,  1.489200000e-01f,  5.818158000e+01f,
                1.491600000e-01f,  5.963238000e+01f,  1.492800000e-01f,  5.899218000e+01f,  1.494000000e-01f,  5.835198000e+01f,
                1.496400000e-01f,  5.980662000e+01f,  1.497600000e-01f,  5.916450000e+01f,  1.498800000e-01f,  5.852238000e+01f,
                1.501200000e-01f,  5.998086000e+01f,  1.502400000e-01f,  5.933682000e+01f,  1.503600000e-01f,  5.869278000e+01f,
                1.506000000e-01f,  6.015510000e+01f,  1.507200000e-01f,  5.950914000e+01f,  1.508400000e-01f,  5.886318000e+01f,
                1.510800000e-01f,  6.032934000e+01f,  1.512000000e-01f,  5.968146000e+01f,  1.513200000e-01f,  5.903358000e+01f,
                1.515600000e-01f,  6.050358000e+01f,  1.516800000e-01f,  5.985378000e+01f,  1.518000000e-01f,  5.920398000e+01f,
                1.520400000e-01f,  6.067782000e+01f,  1.521600000e-01f,  6.002610000e+01f,  1.522800000e-01f,  5.937438000e+01f,
                1.020000000e-01f,  3.789284000e+01f,  1.020800000e-01f,  3.745772000e+01f,  1.021600000e-01f,  3.702260000e+01f,
                5.132000000e-02f,  1.766434000e+01f,  5.136000000e-02f,  1.744646000e+01f,  5.140000000e-02f,  1.722858000e+01f,
                5.052000000e-02f,  2.295922000e+01f,  5.056000000e-02f,  2.274070000e+01f,  5.060000000e-02f,  2.252218000e+01f,
                1.016800000e-01f,  4.336196000e+01f,  1.017600000e-01f,  4.292428000e+01f,  1.018400000e-01f,  4.248660000e+01f,
                1.534800000e-01f,  6.120054000e+01f,  1.536000000e-01f,  6.054306000e+01f,  1.537200000e-01f,  5.988558000e+01f,
                1.539600000e-01f,  6.137478000e+01f,  1.540800000e-01f,  6.071538000e+01f,  1.542000000e-01f,  6.005598000e+01f,
                1.544400000e-01f,  6.154902000e+01f,  1.545600000e-01f,  6.088770000e+01f,  1.546800000e-01f,  6.022638000e+01f,
                1.549200000e-01f,  6.172326000e+01f,  1.550400000e-01f,  6.106002000e+01f,  1.551600000e-01f,  6.039678000e+01f,
                1.554000000e-01f,  6.189750000e+01f,  1.555200000e-01f,  6.123234000e+01f,  1.556400000e-01f,  6.056718000e+01f,
                1.558800000e-01f,  6.207174000e+01f,  1.560000000e-01f,  6.140466000e+01f,  1.561200000e-01f,  6.073758000e+01f,
                1.563600000e-01f,  6.224598000e+01f,  1.564800000e-01f,  6.157698000e+01f,  1.566000000e-01f,  6.090798000e+01f,
                1.568400000e-01f,  6.242022000e+01f,  1.569600000e-01f,  6.174930000e+01f,  1.570800000e-01f,  6.107838000e+01f,
                1.573200000e-01f,  6.259446000e+01f,  1.574400000e-01f,  6.192162000e+01f,  1.575600000e-01f,  6.124878000e+01f,
                1.055200000e-01f,  3.908612000e+01f,  1.056000000e-01f,  3.863692000e+01f,  1.056800000e-01f,  3.818772000e+01f,
                5.308000000e-02f,  1.821874000e+01f,  5.312000000e-02f,  1.799382000e+01f,  5.316000000e-02f,  1.776890000e+01f,
                5.228000000e-02f,  2.368258000e+01f,  5.232000000e-02f,  2.345702000e+01f,  5.236000000e-02f,  2.323146000e+01f,
                1.052000000e-01f,  4.472420000e+01f,  1.052800000e-01f,  4.427244000e+01f,  1.053600000e-01f,  4.382068000e+01f,
                1.587600000e-01f,  6.311718000e+01f,  1.588800000e-01f,  6.243858000e+01f,  1.590000000e-01f,  6.175998000e+01f,
                1.592400000e-01f,  6.329142000e+01f,  1.593600000e-01f,  6.261090000e+01f,  1.594800000e-01f,  6.193038000e+01f,
                1.597200000e-01f,  6.346566000e+01f,  1.598400000e-01f,  6.278322000e+01f,  1.599600000e-01f,  6.210078000e+01f,
                1.602000000e-01f,  6.363990000e+01f,  1.603200000e-01f,  6.295554000e+01f,  1.604400000e-01f,  6.227118000e+01f,
                1.606800000e-01f,  6.381414000e+01f,  1.608000000e-01f,  6.312786000e+01f,  1.609200000e-01f,  6.244158000e+01f,
                1.611600000e-01f,  6.398838000e+01f,  1.612800000e-01f,  6.330018000e+01f,  1.614000000e-01f,  6.261198000e+01f,
                1.616400000e-01f,  6.416262000e+01f,  1.617600000e-01f,  6.347250000e+01f,  1.618800000e-01f,  6.278238000e+01f,
                1.621200000e-01f,  6.433686000e+01f,  1.622400000e-01f,  6.364482000e+01f,  1.623600000e-01f,  6.295278000e+01f,
                1.626000000e-01f,  6.451110000e+01f,  1.627200000e-01f,  6.381714000e+01f,  1.628400000e-01f,  6.312318000e+01f,
                1.090400000e-01f,  4.027940000e+01f,  1.091200000e-01f,  3.981612000e+01f,  1.092000000e-01f,  3.935284000e+01f,
                5.484000000e-02f,  1.877314000e+01f,  5.488000000e-02f,  1.854118000e+01f,  5.492000000e-02f,  1.830922000e+01f,
                5.404000000e-02f,  2.440594000e+01f,  5.408000000e-02f,  2.417334000e+01f,  5.412000000e-02f,  2.394074000e+01f,
                1.087200000e-01f,  4.608644000e+01f,  1.088000000e-01f,  4.562060000e+01f,  1.088800000e-01f,  4.515476000e+01f,
                1.640400000e-01f,  6.503382000e+01f,  1.641600000e-01f,  6.433410000e+01f,  1.642800000e-01f,  6.363438000e+01f,
                1.645200000e-01f,  6.520806000e+01f,  1.646400000e-01f,  6.450642000e+01f,  1.647600000e-01f,  6.380478000e+01f,
                1.650000000e-01f,  6.538230000e+01f,  1.651200000e-01f,  6.467874000e+01f,  1.652400000e-01f,  6.397518000e+01f,
                1.654800000e-01f,  6.555654000e+01f,  1.656000000e-01f,  6.485106000e+01f,  1.657200000e-01f,  6.414558000e+01f,
                1.659600000e-01f,  6.573078000e+01f,  1.660800000e-01f,  6.502338000e+01f,  1.662000000e-01f,  6.431598000e+01f,
                1.664400000e-01f,  6.590502000e+01f,  1.665600000e-01f,  6.519570000e+01f,  1.666800000e-01f,  6.448638000e+01f,
                1.669200000e-01f,  6.607926000e+01f,  1.670400000e-01f,  6.536802000e+01f,  1.671600000e-01f,  6.465678000e+01f,
                1.674000000e-01f,  6.625350000e+01f,  1.675200000e-01f,  6.554034000e+01f,  1.676400000e-01f,  6.482718000e+01f,
                1.678800000e-01f,  6.642774000e+01f,  1.680000000e-01f,  6.571266000e+01f,  1.681200000e-01f,  6.499758000e+01f,
                1.125600000e-01f,  4.147268000e+01f,  1.126400000e-01f,  4.099532000e+01f,  1.127200000e-01f,  4.051796000e+01f,
                5.660000000e-02f,  1.932754000e+01f,  5.664000000e-02f,  1.908854000e+01f,  5.668000000e-02f,  1.884954000e+01f,
                5.580000000e-02f,  2.512930000e+01f,  5.584000000e-02f,  2.488966000e+01f,  5.588000000e-02f,  2.465002000e+01f,
                1.122400000e-01f,  4.744868000e+01f,  1.123200000e-01f,  4.696876000e+01f,  1.124000000e-01f,  4.648884000e+01f,
                1.693200000e-01f,  6.695046000e+01f,  1.694400000e-01f,  6.622962000e+01f,  1.695600000e-01f,  6.550878000e+01f,
                1.698000000e-01f,  6.712470000e+01f,  1.699200000e-01f,  6.640194000e+01f,  1.700400000e-01f,  6.567918000e+01f,
                1.702800000e-01f,  6.729894000e+01f,  1.704000000e-01f,  6.657426000e+01f,  1.705200000e-01f,  6.584958000e+01f,
                1.707600000e-01f,  6.747318000e+01f,  1.708800000e-01f,  6.674658000e+01f,  1.710000000e-01f,  6.601998000e+01f,
                1.712400000e-01f,  6.764742000e+01f,  1.713600000e-01f,  6.691890000e+01f,  1.714800000e-01f,  6.619038000e+01f,
                1.717200000e-01f,  6.782166000e+01f,  1.718400000e-01f,  6.709122000e+01f,  1.719600000e-01f,  6.636078000e+01f,
                1.722000000e-01f,  6.799590000e+01f,  1.723200000e-01f,  6.726354000e+01f,  1.724400000e-01f,  6.653118000e+01f,
                1.726800000e-01f,  6.817014000e+01f,  1.728000000e-01f,  6.743586000e+01f,  1.729200000e-01f,  6.670158000e+01f,
                1.731600000e-01f,  6.834438000e+01f,  1.732800000e-01f,  6.760818000e+01f,  1.734000000e-01f,  6.687198000e+01f,
                1.160800000e-01f,  4.266596000e+01f,  1.161600000e-01f,  4.217452000e+01f,  1.162400000e-01f,  4.168308000e+01f,
                5.836000000e-02f,  1.988194000e+01f,  5.840000000e-02f,  1.963590000e+01f,  5.844000000e-02f,  1.938986000e+01f,
                5.756000000e-02f,  2.585266000e+01f,  5.760000000e-02f,  2.560598000e+01f,  5.764000000e-02f,  2.535930000e+01f,
                1.157600000e-01f,  4.881092000e+01f,  1.158400000e-01f,  4.831692000e+01f,  1.159200000e-01f,  4.782292000e+01f,
                1.746000000e-01f,  6.886710000e+01f,  1.747200000e-01f,  6.812514000e+01f,  1.748400000e-01f,  6.738318000e+01f,
                1.750800000e-01f,  6.904134000e+01f,  1.752000000e-01f,  6.829746000e+01f,  1.753200000e-01f,  6.755358000e+01f,
                1.755600000e-01f,  6.921558000e+01f,  1.756800000e-01f,  6.846978000e+01f,  1.758000000e-01f,  6.772398000e+01f,
                1.760400000e-01f,  6.938982000e+01f,  1.761600000e-01f,  6.864210000e+01f,  1.762800000e-01f,  6.789438000e+01f,
                1.765200000e-01f,  6.956406000e+01f,  1.766400000e-01f,  6.881442000e+01f,  1.767600000e-01f,  6.806478000e+01f,
                1.770000000e-01f,  6.973830000e+01f,  1.771200000e-01f,  6.898674000e+01f,  1.772400000e-01f,  6.823518000e+01f,
                1.774800000e-01f,  6.991254000e+01f,  1.776000000e-01f,  6.915906000e+01f,  1.777200000e-01f,  6.840558000e+01f,
                1.779600000e-01f,  7.008678000e+01f,  1.780800000e-01f,  6.933138000e+01f,  1.782000000e-01f,  6.857598000e+01f,
                1.784400000e-01f,  7.026102000e+01f,  1.785600000e-01f,  6.950370000e+01f,  1.786800000e-01f,  6.874638000e+01f,
                1.196000000e-01f,  4.385924000e+01f,  1.196800000e-01f,  4.335372000e+01f,  1.197600000e-01f,  4.284820000e+01f,
                6.012000000e-02f,  2.043634000e+01f,  6.016000000e-02f,  2.018326000e+01f,  6.020000000e-02f,  1.993018000e+01f,
                5.932000000e-02f,  2.657602000e+01f,  5.936000000e-02f,  2.632230000e+01f,  5.940000000e-02f,  2.606858000e+01f,
                1.192800000e-01f,  5.017316000e+01f,  1.193600000e-01f,  4.966508000e+01f,  1.194400000e-01f,  4.915700000e+01f,
                1.798800000e-01f,  7.078374000e+01f,  1.800000000e-01f,  7.002066000e+01f,  1.801200000e-01f,  6.925758000e+01f,
                1.803600000e-01f,  7.095798000e+01f,  1.804800000e-01f,  7.019298000e+01f,  1.806000000e-01f,  6.942798000e+01f,
                1.808400000e-01f,  7.113222000e+01f,  1.809600000e-01f,  7.036530000e+01f,  1.810800000e-01f,  6.959838000e+01f,
                1.813200000e-01f,  7.130646000e+01f,  1.814400000e-01f,  7.053762000e+01f,  1.815600000e-01f,  6.976878000e+01f,
                1.818000000e-01f,  7.148070000e+01f,  1.819200000e-01f,  7.070994000e+01f,  1.820400000e-01f,  6.993918000e+01f,
                1.822800000e-01f,  7.165494000e+01f,  1.824000000e-01f,  7.088226000e+01f,  1.825200000e-01f,  7.010958000e+01f,
                1.827600000e-01f,  7.182918000e+01f,  1.828800000e-01f,  7.105458000e+01f,  1.830000000e-01f,  7.027998000e+01f,
                1.832400000e-01f,  7.200342000e+01f,  1.833600000e-01f,  7.122690000e+01f,  1.834800000e-01f,  7.045038000e+01f,
                1.837200000e-01f,  7.217766000e+01f,  1.838400000e-01f,  7.139922000e+01f,  1.839600000e-01f,  7.062078000e+01f,
                1.231200000e-01f,  4.505252000e+01f,  1.232000000e-01f,  4.453292000e+01f,  1.232800000e-01f,  4.401332000e+01f,
                6.188000000e-02f,  2.099074000e+01f,  6.192000000e-02f,  2.073062000e+01f,  6.196000000e-02f,  2.047050000e+01f,
                4.873600000e-02f,  1.769384000e+01f,  4.876800000e-02f,  1.748804800e+01f,  4.880000000e-02f,  1.728225600e+01f,
                9.798400000e-02f,  3.296156800e+01f,  9.804800000e-02f,  3.254947200e+01f,  9.811200000e-02f,  3.213737600e+01f,
                1.477440000e-01f,  4.579704000e+01f,  1.478400000e-01f,  4.517812800e+01f,  1.479360000e-01f,  4.455921600e+01f,
                1.481280000e-01f,  4.590878400e+01f,  1.482240000e-01f,  4.528833600e+01f,  1.483200000e-01f,  4.466788800e+01f,
                1.485120000e-01f,  4.602052800e+01f,  1.486080000e-01f,  4.539854400e+01f,  1.487040000e-01f,  4.477656000e+01f,
                1.488960000e-01f,  4.613227200e+01f,  1.489920000e-01f,  4.550875200e+01f,  1.490880000e-01f,  4.488523200e+01f,
                1.492800000e-01f,  4.624401600e+01f,  1.493760000e-01f,  4.561896000e+01f,  1.494720000e-01f,  4.499390400e+01f,
                1.496640000e-01f,  4.635576000e+01f,  1.497600000e-01f,  4.572916800e+01f,  1.498560000e-01f,  4.510257600e+01f,
                1.500480000e-01f,  4.646750400e+01f,  1.501440000e-01f,  4.583937600e+01f,  1.502400000e-01f,  4.521124800e+01f,
                1.504320000e-01f,  4.657924800e+01f,  1.505280000e-01f,  4.594958400e+01f,  1.506240000e-01f,  4.531992000e+01f,
                1.508160000e-01f,  4.669099200e+01f,  1.509120000e-01f,  4.605979200e+01f,  1.510080000e-01f,  4.542859200e+01f,
                1.010560000e-01f,  2.863158400e+01f,  1.011200000e-01f,  2.821027200e+01f,  1.011840000e-01f,  2.778896000e+01f,
                5.078400000e-02f,  1.306587200e+01f,  5.081600000e-02f,  1.285496000e+01f,  5.084800000e-02f,  1.264404800e+01f,
                3.751200000e-02f,  1.054446000e+01f,  3.753600000e-02f,  1.038800400e+01f,  3.756000000e-02f,  1.023154800e+01f,
                7.540800000e-02f,  1.923708000e+01f,  7.545600000e-02f,  1.892378400e+01f,  7.550400000e-02f,  1.861048800e+01f,
                1.136880000e-01f,  2.607325200e+01f,  1.137600000e-01f,  2.560273200e+01f,  1.138320000e-01f,  2.513221200e+01f,
                1.139760000e-01f,  2.613632400e+01f,  1.140480000e-01f,  2.566465200e+01f,  1.141200000e-01f,  2.519298000e+01f,
                1.142640000e-01f,  2.619939600e+01f,  1.143360000e-01f,  2.572657200e+01f,  1.144080000e-01f,  2.525374800e+01f,
                1.145520000e-01f,  2.626246800e+01f,  1.146240000e-01f,  2.578849200e+01f,  1.146960000e-01f,  2.531451600e+01f,
                1.148400000e-01f,  2.632554000e+01f,  1.149120000e-01f,  2.585041200e+01f,  1.149840000e-01f,  2.537528400e+01f,
                1.151280000e-01f,  2.638861200e+01f,  1.152000000e-01f,  2.591233200e+01f,  1.152720000e-01f,  2.543605200e+01f,
                1.154160000e-01f,  2.645168400e+01f,  1.154880000e-01f,  2.597425200e+01f,  1.155600000e-01f,  2.549682000e+01f,
                1.157040000e-01f,  2.651475600e+01f,  1.157760000e-01f,  2.603617200e+01f,  1.158480000e-01f,  2.555758800e+01f,
                1.159920000e-01f,  2.657782800e+01f,  1.160640000e-01f,  2.609809200e+01f,  1.161360000e-01f,  2.561835600e+01f,
                7.771200000e-02f,  1.581448800e+01f,  7.776000000e-02f,  1.549428000e+01f,  7.780800000e-02f,  1.517407200e+01f,
                3.904800000e-02f,  6.953676000e+00f,  3.907200000e-02f,  6.793380000e+00f,  3.909600000e-02f,  6.633084000e+00f,
                2.564800000e-02f,  5.178568000e+00f,  2.566400000e-02f,  5.072856000e+00f,  2.568000000e-02f,  4.967144000e+00f,
                5.155200000e-02f,  9.101072000e+00f,  5.158400000e-02f,  8.889392000e+00f,  5.161600000e-02f,  8.677712000e+00f,
                7.771200000e-02f,  1.176444000e+01f,  7.776000000e-02f,  1.144653600e+01f,  7.780800000e-02f,  1.112863200e+01f,
                7.790400000e-02f,  1.179266400e+01f,  7.795200000e-02f,  1.147399200e+01f,  7.800000000e-02f,  1.115532000e+01f,
                7.809600000e-02f,  1.182088800e+01f,  7.814400000e-02f,  1.150144800e+01f,  7.819200000e-02f,  1.118200800e+01f,
                7.828800000e-02f,  1.184911200e+01f,  7.833600000e-02f,  1.152890400e+01f,  7.838400000e-02f,  1.120869600e+01f,
                7.848000000e-02f,  1.187733600e+01f,  7.852800000e-02f,  1.155636000e+01f,  7.857600000e-02f,  1.123538400e+01f,
                7.867200000e-02f,  1.190556000e+01f,  7.872000000e-02f,  1.158381600e+01f,  7.876800000e-02f,  1.126207200e+01f,
                7.886400000e-02f,  1.193378400e+01f,  7.891200000e-02f,  1.161127200e+01f,  7.896000000e-02f,  1.128876000e+01f,
                7.905600000e-02f,  1.196200800e+01f,  7.910400000e-02f,  1.163872800e+01f,  7.915200000e-02f,  1.131544800e+01f,
                7.924800000e-02f,  1.199023200e+01f,  7.929600000e-02f,  1.166618400e+01f,  7.934400000e-02f,  1.134213600e+01f,
                5.308800000e-02f,  6.702608000e+00f,  5.312000000e-02f,  6.486320000e+00f,  5.315200000e-02f,  6.270032000e+00f,
                2.667200000e-02f,  2.704840000e+00f,  2.668800000e-02f,  2.596568000e+00f,  2.670400000e-02f,  2.488296000e+00f,
                1.314400000e-02f,  1.646852000e+00f,  1.315200000e-02f,  1.593292000e+00f,  1.316000000e-02f,  1.539732000e+00f,
                2.641600000e-02f,  2.654920000e+00f,  2.643200000e-02f,  2.547672000e+00f,  2.644800000e-02f,  2.440424000e+00f,
                3.981600000e-02f,  3.022668000e+00f,  3.984000000e-02f,  2.861604000e+00f,  3.986400000e-02f,  2.700540000e+00f,
                3.991200000e-02f,  3.029868000e+00f,  3.993600000e-02f,  2.868420000e+00f,  3.996000000e-02f,  2.706972000e+00f,
                4.000800000e-02f,  3.037068000e+00f,  4.003200000e-02f,  2.875236000e+00f,  4.005600000e-02f,  2.713404000e+00f,
                4.010400000e-02f,  3.044268000e+00f,  4.012800000e-02f,  2.882052000e+00f,  4.015200000e-02f,  2.719836000e+00f,
                4.020000000e-02f,  3.051468000e+00f,  4.022400000e-02f,  2.888868000e+00f,  4.024800000e-02f,  2.726268000e+00f,
                4.029600000e-02f,  3.058668000e+00f,  4.032000000e-02f,  2.895684000e+00f,  4.034400000e-02f,  2.732700000e+00f,
                4.039200000e-02f,  3.065868000e+00f,  4.041600000e-02f,  2.902500000e+00f,  4.044000000e-02f,  2.739132000e+00f,
                4.048800000e-02f,  3.073068000e+00f,  4.051200000e-02f,  2.909316000e+00f,  4.053600000e-02f,  2.745564000e+00f,
                4.058400000e-02f,  3.080268000e+00f,  4.060800000e-02f,  2.916132000e+00f,  4.063200000e-02f,  2.751996000e+00f,
                2.718400000e-02f,  1.397320000e+00f,  2.720000000e-02f,  1.287768000e+00f,  2.721600000e-02f,  1.178216000e+00f,
                1.365600000e-02f,  3.700520000e-01f,  1.366400000e-02f,  3.152120000e-01f,  1.367200000e-02f,  2.603720000e-01f,

            };

            float[] x_actual = x.ToArray();

            AssertError.Tolerance(x_expect, x_actual, 1e-6f, 1e-4f, $"mismatch value {inchannels},{outchannels},{kwidth},{kheight},{inwidth},{inheight},{batch}");
        }
    }
}
