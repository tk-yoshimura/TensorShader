using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;
using TensorShader;
using TensorShader.Operators.Connection2D;
using TensorShaderCudaBackend.API;

namespace TensorShaderTest.Operators.Connection2D {
    [TestClass]
    public class DeconvolutionTest {
        [TestMethod]
        public void ExecuteFPTest() {
            TensorShaderCudaBackend.Environment.Precision = TensorShaderCudaBackend.Environment.PrecisionMode.Float;
            TensorShaderCudaBackend.Environment.CudnnEnabled = false;

            float max_err = 0;

            foreach (int batch in new int[] { 1, 2 }) {
                foreach ((int inchannels, int outchannels) in new (int, int)[] { (1, 1), (2, 1), (1, 2), (2, 3), (3, 1), (3, 4), (4, 5), (5, 3), (5, 10), (10, 15), (15, 5), (15, 20), (20, 32), (32, 15), (32, 33), (33, 33) }) {
                    foreach (int kheight in new int[] { 1, 3, 5 }) {
                        foreach (int kwidth in new int[] { 1, 3, 5 }) {
                            foreach (int inheight in new int[] { kheight, kheight * 2, 8, 9, 13, 17, 25 }) {
                                foreach (int inwidth in new int[] { kwidth, kwidth * 2, 8, 9, 13, 17, 25 }) {
                                    int outwidth = inwidth - kwidth + 1, outheight = inheight - kheight + 1;

                                    float[] yval = (new float[outwidth * outheight * outchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
                                    float[] wval = (new float[kwidth * kheight * inchannels * outchannels]).Select((_, idx) => (idx + 1) * 1e-3f).Reverse().ToArray();

                                    Map2D y = new(outchannels, outwidth, outheight, batch, yval);
                                    Filter2D w = new(inchannels, outchannels, kwidth, kheight, wval);

                                    Map2D x = Reference(y, w, inwidth, inheight, kwidth, kheight);

                                    OverflowCheckedTensor y_tensor = new(Shape.Map2D(outchannels, outwidth, outheight, batch), yval);
                                    OverflowCheckedTensor w_tensor = new(Shape.Kernel2D(inchannels, outchannels, kwidth, kheight), wval);

                                    OverflowCheckedTensor x_tensor = new(Shape.Map2D(inchannels, inwidth, inheight, batch));

                                    Deconvolution ope = new(outwidth, outheight, outchannels, inchannels, kwidth, kheight, batch);

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

            Console.WriteLine($"maxerr:{max_err}");
        }

        [TestMethod]
        public void ExecuteFFPTest() {
            TensorShaderCudaBackend.Environment.CudnnEnabled = false;
            TensorShaderCudaBackend.Environment.Precision = TensorShaderCudaBackend.Environment.PrecisionMode.FloatFloat;

            float max_err = 0;

            foreach (int batch in new int[] { 1, 2 }) {
                foreach ((int inchannels, int outchannels) in new (int, int)[] { (1, 1), (2, 1), (1, 2), (2, 3), (3, 1), (3, 4), (4, 5), (5, 3), (5, 10), (10, 15), (15, 5), (15, 20), (20, 32), (32, 15), (32, 33), (33, 33) }) {
                    foreach (int kheight in new int[] { 1, 3, 5 }) {
                        foreach (int kwidth in new int[] { 1, 3, 5 }) {
                            foreach (int inheight in new int[] { kheight, kheight * 2, 8, 9, 13, 17, 25 }) {
                                foreach (int inwidth in new int[] { kwidth, kwidth * 2, 8, 9, 13, 17, 25 }) {
                                    int outwidth = inwidth - kwidth + 1, outheight = inheight - kheight + 1;

                                    float[] yval = (new float[outwidth * outheight * outchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
                                    float[] wval = (new float[kwidth * kheight * inchannels * outchannels]).Select((_, idx) => (idx + 1) * 1e-3f).Reverse().ToArray();

                                    Map2D y = new(outchannels, outwidth, outheight, batch, yval);
                                    Filter2D w = new(inchannels, outchannels, kwidth, kheight, wval);

                                    Map2D x = Reference(y, w, inwidth, inheight, kwidth, kheight);

                                    OverflowCheckedTensor y_tensor = new(Shape.Map2D(outchannels, outwidth, outheight, batch), yval);
                                    OverflowCheckedTensor w_tensor = new(Shape.Kernel2D(inchannels, outchannels, kwidth, kheight), wval);

                                    OverflowCheckedTensor x_tensor = new(Shape.Map2D(inchannels, inwidth, inheight, batch));

                                    Deconvolution ope = new(outwidth, outheight, outchannels, inchannels, kwidth, kheight, batch);

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

            Console.WriteLine($"maxerr:{max_err}");
        }

        [TestMethod]
        public void ExecuteCudnnTest() {
            if (!TensorShaderCudaBackend.Environment.CudnnExists) {
                Console.WriteLine("test was skipped. Cudnn library not exists.");
                Assert.Inconclusive();
            }

            TensorShaderCudaBackend.Environment.Precision = TensorShaderCudaBackend.Environment.PrecisionMode.Float;
            TensorShaderCudaBackend.Environment.CudnnEnabled = true;

            float max_err = 0;

            foreach (int batch in new int[] { 1, 2 }) {
                foreach ((int inchannels, int outchannels) in new (int, int)[] { (1, 1), (2, 1), (1, 2), (2, 3), (3, 1), (3, 4), (4, 5), (5, 3), (5, 10), (10, 15), (15, 5), (15, 20), (20, 32), (32, 15), (32, 33), (33, 33) }) {
                    foreach (int kheight in new int[] { 1, 3, 5 }) {
                        foreach (int kwidth in new int[] { 1, 3, 5 }) {
                            foreach (int inheight in new int[] { kheight, kheight * 2, 8, 9, 13, 17, 25 }) {
                                foreach (int inwidth in new int[] { kwidth, kwidth * 2, 8, 9, 13, 17, 25 }) {
                                    int outwidth = inwidth - kwidth + 1, outheight = inheight - kheight + 1;

                                    float[] yval = (new float[outwidth * outheight * outchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
                                    float[] wval = (new float[kwidth * kheight * inchannels * outchannels]).Select((_, idx) => (idx + 1) * 1e-3f).Reverse().ToArray();

                                    Map2D y = new(outchannels, outwidth, outheight, batch, yval);
                                    Filter2D w = new(inchannels, outchannels, kwidth, kheight, wval);

                                    Map2D x = Reference(y, w, inwidth, inheight, kwidth, kheight);

                                    OverflowCheckedTensor y_tensor = new(Shape.Map2D(outchannels, outwidth, outheight, batch), yval);
                                    OverflowCheckedTensor w_tensor = new(Shape.Kernel2D(inchannels, outchannels, kwidth, kheight), wval);

                                    OverflowCheckedTensor x_tensor = new(Shape.Map2D(inchannels, inwidth, inheight, batch));

                                    Deconvolution ope = new(outwidth, outheight, outchannels, inchannels, kwidth, kheight, batch);

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

            Console.WriteLine($"maxerr:{max_err}");
        }

        [TestMethod]
        public void LargeMapTest() {
            TensorShaderCudaBackend.Environment.CudnnEnabled = false;
            TensorShaderCudaBackend.Environment.Precision = TensorShaderCudaBackend.Environment.PrecisionMode.FloatFloat;

            float max_err = 0;

            Random random = new(1234);

            int batch = 3;
            int inchannels = 49, outchannels = 50;
            int kwidth = 5, kheight = 3;
            int inwidth = 250, inheight = 196;
            int outwidth = inwidth - kwidth + 1, outheight = inheight - kheight + 1;

            float[] yval = (new float[outwidth * outheight * outchannels * batch]).Select((_, idx) => (float)random.NextDouble() * 1e-2f).ToArray();
            float[] wval = (new float[kwidth * kheight * inchannels * outchannels]).Select((_, idx) => (float)random.NextDouble() * 1e-2f).ToArray();

            Map2D y = new(outchannels, outwidth, outheight, batch, yval);
            Filter2D w = new(inchannels, outchannels, kwidth, kheight, wval);

            Map2D x = Reference(y, w, inwidth, inheight, kwidth, kheight);

            OverflowCheckedTensor y_tensor = new(Shape.Map2D(outchannels, outwidth, outheight, batch), yval);
            OverflowCheckedTensor w_tensor = new(Shape.Kernel2D(inchannels, outchannels, kwidth, kheight), wval);

            OverflowCheckedTensor x_tensor = new(Shape.Map2D(inchannels, inwidth, inheight, batch));

            Deconvolution ope = new(outwidth, outheight, outchannels, inchannels, kwidth, kheight, batch);

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

            int inwidth = 512, inheight = 512, inchannels = 31, outchannels = 31, ksize = 3;
            int outwidth = inwidth - ksize + 1, outheight = inheight - ksize + 1;

            OverflowCheckedTensor y_tensor = new(Shape.Map2D(outchannels, outwidth, outheight));
            OverflowCheckedTensor w_tensor = new(Shape.Kernel2D(inchannels, outchannels, ksize, ksize));

            OverflowCheckedTensor x_tensor = new(Shape.Map2D(inchannels, inwidth, inheight));

            Deconvolution ope = new(outwidth, outheight, outchannels, inchannels, ksize, ksize);

            ope.Execute(y_tensor, w_tensor, x_tensor);

            Cuda.Profiler.Initialize("../../../../profiler.nvsetting", "../../nvprofiles/deconvolution_2d_fp.nvvp");
            Cuda.Profiler.Start();

            ope.Execute(y_tensor, w_tensor, x_tensor);

            Cuda.Profiler.Stop();
        }

        [TestMethod]
        public void SpeedFFPTest() {
            TensorShaderCudaBackend.Environment.CudnnEnabled = false;
            TensorShaderCudaBackend.Environment.Precision = TensorShaderCudaBackend.Environment.PrecisionMode.FloatFloat;

            int inwidth = 512, inheight = 512, inchannels = 31, outchannels = 31, ksize = 3;
            int outwidth = inwidth - ksize + 1, outheight = inheight - ksize + 1;

            OverflowCheckedTensor y_tensor = new(Shape.Map2D(outchannels, outwidth, outheight));
            OverflowCheckedTensor w_tensor = new(Shape.Kernel2D(inchannels, outchannels, ksize, ksize));

            OverflowCheckedTensor x_tensor = new(Shape.Map2D(inchannels, inwidth, inheight));

            Deconvolution ope = new(outwidth, outheight, outchannels, inchannels, ksize, ksize);

            ope.Execute(y_tensor, w_tensor, x_tensor);

            Cuda.Profiler.Initialize("../../../../profiler.nvsetting", "../../nvprofiles/deconvolution_2d_ffp.nvvp");
            Cuda.Profiler.Start();

            ope.Execute(y_tensor, w_tensor, x_tensor);

            Cuda.Profiler.Stop();
        }

        [TestMethod]
        public void SpeedCudnnTest() {
            if (!TensorShaderCudaBackend.Environment.CudnnExists) {
                Console.WriteLine("test was skipped. Cudnn library not exists.");
                Assert.Inconclusive();
            }

            TensorShaderCudaBackend.Environment.Precision = TensorShaderCudaBackend.Environment.PrecisionMode.Float;
            TensorShaderCudaBackend.Environment.CudnnEnabled = true;

            int inwidth = 512, inheight = 512, inchannels = 31, outchannels = 31, ksize = 3;
            int outwidth = inwidth - ksize + 1, outheight = inheight - ksize + 1;

            OverflowCheckedTensor y_tensor = new(Shape.Map2D(outchannels, outwidth, outheight));
            OverflowCheckedTensor w_tensor = new(Shape.Kernel2D(inchannels, outchannels, ksize, ksize));

            OverflowCheckedTensor x_tensor = new(Shape.Map2D(inchannels, inwidth, inheight));

            Deconvolution ope = new(outwidth, outheight, outchannels, inchannels, ksize, ksize);

            ope.Execute(y_tensor, w_tensor, x_tensor);

            Cuda.Profiler.Initialize("../../../../profiler.nvsetting", "../../nvprofiles/deconvolution_2d_cudnn.nvvp");
            Cuda.Profiler.Start();

            ope.Execute(y_tensor, w_tensor, x_tensor);

            Cuda.Profiler.Stop();
        }

        public static Map2D Reference(Map2D y, Filter2D w, int inw, int inh, int kwidth, int kheight) {
            int inchannels = w.InChannels, outchannels = w.OutChannels, batch = y.Batch;
            int outw = inw - kwidth + 1, outh = inh - kheight + 1;

            if (y.Width != outw || y.Height != outh) {
                throw new ArgumentException("mismatch shape");
            }

            Map2D x = new(inchannels, inw, inh, batch);

            for (int kx, ky = 0; ky < kheight; ky++) {
                for (kx = 0; kx < kwidth; kx++) {
                    for (int th = 0; th < batch; th++) {
                        for (int ox, oy = 0; oy < outh; oy++) {
                            for (ox = 0; ox < outw; ox++) {
                                for (int outch = 0; outch < outchannels; outch++) {
                                    double v = y[outch, ox, oy, th];

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
            int inchannels = 7, outchannels = 11, kwidth = 3, kheight = 5, inwidth = 13, inheight = 17, batch = 2;
            int outwidth = inwidth - kwidth + 1, outheight = inheight - kheight + 1;

            float[] yval = (new float[outwidth * outheight * outchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
            float[] wval = (new float[kwidth * kheight * outchannels * inchannels]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

            Map2D y = new(outchannels, outwidth, outheight, batch, yval);
            Filter2D w = new(inchannels, outchannels, kwidth, kheight, wval);

            Map2D x = Reference(y, w, inwidth, inheight, kwidth, kheight);

            float[] x_expect = {
                6.077500000e-02f, 6.072000000e-02f, 6.066500000e-02f, 6.061000000e-02f, 6.055500000e-02f, 6.050000000e-02f, 6.044500000e-02f,
                2.527140000e-01f, 2.524830000e-01f, 2.522520000e-01f, 2.520210000e-01f, 2.517900000e-01f, 2.515590000e-01f, 2.513280000e-01f,
                5.665000000e-01f, 5.659720000e-01f, 5.654440000e-01f, 5.649160000e-01f, 5.643880000e-01f, 5.638600000e-01f, 5.633320000e-01f,
                9.447460000e-01f, 9.438550000e-01f, 9.429640000e-01f, 9.420730000e-01f, 9.411820000e-01f, 9.402910000e-01f, 9.394000000e-01f,
                1.322992000e+00f, 1.321738000e+00f, 1.320484000e+00f, 1.319230000e+00f, 1.317976000e+00f, 1.316722000e+00f, 1.315468000e+00f,
                1.701238000e+00f, 1.699621000e+00f, 1.698004000e+00f, 1.696387000e+00f, 1.694770000e+00f, 1.693153000e+00f, 1.691536000e+00f,
                2.079484000e+00f, 2.077504000e+00f, 2.075524000e+00f, 2.073544000e+00f, 2.071564000e+00f, 2.069584000e+00f, 2.067604000e+00f,
                2.457730000e+00f, 2.455387000e+00f, 2.453044000e+00f, 2.450701000e+00f, 2.448358000e+00f, 2.446015000e+00f, 2.443672000e+00f,
                2.835976000e+00f, 2.833270000e+00f, 2.830564000e+00f, 2.827858000e+00f, 2.825152000e+00f, 2.822446000e+00f, 2.819740000e+00f,
                3.214222000e+00f, 3.211153000e+00f, 3.208084000e+00f, 3.205015000e+00f, 3.201946000e+00f, 3.198877000e+00f, 3.195808000e+00f,
                3.592468000e+00f, 3.589036000e+00f, 3.585604000e+00f, 3.582172000e+00f, 3.578740000e+00f, 3.575308000e+00f, 3.571876000e+00f,
                2.420550000e+00f, 2.418141000e+00f, 2.415732000e+00f, 2.413323000e+00f, 2.410914000e+00f, 2.408505000e+00f, 2.406096000e+00f,
                1.219955000e+00f, 1.218690000e+00f, 1.217425000e+00f, 1.216160000e+00f, 1.214895000e+00f, 1.213630000e+00f, 1.212365000e+00f,
                1.598234000e+00f, 1.596793000e+00f, 1.595352000e+00f, 1.593911000e+00f, 1.592470000e+00f, 1.591029000e+00f, 1.589588000e+00f,
                3.328358000e+00f, 3.325234000e+00f, 3.322110000e+00f, 3.318986000e+00f, 3.315862000e+00f, 3.312738000e+00f, 3.309614000e+00f,
                5.171738000e+00f, 5.166689000e+00f, 5.161640000e+00f, 5.156591000e+00f, 5.151542000e+00f, 5.146493000e+00f, 5.141444000e+00f,
                5.844377000e+00f, 5.838602000e+00f, 5.832827000e+00f, 5.827052000e+00f, 5.821277000e+00f, 5.815502000e+00f, 5.809727000e+00f,
                6.517016000e+00f, 6.510515000e+00f, 6.504014000e+00f, 6.497513000e+00f, 6.491012000e+00f, 6.484511000e+00f, 6.478010000e+00f,
                7.189655000e+00f, 7.182428000e+00f, 7.175201000e+00f, 7.167974000e+00f, 7.160747000e+00f, 7.153520000e+00f, 7.146293000e+00f,
                7.862294000e+00f, 7.854341000e+00f, 7.846388000e+00f, 7.838435000e+00f, 7.830482000e+00f, 7.822529000e+00f, 7.814576000e+00f,
                8.534933000e+00f, 8.526254000e+00f, 8.517575000e+00f, 8.508896000e+00f, 8.500217000e+00f, 8.491538000e+00f, 8.482859000e+00f,
                9.207572000e+00f, 9.198167000e+00f, 9.188762000e+00f, 9.179357000e+00f, 9.169952000e+00f, 9.160547000e+00f, 9.151142000e+00f,
                9.880211000e+00f, 9.870080000e+00f, 9.859949000e+00f, 9.849818000e+00f, 9.839687000e+00f, 9.829556000e+00f, 9.819425000e+00f,
                1.055285000e+01f, 1.054199300e+01f, 1.053113600e+01f, 1.052027900e+01f, 1.050942200e+01f, 1.049856500e+01f, 1.048770800e+01f,
                6.955938000e+00f, 6.948458000e+00f, 6.940978000e+00f, 6.933498000e+00f, 6.926018000e+00f, 6.918538000e+00f, 6.911058000e+00f,
                3.432110000e+00f, 3.428249000e+00f, 3.424388000e+00f, 3.420527000e+00f, 3.416666000e+00f, 3.412805000e+00f, 3.408944000e+00f,
                4.304916000e+00f, 4.300758000e+00f, 4.296600000e+00f, 4.292442000e+00f, 4.288284000e+00f, 4.284126000e+00f, 4.279968000e+00f,
                8.612010000e+00f, 8.603331000e+00f, 8.594652000e+00f, 8.585973000e+00f, 8.577294000e+00f, 8.568615000e+00f, 8.559936000e+00f,
                1.289333100e+01f, 1.287976800e+01f, 1.286620500e+01f, 1.285264200e+01f, 1.283907900e+01f, 1.282551600e+01f, 1.281195300e+01f,
                1.377651000e+01f, 1.376185800e+01f, 1.374720600e+01f, 1.373255400e+01f, 1.371790200e+01f, 1.370325000e+01f, 1.368859800e+01f,
                1.465968900e+01f, 1.464394800e+01f, 1.462820700e+01f, 1.461246600e+01f, 1.459672500e+01f, 1.458098400e+01f, 1.456524300e+01f,
                1.554286800e+01f, 1.552603800e+01f, 1.550920800e+01f, 1.549237800e+01f, 1.547554800e+01f, 1.545871800e+01f, 1.544188800e+01f,
                1.642604700e+01f, 1.640812800e+01f, 1.639020900e+01f, 1.637229000e+01f, 1.635437100e+01f, 1.633645200e+01f, 1.631853300e+01f,
                1.730922600e+01f, 1.729021800e+01f, 1.727121000e+01f, 1.725220200e+01f, 1.723319400e+01f, 1.721418600e+01f, 1.719517800e+01f,
                1.819240500e+01f, 1.817230800e+01f, 1.815221100e+01f, 1.813211400e+01f, 1.811201700e+01f, 1.809192000e+01f, 1.807182300e+01f,
                1.907558400e+01f, 1.905439800e+01f, 1.903321200e+01f, 1.901202600e+01f, 1.899084000e+01f, 1.896965400e+01f, 1.894846800e+01f,
                1.995876300e+01f, 1.993648800e+01f, 1.991421300e+01f, 1.989193800e+01f, 1.986966300e+01f, 1.984738800e+01f, 1.982511300e+01f,
                1.299124200e+01f, 1.297602900e+01f, 1.296081600e+01f, 1.294560300e+01f, 1.293039000e+01f, 1.291517700e+01f, 1.289996400e+01f,
                6.329004000e+00f, 6.321216000e+00f, 6.313428000e+00f, 6.305640000e+00f, 6.297852000e+00f, 6.290064000e+00f, 6.282276000e+00f,
                7.873360000e+00f, 7.865154000e+00f, 7.856948000e+00f, 7.848742000e+00f, 7.840536000e+00f, 7.832330000e+00f, 7.824124000e+00f,
                1.548874800e+01f, 1.547185200e+01f, 1.545495600e+01f, 1.543806000e+01f, 1.542116400e+01f, 1.540426800e+01f, 1.538737200e+01f,
                2.280889600e+01f, 2.278282600e+01f, 2.275675600e+01f, 2.273068600e+01f, 2.270461600e+01f, 2.267854600e+01f, 2.265247600e+01f,
                2.381876200e+01f, 2.379124000e+01f, 2.376371800e+01f, 2.373619600e+01f, 2.370867400e+01f, 2.368115200e+01f, 2.365363000e+01f,
                2.482862800e+01f, 2.479965400e+01f, 2.477068000e+01f, 2.474170600e+01f, 2.471273200e+01f, 2.468375800e+01f, 2.465478400e+01f,
                2.583849400e+01f, 2.580806800e+01f, 2.577764200e+01f, 2.574721600e+01f, 2.571679000e+01f, 2.568636400e+01f, 2.565593800e+01f,
                2.684836000e+01f, 2.681648200e+01f, 2.678460400e+01f, 2.675272600e+01f, 2.672084800e+01f, 2.668897000e+01f, 2.665709200e+01f,
                2.785822600e+01f, 2.782489600e+01f, 2.779156600e+01f, 2.775823600e+01f, 2.772490600e+01f, 2.769157600e+01f, 2.765824600e+01f,
                2.886809200e+01f, 2.883331000e+01f, 2.879852800e+01f, 2.876374600e+01f, 2.872896400e+01f, 2.869418200e+01f, 2.865940000e+01f,
                2.987795800e+01f, 2.984172400e+01f, 2.980549000e+01f, 2.976925600e+01f, 2.973302200e+01f, 2.969678800e+01f, 2.966055400e+01f,
                3.088782400e+01f, 3.085013800e+01f, 3.081245200e+01f, 3.077476600e+01f, 3.073708000e+01f, 3.069939400e+01f, 3.066170800e+01f,
                1.991154000e+01f, 1.988593200e+01f, 1.986032400e+01f, 1.983471600e+01f, 1.980910800e+01f, 1.978350000e+01f, 1.975789200e+01f,
                9.603176000e+00f, 9.590130000e+00f, 9.577084000e+00f, 9.564038000e+00f, 9.550992000e+00f, 9.537946000e+00f, 9.524900000e+00f,
                1.199610500e+01f, 1.198252000e+01f, 1.196893500e+01f, 1.195535000e+01f, 1.194176500e+01f, 1.192818000e+01f, 1.191459500e+01f,
                2.334365000e+01f, 2.331587500e+01f, 2.328810000e+01f, 2.326032500e+01f, 2.323255000e+01f, 2.320477500e+01f, 2.317700000e+01f,
                3.399605000e+01f, 3.395348000e+01f, 3.391091000e+01f, 3.386834000e+01f, 3.382577000e+01f, 3.378320000e+01f, 3.374063000e+01f,
                3.504875000e+01f, 3.500436500e+01f, 3.495998000e+01f, 3.491559500e+01f, 3.487121000e+01f, 3.482682500e+01f, 3.478244000e+01f,
                3.610145000e+01f, 3.605525000e+01f, 3.600905000e+01f, 3.596285000e+01f, 3.591665000e+01f, 3.587045000e+01f, 3.582425000e+01f,
                3.715415000e+01f, 3.710613500e+01f, 3.705812000e+01f, 3.701010500e+01f, 3.696209000e+01f, 3.691407500e+01f, 3.686606000e+01f,
                3.820685000e+01f, 3.815702000e+01f, 3.810719000e+01f, 3.805736000e+01f, 3.800753000e+01f, 3.795770000e+01f, 3.790787000e+01f,
                3.925955000e+01f, 3.920790500e+01f, 3.915626000e+01f, 3.910461500e+01f, 3.905297000e+01f, 3.900132500e+01f, 3.894968000e+01f,
                4.031225000e+01f, 4.025879000e+01f, 4.020533000e+01f, 4.015187000e+01f, 4.009841000e+01f, 4.004495000e+01f, 3.999149000e+01f,
                4.136495000e+01f, 4.130967500e+01f, 4.125440000e+01f, 4.119912500e+01f, 4.114385000e+01f, 4.108857500e+01f, 4.103330000e+01f,
                4.241765000e+01f, 4.236056000e+01f, 4.230347000e+01f, 4.224638000e+01f, 4.218929000e+01f, 4.213220000e+01f, 4.207511000e+01f,
                2.710191000e+01f, 2.706324500e+01f, 2.702458000e+01f, 2.698591500e+01f, 2.694725000e+01f, 2.690858500e+01f, 2.686992000e+01f,
                1.294716500e+01f, 1.292753000e+01f, 1.290789500e+01f, 1.288826000e+01f, 1.286862500e+01f, 1.284899000e+01f, 1.282935500e+01f,
                1.636844000e+01f, 1.634820000e+01f, 1.632796000e+01f, 1.630772000e+01f, 1.628748000e+01f, 1.626724000e+01f, 1.624700000e+01f,
                3.157588500e+01f, 3.153480000e+01f, 3.149371500e+01f, 3.145263000e+01f, 3.141154500e+01f, 3.137046000e+01f, 3.132937500e+01f,
                4.557575000e+01f, 4.551321500e+01f, 4.545068000e+01f, 4.538814500e+01f, 4.532561000e+01f, 4.526307500e+01f, 4.520054000e+01f,
                4.662845000e+01f, 4.656410000e+01f, 4.649975000e+01f, 4.643540000e+01f, 4.637105000e+01f, 4.630670000e+01f, 4.624235000e+01f,
                4.768115000e+01f, 4.761498500e+01f, 4.754882000e+01f, 4.748265500e+01f, 4.741649000e+01f, 4.735032500e+01f, 4.728416000e+01f,
                4.873385000e+01f, 4.866587000e+01f, 4.859789000e+01f, 4.852991000e+01f, 4.846193000e+01f, 4.839395000e+01f, 4.832597000e+01f,
                4.978655000e+01f, 4.971675500e+01f, 4.964696000e+01f, 4.957716500e+01f, 4.950737000e+01f, 4.943757500e+01f, 4.936778000e+01f,
                5.083925000e+01f, 5.076764000e+01f, 5.069603000e+01f, 5.062442000e+01f, 5.055281000e+01f, 5.048120000e+01f, 5.040959000e+01f,
                5.189195000e+01f, 5.181852500e+01f, 5.174510000e+01f, 5.167167500e+01f, 5.159825000e+01f, 5.152482500e+01f, 5.145140000e+01f,
                5.294465000e+01f, 5.286941000e+01f, 5.279417000e+01f, 5.271893000e+01f, 5.264369000e+01f, 5.256845000e+01f, 5.249321000e+01f,
                5.399735000e+01f, 5.392029500e+01f, 5.384324000e+01f, 5.376618500e+01f, 5.368913000e+01f, 5.361207500e+01f, 5.353502000e+01f,
                3.430927500e+01f, 3.425730000e+01f, 3.420532500e+01f, 3.415335000e+01f, 3.410137500e+01f, 3.404940000e+01f, 3.399742500e+01f,
                1.629463000e+01f, 1.626834000e+01f, 1.624205000e+01f, 1.621576000e+01f, 1.618947000e+01f, 1.616318000e+01f, 1.613689000e+01f,
                2.074077500e+01f, 2.071388000e+01f, 2.068698500e+01f, 2.066009000e+01f, 2.063319500e+01f, 2.060630000e+01f, 2.057940500e+01f,
                3.980812000e+01f, 3.975372500e+01f, 3.969933000e+01f, 3.964493500e+01f, 3.959054000e+01f, 3.953614500e+01f, 3.948175000e+01f,
                5.715545000e+01f, 5.707295000e+01f, 5.699045000e+01f, 5.690795000e+01f, 5.682545000e+01f, 5.674295000e+01f, 5.666045000e+01f,
                5.820815000e+01f, 5.812383500e+01f, 5.803952000e+01f, 5.795520500e+01f, 5.787089000e+01f, 5.778657500e+01f, 5.770226000e+01f,
                5.926085000e+01f, 5.917472000e+01f, 5.908859000e+01f, 5.900246000e+01f, 5.891633000e+01f, 5.883020000e+01f, 5.874407000e+01f,
                6.031355000e+01f, 6.022560500e+01f, 6.013766000e+01f, 6.004971500e+01f, 5.996177000e+01f, 5.987382500e+01f, 5.978588000e+01f,
                6.136625000e+01f, 6.127649000e+01f, 6.118673000e+01f, 6.109697000e+01f, 6.100721000e+01f, 6.091745000e+01f, 6.082769000e+01f,
                6.241895000e+01f, 6.232737500e+01f, 6.223580000e+01f, 6.214422500e+01f, 6.205265000e+01f, 6.196107500e+01f, 6.186950000e+01f,
                6.347165000e+01f, 6.337826000e+01f, 6.328487000e+01f, 6.319148000e+01f, 6.309809000e+01f, 6.300470000e+01f, 6.291131000e+01f,
                6.452435000e+01f, 6.442914500e+01f, 6.433394000e+01f, 6.423873500e+01f, 6.414353000e+01f, 6.404832500e+01f, 6.395312000e+01f,
                6.557705000e+01f, 6.548003000e+01f, 6.538301000e+01f, 6.528599000e+01f, 6.518897000e+01f, 6.509195000e+01f, 6.499493000e+01f,
                4.151664000e+01f, 4.145135500e+01f, 4.138607000e+01f, 4.132078500e+01f, 4.125550000e+01f, 4.119021500e+01f, 4.112493000e+01f,
                1.964209500e+01f, 1.960915000e+01f, 1.957620500e+01f, 1.954326000e+01f, 1.951031500e+01f, 1.947737000e+01f, 1.944442500e+01f,
                2.511311000e+01f, 2.507956000e+01f, 2.504601000e+01f, 2.501246000e+01f, 2.497891000e+01f, 2.494536000e+01f, 2.491181000e+01f,
                4.804035500e+01f, 4.797265000e+01f, 4.790494500e+01f, 4.783724000e+01f, 4.776953500e+01f, 4.770183000e+01f, 4.763412500e+01f,
                6.873515000e+01f, 6.863268500e+01f, 6.853022000e+01f, 6.842775500e+01f, 6.832529000e+01f, 6.822282500e+01f, 6.812036000e+01f,
                6.978785000e+01f, 6.968357000e+01f, 6.957929000e+01f, 6.947501000e+01f, 6.937073000e+01f, 6.926645000e+01f, 6.916217000e+01f,
                7.084055000e+01f, 7.073445500e+01f, 7.062836000e+01f, 7.052226500e+01f, 7.041617000e+01f, 7.031007500e+01f, 7.020398000e+01f,
                7.189325000e+01f, 7.178534000e+01f, 7.167743000e+01f, 7.156952000e+01f, 7.146161000e+01f, 7.135370000e+01f, 7.124579000e+01f,
                7.294595000e+01f, 7.283622500e+01f, 7.272650000e+01f, 7.261677500e+01f, 7.250705000e+01f, 7.239732500e+01f, 7.228760000e+01f,
                7.399865000e+01f, 7.388711000e+01f, 7.377557000e+01f, 7.366403000e+01f, 7.355249000e+01f, 7.344095000e+01f, 7.332941000e+01f,
                7.505135000e+01f, 7.493799500e+01f, 7.482464000e+01f, 7.471128500e+01f, 7.459793000e+01f, 7.448457500e+01f, 7.437122000e+01f,
                7.610405000e+01f, 7.598888000e+01f, 7.587371000e+01f, 7.575854000e+01f, 7.564337000e+01f, 7.552820000e+01f, 7.541303000e+01f,
                7.715675000e+01f, 7.703976500e+01f, 7.692278000e+01f, 7.680579500e+01f, 7.668881000e+01f, 7.657182500e+01f, 7.645484000e+01f,
                4.872400500e+01f, 4.864541000e+01f, 4.856681500e+01f, 4.848822000e+01f, 4.840962500e+01f, 4.833103000e+01f, 4.825243500e+01f,
                2.298956000e+01f, 2.294996000e+01f, 2.291036000e+01f, 2.287076000e+01f, 2.283116000e+01f, 2.279156000e+01f, 2.275196000e+01f,
                2.948544500e+01f, 2.944524000e+01f, 2.940503500e+01f, 2.936483000e+01f, 2.932462500e+01f, 2.928442000e+01f, 2.924421500e+01f,
                5.627259000e+01f, 5.619157500e+01f, 5.611056000e+01f, 5.602954500e+01f, 5.594853000e+01f, 5.586751500e+01f, 5.578650000e+01f,
                8.031485000e+01f, 8.019242000e+01f, 8.006999000e+01f, 7.994756000e+01f, 7.982513000e+01f, 7.970270000e+01f, 7.958027000e+01f,
                8.136755000e+01f, 8.124330500e+01f, 8.111906000e+01f, 8.099481500e+01f, 8.087057000e+01f, 8.074632500e+01f, 8.062208000e+01f,
                8.242025000e+01f, 8.229419000e+01f, 8.216813000e+01f, 8.204207000e+01f, 8.191601000e+01f, 8.178995000e+01f, 8.166389000e+01f,
                8.347295000e+01f, 8.334507500e+01f, 8.321720000e+01f, 8.308932500e+01f, 8.296145000e+01f, 8.283357500e+01f, 8.270570000e+01f,
                8.452565000e+01f, 8.439596000e+01f, 8.426627000e+01f, 8.413658000e+01f, 8.400689000e+01f, 8.387720000e+01f, 8.374751000e+01f,
                8.557835000e+01f, 8.544684500e+01f, 8.531534000e+01f, 8.518383500e+01f, 8.505233000e+01f, 8.492082500e+01f, 8.478932000e+01f,
                8.663105000e+01f, 8.649773000e+01f, 8.636441000e+01f, 8.623109000e+01f, 8.609777000e+01f, 8.596445000e+01f, 8.583113000e+01f,
                8.768375000e+01f, 8.754861500e+01f, 8.741348000e+01f, 8.727834500e+01f, 8.714321000e+01f, 8.700807500e+01f, 8.687294000e+01f,
                8.873645000e+01f, 8.859950000e+01f, 8.846255000e+01f, 8.832560000e+01f, 8.818865000e+01f, 8.805170000e+01f, 8.791475000e+01f,
                5.593137000e+01f, 5.583946500e+01f, 5.574756000e+01f, 5.565565500e+01f, 5.556375000e+01f, 5.547184500e+01f, 5.537994000e+01f,
                2.633702500e+01f, 2.629077000e+01f, 2.624451500e+01f, 2.619826000e+01f, 2.615200500e+01f, 2.610575000e+01f, 2.605949500e+01f,
                3.385778000e+01f, 3.381092000e+01f, 3.376406000e+01f, 3.371720000e+01f, 3.367034000e+01f, 3.362348000e+01f, 3.357662000e+01f,
                6.450482500e+01f, 6.441050000e+01f, 6.431617500e+01f, 6.422185000e+01f, 6.412752500e+01f, 6.403320000e+01f, 6.393887500e+01f,
                9.189455000e+01f, 9.175215500e+01f, 9.160976000e+01f, 9.146736500e+01f, 9.132497000e+01f, 9.118257500e+01f, 9.104018000e+01f,
                9.294725000e+01f, 9.280304000e+01f, 9.265883000e+01f, 9.251462000e+01f, 9.237041000e+01f, 9.222620000e+01f, 9.208199000e+01f,
                9.399995000e+01f, 9.385392500e+01f, 9.370790000e+01f, 9.356187500e+01f, 9.341585000e+01f, 9.326982500e+01f, 9.312380000e+01f,
                9.505265000e+01f, 9.490481000e+01f, 9.475697000e+01f, 9.460913000e+01f, 9.446129000e+01f, 9.431345000e+01f, 9.416561000e+01f,
                9.610535000e+01f, 9.595569500e+01f, 9.580604000e+01f, 9.565638500e+01f, 9.550673000e+01f, 9.535707500e+01f, 9.520742000e+01f,
                9.715805000e+01f, 9.700658000e+01f, 9.685511000e+01f, 9.670364000e+01f, 9.655217000e+01f, 9.640070000e+01f, 9.624923000e+01f,
                9.821075000e+01f, 9.805746500e+01f, 9.790418000e+01f, 9.775089500e+01f, 9.759761000e+01f, 9.744432500e+01f, 9.729104000e+01f,
                9.926345000e+01f, 9.910835000e+01f, 9.895325000e+01f, 9.879815000e+01f, 9.864305000e+01f, 9.848795000e+01f, 9.833285000e+01f,
                1.003161500e+02f, 1.001592350e+02f, 1.000023200e+02f, 9.984540500e+01f, 9.968849000e+01f, 9.953157500e+01f, 9.937466000e+01f,
                6.313873500e+01f, 6.303352000e+01f, 6.292830500e+01f, 6.282309000e+01f, 6.271787500e+01f, 6.261266000e+01f, 6.250744500e+01f,
                2.968449000e+01f, 2.963158000e+01f, 2.957867000e+01f, 2.952576000e+01f, 2.947285000e+01f, 2.941994000e+01f, 2.936703000e+01f,
                3.823011500e+01f, 3.817660000e+01f, 3.812308500e+01f, 3.806957000e+01f, 3.801605500e+01f, 3.796254000e+01f, 3.790902500e+01f,
                7.273706000e+01f, 7.262942500e+01f, 7.252179000e+01f, 7.241415500e+01f, 7.230652000e+01f, 7.219888500e+01f, 7.209125000e+01f,
                1.034742500e+02f, 1.033118900e+02f, 1.031495300e+02f, 1.029871700e+02f, 1.028248100e+02f, 1.026624500e+02f, 1.025000900e+02f,
                1.045269500e+02f, 1.043627750e+02f, 1.041986000e+02f, 1.040344250e+02f, 1.038702500e+02f, 1.037060750e+02f, 1.035419000e+02f,
                1.055796500e+02f, 1.054136600e+02f, 1.052476700e+02f, 1.050816800e+02f, 1.049156900e+02f, 1.047497000e+02f, 1.045837100e+02f,
                1.066323500e+02f, 1.064645450e+02f, 1.062967400e+02f, 1.061289350e+02f, 1.059611300e+02f, 1.057933250e+02f, 1.056255200e+02f,
                1.076850500e+02f, 1.075154300e+02f, 1.073458100e+02f, 1.071761900e+02f, 1.070065700e+02f, 1.068369500e+02f, 1.066673300e+02f,
                1.087377500e+02f, 1.085663150e+02f, 1.083948800e+02f, 1.082234450e+02f, 1.080520100e+02f, 1.078805750e+02f, 1.077091400e+02f,
                1.097904500e+02f, 1.096172000e+02f, 1.094439500e+02f, 1.092707000e+02f, 1.090974500e+02f, 1.089242000e+02f, 1.087509500e+02f,
                1.108431500e+02f, 1.106680850e+02f, 1.104930200e+02f, 1.103179550e+02f, 1.101428900e+02f, 1.099678250e+02f, 1.097927600e+02f,
                1.118958500e+02f, 1.117189700e+02f, 1.115420900e+02f, 1.113652100e+02f, 1.111883300e+02f, 1.110114500e+02f, 1.108345700e+02f,
                7.034610000e+01f, 7.022757500e+01f, 7.010905000e+01f, 6.999052500e+01f, 6.987200000e+01f, 6.975347500e+01f, 6.963495000e+01f,
                3.303195500e+01f, 3.297239000e+01f, 3.291282500e+01f, 3.285326000e+01f, 3.279369500e+01f, 3.273413000e+01f, 3.267456500e+01f,
                4.260245000e+01f, 4.254228000e+01f, 4.248211000e+01f, 4.242194000e+01f, 4.236177000e+01f, 4.230160000e+01f, 4.224143000e+01f,
                8.096929500e+01f, 8.084835000e+01f, 8.072740500e+01f, 8.060646000e+01f, 8.048551500e+01f, 8.036457000e+01f, 8.024362500e+01f,
                1.150539500e+02f, 1.148716250e+02f, 1.146893000e+02f, 1.145069750e+02f, 1.143246500e+02f, 1.141423250e+02f, 1.139600000e+02f,
                1.161066500e+02f, 1.159225100e+02f, 1.157383700e+02f, 1.155542300e+02f, 1.153700900e+02f, 1.151859500e+02f, 1.150018100e+02f,
                1.171593500e+02f, 1.169733950e+02f, 1.167874400e+02f, 1.166014850e+02f, 1.164155300e+02f, 1.162295750e+02f, 1.160436200e+02f,
                1.182120500e+02f, 1.180242800e+02f, 1.178365100e+02f, 1.176487400e+02f, 1.174609700e+02f, 1.172732000e+02f, 1.170854300e+02f,
                1.192647500e+02f, 1.190751650e+02f, 1.188855800e+02f, 1.186959950e+02f, 1.185064100e+02f, 1.183168250e+02f, 1.181272400e+02f,
                1.203174500e+02f, 1.201260500e+02f, 1.199346500e+02f, 1.197432500e+02f, 1.195518500e+02f, 1.193604500e+02f, 1.191690500e+02f,
                1.213701500e+02f, 1.211769350e+02f, 1.209837200e+02f, 1.207905050e+02f, 1.205972900e+02f, 1.204040750e+02f, 1.202108600e+02f,
                1.224228500e+02f, 1.222278200e+02f, 1.220327900e+02f, 1.218377600e+02f, 1.216427300e+02f, 1.214477000e+02f, 1.212526700e+02f,
                1.234755500e+02f, 1.232787050e+02f, 1.230818600e+02f, 1.228850150e+02f, 1.226881700e+02f, 1.224913250e+02f, 1.222944800e+02f,
                7.755346500e+01f, 7.742163000e+01f, 7.728979500e+01f, 7.715796000e+01f, 7.702612500e+01f, 7.689429000e+01f, 7.676245500e+01f,
                3.637942000e+01f, 3.631320000e+01f, 3.624698000e+01f, 3.618076000e+01f, 3.611454000e+01f, 3.604832000e+01f, 3.598210000e+01f,
                4.697478500e+01f, 4.690796000e+01f, 4.684113500e+01f, 4.677431000e+01f, 4.670748500e+01f, 4.664066000e+01f, 4.657383500e+01f,
                8.920153000e+01f, 8.906727500e+01f, 8.893302000e+01f, 8.879876500e+01f, 8.866451000e+01f, 8.853025500e+01f, 8.839600000e+01f,
                1.266336500e+02f, 1.264313600e+02f, 1.262290700e+02f, 1.260267800e+02f, 1.258244900e+02f, 1.256222000e+02f, 1.254199100e+02f,
                1.276863500e+02f, 1.274822450e+02f, 1.272781400e+02f, 1.270740350e+02f, 1.268699300e+02f, 1.266658250e+02f, 1.264617200e+02f,
                1.287390500e+02f, 1.285331300e+02f, 1.283272100e+02f, 1.281212900e+02f, 1.279153700e+02f, 1.277094500e+02f, 1.275035300e+02f,
                1.297917500e+02f, 1.295840150e+02f, 1.293762800e+02f, 1.291685450e+02f, 1.289608100e+02f, 1.287530750e+02f, 1.285453400e+02f,
                1.308444500e+02f, 1.306349000e+02f, 1.304253500e+02f, 1.302158000e+02f, 1.300062500e+02f, 1.297967000e+02f, 1.295871500e+02f,
                1.318971500e+02f, 1.316857850e+02f, 1.314744200e+02f, 1.312630550e+02f, 1.310516900e+02f, 1.308403250e+02f, 1.306289600e+02f,
                1.329498500e+02f, 1.327366700e+02f, 1.325234900e+02f, 1.323103100e+02f, 1.320971300e+02f, 1.318839500e+02f, 1.316707700e+02f,
                1.340025500e+02f, 1.337875550e+02f, 1.335725600e+02f, 1.333575650e+02f, 1.331425700e+02f, 1.329275750e+02f, 1.327125800e+02f,
                1.350552500e+02f, 1.348384400e+02f, 1.346216300e+02f, 1.344048200e+02f, 1.341880100e+02f, 1.339712000e+02f, 1.337543900e+02f,
                8.476083000e+01f, 8.461568500e+01f, 8.447054000e+01f, 8.432539500e+01f, 8.418025000e+01f, 8.403510500e+01f, 8.388996000e+01f,
                3.972688500e+01f, 3.965401000e+01f, 3.958113500e+01f, 3.950826000e+01f, 3.943538500e+01f, 3.936251000e+01f, 3.928963500e+01f,
                3.192428800e+01f, 3.186816600e+01f, 3.181204400e+01f, 3.175592200e+01f, 3.169980000e+01f, 3.164367800e+01f, 3.158755600e+01f,
                5.978926800e+01f, 5.967654000e+01f, 5.956381200e+01f, 5.945108400e+01f, 5.933835600e+01f, 5.922562800e+01f, 5.911290000e+01f,
                8.355767200e+01f, 8.338785400e+01f, 8.321803600e+01f, 8.304821800e+01f, 8.287840000e+01f, 8.270858200e+01f, 8.253876400e+01f,
                8.423212600e+01f, 8.406085600e+01f, 8.388958600e+01f, 8.371831600e+01f, 8.354704600e+01f, 8.337577600e+01f, 8.320450600e+01f,
                8.490658000e+01f, 8.473385800e+01f, 8.456113600e+01f, 8.438841400e+01f, 8.421569200e+01f, 8.404297000e+01f, 8.387024800e+01f,
                8.558103400e+01f, 8.540686000e+01f, 8.523268600e+01f, 8.505851200e+01f, 8.488433800e+01f, 8.471016400e+01f, 8.453599000e+01f,
                8.625548800e+01f, 8.607986200e+01f, 8.590423600e+01f, 8.572861000e+01f, 8.555298400e+01f, 8.537735800e+01f, 8.520173200e+01f,
                8.692994200e+01f, 8.675286400e+01f, 8.657578600e+01f, 8.639870800e+01f, 8.622163000e+01f, 8.604455200e+01f, 8.586747400e+01f,
                8.760439600e+01f, 8.742586600e+01f, 8.724733600e+01f, 8.706880600e+01f, 8.689027600e+01f, 8.671174600e+01f, 8.653321600e+01f,
                8.827885000e+01f, 8.809886800e+01f, 8.791888600e+01f, 8.773890400e+01f, 8.755892200e+01f, 8.737894000e+01f, 8.719895800e+01f,
                8.895330400e+01f, 8.877187000e+01f, 8.859043600e+01f, 8.840900200e+01f, 8.822756800e+01f, 8.804613400e+01f, 8.786470000e+01f,
                5.482052400e+01f, 5.469908400e+01f, 5.457764400e+01f, 5.445620400e+01f, 5.433476400e+01f, 5.421332400e+01f, 5.409188400e+01f,
                2.515700000e+01f, 2.509603800e+01f, 2.503507600e+01f, 2.497411400e+01f, 2.491315200e+01f, 2.485219000e+01f, 2.479122800e+01f,
                1.939410000e+01f, 1.935001200e+01f, 1.930592400e+01f, 1.926183600e+01f, 1.921774800e+01f, 1.917366000e+01f, 1.912957200e+01f,
                3.554806200e+01f, 3.545952300e+01f, 3.537098400e+01f, 3.528244500e+01f, 3.519390600e+01f, 3.510536700e+01f, 3.501682800e+01f,
                4.843393500e+01f, 4.830058200e+01f, 4.816722900e+01f, 4.803387600e+01f, 4.790052300e+01f, 4.776717000e+01f, 4.763381700e+01f,
                4.881399600e+01f, 4.867955400e+01f, 4.854511200e+01f, 4.841067000e+01f, 4.827622800e+01f, 4.814178600e+01f, 4.800734400e+01f,
                4.919405700e+01f, 4.905852600e+01f, 4.892299500e+01f, 4.878746400e+01f, 4.865193300e+01f, 4.851640200e+01f, 4.838087100e+01f,
                4.957411800e+01f, 4.943749800e+01f, 4.930087800e+01f, 4.916425800e+01f, 4.902763800e+01f, 4.889101800e+01f, 4.875439800e+01f,
                4.995417900e+01f, 4.981647000e+01f, 4.967876100e+01f, 4.954105200e+01f, 4.940334300e+01f, 4.926563400e+01f, 4.912792500e+01f,
                5.033424000e+01f, 5.019544200e+01f, 5.005664400e+01f, 4.991784600e+01f, 4.977904800e+01f, 4.964025000e+01f, 4.950145200e+01f,
                5.071430100e+01f, 5.057441400e+01f, 5.043452700e+01f, 5.029464000e+01f, 5.015475300e+01f, 5.001486600e+01f, 4.987497900e+01f,
                5.109436200e+01f, 5.095338600e+01f, 5.081241000e+01f, 5.067143400e+01f, 5.053045800e+01f, 5.038948200e+01f, 5.024850600e+01f,
                5.147442300e+01f, 5.133235800e+01f, 5.119029300e+01f, 5.104822800e+01f, 5.090616300e+01f, 5.076409800e+01f, 5.062203300e+01f,
                3.075936600e+01f, 3.066429300e+01f, 3.056922000e+01f, 3.047414700e+01f, 3.037907400e+01f, 3.028400100e+01f, 3.018892800e+01f,
                1.359190800e+01f, 1.354419000e+01f, 1.349647200e+01f, 1.344875400e+01f, 1.340103600e+01f, 1.335331800e+01f, 1.330560000e+01f,
                9.691682000e+00f, 9.660959000e+00f, 9.630236000e+00f, 9.599513000e+00f, 9.568790000e+00f, 9.538067000e+00f, 9.507344000e+00f,
                1.709283400e+01f, 1.703114600e+01f, 1.696945800e+01f, 1.690777000e+01f, 1.684608200e+01f, 1.678439400e+01f, 1.672270600e+01f,
                2.218482200e+01f, 2.209192700e+01f, 2.199903200e+01f, 2.190613700e+01f, 2.181324200e+01f, 2.172034700e+01f, 2.162745200e+01f,
                2.235434300e+01f, 2.226072200e+01f, 2.216710100e+01f, 2.207348000e+01f, 2.197985900e+01f, 2.188623800e+01f, 2.179261700e+01f,
                2.252386400e+01f, 2.242951700e+01f, 2.233517000e+01f, 2.224082300e+01f, 2.214647600e+01f, 2.205212900e+01f, 2.195778200e+01f,
                2.269338500e+01f, 2.259831200e+01f, 2.250323900e+01f, 2.240816600e+01f, 2.231309300e+01f, 2.221802000e+01f, 2.212294700e+01f,
                2.286290600e+01f, 2.276710700e+01f, 2.267130800e+01f, 2.257550900e+01f, 2.247971000e+01f, 2.238391100e+01f, 2.228811200e+01f,
                2.303242700e+01f, 2.293590200e+01f, 2.283937700e+01f, 2.274285200e+01f, 2.264632700e+01f, 2.254980200e+01f, 2.245327700e+01f,
                2.320194800e+01f, 2.310469700e+01f, 2.300744600e+01f, 2.291019500e+01f, 2.281294400e+01f, 2.271569300e+01f, 2.261844200e+01f,
                2.337146900e+01f, 2.327349200e+01f, 2.317551500e+01f, 2.307753800e+01f, 2.297956100e+01f, 2.288158400e+01f, 2.278360700e+01f,
                2.354099000e+01f, 2.344228700e+01f, 2.334358400e+01f, 2.324488100e+01f, 2.314617800e+01f, 2.304747500e+01f, 2.294877200e+01f,
                1.319227800e+01f, 1.312623400e+01f, 1.306019000e+01f, 1.299414600e+01f, 1.292810200e+01f, 1.286205800e+01f, 1.279601400e+01f,
                5.339070000e+00f, 5.305927000e+00f, 5.272784000e+00f, 5.239641000e+00f, 5.206498000e+00f, 5.173355000e+00f, 5.140212000e+00f,
                3.124495000e+00f, 3.108468000e+00f, 3.092441000e+00f, 3.076414000e+00f, 3.060387000e+00f, 3.044360000e+00f, 3.028333000e+00f,
                5.038506000e+00f, 5.006331000e+00f, 4.974156000e+00f, 4.941981000e+00f, 4.909806000e+00f, 4.877631000e+00f, 4.845456000e+00f,
                5.732716000e+00f, 5.684272000e+00f, 5.635828000e+00f, 5.587384000e+00f, 5.538940000e+00f, 5.490496000e+00f, 5.442052000e+00f,
                5.775550000e+00f, 5.726743000e+00f, 5.677936000e+00f, 5.629129000e+00f, 5.580322000e+00f, 5.531515000e+00f, 5.482708000e+00f,
                5.818384000e+00f, 5.769214000e+00f, 5.720044000e+00f, 5.670874000e+00f, 5.621704000e+00f, 5.572534000e+00f, 5.523364000e+00f,
                5.861218000e+00f, 5.811685000e+00f, 5.762152000e+00f, 5.712619000e+00f, 5.663086000e+00f, 5.613553000e+00f, 5.564020000e+00f,
                5.904052000e+00f, 5.854156000e+00f, 5.804260000e+00f, 5.754364000e+00f, 5.704468000e+00f, 5.654572000e+00f, 5.604676000e+00f,
                5.946886000e+00f, 5.896627000e+00f, 5.846368000e+00f, 5.796109000e+00f, 5.745850000e+00f, 5.695591000e+00f, 5.645332000e+00f,
                5.989720000e+00f, 5.939098000e+00f, 5.888476000e+00f, 5.837854000e+00f, 5.787232000e+00f, 5.736610000e+00f, 5.685988000e+00f,
                6.032554000e+00f, 5.981569000e+00f, 5.930584000e+00f, 5.879599000e+00f, 5.828614000e+00f, 5.777629000e+00f, 5.726644000e+00f,
                6.075388000e+00f, 6.024040000e+00f, 5.972692000e+00f, 5.921344000e+00f, 5.869996000e+00f, 5.818648000e+00f, 5.767300000e+00f,
                2.734182000e+00f, 2.699829000e+00f, 2.665476000e+00f, 2.631123000e+00f, 2.596770000e+00f, 2.562417000e+00f, 2.528064000e+00f,
                7.059470000e-01f, 6.887100000e-01f, 6.714730000e-01f, 6.542360000e-01f, 6.369990000e-01f, 6.197620000e-01f, 6.025250000e-01f,
                1.942283200e+01f, 1.940547400e+01f, 1.938811600e+01f, 1.937075800e+01f, 1.935340000e+01f, 1.933604200e+01f, 1.931868400e+01f,
                3.764449700e+01f, 3.760966000e+01f, 3.757482300e+01f, 3.753998600e+01f, 3.750514900e+01f, 3.747031200e+01f, 3.743547500e+01f,
                5.465567800e+01f, 5.460324100e+01f, 5.455080400e+01f, 5.449836700e+01f, 5.444593000e+01f, 5.439349300e+01f, 5.434105600e+01f,
                5.503392400e+01f, 5.498112400e+01f, 5.492832400e+01f, 5.487552400e+01f, 5.482272400e+01f, 5.476992400e+01f, 5.471712400e+01f,
                5.541217000e+01f, 5.535900700e+01f, 5.530584400e+01f, 5.525268100e+01f, 5.519951800e+01f, 5.514635500e+01f, 5.509319200e+01f,
                5.579041600e+01f, 5.573689000e+01f, 5.568336400e+01f, 5.562983800e+01f, 5.557631200e+01f, 5.552278600e+01f, 5.546926000e+01f,
                5.616866200e+01f, 5.611477300e+01f, 5.606088400e+01f, 5.600699500e+01f, 5.595310600e+01f, 5.589921700e+01f, 5.584532800e+01f,
                5.654690800e+01f, 5.649265600e+01f, 5.643840400e+01f, 5.638415200e+01f, 5.632990000e+01f, 5.627564800e+01f, 5.622139600e+01f,
                5.692515400e+01f, 5.687053900e+01f, 5.681592400e+01f, 5.676130900e+01f, 5.670669400e+01f, 5.665207900e+01f, 5.659746400e+01f,
                5.730340000e+01f, 5.724842200e+01f, 5.719344400e+01f, 5.713846600e+01f, 5.708348800e+01f, 5.702851000e+01f, 5.697353200e+01f,
                5.768164600e+01f, 5.762630500e+01f, 5.757096400e+01f, 5.751562300e+01f, 5.746028200e+01f, 5.740494100e+01f, 5.734960000e+01f,
                3.714767100e+01f, 3.711065600e+01f, 3.707364100e+01f, 3.703662600e+01f, 3.699961100e+01f, 3.696259600e+01f, 3.692558100e+01f,
                1.791735000e+01f, 1.789878200e+01f, 1.788021400e+01f, 1.786164600e+01f, 1.784307800e+01f, 1.782451000e+01f, 1.780594200e+01f,
                3.632535500e+01f, 3.628930800e+01f, 3.625326100e+01f, 3.621721400e+01f, 3.618116700e+01f, 3.614512000e+01f, 3.610907300e+01f,
                7.011793800e+01f, 7.004560200e+01f, 6.997326600e+01f, 6.990093000e+01f, 6.982859400e+01f, 6.975625800e+01f, 6.968392200e+01f,
                1.013591150e+02f, 1.012502480e+02f, 1.011413810e+02f, 1.010325140e+02f, 1.009236470e+02f, 1.008147800e+02f, 1.007059130e+02f,
                1.020317540e+02f, 1.019221610e+02f, 1.018125680e+02f, 1.017029750e+02f, 1.015933820e+02f, 1.014837890e+02f, 1.013741960e+02f,
                1.027043930e+02f, 1.025940740e+02f, 1.024837550e+02f, 1.023734360e+02f, 1.022631170e+02f, 1.021527980e+02f, 1.020424790e+02f,
                1.033770320e+02f, 1.032659870e+02f, 1.031549420e+02f, 1.030438970e+02f, 1.029328520e+02f, 1.028218070e+02f, 1.027107620e+02f,
                1.040496710e+02f, 1.039379000e+02f, 1.038261290e+02f, 1.037143580e+02f, 1.036025870e+02f, 1.034908160e+02f, 1.033790450e+02f,
                1.047223100e+02f, 1.046098130e+02f, 1.044973160e+02f, 1.043848190e+02f, 1.042723220e+02f, 1.041598250e+02f, 1.040473280e+02f,
                1.053949490e+02f, 1.052817260e+02f, 1.051685030e+02f, 1.050552800e+02f, 1.049420570e+02f, 1.048288340e+02f, 1.047156110e+02f,
                1.060675880e+02f, 1.059536390e+02f, 1.058396900e+02f, 1.057257410e+02f, 1.056117920e+02f, 1.054978430e+02f, 1.053838940e+02f,
                1.067402270e+02f, 1.066255520e+02f, 1.065108770e+02f, 1.063962020e+02f, 1.062815270e+02f, 1.061668520e+02f, 1.060521770e+02f,
                6.841619400e+01f, 6.833950200e+01f, 6.826281000e+01f, 6.818611800e+01f, 6.810942600e+01f, 6.803273400e+01f, 6.795604200e+01f,
                3.282990700e+01f, 3.279144000e+01f, 3.275297300e+01f, 3.271450600e+01f, 3.267603900e+01f, 3.263757200e+01f, 3.259910500e+01f,
                5.040010800e+01f, 5.034404100e+01f, 5.028797400e+01f, 5.023190700e+01f, 5.017584000e+01f, 5.011977300e+01f, 5.006370600e+01f,
                9.680540100e+01f, 9.669290400e+01f, 9.658040700e+01f, 9.646791000e+01f, 9.635541300e+01f, 9.624291600e+01f, 9.613041900e+01f,
                1.391879280e+02f, 1.390186380e+02f, 1.388493480e+02f, 1.386800580e+02f, 1.385107680e+02f, 1.383414780e+02f, 1.381721880e+02f,
                1.400711070e+02f, 1.399007280e+02f, 1.397303490e+02f, 1.395599700e+02f, 1.393895910e+02f, 1.392192120e+02f, 1.390488330e+02f,
                1.409542860e+02f, 1.407828180e+02f, 1.406113500e+02f, 1.404398820e+02f, 1.402684140e+02f, 1.400969460e+02f, 1.399254780e+02f,
                1.418374650e+02f, 1.416649080e+02f, 1.414923510e+02f, 1.413197940e+02f, 1.411472370e+02f, 1.409746800e+02f, 1.408021230e+02f,
                1.427206440e+02f, 1.425469980e+02f, 1.423733520e+02f, 1.421997060e+02f, 1.420260600e+02f, 1.418524140e+02f, 1.416787680e+02f,
                1.436038230e+02f, 1.434290880e+02f, 1.432543530e+02f, 1.430796180e+02f, 1.429048830e+02f, 1.427301480e+02f, 1.425554130e+02f,
                1.444870020e+02f, 1.443111780e+02f, 1.441353540e+02f, 1.439595300e+02f, 1.437837060e+02f, 1.436078820e+02f, 1.434320580e+02f,
                1.453701810e+02f, 1.451932680e+02f, 1.450163550e+02f, 1.448394420e+02f, 1.446625290e+02f, 1.444856160e+02f, 1.443087030e+02f,
                1.462533600e+02f, 1.460753580e+02f, 1.458973560e+02f, 1.457193540e+02f, 1.455413520e+02f, 1.453633500e+02f, 1.451853480e+02f,
                9.319064700e+01f, 9.307161600e+01f, 9.295258500e+01f, 9.283355400e+01f, 9.271452300e+01f, 9.259549200e+01f, 9.247646100e+01f,
                4.443021000e+01f, 4.437051300e+01f, 4.431081600e+01f, 4.425111900e+01f, 4.419142200e+01f, 4.413172500e+01f, 4.407202800e+01f,
                6.133963000e+01f, 6.126221200e+01f, 6.118479400e+01f, 6.110737600e+01f, 6.102995800e+01f, 6.095254000e+01f, 6.087512200e+01f,
                1.170919640e+02f, 1.169366440e+02f, 1.167813240e+02f, 1.166260040e+02f, 1.164706840e+02f, 1.163153640e+02f, 1.161600440e+02f,
                1.672197340e+02f, 1.669860280e+02f, 1.667523220e+02f, 1.665186160e+02f, 1.662849100e+02f, 1.660512040e+02f, 1.658174980e+02f,
                1.682296000e+02f, 1.679944420e+02f, 1.677592840e+02f, 1.675241260e+02f, 1.672889680e+02f, 1.670538100e+02f, 1.668186520e+02f,
                1.692394660e+02f, 1.690028560e+02f, 1.687662460e+02f, 1.685296360e+02f, 1.682930260e+02f, 1.680564160e+02f, 1.678198060e+02f,
                1.702493320e+02f, 1.700112700e+02f, 1.697732080e+02f, 1.695351460e+02f, 1.692970840e+02f, 1.690590220e+02f, 1.688209600e+02f,
                1.712591980e+02f, 1.710196840e+02f, 1.707801700e+02f, 1.705406560e+02f, 1.703011420e+02f, 1.700616280e+02f, 1.698221140e+02f,
                1.722690640e+02f, 1.720280980e+02f, 1.717871320e+02f, 1.715461660e+02f, 1.713052000e+02f, 1.710642340e+02f, 1.708232680e+02f,
                1.732789300e+02f, 1.730365120e+02f, 1.727940940e+02f, 1.725516760e+02f, 1.723092580e+02f, 1.720668400e+02f, 1.718244220e+02f,
                1.742887960e+02f, 1.740449260e+02f, 1.738010560e+02f, 1.735571860e+02f, 1.733133160e+02f, 1.730694460e+02f, 1.728255760e+02f,
                1.752986620e+02f, 1.750533400e+02f, 1.748080180e+02f, 1.745626960e+02f, 1.743173740e+02f, 1.740720520e+02f, 1.738267300e+02f,
                1.108561080e+02f, 1.106920760e+02f, 1.105280440e+02f, 1.103640120e+02f, 1.101999800e+02f, 1.100359480e+02f, 1.098719160e+02f,
                5.241079800e+01f, 5.232854000e+01f, 5.224628200e+01f, 5.216402400e+01f, 5.208176600e+01f, 5.199950800e+01f, 5.191725000e+01f,
                6.883646000e+01f, 6.873636000e+01f, 6.863626000e+01f, 6.853616000e+01f, 6.843606000e+01f, 6.833596000e+01f, 6.823586000e+01f,
                1.303627050e+02f, 1.301619000e+02f, 1.299610950e+02f, 1.297602900e+02f, 1.295594850e+02f, 1.293586800e+02f, 1.291578750e+02f,
                1.845321500e+02f, 1.842300350e+02f, 1.839279200e+02f, 1.836258050e+02f, 1.833236900e+02f, 1.830215750e+02f, 1.827194600e+02f,
                1.855848500e+02f, 1.852809200e+02f, 1.849769900e+02f, 1.846730600e+02f, 1.843691300e+02f, 1.840652000e+02f, 1.837612700e+02f,
                1.866375500e+02f, 1.863318050e+02f, 1.860260600e+02f, 1.857203150e+02f, 1.854145700e+02f, 1.851088250e+02f, 1.848030800e+02f,
                1.876902500e+02f, 1.873826900e+02f, 1.870751300e+02f, 1.867675700e+02f, 1.864600100e+02f, 1.861524500e+02f, 1.858448900e+02f,
                1.887429500e+02f, 1.884335750e+02f, 1.881242000e+02f, 1.878148250e+02f, 1.875054500e+02f, 1.871960750e+02f, 1.868867000e+02f,
                1.897956500e+02f, 1.894844600e+02f, 1.891732700e+02f, 1.888620800e+02f, 1.885508900e+02f, 1.882397000e+02f, 1.879285100e+02f,
                1.908483500e+02f, 1.905353450e+02f, 1.902223400e+02f, 1.899093350e+02f, 1.895963300e+02f, 1.892833250e+02f, 1.889703200e+02f,
                1.919010500e+02f, 1.915862300e+02f, 1.912714100e+02f, 1.909565900e+02f, 1.906417700e+02f, 1.903269500e+02f, 1.900121300e+02f,
                1.929537500e+02f, 1.926371150e+02f, 1.923204800e+02f, 1.920038450e+02f, 1.916872100e+02f, 1.913705750e+02f, 1.910539400e+02f,
                1.207976550e+02f, 1.205859600e+02f, 1.203742650e+02f, 1.201625700e+02f, 1.199508750e+02f, 1.197391800e+02f, 1.195274850e+02f,
                5.646421000e+01f, 5.635806000e+01f, 5.625191000e+01f, 5.614576000e+01f, 5.603961000e+01f, 5.593346000e+01f, 5.582731000e+01f,
                7.320879500e+01f, 7.310204000e+01f, 7.299528500e+01f, 7.288853000e+01f, 7.278177500e+01f, 7.267502000e+01f, 7.256826500e+01f,
                1.385949400e+02f, 1.383808250e+02f, 1.381667100e+02f, 1.379525950e+02f, 1.377384800e+02f, 1.375243650e+02f, 1.373102500e+02f,
                1.961118500e+02f, 1.957897700e+02f, 1.954676900e+02f, 1.951456100e+02f, 1.948235300e+02f, 1.945014500e+02f, 1.941793700e+02f,
                1.971645500e+02f, 1.968406550e+02f, 1.965167600e+02f, 1.961928650e+02f, 1.958689700e+02f, 1.955450750e+02f, 1.952211800e+02f,
                1.982172500e+02f, 1.978915400e+02f, 1.975658300e+02f, 1.972401200e+02f, 1.969144100e+02f, 1.965887000e+02f, 1.962629900e+02f,
                1.992699500e+02f, 1.989424250e+02f, 1.986149000e+02f, 1.982873750e+02f, 1.979598500e+02f, 1.976323250e+02f, 1.973048000e+02f,
                2.003226500e+02f, 1.999933100e+02f, 1.996639700e+02f, 1.993346300e+02f, 1.990052900e+02f, 1.986759500e+02f, 1.983466100e+02f,
                2.013753500e+02f, 2.010441950e+02f, 2.007130400e+02f, 2.003818850e+02f, 2.000507300e+02f, 1.997195750e+02f, 1.993884200e+02f,
                2.024280500e+02f, 2.020950800e+02f, 2.017621100e+02f, 2.014291400e+02f, 2.010961700e+02f, 2.007632000e+02f, 2.004302300e+02f,
                2.034807500e+02f, 2.031459650e+02f, 2.028111800e+02f, 2.024763950e+02f, 2.021416100e+02f, 2.018068250e+02f, 2.014720400e+02f,
                2.045334500e+02f, 2.041968500e+02f, 2.038602500e+02f, 2.035236500e+02f, 2.031870500e+02f, 2.028504500e+02f, 2.025138500e+02f,
                1.280050200e+02f, 1.277800150e+02f, 1.275550100e+02f, 1.273300050e+02f, 1.271050000e+02f, 1.268799950e+02f, 1.266549900e+02f,
                5.981167500e+01f, 5.969887000e+01f, 5.958606500e+01f, 5.947326000e+01f, 5.936045500e+01f, 5.924765000e+01f, 5.913484500e+01f,
                7.758113000e+01f, 7.746772000e+01f, 7.735431000e+01f, 7.724090000e+01f, 7.712749000e+01f, 7.701408000e+01f, 7.690067000e+01f,
                1.468271750e+02f, 1.465997500e+02f, 1.463723250e+02f, 1.461449000e+02f, 1.459174750e+02f, 1.456900500e+02f, 1.454626250e+02f,
                2.076915500e+02f, 2.073495050e+02f, 2.070074600e+02f, 2.066654150e+02f, 2.063233700e+02f, 2.059813250e+02f, 2.056392800e+02f,
                2.087442500e+02f, 2.084003900e+02f, 2.080565300e+02f, 2.077126700e+02f, 2.073688100e+02f, 2.070249500e+02f, 2.066810900e+02f,
                2.097969500e+02f, 2.094512750e+02f, 2.091056000e+02f, 2.087599250e+02f, 2.084142500e+02f, 2.080685750e+02f, 2.077229000e+02f,
                2.108496500e+02f, 2.105021600e+02f, 2.101546700e+02f, 2.098071800e+02f, 2.094596900e+02f, 2.091122000e+02f, 2.087647100e+02f,
                2.119023500e+02f, 2.115530450e+02f, 2.112037400e+02f, 2.108544350e+02f, 2.105051300e+02f, 2.101558250e+02f, 2.098065200e+02f,
                2.129550500e+02f, 2.126039300e+02f, 2.122528100e+02f, 2.119016900e+02f, 2.115505700e+02f, 2.111994500e+02f, 2.108483300e+02f,
                2.140077500e+02f, 2.136548150e+02f, 2.133018800e+02f, 2.129489450e+02f, 2.125960100e+02f, 2.122430750e+02f, 2.118901400e+02f,
                2.150604500e+02f, 2.147057000e+02f, 2.143509500e+02f, 2.139962000e+02f, 2.136414500e+02f, 2.132867000e+02f, 2.129319500e+02f,
                2.161131500e+02f, 2.157565850e+02f, 2.154000200e+02f, 2.150434550e+02f, 2.146868900e+02f, 2.143303250e+02f, 2.139737600e+02f,
                1.352123850e+02f, 1.349740700e+02f, 1.347357550e+02f, 1.344974400e+02f, 1.342591250e+02f, 1.340208100e+02f, 1.337824950e+02f,
                6.315914000e+01f, 6.303968000e+01f, 6.292022000e+01f, 6.280076000e+01f, 6.268130000e+01f, 6.256184000e+01f, 6.244238000e+01f,
                8.195346500e+01f, 8.183340000e+01f, 8.171333500e+01f, 8.159327000e+01f, 8.147320500e+01f, 8.135314000e+01f, 8.123307500e+01f,
                1.550594100e+02f, 1.548186750e+02f, 1.545779400e+02f, 1.543372050e+02f, 1.540964700e+02f, 1.538557350e+02f, 1.536150000e+02f,
                2.192712500e+02f, 2.189092400e+02f, 2.185472300e+02f, 2.181852200e+02f, 2.178232100e+02f, 2.174612000e+02f, 2.170991900e+02f,
                2.203239500e+02f, 2.199601250e+02f, 2.195963000e+02f, 2.192324750e+02f, 2.188686500e+02f, 2.185048250e+02f, 2.181410000e+02f,
                2.213766500e+02f, 2.210110100e+02f, 2.206453700e+02f, 2.202797300e+02f, 2.199140900e+02f, 2.195484500e+02f, 2.191828100e+02f,
                2.224293500e+02f, 2.220618950e+02f, 2.216944400e+02f, 2.213269850e+02f, 2.209595300e+02f, 2.205920750e+02f, 2.202246200e+02f,
                2.234820500e+02f, 2.231127800e+02f, 2.227435100e+02f, 2.223742400e+02f, 2.220049700e+02f, 2.216357000e+02f, 2.212664300e+02f,
                2.245347500e+02f, 2.241636650e+02f, 2.237925800e+02f, 2.234214950e+02f, 2.230504100e+02f, 2.226793250e+02f, 2.223082400e+02f,
                2.255874500e+02f, 2.252145500e+02f, 2.248416500e+02f, 2.244687500e+02f, 2.240958500e+02f, 2.237229500e+02f, 2.233500500e+02f,
                2.266401500e+02f, 2.262654350e+02f, 2.258907200e+02f, 2.255160050e+02f, 2.251412900e+02f, 2.247665750e+02f, 2.243918600e+02f,
                2.276928500e+02f, 2.273163200e+02f, 2.269397900e+02f, 2.265632600e+02f, 2.261867300e+02f, 2.258102000e+02f, 2.254336700e+02f,
                1.424197500e+02f, 1.421681250e+02f, 1.419165000e+02f, 1.416648750e+02f, 1.414132500e+02f, 1.411616250e+02f, 1.409100000e+02f,
                6.650660500e+01f, 6.638049000e+01f, 6.625437500e+01f, 6.612826000e+01f, 6.600214500e+01f, 6.587603000e+01f, 6.574991500e+01f,
                8.632580000e+01f, 8.619908000e+01f, 8.607236000e+01f, 8.594564000e+01f, 8.581892000e+01f, 8.569220000e+01f, 8.556548000e+01f,
                1.632916450e+02f, 1.630376000e+02f, 1.627835550e+02f, 1.625295100e+02f, 1.622754650e+02f, 1.620214200e+02f, 1.617673750e+02f,
                2.308509500e+02f, 2.304689750e+02f, 2.300870000e+02f, 2.297050250e+02f, 2.293230500e+02f, 2.289410750e+02f, 2.285591000e+02f,
                2.319036500e+02f, 2.315198600e+02f, 2.311360700e+02f, 2.307522800e+02f, 2.303684900e+02f, 2.299847000e+02f, 2.296009100e+02f,
                2.329563500e+02f, 2.325707450e+02f, 2.321851400e+02f, 2.317995350e+02f, 2.314139300e+02f, 2.310283250e+02f, 2.306427200e+02f,
                2.340090500e+02f, 2.336216300e+02f, 2.332342100e+02f, 2.328467900e+02f, 2.324593700e+02f, 2.320719500e+02f, 2.316845300e+02f,
                2.350617500e+02f, 2.346725150e+02f, 2.342832800e+02f, 2.338940450e+02f, 2.335048100e+02f, 2.331155750e+02f, 2.327263400e+02f,
                2.361144500e+02f, 2.357234000e+02f, 2.353323500e+02f, 2.349413000e+02f, 2.345502500e+02f, 2.341592000e+02f, 2.337681500e+02f,
                2.371671500e+02f, 2.367742850e+02f, 2.363814200e+02f, 2.359885550e+02f, 2.355956900e+02f, 2.352028250e+02f, 2.348099600e+02f,
                2.382198500e+02f, 2.378251700e+02f, 2.374304900e+02f, 2.370358100e+02f, 2.366411300e+02f, 2.362464500e+02f, 2.358517700e+02f,
                2.392725500e+02f, 2.388760550e+02f, 2.384795600e+02f, 2.380830650e+02f, 2.376865700e+02f, 2.372900750e+02f, 2.368935800e+02f,
                1.496271150e+02f, 1.493621800e+02f, 1.490972450e+02f, 1.488323100e+02f, 1.485673750e+02f, 1.483024400e+02f, 1.480375050e+02f,
                6.985407000e+01f, 6.972130000e+01f, 6.958853000e+01f, 6.945576000e+01f, 6.932299000e+01f, 6.919022000e+01f, 6.905745000e+01f,
                9.069813500e+01f, 9.056476000e+01f, 9.043138500e+01f, 9.029801000e+01f, 9.016463500e+01f, 9.003126000e+01f, 8.989788500e+01f,
                1.715238800e+02f, 1.712565250e+02f, 1.709891700e+02f, 1.707218150e+02f, 1.704544600e+02f, 1.701871050e+02f, 1.699197500e+02f,
                2.424306500e+02f, 2.420287100e+02f, 2.416267700e+02f, 2.412248300e+02f, 2.408228900e+02f, 2.404209500e+02f, 2.400190100e+02f,
                2.434833500e+02f, 2.430795950e+02f, 2.426758400e+02f, 2.422720850e+02f, 2.418683300e+02f, 2.414645750e+02f, 2.410608200e+02f,
                2.445360500e+02f, 2.441304800e+02f, 2.437249100e+02f, 2.433193400e+02f, 2.429137700e+02f, 2.425082000e+02f, 2.421026300e+02f,
                2.455887500e+02f, 2.451813650e+02f, 2.447739800e+02f, 2.443665950e+02f, 2.439592100e+02f, 2.435518250e+02f, 2.431444400e+02f,
                2.466414500e+02f, 2.462322500e+02f, 2.458230500e+02f, 2.454138500e+02f, 2.450046500e+02f, 2.445954500e+02f, 2.441862500e+02f,
                2.476941500e+02f, 2.472831350e+02f, 2.468721200e+02f, 2.464611050e+02f, 2.460500900e+02f, 2.456390750e+02f, 2.452280600e+02f,
                2.487468500e+02f, 2.483340200e+02f, 2.479211900e+02f, 2.475083600e+02f, 2.470955300e+02f, 2.466827000e+02f, 2.462698700e+02f,
                2.497995500e+02f, 2.493849050e+02f, 2.489702600e+02f, 2.485556150e+02f, 2.481409700e+02f, 2.477263250e+02f, 2.473116800e+02f,
                2.508522500e+02f, 2.504357900e+02f, 2.500193300e+02f, 2.496028700e+02f, 2.491864100e+02f, 2.487699500e+02f, 2.483534900e+02f,
                1.568344800e+02f, 1.565562350e+02f, 1.562779900e+02f, 1.559997450e+02f, 1.557215000e+02f, 1.554432550e+02f, 1.551650100e+02f,
                7.320153500e+01f, 7.306211000e+01f, 7.292268500e+01f, 7.278326000e+01f, 7.264383500e+01f, 7.250441000e+01f, 7.236498500e+01f,
                9.507047000e+01f, 9.493044000e+01f, 9.479041000e+01f, 9.465038000e+01f, 9.451035000e+01f, 9.437032000e+01f, 9.423029000e+01f,
                1.797561150e+02f, 1.794754500e+02f, 1.791947850e+02f, 1.789141200e+02f, 1.786334550e+02f, 1.783527900e+02f, 1.780721250e+02f,
                2.540103500e+02f, 2.535884450e+02f, 2.531665400e+02f, 2.527446350e+02f, 2.523227300e+02f, 2.519008250e+02f, 2.514789200e+02f,
                2.550630500e+02f, 2.546393300e+02f, 2.542156100e+02f, 2.537918900e+02f, 2.533681700e+02f, 2.529444500e+02f, 2.525207300e+02f,
                2.561157500e+02f, 2.556902150e+02f, 2.552646800e+02f, 2.548391450e+02f, 2.544136100e+02f, 2.539880750e+02f, 2.535625400e+02f,
                2.571684500e+02f, 2.567411000e+02f, 2.563137500e+02f, 2.558864000e+02f, 2.554590500e+02f, 2.550317000e+02f, 2.546043500e+02f,
                2.582211500e+02f, 2.577919850e+02f, 2.573628200e+02f, 2.569336550e+02f, 2.565044900e+02f, 2.560753250e+02f, 2.556461600e+02f,
                2.592738500e+02f, 2.588428700e+02f, 2.584118900e+02f, 2.579809100e+02f, 2.575499300e+02f, 2.571189500e+02f, 2.566879700e+02f,
                2.603265500e+02f, 2.598937550e+02f, 2.594609600e+02f, 2.590281650e+02f, 2.585953700e+02f, 2.581625750e+02f, 2.577297800e+02f,
                2.613792500e+02f, 2.609446400e+02f, 2.605100300e+02f, 2.600754200e+02f, 2.596408100e+02f, 2.592062000e+02f, 2.587715900e+02f,
                2.624319500e+02f, 2.619955250e+02f, 2.615591000e+02f, 2.611226750e+02f, 2.606862500e+02f, 2.602498250e+02f, 2.598134000e+02f,
                1.640418450e+02f, 1.637502900e+02f, 1.634587350e+02f, 1.631671800e+02f, 1.628756250e+02f, 1.625840700e+02f, 1.622925150e+02f,
                7.654900000e+01f, 7.640292000e+01f, 7.625684000e+01f, 7.611076000e+01f, 7.596468000e+01f, 7.581860000e+01f, 7.567252000e+01f,
                9.944280500e+01f, 9.929612000e+01f, 9.914943500e+01f, 9.900275000e+01f, 9.885606500e+01f, 9.870938000e+01f, 9.856269500e+01f,
                1.879883500e+02f, 1.876943750e+02f, 1.874004000e+02f, 1.871064250e+02f, 1.868124500e+02f, 1.865184750e+02f, 1.862245000e+02f,
                2.655900500e+02f, 2.651481800e+02f, 2.647063100e+02f, 2.642644400e+02f, 2.638225700e+02f, 2.633807000e+02f, 2.629388300e+02f,
                2.666427500e+02f, 2.661990650e+02f, 2.657553800e+02f, 2.653116950e+02f, 2.648680100e+02f, 2.644243250e+02f, 2.639806400e+02f,
                2.676954500e+02f, 2.672499500e+02f, 2.668044500e+02f, 2.663589500e+02f, 2.659134500e+02f, 2.654679500e+02f, 2.650224500e+02f,
                2.687481500e+02f, 2.683008350e+02f, 2.678535200e+02f, 2.674062050e+02f, 2.669588900e+02f, 2.665115750e+02f, 2.660642600e+02f,
                2.698008500e+02f, 2.693517200e+02f, 2.689025900e+02f, 2.684534600e+02f, 2.680043300e+02f, 2.675552000e+02f, 2.671060700e+02f,
                2.708535500e+02f, 2.704026050e+02f, 2.699516600e+02f, 2.695007150e+02f, 2.690497700e+02f, 2.685988250e+02f, 2.681478800e+02f,
                2.719062500e+02f, 2.714534900e+02f, 2.710007300e+02f, 2.705479700e+02f, 2.700952100e+02f, 2.696424500e+02f, 2.691896900e+02f,
                2.729589500e+02f, 2.725043750e+02f, 2.720498000e+02f, 2.715952250e+02f, 2.711406500e+02f, 2.706860750e+02f, 2.702315000e+02f,
                2.740116500e+02f, 2.735552600e+02f, 2.730988700e+02f, 2.726424800e+02f, 2.721860900e+02f, 2.717297000e+02f, 2.712733100e+02f,
                1.712492100e+02f, 1.709443450e+02f, 1.706394800e+02f, 1.703346150e+02f, 1.700297500e+02f, 1.697248850e+02f, 1.694200200e+02f,
                7.989646500e+01f, 7.974373000e+01f, 7.959099500e+01f, 7.943826000e+01f, 7.928552500e+01f, 7.913279000e+01f, 7.898005500e+01f,
                1.038151400e+02f, 1.036618000e+02f, 1.035084600e+02f, 1.033551200e+02f, 1.032017800e+02f, 1.030484400e+02f, 1.028951000e+02f,
                1.962205850e+02f, 1.959133000e+02f, 1.956060150e+02f, 1.952987300e+02f, 1.949914450e+02f, 1.946841600e+02f, 1.943768750e+02f,
                2.771697500e+02f, 2.767079150e+02f, 2.762460800e+02f, 2.757842450e+02f, 2.753224100e+02f, 2.748605750e+02f, 2.743987400e+02f,
                2.782224500e+02f, 2.777588000e+02f, 2.772951500e+02f, 2.768315000e+02f, 2.763678500e+02f, 2.759042000e+02f, 2.754405500e+02f,
                2.792751500e+02f, 2.788096850e+02f, 2.783442200e+02f, 2.778787550e+02f, 2.774132900e+02f, 2.769478250e+02f, 2.764823600e+02f,
                2.803278500e+02f, 2.798605700e+02f, 2.793932900e+02f, 2.789260100e+02f, 2.784587300e+02f, 2.779914500e+02f, 2.775241700e+02f,
                2.813805500e+02f, 2.809114550e+02f, 2.804423600e+02f, 2.799732650e+02f, 2.795041700e+02f, 2.790350750e+02f, 2.785659800e+02f,
                2.824332500e+02f, 2.819623400e+02f, 2.814914300e+02f, 2.810205200e+02f, 2.805496100e+02f, 2.800787000e+02f, 2.796077900e+02f,
                2.834859500e+02f, 2.830132250e+02f, 2.825405000e+02f, 2.820677750e+02f, 2.815950500e+02f, 2.811223250e+02f, 2.806496000e+02f,
                2.845386500e+02f, 2.840641100e+02f, 2.835895700e+02f, 2.831150300e+02f, 2.826404900e+02f, 2.821659500e+02f, 2.816914100e+02f,
                2.855913500e+02f, 2.851149950e+02f, 2.846386400e+02f, 2.841622850e+02f, 2.836859300e+02f, 2.832095750e+02f, 2.827332200e+02f,
                1.784565750e+02f, 1.781384000e+02f, 1.778202250e+02f, 1.775020500e+02f, 1.771838750e+02f, 1.768657000e+02f, 1.765475250e+02f,
                8.324393000e+01f, 8.308454000e+01f, 8.292515000e+01f, 8.276576000e+01f, 8.260637000e+01f, 8.244698000e+01f, 8.228759000e+01f,
                6.940258600e+01f, 6.927725200e+01f, 6.915191800e+01f, 6.902658400e+01f, 6.890125000e+01f, 6.877591600e+01f, 6.865058200e+01f,
                1.294165400e+02f, 1.291653880e+02f, 1.289142360e+02f, 1.286630840e+02f, 1.284119320e+02f, 1.281607800e+02f, 1.279096280e+02f,
                1.800045940e+02f, 1.796271400e+02f, 1.792496860e+02f, 1.788722320e+02f, 1.784947780e+02f, 1.781173240e+02f, 1.777398700e+02f,
                1.806790480e+02f, 1.803001420e+02f, 1.799212360e+02f, 1.795423300e+02f, 1.791634240e+02f, 1.787845180e+02f, 1.784056120e+02f,
                1.813535020e+02f, 1.809731440e+02f, 1.805927860e+02f, 1.802124280e+02f, 1.798320700e+02f, 1.794517120e+02f, 1.790713540e+02f,
                1.820279560e+02f, 1.816461460e+02f, 1.812643360e+02f, 1.808825260e+02f, 1.805007160e+02f, 1.801189060e+02f, 1.797370960e+02f,
                1.827024100e+02f, 1.823191480e+02f, 1.819358860e+02f, 1.815526240e+02f, 1.811693620e+02f, 1.807861000e+02f, 1.804028380e+02f,
                1.833768640e+02f, 1.829921500e+02f, 1.826074360e+02f, 1.822227220e+02f, 1.818380080e+02f, 1.814532940e+02f, 1.810685800e+02f,
                1.840513180e+02f, 1.836651520e+02f, 1.832789860e+02f, 1.828928200e+02f, 1.825066540e+02f, 1.821204880e+02f, 1.817343220e+02f,
                1.847257720e+02f, 1.843381540e+02f, 1.839505360e+02f, 1.835629180e+02f, 1.831753000e+02f, 1.827876820e+02f, 1.824000640e+02f,
                1.854002260e+02f, 1.850111560e+02f, 1.846220860e+02f, 1.842330160e+02f, 1.838439460e+02f, 1.834548760e+02f, 1.830658060e+02f,
                1.137891480e+02f, 1.135292840e+02f, 1.132694200e+02f, 1.130095560e+02f, 1.127496920e+02f, 1.124898280e+02f, 1.122299640e+02f,
                5.197665000e+01f, 5.184647600e+01f, 5.171630200e+01f, 5.158612800e+01f, 5.145595400e+01f, 5.132578000e+01f, 5.119560600e+01f,
                4.150733400e+01f, 4.141133700e+01f, 4.131534000e+01f, 4.121934300e+01f, 4.112334600e+01f, 4.102734900e+01f, 4.093135200e+01f,
                7.577753700e+01f, 7.558518000e+01f, 7.539282300e+01f, 7.520046600e+01f, 7.500810900e+01f, 7.481575200e+01f, 7.462339500e+01f,
                1.027826580e+02f, 1.024935780e+02f, 1.022044980e+02f, 1.019154180e+02f, 1.016263380e+02f, 1.013372580e+02f, 1.010481780e+02f,
                1.031627190e+02f, 1.028725500e+02f, 1.025823810e+02f, 1.022922120e+02f, 1.020020430e+02f, 1.017118740e+02f, 1.014217050e+02f,
                1.035427800e+02f, 1.032515220e+02f, 1.029602640e+02f, 1.026690060e+02f, 1.023777480e+02f, 1.020864900e+02f, 1.017952320e+02f,
                1.039228410e+02f, 1.036304940e+02f, 1.033381470e+02f, 1.030458000e+02f, 1.027534530e+02f, 1.024611060e+02f, 1.021687590e+02f,
                1.043029020e+02f, 1.040094660e+02f, 1.037160300e+02f, 1.034225940e+02f, 1.031291580e+02f, 1.028357220e+02f, 1.025422860e+02f,
                1.046829630e+02f, 1.043884380e+02f, 1.040939130e+02f, 1.037993880e+02f, 1.035048630e+02f, 1.032103380e+02f, 1.029158130e+02f,
                1.050630240e+02f, 1.047674100e+02f, 1.044717960e+02f, 1.041761820e+02f, 1.038805680e+02f, 1.035849540e+02f, 1.032893400e+02f,
                1.054430850e+02f, 1.051463820e+02f, 1.048496790e+02f, 1.045529760e+02f, 1.042562730e+02f, 1.039595700e+02f, 1.036628670e+02f,
                1.058231460e+02f, 1.055253540e+02f, 1.052275620e+02f, 1.049297700e+02f, 1.046319780e+02f, 1.043341860e+02f, 1.040363940e+02f,
                6.299485500e+01f, 6.279596400e+01f, 6.259707300e+01f, 6.239818200e+01f, 6.219929100e+01f, 6.200040000e+01f, 6.180150900e+01f,
                2.771115600e+01f, 2.761152900e+01f, 2.751190200e+01f, 2.741227500e+01f, 2.731264800e+01f, 2.721302100e+01f, 2.711339400e+01f,
                2.043684500e+01f, 2.037151600e+01f, 2.030618700e+01f, 2.024085800e+01f, 2.017552900e+01f, 2.011020000e+01f, 2.004487100e+01f,
                3.591849800e+01f, 3.578759800e+01f, 3.565669800e+01f, 3.552579800e+01f, 3.539489800e+01f, 3.526399800e+01f, 3.513309800e+01f,
                4.642632500e+01f, 4.622961200e+01f, 4.603289900e+01f, 4.583618600e+01f, 4.563947300e+01f, 4.544276000e+01f, 4.524604700e+01f,
                4.659584600e+01f, 4.639840700e+01f, 4.620096800e+01f, 4.600352900e+01f, 4.580609000e+01f, 4.560865100e+01f, 4.541121200e+01f,
                4.676536700e+01f, 4.656720200e+01f, 4.636903700e+01f, 4.617087200e+01f, 4.597270700e+01f, 4.577454200e+01f, 4.557637700e+01f,
                4.693488800e+01f, 4.673599700e+01f, 4.653710600e+01f, 4.633821500e+01f, 4.613932400e+01f, 4.594043300e+01f, 4.574154200e+01f,
                4.710440900e+01f, 4.690479200e+01f, 4.670517500e+01f, 4.650555800e+01f, 4.630594100e+01f, 4.610632400e+01f, 4.590670700e+01f,
                4.727393000e+01f, 4.707358700e+01f, 4.687324400e+01f, 4.667290100e+01f, 4.647255800e+01f, 4.627221500e+01f, 4.607187200e+01f,
                4.744345100e+01f, 4.724238200e+01f, 4.704131300e+01f, 4.684024400e+01f, 4.663917500e+01f, 4.643810600e+01f, 4.623703700e+01f,
                4.761297200e+01f, 4.741117700e+01f, 4.720938200e+01f, 4.700758700e+01f, 4.680579200e+01f, 4.660399700e+01f, 4.640220200e+01f,
                4.778249300e+01f, 4.757997200e+01f, 4.737745100e+01f, 4.717493000e+01f, 4.697240900e+01f, 4.676988800e+01f, 4.656736700e+01f,
                2.668861800e+01f, 2.655336200e+01f, 2.641810600e+01f, 2.628285000e+01f, 2.614759400e+01f, 2.601233800e+01f, 2.587708200e+01f,
                1.075490900e+01f, 1.068716000e+01f, 1.061941100e+01f, 1.055166200e+01f, 1.048391300e+01f, 1.041616400e+01f, 1.034841500e+01f,
                6.498580000e+00f, 6.465250000e+00f, 6.431920000e+00f, 6.398590000e+00f, 6.365260000e+00f, 6.331930000e+00f, 6.298600000e+00f,
                1.045434500e+01f, 1.038756400e+01f, 1.032078300e+01f, 1.025400200e+01f, 1.018722100e+01f, 1.012044000e+01f, 1.005365900e+01f,
                1.185797800e+01f, 1.175762500e+01f, 1.165727200e+01f, 1.155691900e+01f, 1.145656600e+01f, 1.135621300e+01f, 1.125586000e+01f,
                1.190081200e+01f, 1.180009600e+01f, 1.169938000e+01f, 1.159866400e+01f, 1.149794800e+01f, 1.139723200e+01f, 1.129651600e+01f,
                1.194364600e+01f, 1.184256700e+01f, 1.174148800e+01f, 1.164040900e+01f, 1.153933000e+01f, 1.143825100e+01f, 1.133717200e+01f,
                1.198648000e+01f, 1.188503800e+01f, 1.178359600e+01f, 1.168215400e+01f, 1.158071200e+01f, 1.147927000e+01f, 1.137782800e+01f,
                1.202931400e+01f, 1.192750900e+01f, 1.182570400e+01f, 1.172389900e+01f, 1.162209400e+01f, 1.152028900e+01f, 1.141848400e+01f,
                1.207214800e+01f, 1.196998000e+01f, 1.186781200e+01f, 1.176564400e+01f, 1.166347600e+01f, 1.156130800e+01f, 1.145914000e+01f,
                1.211498200e+01f, 1.201245100e+01f, 1.190992000e+01f, 1.180738900e+01f, 1.170485800e+01f, 1.160232700e+01f, 1.149979600e+01f,
                1.215781600e+01f, 1.205492200e+01f, 1.195202800e+01f, 1.184913400e+01f, 1.174624000e+01f, 1.164334600e+01f, 1.154045200e+01f,
                1.220065000e+01f, 1.209739300e+01f, 1.199413600e+01f, 1.189087900e+01f, 1.178762200e+01f, 1.168436500e+01f, 1.158110800e+01f,
                5.485359000e+00f, 5.416400000e+00f, 5.347441000e+00f, 5.278482000e+00f, 5.209523000e+00f, 5.140564000e+00f, 5.071605000e+00f,
                1.415370000e+00f, 1.380830000e+00f, 1.346290000e+00f, 1.311750000e+00f, 1.277210000e+00f, 1.242670000e+00f, 1.208130000e+00f,

            };

            float[] x_actual = x.ToArray();

            AssertError.Tolerance(x_expect, x_actual, 1e-7f, 1e-5f, $"mismatch value {inchannels},{outchannels},{kwidth},{kheight},{inwidth},{inheight},{batch}");
        }
    }
}
