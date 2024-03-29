using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;
using TensorShader;
using TensorShader.Operators.Connection2D;
using TensorShaderCudaBackend.API;

namespace TensorShaderTest.Operators.Connection2D {
    [TestClass]
    public class ConvolutionTest {
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

                                    float[] xval = (new float[inwidth * inheight * inchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
                                    float[] wval = (new float[kwidth * kheight * inchannels * outchannels]).Select((_, idx) => (idx + 1) * 1e-3f).Reverse().ToArray();

                                    Map2D x = new(inchannels, inwidth, inheight, batch, xval);
                                    Filter2D w = new(inchannels, outchannels, kwidth, kheight, wval);

                                    Map2D y = Reference(x, w, kwidth, kheight);

                                    OverflowCheckedTensor x_tensor = new(Shape.Map2D(inchannels, inwidth, inheight, batch), xval);
                                    OverflowCheckedTensor w_tensor = new(Shape.Kernel2D(inchannels, outchannels, kwidth, kheight), wval);

                                    OverflowCheckedTensor y_tensor = new(Shape.Map2D(outchannels, outwidth, outheight, batch));

                                    Convolution ope = new(inwidth, inheight, inchannels, outchannels, kwidth, kheight, batch);

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

                                    float[] xval = (new float[inwidth * inheight * inchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
                                    float[] wval = (new float[kwidth * kheight * inchannels * outchannels]).Select((_, idx) => (idx + 1) * 1e-3f).Reverse().ToArray();

                                    Map2D x = new(inchannels, inwidth, inheight, batch, xval);
                                    Filter2D w = new(inchannels, outchannels, kwidth, kheight, wval);

                                    Map2D y = Reference(x, w, kwidth, kheight);

                                    OverflowCheckedTensor x_tensor = new(Shape.Map2D(inchannels, inwidth, inheight, batch), xval);
                                    OverflowCheckedTensor w_tensor = new(Shape.Kernel2D(inchannels, outchannels, kwidth, kheight), wval);

                                    OverflowCheckedTensor y_tensor = new(Shape.Map2D(outchannels, outwidth, outheight, batch));

                                    Convolution ope = new(inwidth, inheight, inchannels, outchannels, kwidth, kheight, batch);

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

                                    float[] xval = (new float[inwidth * inheight * inchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
                                    float[] wval = (new float[kwidth * kheight * inchannels * outchannels]).Select((_, idx) => (idx + 1) * 1e-3f).Reverse().ToArray();

                                    Map2D x = new(inchannels, inwidth, inheight, batch, xval);
                                    Filter2D w = new(inchannels, outchannels, kwidth, kheight, wval);

                                    Map2D y = Reference(x, w, kwidth, kheight);

                                    OverflowCheckedTensor x_tensor = new(Shape.Map2D(inchannels, inwidth, inheight, batch), xval);
                                    OverflowCheckedTensor w_tensor = new(Shape.Kernel2D(inchannels, outchannels, kwidth, kheight), wval);

                                    OverflowCheckedTensor y_tensor = new(Shape.Map2D(outchannels, outwidth, outheight, batch));

                                    Convolution ope = new(inwidth, inheight, inchannels, outchannels, kwidth, kheight, batch);

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

            float[] xval = (new float[inwidth * inheight * inchannels * batch]).Select((_, idx) => (float)random.NextDouble() * 1e-2f).ToArray();
            float[] wval = (new float[kwidth * kheight * inchannels * outchannels]).Select((_, idx) => (float)random.NextDouble() * 1e-2f).ToArray();

            Map2D x = new(inchannels, inwidth, inheight, batch, xval);
            Filter2D w = new(inchannels, outchannels, kwidth, kheight, wval);

            Map2D y = Reference(x, w, kwidth, kheight);

            OverflowCheckedTensor x_tensor = new(Shape.Map2D(inchannels, inwidth, inheight, batch), xval);
            OverflowCheckedTensor w_tensor = new(Shape.Kernel2D(inchannels, outchannels, kwidth, kheight), wval);

            OverflowCheckedTensor y_tensor = new(Shape.Map2D(outchannels, outwidth, outheight, batch));

            Convolution ope = new(inwidth, inheight, inchannels, outchannels, kwidth, kheight, batch);

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

            int inwidth = 512, inheight = 512, inchannels = 31, outchannels = 31, ksize = 3;
            int outwidth = inwidth - ksize + 1, outheight = inheight - ksize + 1;

            OverflowCheckedTensor x_tensor = new(Shape.Map2D(inchannels, inwidth, inheight));
            OverflowCheckedTensor w_tensor = new(Shape.Kernel2D(inchannels, outchannels, ksize, ksize));

            OverflowCheckedTensor y_tensor = new(Shape.Map2D(outchannels, outwidth, outheight));

            Convolution ope = new(inwidth, inheight, inchannels, outchannels, ksize, ksize);

            Cuda.Profiler.Initialize("../../../../profiler.nvsetting", "../../nvprofiles/convolution_2d_fp.nvvp");
            Cuda.Profiler.Start();

            ope.Execute(x_tensor, w_tensor, y_tensor);

            Cuda.Profiler.Stop();
        }

        [TestMethod]
        public void SpeedFFPTest() {
            TensorShaderCudaBackend.Environment.CudnnEnabled = false;
            TensorShaderCudaBackend.Environment.Precision = TensorShaderCudaBackend.Environment.PrecisionMode.FloatFloat;

            int inwidth = 512, inheight = 512, inchannels = 31, outchannels = 31, ksize = 3;
            int outwidth = inwidth - ksize + 1, outheight = inheight - ksize + 1;

            OverflowCheckedTensor x_tensor = new(Shape.Map2D(inchannels, inwidth, inheight));
            OverflowCheckedTensor w_tensor = new(Shape.Kernel2D(inchannels, outchannels, ksize, ksize));

            OverflowCheckedTensor y_tensor = new(Shape.Map2D(outchannels, outwidth, outheight));

            Convolution ope = new(inwidth, inheight, inchannels, outchannels, ksize, ksize);

            Cuda.Profiler.Initialize("../../../../profiler.nvsetting", "../../nvprofiles/convolution_2d_ffp.nvvp");
            Cuda.Profiler.Start();

            ope.Execute(x_tensor, w_tensor, y_tensor);

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

            OverflowCheckedTensor x_tensor = new(Shape.Map2D(inchannels, inwidth, inheight));
            OverflowCheckedTensor w_tensor = new(Shape.Kernel2D(inchannels, outchannels, ksize, ksize));

            OverflowCheckedTensor y_tensor = new(Shape.Map2D(outchannels, outwidth, outheight));

            Convolution ope = new(inwidth, inheight, inchannels, outchannels, ksize, ksize);

            ope.Execute(x_tensor, w_tensor, y_tensor);

            Cuda.Profiler.Initialize("../../../../profiler.nvsetting", "../../nvprofiles/convolution_2d_cudnn.nvvp");
            Cuda.Profiler.Start();

            ope.Execute(x_tensor, w_tensor, y_tensor);

            Cuda.Profiler.Stop();
        }

        public static Map2D Reference(Map2D x, Filter2D w, int kwidth, int kheight) {
            int inchannels = x.Channels, outchannels = w.OutChannels, batch = x.Batch;
            int inw = x.Width, inh = x.Height, outw = inw - kwidth + 1, outh = inh - kheight + 1;

            Map2D y = new(outchannels, outw, outh, batch);

            for (int kx, ky = 0; ky < kheight; ky++) {
                for (kx = 0; kx < kwidth; kx++) {
                    for (int th = 0; th < batch; th++) {
                        for (int ox, oy = 0; oy < outh; oy++) {
                            for (ox = 0; ox < outw; ox++) {
                                for (int outch = 0; outch < outchannels; outch++) {
                                    double sum = y[outch, ox, oy, th];

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
            int inchannels = 7, outchannels = 11, kwidth = 3, kheight = 5, inwidth = 13, inheight = 17, batch = 2;
            int outwidth = inwidth - kwidth + 1, outheight = inheight - kheight + 1;

            float[] xval = (new float[batch * inwidth * inheight * inchannels]).Select((_, idx) => idx * 1e-3f).ToArray();
            float[] wval = (new float[kwidth * kheight * outchannels * inchannels]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

            Map2D x = new(inchannels, inwidth, inheight, batch, xval);
            Filter2D w = new(inchannels, outchannels, kwidth, kheight, wval);

            Map2D y = Reference(x, w, kwidth, kheight);

            float[] y_expect = {
                7.885360000e+00f, 7.744240000e+00f, 7.603120000e+00f, 7.462000000e+00f, 7.320880000e+00f, 7.179760000e+00f, 7.038640000e+00f, 6.897520000e+00f, 6.756400000e+00f, 6.615280000e+00f, 6.474160000e+00f,
                8.335180000e+00f, 8.188915000e+00f, 8.042650000e+00f, 7.896385000e+00f, 7.750120000e+00f, 7.603855000e+00f, 7.457590000e+00f, 7.311325000e+00f, 7.165060000e+00f, 7.018795000e+00f, 6.872530000e+00f,
                8.785000000e+00f, 8.633590000e+00f, 8.482180000e+00f, 8.330770000e+00f, 8.179360000e+00f, 8.027950000e+00f, 7.876540000e+00f, 7.725130000e+00f, 7.573720000e+00f, 7.422310000e+00f, 7.270900000e+00f,
                9.234820000e+00f, 9.078265000e+00f, 8.921710000e+00f, 8.765155000e+00f, 8.608600000e+00f, 8.452045000e+00f, 8.295490000e+00f, 8.138935000e+00f, 7.982380000e+00f, 7.825825000e+00f, 7.669270000e+00f,
                9.684640000e+00f, 9.522940000e+00f, 9.361240000e+00f, 9.199540000e+00f, 9.037840000e+00f, 8.876140000e+00f, 8.714440000e+00f, 8.552740000e+00f, 8.391040000e+00f, 8.229340000e+00f, 8.067640000e+00f,
                1.013446000e+01f, 9.967615000e+00f, 9.800770000e+00f, 9.633925000e+00f, 9.467080000e+00f, 9.300235000e+00f, 9.133390000e+00f, 8.966545000e+00f, 8.799700000e+00f, 8.632855000e+00f, 8.466010000e+00f,
                1.058428000e+01f, 1.041229000e+01f, 1.024030000e+01f, 1.006831000e+01f, 9.896320000e+00f, 9.724330000e+00f, 9.552340000e+00f, 9.380350000e+00f, 9.208360000e+00f, 9.036370000e+00f, 8.864380000e+00f,
                1.103410000e+01f, 1.085696500e+01f, 1.067983000e+01f, 1.050269500e+01f, 1.032556000e+01f, 1.014842500e+01f, 9.971290000e+00f, 9.794155000e+00f, 9.617020000e+00f, 9.439885000e+00f, 9.262750000e+00f,
                1.148392000e+01f, 1.130164000e+01f, 1.111936000e+01f, 1.093708000e+01f, 1.075480000e+01f, 1.057252000e+01f, 1.039024000e+01f, 1.020796000e+01f, 1.002568000e+01f, 9.843400000e+00f, 9.661120000e+00f,
                1.193374000e+01f, 1.174631500e+01f, 1.155889000e+01f, 1.137146500e+01f, 1.118404000e+01f, 1.099661500e+01f, 1.080919000e+01f, 1.062176500e+01f, 1.043434000e+01f, 1.024691500e+01f, 1.005949000e+01f,
                1.238356000e+01f, 1.219099000e+01f, 1.199842000e+01f, 1.180585000e+01f, 1.161328000e+01f, 1.142071000e+01f, 1.122814000e+01f, 1.103557000e+01f, 1.084300000e+01f, 1.065043000e+01f, 1.045786000e+01f,
                1.373302000e+01f, 1.352501500e+01f, 1.331701000e+01f, 1.310900500e+01f, 1.290100000e+01f, 1.269299500e+01f, 1.248499000e+01f, 1.227698500e+01f, 1.206898000e+01f, 1.186097500e+01f, 1.165297000e+01f,
                1.418284000e+01f, 1.396969000e+01f, 1.375654000e+01f, 1.354339000e+01f, 1.333024000e+01f, 1.311709000e+01f, 1.290394000e+01f, 1.269079000e+01f, 1.247764000e+01f, 1.226449000e+01f, 1.205134000e+01f,
                1.463266000e+01f, 1.441436500e+01f, 1.419607000e+01f, 1.397777500e+01f, 1.375948000e+01f, 1.354118500e+01f, 1.332289000e+01f, 1.310459500e+01f, 1.288630000e+01f, 1.266800500e+01f, 1.244971000e+01f,
                1.508248000e+01f, 1.485904000e+01f, 1.463560000e+01f, 1.441216000e+01f, 1.418872000e+01f, 1.396528000e+01f, 1.374184000e+01f, 1.351840000e+01f, 1.329496000e+01f, 1.307152000e+01f, 1.284808000e+01f,
                1.553230000e+01f, 1.530371500e+01f, 1.507513000e+01f, 1.484654500e+01f, 1.461796000e+01f, 1.438937500e+01f, 1.416079000e+01f, 1.393220500e+01f, 1.370362000e+01f, 1.347503500e+01f, 1.324645000e+01f,
                1.598212000e+01f, 1.574839000e+01f, 1.551466000e+01f, 1.528093000e+01f, 1.504720000e+01f, 1.481347000e+01f, 1.457974000e+01f, 1.434601000e+01f, 1.411228000e+01f, 1.387855000e+01f, 1.364482000e+01f,
                1.643194000e+01f, 1.619306500e+01f, 1.595419000e+01f, 1.571531500e+01f, 1.547644000e+01f, 1.523756500e+01f, 1.499869000e+01f, 1.475981500e+01f, 1.452094000e+01f, 1.428206500e+01f, 1.404319000e+01f,
                1.688176000e+01f, 1.663774000e+01f, 1.639372000e+01f, 1.614970000e+01f, 1.590568000e+01f, 1.566166000e+01f, 1.541764000e+01f, 1.517362000e+01f, 1.492960000e+01f, 1.468558000e+01f, 1.444156000e+01f,
                1.733158000e+01f, 1.708241500e+01f, 1.683325000e+01f, 1.658408500e+01f, 1.633492000e+01f, 1.608575500e+01f, 1.583659000e+01f, 1.558742500e+01f, 1.533826000e+01f, 1.508909500e+01f, 1.483993000e+01f,
                1.778140000e+01f, 1.752709000e+01f, 1.727278000e+01f, 1.701847000e+01f, 1.676416000e+01f, 1.650985000e+01f, 1.625554000e+01f, 1.600123000e+01f, 1.574692000e+01f, 1.549261000e+01f, 1.523830000e+01f,
                1.823122000e+01f, 1.797176500e+01f, 1.771231000e+01f, 1.745285500e+01f, 1.719340000e+01f, 1.693394500e+01f, 1.667449000e+01f, 1.641503500e+01f, 1.615558000e+01f, 1.589612500e+01f, 1.563667000e+01f,
                1.958068000e+01f, 1.930579000e+01f, 1.903090000e+01f, 1.875601000e+01f, 1.848112000e+01f, 1.820623000e+01f, 1.793134000e+01f, 1.765645000e+01f, 1.738156000e+01f, 1.710667000e+01f, 1.683178000e+01f,
                2.003050000e+01f, 1.975046500e+01f, 1.947043000e+01f, 1.919039500e+01f, 1.891036000e+01f, 1.863032500e+01f, 1.835029000e+01f, 1.807025500e+01f, 1.779022000e+01f, 1.751018500e+01f, 1.723015000e+01f,
                2.048032000e+01f, 2.019514000e+01f, 1.990996000e+01f, 1.962478000e+01f, 1.933960000e+01f, 1.905442000e+01f, 1.876924000e+01f, 1.848406000e+01f, 1.819888000e+01f, 1.791370000e+01f, 1.762852000e+01f,
                2.093014000e+01f, 2.063981500e+01f, 2.034949000e+01f, 2.005916500e+01f, 1.976884000e+01f, 1.947851500e+01f, 1.918819000e+01f, 1.889786500e+01f, 1.860754000e+01f, 1.831721500e+01f, 1.802689000e+01f,
                2.137996000e+01f, 2.108449000e+01f, 2.078902000e+01f, 2.049355000e+01f, 2.019808000e+01f, 1.990261000e+01f, 1.960714000e+01f, 1.931167000e+01f, 1.901620000e+01f, 1.872073000e+01f, 1.842526000e+01f,
                2.182978000e+01f, 2.152916500e+01f, 2.122855000e+01f, 2.092793500e+01f, 2.062732000e+01f, 2.032670500e+01f, 2.002609000e+01f, 1.972547500e+01f, 1.942486000e+01f, 1.912424500e+01f, 1.882363000e+01f,
                2.227960000e+01f, 2.197384000e+01f, 2.166808000e+01f, 2.136232000e+01f, 2.105656000e+01f, 2.075080000e+01f, 2.044504000e+01f, 2.013928000e+01f, 1.983352000e+01f, 1.952776000e+01f, 1.922200000e+01f,
                2.272942000e+01f, 2.241851500e+01f, 2.210761000e+01f, 2.179670500e+01f, 2.148580000e+01f, 2.117489500e+01f, 2.086399000e+01f, 2.055308500e+01f, 2.024218000e+01f, 1.993127500e+01f, 1.962037000e+01f,
                2.317924000e+01f, 2.286319000e+01f, 2.254714000e+01f, 2.223109000e+01f, 2.191504000e+01f, 2.159899000e+01f, 2.128294000e+01f, 2.096689000e+01f, 2.065084000e+01f, 2.033479000e+01f, 2.001874000e+01f,
                2.362906000e+01f, 2.330786500e+01f, 2.298667000e+01f, 2.266547500e+01f, 2.234428000e+01f, 2.202308500e+01f, 2.170189000e+01f, 2.138069500e+01f, 2.105950000e+01f, 2.073830500e+01f, 2.041711000e+01f,
                2.407888000e+01f, 2.375254000e+01f, 2.342620000e+01f, 2.309986000e+01f, 2.277352000e+01f, 2.244718000e+01f, 2.212084000e+01f, 2.179450000e+01f, 2.146816000e+01f, 2.114182000e+01f, 2.081548000e+01f,
                2.542834000e+01f, 2.508656500e+01f, 2.474479000e+01f, 2.440301500e+01f, 2.406124000e+01f, 2.371946500e+01f, 2.337769000e+01f, 2.303591500e+01f, 2.269414000e+01f, 2.235236500e+01f, 2.201059000e+01f,
                2.587816000e+01f, 2.553124000e+01f, 2.518432000e+01f, 2.483740000e+01f, 2.449048000e+01f, 2.414356000e+01f, 2.379664000e+01f, 2.344972000e+01f, 2.310280000e+01f, 2.275588000e+01f, 2.240896000e+01f,
                2.632798000e+01f, 2.597591500e+01f, 2.562385000e+01f, 2.527178500e+01f, 2.491972000e+01f, 2.456765500e+01f, 2.421559000e+01f, 2.386352500e+01f, 2.351146000e+01f, 2.315939500e+01f, 2.280733000e+01f,
                2.677780000e+01f, 2.642059000e+01f, 2.606338000e+01f, 2.570617000e+01f, 2.534896000e+01f, 2.499175000e+01f, 2.463454000e+01f, 2.427733000e+01f, 2.392012000e+01f, 2.356291000e+01f, 2.320570000e+01f,
                2.722762000e+01f, 2.686526500e+01f, 2.650291000e+01f, 2.614055500e+01f, 2.577820000e+01f, 2.541584500e+01f, 2.505349000e+01f, 2.469113500e+01f, 2.432878000e+01f, 2.396642500e+01f, 2.360407000e+01f,
                2.767744000e+01f, 2.730994000e+01f, 2.694244000e+01f, 2.657494000e+01f, 2.620744000e+01f, 2.583994000e+01f, 2.547244000e+01f, 2.510494000e+01f, 2.473744000e+01f, 2.436994000e+01f, 2.400244000e+01f,
                2.812726000e+01f, 2.775461500e+01f, 2.738197000e+01f, 2.700932500e+01f, 2.663668000e+01f, 2.626403500e+01f, 2.589139000e+01f, 2.551874500e+01f, 2.514610000e+01f, 2.477345500e+01f, 2.440081000e+01f,
                2.857708000e+01f, 2.819929000e+01f, 2.782150000e+01f, 2.744371000e+01f, 2.706592000e+01f, 2.668813000e+01f, 2.631034000e+01f, 2.593255000e+01f, 2.555476000e+01f, 2.517697000e+01f, 2.479918000e+01f,
                2.902690000e+01f, 2.864396500e+01f, 2.826103000e+01f, 2.787809500e+01f, 2.749516000e+01f, 2.711222500e+01f, 2.672929000e+01f, 2.634635500e+01f, 2.596342000e+01f, 2.558048500e+01f, 2.519755000e+01f,
                2.947672000e+01f, 2.908864000e+01f, 2.870056000e+01f, 2.831248000e+01f, 2.792440000e+01f, 2.753632000e+01f, 2.714824000e+01f, 2.676016000e+01f, 2.637208000e+01f, 2.598400000e+01f, 2.559592000e+01f,
                2.992654000e+01f, 2.953331500e+01f, 2.914009000e+01f, 2.874686500e+01f, 2.835364000e+01f, 2.796041500e+01f, 2.756719000e+01f, 2.717396500e+01f, 2.678074000e+01f, 2.638751500e+01f, 2.599429000e+01f,
                3.127600000e+01f, 3.086734000e+01f, 3.045868000e+01f, 3.005002000e+01f, 2.964136000e+01f, 2.923270000e+01f, 2.882404000e+01f, 2.841538000e+01f, 2.800672000e+01f, 2.759806000e+01f, 2.718940000e+01f,
                3.172582000e+01f, 3.131201500e+01f, 3.089821000e+01f, 3.048440500e+01f, 3.007060000e+01f, 2.965679500e+01f, 2.924299000e+01f, 2.882918500e+01f, 2.841538000e+01f, 2.800157500e+01f, 2.758777000e+01f,
                3.217564000e+01f, 3.175669000e+01f, 3.133774000e+01f, 3.091879000e+01f, 3.049984000e+01f, 3.008089000e+01f, 2.966194000e+01f, 2.924299000e+01f, 2.882404000e+01f, 2.840509000e+01f, 2.798614000e+01f,
                3.262546000e+01f, 3.220136500e+01f, 3.177727000e+01f, 3.135317500e+01f, 3.092908000e+01f, 3.050498500e+01f, 3.008089000e+01f, 2.965679500e+01f, 2.923270000e+01f, 2.880860500e+01f, 2.838451000e+01f,
                3.307528000e+01f, 3.264604000e+01f, 3.221680000e+01f, 3.178756000e+01f, 3.135832000e+01f, 3.092908000e+01f, 3.049984000e+01f, 3.007060000e+01f, 2.964136000e+01f, 2.921212000e+01f, 2.878288000e+01f,
                3.352510000e+01f, 3.309071500e+01f, 3.265633000e+01f, 3.222194500e+01f, 3.178756000e+01f, 3.135317500e+01f, 3.091879000e+01f, 3.048440500e+01f, 3.005002000e+01f, 2.961563500e+01f, 2.918125000e+01f,
                3.397492000e+01f, 3.353539000e+01f, 3.309586000e+01f, 3.265633000e+01f, 3.221680000e+01f, 3.177727000e+01f, 3.133774000e+01f, 3.089821000e+01f, 3.045868000e+01f, 3.001915000e+01f, 2.957962000e+01f,
                3.442474000e+01f, 3.398006500e+01f, 3.353539000e+01f, 3.309071500e+01f, 3.264604000e+01f, 3.220136500e+01f, 3.175669000e+01f, 3.131201500e+01f, 3.086734000e+01f, 3.042266500e+01f, 2.997799000e+01f,
                3.487456000e+01f, 3.442474000e+01f, 3.397492000e+01f, 3.352510000e+01f, 3.307528000e+01f, 3.262546000e+01f, 3.217564000e+01f, 3.172582000e+01f, 3.127600000e+01f, 3.082618000e+01f, 3.037636000e+01f,
                3.532438000e+01f, 3.486941500e+01f, 3.441445000e+01f, 3.395948500e+01f, 3.350452000e+01f, 3.304955500e+01f, 3.259459000e+01f, 3.213962500e+01f, 3.168466000e+01f, 3.122969500e+01f, 3.077473000e+01f,
                3.577420000e+01f, 3.531409000e+01f, 3.485398000e+01f, 3.439387000e+01f, 3.393376000e+01f, 3.347365000e+01f, 3.301354000e+01f, 3.255343000e+01f, 3.209332000e+01f, 3.163321000e+01f, 3.117310000e+01f,
                3.712366000e+01f, 3.664811500e+01f, 3.617257000e+01f, 3.569702500e+01f, 3.522148000e+01f, 3.474593500e+01f, 3.427039000e+01f, 3.379484500e+01f, 3.331930000e+01f, 3.284375500e+01f, 3.236821000e+01f,
                3.757348000e+01f, 3.709279000e+01f, 3.661210000e+01f, 3.613141000e+01f, 3.565072000e+01f, 3.517003000e+01f, 3.468934000e+01f, 3.420865000e+01f, 3.372796000e+01f, 3.324727000e+01f, 3.276658000e+01f,
                3.802330000e+01f, 3.753746500e+01f, 3.705163000e+01f, 3.656579500e+01f, 3.607996000e+01f, 3.559412500e+01f, 3.510829000e+01f, 3.462245500e+01f, 3.413662000e+01f, 3.365078500e+01f, 3.316495000e+01f,
                3.847312000e+01f, 3.798214000e+01f, 3.749116000e+01f, 3.700018000e+01f, 3.650920000e+01f, 3.601822000e+01f, 3.552724000e+01f, 3.503626000e+01f, 3.454528000e+01f, 3.405430000e+01f, 3.356332000e+01f,
                3.892294000e+01f, 3.842681500e+01f, 3.793069000e+01f, 3.743456500e+01f, 3.693844000e+01f, 3.644231500e+01f, 3.594619000e+01f, 3.545006500e+01f, 3.495394000e+01f, 3.445781500e+01f, 3.396169000e+01f,
                3.937276000e+01f, 3.887149000e+01f, 3.837022000e+01f, 3.786895000e+01f, 3.736768000e+01f, 3.686641000e+01f, 3.636514000e+01f, 3.586387000e+01f, 3.536260000e+01f, 3.486133000e+01f, 3.436006000e+01f,
                3.982258000e+01f, 3.931616500e+01f, 3.880975000e+01f, 3.830333500e+01f, 3.779692000e+01f, 3.729050500e+01f, 3.678409000e+01f, 3.627767500e+01f, 3.577126000e+01f, 3.526484500e+01f, 3.475843000e+01f,
                4.027240000e+01f, 3.976084000e+01f, 3.924928000e+01f, 3.873772000e+01f, 3.822616000e+01f, 3.771460000e+01f, 3.720304000e+01f, 3.669148000e+01f, 3.617992000e+01f, 3.566836000e+01f, 3.515680000e+01f,
                4.072222000e+01f, 4.020551500e+01f, 3.968881000e+01f, 3.917210500e+01f, 3.865540000e+01f, 3.813869500e+01f, 3.762199000e+01f, 3.710528500e+01f, 3.658858000e+01f, 3.607187500e+01f, 3.555517000e+01f,
                4.117204000e+01f, 4.065019000e+01f, 4.012834000e+01f, 3.960649000e+01f, 3.908464000e+01f, 3.856279000e+01f, 3.804094000e+01f, 3.751909000e+01f, 3.699724000e+01f, 3.647539000e+01f, 3.595354000e+01f,
                4.162186000e+01f, 4.109486500e+01f, 4.056787000e+01f, 4.004087500e+01f, 3.951388000e+01f, 3.898688500e+01f, 3.845989000e+01f, 3.793289500e+01f, 3.740590000e+01f, 3.687890500e+01f, 3.635191000e+01f,
                4.297132000e+01f, 4.242889000e+01f, 4.188646000e+01f, 4.134403000e+01f, 4.080160000e+01f, 4.025917000e+01f, 3.971674000e+01f, 3.917431000e+01f, 3.863188000e+01f, 3.808945000e+01f, 3.754702000e+01f,
                4.342114000e+01f, 4.287356500e+01f, 4.232599000e+01f, 4.177841500e+01f, 4.123084000e+01f, 4.068326500e+01f, 4.013569000e+01f, 3.958811500e+01f, 3.904054000e+01f, 3.849296500e+01f, 3.794539000e+01f,
                4.387096000e+01f, 4.331824000e+01f, 4.276552000e+01f, 4.221280000e+01f, 4.166008000e+01f, 4.110736000e+01f, 4.055464000e+01f, 4.000192000e+01f, 3.944920000e+01f, 3.889648000e+01f, 3.834376000e+01f,
                4.432078000e+01f, 4.376291500e+01f, 4.320505000e+01f, 4.264718500e+01f, 4.208932000e+01f, 4.153145500e+01f, 4.097359000e+01f, 4.041572500e+01f, 3.985786000e+01f, 3.929999500e+01f, 3.874213000e+01f,
                4.477060000e+01f, 4.420759000e+01f, 4.364458000e+01f, 4.308157000e+01f, 4.251856000e+01f, 4.195555000e+01f, 4.139254000e+01f, 4.082953000e+01f, 4.026652000e+01f, 3.970351000e+01f, 3.914050000e+01f,
                4.522042000e+01f, 4.465226500e+01f, 4.408411000e+01f, 4.351595500e+01f, 4.294780000e+01f, 4.237964500e+01f, 4.181149000e+01f, 4.124333500e+01f, 4.067518000e+01f, 4.010702500e+01f, 3.953887000e+01f,
                4.567024000e+01f, 4.509694000e+01f, 4.452364000e+01f, 4.395034000e+01f, 4.337704000e+01f, 4.280374000e+01f, 4.223044000e+01f, 4.165714000e+01f, 4.108384000e+01f, 4.051054000e+01f, 3.993724000e+01f,
                4.612006000e+01f, 4.554161500e+01f, 4.496317000e+01f, 4.438472500e+01f, 4.380628000e+01f, 4.322783500e+01f, 4.264939000e+01f, 4.207094500e+01f, 4.149250000e+01f, 4.091405500e+01f, 4.033561000e+01f,
                4.656988000e+01f, 4.598629000e+01f, 4.540270000e+01f, 4.481911000e+01f, 4.423552000e+01f, 4.365193000e+01f, 4.306834000e+01f, 4.248475000e+01f, 4.190116000e+01f, 4.131757000e+01f, 4.073398000e+01f,
                4.701970000e+01f, 4.643096500e+01f, 4.584223000e+01f, 4.525349500e+01f, 4.466476000e+01f, 4.407602500e+01f, 4.348729000e+01f, 4.289855500e+01f, 4.230982000e+01f, 4.172108500e+01f, 4.113235000e+01f,
                4.746952000e+01f, 4.687564000e+01f, 4.628176000e+01f, 4.568788000e+01f, 4.509400000e+01f, 4.450012000e+01f, 4.390624000e+01f, 4.331236000e+01f, 4.271848000e+01f, 4.212460000e+01f, 4.153072000e+01f,
                4.881898000e+01f, 4.820966500e+01f, 4.760035000e+01f, 4.699103500e+01f, 4.638172000e+01f, 4.577240500e+01f, 4.516309000e+01f, 4.455377500e+01f, 4.394446000e+01f, 4.333514500e+01f, 4.272583000e+01f,
                4.926880000e+01f, 4.865434000e+01f, 4.803988000e+01f, 4.742542000e+01f, 4.681096000e+01f, 4.619650000e+01f, 4.558204000e+01f, 4.496758000e+01f, 4.435312000e+01f, 4.373866000e+01f, 4.312420000e+01f,
                4.971862000e+01f, 4.909901500e+01f, 4.847941000e+01f, 4.785980500e+01f, 4.724020000e+01f, 4.662059500e+01f, 4.600099000e+01f, 4.538138500e+01f, 4.476178000e+01f, 4.414217500e+01f, 4.352257000e+01f,
                5.016844000e+01f, 4.954369000e+01f, 4.891894000e+01f, 4.829419000e+01f, 4.766944000e+01f, 4.704469000e+01f, 4.641994000e+01f, 4.579519000e+01f, 4.517044000e+01f, 4.454569000e+01f, 4.392094000e+01f,
                5.061826000e+01f, 4.998836500e+01f, 4.935847000e+01f, 4.872857500e+01f, 4.809868000e+01f, 4.746878500e+01f, 4.683889000e+01f, 4.620899500e+01f, 4.557910000e+01f, 4.494920500e+01f, 4.431931000e+01f,
                5.106808000e+01f, 5.043304000e+01f, 4.979800000e+01f, 4.916296000e+01f, 4.852792000e+01f, 4.789288000e+01f, 4.725784000e+01f, 4.662280000e+01f, 4.598776000e+01f, 4.535272000e+01f, 4.471768000e+01f,
                5.151790000e+01f, 5.087771500e+01f, 5.023753000e+01f, 4.959734500e+01f, 4.895716000e+01f, 4.831697500e+01f, 4.767679000e+01f, 4.703660500e+01f, 4.639642000e+01f, 4.575623500e+01f, 4.511605000e+01f,
                5.196772000e+01f, 5.132239000e+01f, 5.067706000e+01f, 5.003173000e+01f, 4.938640000e+01f, 4.874107000e+01f, 4.809574000e+01f, 4.745041000e+01f, 4.680508000e+01f, 4.615975000e+01f, 4.551442000e+01f,
                5.241754000e+01f, 5.176706500e+01f, 5.111659000e+01f, 5.046611500e+01f, 4.981564000e+01f, 4.916516500e+01f, 4.851469000e+01f, 4.786421500e+01f, 4.721374000e+01f, 4.656326500e+01f, 4.591279000e+01f,
                5.286736000e+01f, 5.221174000e+01f, 5.155612000e+01f, 5.090050000e+01f, 5.024488000e+01f, 4.958926000e+01f, 4.893364000e+01f, 4.827802000e+01f, 4.762240000e+01f, 4.696678000e+01f, 4.631116000e+01f,
                5.331718000e+01f, 5.265641500e+01f, 5.199565000e+01f, 5.133488500e+01f, 5.067412000e+01f, 5.001335500e+01f, 4.935259000e+01f, 4.869182500e+01f, 4.803106000e+01f, 4.737029500e+01f, 4.670953000e+01f,
                5.466664000e+01f, 5.399044000e+01f, 5.331424000e+01f, 5.263804000e+01f, 5.196184000e+01f, 5.128564000e+01f, 5.060944000e+01f, 4.993324000e+01f, 4.925704000e+01f, 4.858084000e+01f, 4.790464000e+01f,
                5.511646000e+01f, 5.443511500e+01f, 5.375377000e+01f, 5.307242500e+01f, 5.239108000e+01f, 5.170973500e+01f, 5.102839000e+01f, 5.034704500e+01f, 4.966570000e+01f, 4.898435500e+01f, 4.830301000e+01f,
                5.556628000e+01f, 5.487979000e+01f, 5.419330000e+01f, 5.350681000e+01f, 5.282032000e+01f, 5.213383000e+01f, 5.144734000e+01f, 5.076085000e+01f, 5.007436000e+01f, 4.938787000e+01f, 4.870138000e+01f,
                5.601610000e+01f, 5.532446500e+01f, 5.463283000e+01f, 5.394119500e+01f, 5.324956000e+01f, 5.255792500e+01f, 5.186629000e+01f, 5.117465500e+01f, 5.048302000e+01f, 4.979138500e+01f, 4.909975000e+01f,
                5.646592000e+01f, 5.576914000e+01f, 5.507236000e+01f, 5.437558000e+01f, 5.367880000e+01f, 5.298202000e+01f, 5.228524000e+01f, 5.158846000e+01f, 5.089168000e+01f, 5.019490000e+01f, 4.949812000e+01f,
                5.691574000e+01f, 5.621381500e+01f, 5.551189000e+01f, 5.480996500e+01f, 5.410804000e+01f, 5.340611500e+01f, 5.270419000e+01f, 5.200226500e+01f, 5.130034000e+01f, 5.059841500e+01f, 4.989649000e+01f,
                5.736556000e+01f, 5.665849000e+01f, 5.595142000e+01f, 5.524435000e+01f, 5.453728000e+01f, 5.383021000e+01f, 5.312314000e+01f, 5.241607000e+01f, 5.170900000e+01f, 5.100193000e+01f, 5.029486000e+01f,
                5.781538000e+01f, 5.710316500e+01f, 5.639095000e+01f, 5.567873500e+01f, 5.496652000e+01f, 5.425430500e+01f, 5.354209000e+01f, 5.282987500e+01f, 5.211766000e+01f, 5.140544500e+01f, 5.069323000e+01f,
                5.826520000e+01f, 5.754784000e+01f, 5.683048000e+01f, 5.611312000e+01f, 5.539576000e+01f, 5.467840000e+01f, 5.396104000e+01f, 5.324368000e+01f, 5.252632000e+01f, 5.180896000e+01f, 5.109160000e+01f,
                5.871502000e+01f, 5.799251500e+01f, 5.727001000e+01f, 5.654750500e+01f, 5.582500000e+01f, 5.510249500e+01f, 5.437999000e+01f, 5.365748500e+01f, 5.293498000e+01f, 5.221247500e+01f, 5.148997000e+01f,
                5.916484000e+01f, 5.843719000e+01f, 5.770954000e+01f, 5.698189000e+01f, 5.625424000e+01f, 5.552659000e+01f, 5.479894000e+01f, 5.407129000e+01f, 5.334364000e+01f, 5.261599000e+01f, 5.188834000e+01f,
                6.051430000e+01f, 5.977121500e+01f, 5.902813000e+01f, 5.828504500e+01f, 5.754196000e+01f, 5.679887500e+01f, 5.605579000e+01f, 5.531270500e+01f, 5.456962000e+01f, 5.382653500e+01f, 5.308345000e+01f,
                6.096412000e+01f, 6.021589000e+01f, 5.946766000e+01f, 5.871943000e+01f, 5.797120000e+01f, 5.722297000e+01f, 5.647474000e+01f, 5.572651000e+01f, 5.497828000e+01f, 5.423005000e+01f, 5.348182000e+01f,
                6.141394000e+01f, 6.066056500e+01f, 5.990719000e+01f, 5.915381500e+01f, 5.840044000e+01f, 5.764706500e+01f, 5.689369000e+01f, 5.614031500e+01f, 5.538694000e+01f, 5.463356500e+01f, 5.388019000e+01f,
                6.186376000e+01f, 6.110524000e+01f, 6.034672000e+01f, 5.958820000e+01f, 5.882968000e+01f, 5.807116000e+01f, 5.731264000e+01f, 5.655412000e+01f, 5.579560000e+01f, 5.503708000e+01f, 5.427856000e+01f,
                6.231358000e+01f, 6.154991500e+01f, 6.078625000e+01f, 6.002258500e+01f, 5.925892000e+01f, 5.849525500e+01f, 5.773159000e+01f, 5.696792500e+01f, 5.620426000e+01f, 5.544059500e+01f, 5.467693000e+01f,
                6.276340000e+01f, 6.199459000e+01f, 6.122578000e+01f, 6.045697000e+01f, 5.968816000e+01f, 5.891935000e+01f, 5.815054000e+01f, 5.738173000e+01f, 5.661292000e+01f, 5.584411000e+01f, 5.507530000e+01f,
                6.321322000e+01f, 6.243926500e+01f, 6.166531000e+01f, 6.089135500e+01f, 6.011740000e+01f, 5.934344500e+01f, 5.856949000e+01f, 5.779553500e+01f, 5.702158000e+01f, 5.624762500e+01f, 5.547367000e+01f,
                6.366304000e+01f, 6.288394000e+01f, 6.210484000e+01f, 6.132574000e+01f, 6.054664000e+01f, 5.976754000e+01f, 5.898844000e+01f, 5.820934000e+01f, 5.743024000e+01f, 5.665114000e+01f, 5.587204000e+01f,
                6.411286000e+01f, 6.332861500e+01f, 6.254437000e+01f, 6.176012500e+01f, 6.097588000e+01f, 6.019163500e+01f, 5.940739000e+01f, 5.862314500e+01f, 5.783890000e+01f, 5.705465500e+01f, 5.627041000e+01f,
                6.456268000e+01f, 6.377329000e+01f, 6.298390000e+01f, 6.219451000e+01f, 6.140512000e+01f, 6.061573000e+01f, 5.982634000e+01f, 5.903695000e+01f, 5.824756000e+01f, 5.745817000e+01f, 5.666878000e+01f,
                6.501250000e+01f, 6.421796500e+01f, 6.342343000e+01f, 6.262889500e+01f, 6.183436000e+01f, 6.103982500e+01f, 6.024529000e+01f, 5.945075500e+01f, 5.865622000e+01f, 5.786168500e+01f, 5.706715000e+01f,
                6.636196000e+01f, 6.555199000e+01f, 6.474202000e+01f, 6.393205000e+01f, 6.312208000e+01f, 6.231211000e+01f, 6.150214000e+01f, 6.069217000e+01f, 5.988220000e+01f, 5.907223000e+01f, 5.826226000e+01f,
                6.681178000e+01f, 6.599666500e+01f, 6.518155000e+01f, 6.436643500e+01f, 6.355132000e+01f, 6.273620500e+01f, 6.192109000e+01f, 6.110597500e+01f, 6.029086000e+01f, 5.947574500e+01f, 5.866063000e+01f,
                6.726160000e+01f, 6.644134000e+01f, 6.562108000e+01f, 6.480082000e+01f, 6.398056000e+01f, 6.316030000e+01f, 6.234004000e+01f, 6.151978000e+01f, 6.069952000e+01f, 5.987926000e+01f, 5.905900000e+01f,
                6.771142000e+01f, 6.688601500e+01f, 6.606061000e+01f, 6.523520500e+01f, 6.440980000e+01f, 6.358439500e+01f, 6.275899000e+01f, 6.193358500e+01f, 6.110818000e+01f, 6.028277500e+01f, 5.945737000e+01f,
                6.816124000e+01f, 6.733069000e+01f, 6.650014000e+01f, 6.566959000e+01f, 6.483904000e+01f, 6.400849000e+01f, 6.317794000e+01f, 6.234739000e+01f, 6.151684000e+01f, 6.068629000e+01f, 5.985574000e+01f,
                6.861106000e+01f, 6.777536500e+01f, 6.693967000e+01f, 6.610397500e+01f, 6.526828000e+01f, 6.443258500e+01f, 6.359689000e+01f, 6.276119500e+01f, 6.192550000e+01f, 6.108980500e+01f, 6.025411000e+01f,
                6.906088000e+01f, 6.822004000e+01f, 6.737920000e+01f, 6.653836000e+01f, 6.569752000e+01f, 6.485668000e+01f, 6.401584000e+01f, 6.317500000e+01f, 6.233416000e+01f, 6.149332000e+01f, 6.065248000e+01f,
                6.951070000e+01f, 6.866471500e+01f, 6.781873000e+01f, 6.697274500e+01f, 6.612676000e+01f, 6.528077500e+01f, 6.443479000e+01f, 6.358880500e+01f, 6.274282000e+01f, 6.189683500e+01f, 6.105085000e+01f,
                6.996052000e+01f, 6.910939000e+01f, 6.825826000e+01f, 6.740713000e+01f, 6.655600000e+01f, 6.570487000e+01f, 6.485374000e+01f, 6.400261000e+01f, 6.315148000e+01f, 6.230035000e+01f, 6.144922000e+01f,
                7.041034000e+01f, 6.955406500e+01f, 6.869779000e+01f, 6.784151500e+01f, 6.698524000e+01f, 6.612896500e+01f, 6.527269000e+01f, 6.441641500e+01f, 6.356014000e+01f, 6.270386500e+01f, 6.184759000e+01f,
                7.086016000e+01f, 6.999874000e+01f, 6.913732000e+01f, 6.827590000e+01f, 6.741448000e+01f, 6.655306000e+01f, 6.569164000e+01f, 6.483022000e+01f, 6.396880000e+01f, 6.310738000e+01f, 6.224596000e+01f,
                7.220962000e+01f, 7.133276500e+01f, 7.045591000e+01f, 6.957905500e+01f, 6.870220000e+01f, 6.782534500e+01f, 6.694849000e+01f, 6.607163500e+01f, 6.519478000e+01f, 6.431792500e+01f, 6.344107000e+01f,
                7.265944000e+01f, 7.177744000e+01f, 7.089544000e+01f, 7.001344000e+01f, 6.913144000e+01f, 6.824944000e+01f, 6.736744000e+01f, 6.648544000e+01f, 6.560344000e+01f, 6.472144000e+01f, 6.383944000e+01f,
                7.310926000e+01f, 7.222211500e+01f, 7.133497000e+01f, 7.044782500e+01f, 6.956068000e+01f, 6.867353500e+01f, 6.778639000e+01f, 6.689924500e+01f, 6.601210000e+01f, 6.512495500e+01f, 6.423781000e+01f,
                7.355908000e+01f, 7.266679000e+01f, 7.177450000e+01f, 7.088221000e+01f, 6.998992000e+01f, 6.909763000e+01f, 6.820534000e+01f, 6.731305000e+01f, 6.642076000e+01f, 6.552847000e+01f, 6.463618000e+01f,
                7.400890000e+01f, 7.311146500e+01f, 7.221403000e+01f, 7.131659500e+01f, 7.041916000e+01f, 6.952172500e+01f, 6.862429000e+01f, 6.772685500e+01f, 6.682942000e+01f, 6.593198500e+01f, 6.503455000e+01f,
                7.445872000e+01f, 7.355614000e+01f, 7.265356000e+01f, 7.175098000e+01f, 7.084840000e+01f, 6.994582000e+01f, 6.904324000e+01f, 6.814066000e+01f, 6.723808000e+01f, 6.633550000e+01f, 6.543292000e+01f,
                7.490854000e+01f, 7.400081500e+01f, 7.309309000e+01f, 7.218536500e+01f, 7.127764000e+01f, 7.036991500e+01f, 6.946219000e+01f, 6.855446500e+01f, 6.764674000e+01f, 6.673901500e+01f, 6.583129000e+01f,
                7.535836000e+01f, 7.444549000e+01f, 7.353262000e+01f, 7.261975000e+01f, 7.170688000e+01f, 7.079401000e+01f, 6.988114000e+01f, 6.896827000e+01f, 6.805540000e+01f, 6.714253000e+01f, 6.622966000e+01f,
                7.580818000e+01f, 7.489016500e+01f, 7.397215000e+01f, 7.305413500e+01f, 7.213612000e+01f, 7.121810500e+01f, 7.030009000e+01f, 6.938207500e+01f, 6.846406000e+01f, 6.754604500e+01f, 6.662803000e+01f,
                7.625800000e+01f, 7.533484000e+01f, 7.441168000e+01f, 7.348852000e+01f, 7.256536000e+01f, 7.164220000e+01f, 7.071904000e+01f, 6.979588000e+01f, 6.887272000e+01f, 6.794956000e+01f, 6.702640000e+01f,
                7.670782000e+01f, 7.577951500e+01f, 7.485121000e+01f, 7.392290500e+01f, 7.299460000e+01f, 7.206629500e+01f, 7.113799000e+01f, 7.020968500e+01f, 6.928138000e+01f, 6.835307500e+01f, 6.742477000e+01f,
                7.805728000e+01f, 7.711354000e+01f, 7.616980000e+01f, 7.522606000e+01f, 7.428232000e+01f, 7.333858000e+01f, 7.239484000e+01f, 7.145110000e+01f, 7.050736000e+01f, 6.956362000e+01f, 6.861988000e+01f,
                7.850710000e+01f, 7.755821500e+01f, 7.660933000e+01f, 7.566044500e+01f, 7.471156000e+01f, 7.376267500e+01f, 7.281379000e+01f, 7.186490500e+01f, 7.091602000e+01f, 6.996713500e+01f, 6.901825000e+01f,
                7.895692000e+01f, 7.800289000e+01f, 7.704886000e+01f, 7.609483000e+01f, 7.514080000e+01f, 7.418677000e+01f, 7.323274000e+01f, 7.227871000e+01f, 7.132468000e+01f, 7.037065000e+01f, 6.941662000e+01f,
                7.940674000e+01f, 7.844756500e+01f, 7.748839000e+01f, 7.652921500e+01f, 7.557004000e+01f, 7.461086500e+01f, 7.365169000e+01f, 7.269251500e+01f, 7.173334000e+01f, 7.077416500e+01f, 6.981499000e+01f,
                7.985656000e+01f, 7.889224000e+01f, 7.792792000e+01f, 7.696360000e+01f, 7.599928000e+01f, 7.503496000e+01f, 7.407064000e+01f, 7.310632000e+01f, 7.214200000e+01f, 7.117768000e+01f, 7.021336000e+01f,
                8.030638000e+01f, 7.933691500e+01f, 7.836745000e+01f, 7.739798500e+01f, 7.642852000e+01f, 7.545905500e+01f, 7.448959000e+01f, 7.352012500e+01f, 7.255066000e+01f, 7.158119500e+01f, 7.061173000e+01f,
                8.075620000e+01f, 7.978159000e+01f, 7.880698000e+01f, 7.783237000e+01f, 7.685776000e+01f, 7.588315000e+01f, 7.490854000e+01f, 7.393393000e+01f, 7.295932000e+01f, 7.198471000e+01f, 7.101010000e+01f,
                8.120602000e+01f, 8.022626500e+01f, 7.924651000e+01f, 7.826675500e+01f, 7.728700000e+01f, 7.630724500e+01f, 7.532749000e+01f, 7.434773500e+01f, 7.336798000e+01f, 7.238822500e+01f, 7.140847000e+01f,
                8.165584000e+01f, 8.067094000e+01f, 7.968604000e+01f, 7.870114000e+01f, 7.771624000e+01f, 7.673134000e+01f, 7.574644000e+01f, 7.476154000e+01f, 7.377664000e+01f, 7.279174000e+01f, 7.180684000e+01f,
                8.210566000e+01f, 8.111561500e+01f, 8.012557000e+01f, 7.913552500e+01f, 7.814548000e+01f, 7.715543500e+01f, 7.616539000e+01f, 7.517534500e+01f, 7.418530000e+01f, 7.319525500e+01f, 7.220521000e+01f,
                8.255548000e+01f, 8.156029000e+01f, 8.056510000e+01f, 7.956991000e+01f, 7.857472000e+01f, 7.757953000e+01f, 7.658434000e+01f, 7.558915000e+01f, 7.459396000e+01f, 7.359877000e+01f, 7.260358000e+01f,
                1.072955800e+02f, 1.060174150e+02f, 1.047392500e+02f, 1.034610850e+02f, 1.021829200e+02f, 1.009047550e+02f, 9.962659000e+01f, 9.834842500e+01f, 9.707026000e+01f, 9.579209500e+01f, 9.451393000e+01f,
                1.077454000e+02f, 1.064620900e+02f, 1.051787800e+02f, 1.038954700e+02f, 1.026121600e+02f, 1.013288500e+02f, 1.000455400e+02f, 9.876223000e+01f, 9.747892000e+01f, 9.619561000e+01f, 9.491230000e+01f,
                1.081952200e+02f, 1.069067650e+02f, 1.056183100e+02f, 1.043298550e+02f, 1.030414000e+02f, 1.017529450e+02f, 1.004644900e+02f, 9.917603500e+01f, 9.788758000e+01f, 9.659912500e+01f, 9.531067000e+01f,
                1.086450400e+02f, 1.073514400e+02f, 1.060578400e+02f, 1.047642400e+02f, 1.034706400e+02f, 1.021770400e+02f, 1.008834400e+02f, 9.958984000e+01f, 9.829624000e+01f, 9.700264000e+01f, 9.570904000e+01f,
                1.090948600e+02f, 1.077961150e+02f, 1.064973700e+02f, 1.051986250e+02f, 1.038998800e+02f, 1.026011350e+02f, 1.013023900e+02f, 1.000036450e+02f, 9.870490000e+01f, 9.740615500e+01f, 9.610741000e+01f,
                1.095446800e+02f, 1.082407900e+02f, 1.069369000e+02f, 1.056330100e+02f, 1.043291200e+02f, 1.030252300e+02f, 1.017213400e+02f, 1.004174500e+02f, 9.911356000e+01f, 9.780967000e+01f, 9.650578000e+01f,
                1.099945000e+02f, 1.086854650e+02f, 1.073764300e+02f, 1.060673950e+02f, 1.047583600e+02f, 1.034493250e+02f, 1.021402900e+02f, 1.008312550e+02f, 9.952222000e+01f, 9.821318500e+01f, 9.690415000e+01f,
                1.104443200e+02f, 1.091301400e+02f, 1.078159600e+02f, 1.065017800e+02f, 1.051876000e+02f, 1.038734200e+02f, 1.025592400e+02f, 1.012450600e+02f, 9.993088000e+01f, 9.861670000e+01f, 9.730252000e+01f,
                1.108941400e+02f, 1.095748150e+02f, 1.082554900e+02f, 1.069361650e+02f, 1.056168400e+02f, 1.042975150e+02f, 1.029781900e+02f, 1.016588650e+02f, 1.003395400e+02f, 9.902021500e+01f, 9.770089000e+01f,
                1.113439600e+02f, 1.100194900e+02f, 1.086950200e+02f, 1.073705500e+02f, 1.060460800e+02f, 1.047216100e+02f, 1.033971400e+02f, 1.020726700e+02f, 1.007482000e+02f, 9.942373000e+01f, 9.809926000e+01f,
                1.117937800e+02f, 1.104641650e+02f, 1.091345500e+02f, 1.078049350e+02f, 1.064753200e+02f, 1.051457050e+02f, 1.038160900e+02f, 1.024864750e+02f, 1.011568600e+02f, 9.982724500e+01f, 9.849763000e+01f,
                1.131432400e+02f, 1.117981900e+02f, 1.104531400e+02f, 1.091080900e+02f, 1.077630400e+02f, 1.064179900e+02f, 1.050729400e+02f, 1.037278900e+02f, 1.023828400e+02f, 1.010377900e+02f, 9.969274000e+01f,
                1.135930600e+02f, 1.122428650e+02f, 1.108926700e+02f, 1.095424750e+02f, 1.081922800e+02f, 1.068420850e+02f, 1.054918900e+02f, 1.041416950e+02f, 1.027915000e+02f, 1.014413050e+02f, 1.000911100e+02f,
                1.140428800e+02f, 1.126875400e+02f, 1.113322000e+02f, 1.099768600e+02f, 1.086215200e+02f, 1.072661800e+02f, 1.059108400e+02f, 1.045555000e+02f, 1.032001600e+02f, 1.018448200e+02f, 1.004894800e+02f,
                1.144927000e+02f, 1.131322150e+02f, 1.117717300e+02f, 1.104112450e+02f, 1.090507600e+02f, 1.076902750e+02f, 1.063297900e+02f, 1.049693050e+02f, 1.036088200e+02f, 1.022483350e+02f, 1.008878500e+02f,
                1.149425200e+02f, 1.135768900e+02f, 1.122112600e+02f, 1.108456300e+02f, 1.094800000e+02f, 1.081143700e+02f, 1.067487400e+02f, 1.053831100e+02f, 1.040174800e+02f, 1.026518500e+02f, 1.012862200e+02f,
                1.153923400e+02f, 1.140215650e+02f, 1.126507900e+02f, 1.112800150e+02f, 1.099092400e+02f, 1.085384650e+02f, 1.071676900e+02f, 1.057969150e+02f, 1.044261400e+02f, 1.030553650e+02f, 1.016845900e+02f,
                1.158421600e+02f, 1.144662400e+02f, 1.130903200e+02f, 1.117144000e+02f, 1.103384800e+02f, 1.089625600e+02f, 1.075866400e+02f, 1.062107200e+02f, 1.048348000e+02f, 1.034588800e+02f, 1.020829600e+02f,
                1.162919800e+02f, 1.149109150e+02f, 1.135298500e+02f, 1.121487850e+02f, 1.107677200e+02f, 1.093866550e+02f, 1.080055900e+02f, 1.066245250e+02f, 1.052434600e+02f, 1.038623950e+02f, 1.024813300e+02f,
                1.167418000e+02f, 1.153555900e+02f, 1.139693800e+02f, 1.125831700e+02f, 1.111969600e+02f, 1.098107500e+02f, 1.084245400e+02f, 1.070383300e+02f, 1.056521200e+02f, 1.042659100e+02f, 1.028797000e+02f,
                1.171916200e+02f, 1.158002650e+02f, 1.144089100e+02f, 1.130175550e+02f, 1.116262000e+02f, 1.102348450e+02f, 1.088434900e+02f, 1.074521350e+02f, 1.060607800e+02f, 1.046694250e+02f, 1.032780700e+02f,
                1.176414400e+02f, 1.162449400e+02f, 1.148484400e+02f, 1.134519400e+02f, 1.120554400e+02f, 1.106589400e+02f, 1.092624400e+02f, 1.078659400e+02f, 1.064694400e+02f, 1.050729400e+02f, 1.036764400e+02f,
                1.189909000e+02f, 1.175789650e+02f, 1.161670300e+02f, 1.147550950e+02f, 1.133431600e+02f, 1.119312250e+02f, 1.105192900e+02f, 1.091073550e+02f, 1.076954200e+02f, 1.062834850e+02f, 1.048715500e+02f,
                1.194407200e+02f, 1.180236400e+02f, 1.166065600e+02f, 1.151894800e+02f, 1.137724000e+02f, 1.123553200e+02f, 1.109382400e+02f, 1.095211600e+02f, 1.081040800e+02f, 1.066870000e+02f, 1.052699200e+02f,
                1.198905400e+02f, 1.184683150e+02f, 1.170460900e+02f, 1.156238650e+02f, 1.142016400e+02f, 1.127794150e+02f, 1.113571900e+02f, 1.099349650e+02f, 1.085127400e+02f, 1.070905150e+02f, 1.056682900e+02f,
                1.203403600e+02f, 1.189129900e+02f, 1.174856200e+02f, 1.160582500e+02f, 1.146308800e+02f, 1.132035100e+02f, 1.117761400e+02f, 1.103487700e+02f, 1.089214000e+02f, 1.074940300e+02f, 1.060666600e+02f,
                1.207901800e+02f, 1.193576650e+02f, 1.179251500e+02f, 1.164926350e+02f, 1.150601200e+02f, 1.136276050e+02f, 1.121950900e+02f, 1.107625750e+02f, 1.093300600e+02f, 1.078975450e+02f, 1.064650300e+02f,
                1.212400000e+02f, 1.198023400e+02f, 1.183646800e+02f, 1.169270200e+02f, 1.154893600e+02f, 1.140517000e+02f, 1.126140400e+02f, 1.111763800e+02f, 1.097387200e+02f, 1.083010600e+02f, 1.068634000e+02f,
                1.216898200e+02f, 1.202470150e+02f, 1.188042100e+02f, 1.173614050e+02f, 1.159186000e+02f, 1.144757950e+02f, 1.130329900e+02f, 1.115901850e+02f, 1.101473800e+02f, 1.087045750e+02f, 1.072617700e+02f,
                1.221396400e+02f, 1.206916900e+02f, 1.192437400e+02f, 1.177957900e+02f, 1.163478400e+02f, 1.148998900e+02f, 1.134519400e+02f, 1.120039900e+02f, 1.105560400e+02f, 1.091080900e+02f, 1.076601400e+02f,
                1.225894600e+02f, 1.211363650e+02f, 1.196832700e+02f, 1.182301750e+02f, 1.167770800e+02f, 1.153239850e+02f, 1.138708900e+02f, 1.124177950e+02f, 1.109647000e+02f, 1.095116050e+02f, 1.080585100e+02f,
                1.230392800e+02f, 1.215810400e+02f, 1.201228000e+02f, 1.186645600e+02f, 1.172063200e+02f, 1.157480800e+02f, 1.142898400e+02f, 1.128316000e+02f, 1.113733600e+02f, 1.099151200e+02f, 1.084568800e+02f,
                1.234891000e+02f, 1.220257150e+02f, 1.205623300e+02f, 1.190989450e+02f, 1.176355600e+02f, 1.161721750e+02f, 1.147087900e+02f, 1.132454050e+02f, 1.117820200e+02f, 1.103186350e+02f, 1.088552500e+02f,
                1.248385600e+02f, 1.233597400e+02f, 1.218809200e+02f, 1.204021000e+02f, 1.189232800e+02f, 1.174444600e+02f, 1.159656400e+02f, 1.144868200e+02f, 1.130080000e+02f, 1.115291800e+02f, 1.100503600e+02f,
                1.252883800e+02f, 1.238044150e+02f, 1.223204500e+02f, 1.208364850e+02f, 1.193525200e+02f, 1.178685550e+02f, 1.163845900e+02f, 1.149006250e+02f, 1.134166600e+02f, 1.119326950e+02f, 1.104487300e+02f,
                1.257382000e+02f, 1.242490900e+02f, 1.227599800e+02f, 1.212708700e+02f, 1.197817600e+02f, 1.182926500e+02f, 1.168035400e+02f, 1.153144300e+02f, 1.138253200e+02f, 1.123362100e+02f, 1.108471000e+02f,
                1.261880200e+02f, 1.246937650e+02f, 1.231995100e+02f, 1.217052550e+02f, 1.202110000e+02f, 1.187167450e+02f, 1.172224900e+02f, 1.157282350e+02f, 1.142339800e+02f, 1.127397250e+02f, 1.112454700e+02f,
                1.266378400e+02f, 1.251384400e+02f, 1.236390400e+02f, 1.221396400e+02f, 1.206402400e+02f, 1.191408400e+02f, 1.176414400e+02f, 1.161420400e+02f, 1.146426400e+02f, 1.131432400e+02f, 1.116438400e+02f,
                1.270876600e+02f, 1.255831150e+02f, 1.240785700e+02f, 1.225740250e+02f, 1.210694800e+02f, 1.195649350e+02f, 1.180603900e+02f, 1.165558450e+02f, 1.150513000e+02f, 1.135467550e+02f, 1.120422100e+02f,
                1.275374800e+02f, 1.260277900e+02f, 1.245181000e+02f, 1.230084100e+02f, 1.214987200e+02f, 1.199890300e+02f, 1.184793400e+02f, 1.169696500e+02f, 1.154599600e+02f, 1.139502700e+02f, 1.124405800e+02f,
                1.279873000e+02f, 1.264724650e+02f, 1.249576300e+02f, 1.234427950e+02f, 1.219279600e+02f, 1.204131250e+02f, 1.188982900e+02f, 1.173834550e+02f, 1.158686200e+02f, 1.143537850e+02f, 1.128389500e+02f,
                1.284371200e+02f, 1.269171400e+02f, 1.253971600e+02f, 1.238771800e+02f, 1.223572000e+02f, 1.208372200e+02f, 1.193172400e+02f, 1.177972600e+02f, 1.162772800e+02f, 1.147573000e+02f, 1.132373200e+02f,
                1.288869400e+02f, 1.273618150e+02f, 1.258366900e+02f, 1.243115650e+02f, 1.227864400e+02f, 1.212613150e+02f, 1.197361900e+02f, 1.182110650e+02f, 1.166859400e+02f, 1.151608150e+02f, 1.136356900e+02f,
                1.293367600e+02f, 1.278064900e+02f, 1.262762200e+02f, 1.247459500e+02f, 1.232156800e+02f, 1.216854100e+02f, 1.201551400e+02f, 1.186248700e+02f, 1.170946000e+02f, 1.155643300e+02f, 1.140340600e+02f,
                1.306862200e+02f, 1.291405150e+02f, 1.275948100e+02f, 1.260491050e+02f, 1.245034000e+02f, 1.229576950e+02f, 1.214119900e+02f, 1.198662850e+02f, 1.183205800e+02f, 1.167748750e+02f, 1.152291700e+02f,
                1.311360400e+02f, 1.295851900e+02f, 1.280343400e+02f, 1.264834900e+02f, 1.249326400e+02f, 1.233817900e+02f, 1.218309400e+02f, 1.202800900e+02f, 1.187292400e+02f, 1.171783900e+02f, 1.156275400e+02f,
                1.315858600e+02f, 1.300298650e+02f, 1.284738700e+02f, 1.269178750e+02f, 1.253618800e+02f, 1.238058850e+02f, 1.222498900e+02f, 1.206938950e+02f, 1.191379000e+02f, 1.175819050e+02f, 1.160259100e+02f,
                1.320356800e+02f, 1.304745400e+02f, 1.289134000e+02f, 1.273522600e+02f, 1.257911200e+02f, 1.242299800e+02f, 1.226688400e+02f, 1.211077000e+02f, 1.195465600e+02f, 1.179854200e+02f, 1.164242800e+02f,
                1.324855000e+02f, 1.309192150e+02f, 1.293529300e+02f, 1.277866450e+02f, 1.262203600e+02f, 1.246540750e+02f, 1.230877900e+02f, 1.215215050e+02f, 1.199552200e+02f, 1.183889350e+02f, 1.168226500e+02f,
                1.329353200e+02f, 1.313638900e+02f, 1.297924600e+02f, 1.282210300e+02f, 1.266496000e+02f, 1.250781700e+02f, 1.235067400e+02f, 1.219353100e+02f, 1.203638800e+02f, 1.187924500e+02f, 1.172210200e+02f,
                1.333851400e+02f, 1.318085650e+02f, 1.302319900e+02f, 1.286554150e+02f, 1.270788400e+02f, 1.255022650e+02f, 1.239256900e+02f, 1.223491150e+02f, 1.207725400e+02f, 1.191959650e+02f, 1.176193900e+02f,
                1.338349600e+02f, 1.322532400e+02f, 1.306715200e+02f, 1.290898000e+02f, 1.275080800e+02f, 1.259263600e+02f, 1.243446400e+02f, 1.227629200e+02f, 1.211812000e+02f, 1.195994800e+02f, 1.180177600e+02f,
                1.342847800e+02f, 1.326979150e+02f, 1.311110500e+02f, 1.295241850e+02f, 1.279373200e+02f, 1.263504550e+02f, 1.247635900e+02f, 1.231767250e+02f, 1.215898600e+02f, 1.200029950e+02f, 1.184161300e+02f,
                1.347346000e+02f, 1.331425900e+02f, 1.315505800e+02f, 1.299585700e+02f, 1.283665600e+02f, 1.267745500e+02f, 1.251825400e+02f, 1.235905300e+02f, 1.219985200e+02f, 1.204065100e+02f, 1.188145000e+02f,
                1.351844200e+02f, 1.335872650e+02f, 1.319901100e+02f, 1.303929550e+02f, 1.287958000e+02f, 1.271986450e+02f, 1.256014900e+02f, 1.240043350e+02f, 1.224071800e+02f, 1.208100250e+02f, 1.192128700e+02f,
                1.365338800e+02f, 1.349212900e+02f, 1.333087000e+02f, 1.316961100e+02f, 1.300835200e+02f, 1.284709300e+02f, 1.268583400e+02f, 1.252457500e+02f, 1.236331600e+02f, 1.220205700e+02f, 1.204079800e+02f,
                1.369837000e+02f, 1.353659650e+02f, 1.337482300e+02f, 1.321304950e+02f, 1.305127600e+02f, 1.288950250e+02f, 1.272772900e+02f, 1.256595550e+02f, 1.240418200e+02f, 1.224240850e+02f, 1.208063500e+02f,
                1.374335200e+02f, 1.358106400e+02f, 1.341877600e+02f, 1.325648800e+02f, 1.309420000e+02f, 1.293191200e+02f, 1.276962400e+02f, 1.260733600e+02f, 1.244504800e+02f, 1.228276000e+02f, 1.212047200e+02f,
                1.378833400e+02f, 1.362553150e+02f, 1.346272900e+02f, 1.329992650e+02f, 1.313712400e+02f, 1.297432150e+02f, 1.281151900e+02f, 1.264871650e+02f, 1.248591400e+02f, 1.232311150e+02f, 1.216030900e+02f,
                1.383331600e+02f, 1.366999900e+02f, 1.350668200e+02f, 1.334336500e+02f, 1.318004800e+02f, 1.301673100e+02f, 1.285341400e+02f, 1.269009700e+02f, 1.252678000e+02f, 1.236346300e+02f, 1.220014600e+02f,
                1.387829800e+02f, 1.371446650e+02f, 1.355063500e+02f, 1.338680350e+02f, 1.322297200e+02f, 1.305914050e+02f, 1.289530900e+02f, 1.273147750e+02f, 1.256764600e+02f, 1.240381450e+02f, 1.223998300e+02f,
                1.392328000e+02f, 1.375893400e+02f, 1.359458800e+02f, 1.343024200e+02f, 1.326589600e+02f, 1.310155000e+02f, 1.293720400e+02f, 1.277285800e+02f, 1.260851200e+02f, 1.244416600e+02f, 1.227982000e+02f,
                1.396826200e+02f, 1.380340150e+02f, 1.363854100e+02f, 1.347368050e+02f, 1.330882000e+02f, 1.314395950e+02f, 1.297909900e+02f, 1.281423850e+02f, 1.264937800e+02f, 1.248451750e+02f, 1.231965700e+02f,
                1.401324400e+02f, 1.384786900e+02f, 1.368249400e+02f, 1.351711900e+02f, 1.335174400e+02f, 1.318636900e+02f, 1.302099400e+02f, 1.285561900e+02f, 1.269024400e+02f, 1.252486900e+02f, 1.235949400e+02f,
                1.405822600e+02f, 1.389233650e+02f, 1.372644700e+02f, 1.356055750e+02f, 1.339466800e+02f, 1.322877850e+02f, 1.306288900e+02f, 1.289699950e+02f, 1.273111000e+02f, 1.256522050e+02f, 1.239933100e+02f,
                1.410320800e+02f, 1.393680400e+02f, 1.377040000e+02f, 1.360399600e+02f, 1.343759200e+02f, 1.327118800e+02f, 1.310478400e+02f, 1.293838000e+02f, 1.277197600e+02f, 1.260557200e+02f, 1.243916800e+02f,
                1.423815400e+02f, 1.407020650e+02f, 1.390225900e+02f, 1.373431150e+02f, 1.356636400e+02f, 1.339841650e+02f, 1.323046900e+02f, 1.306252150e+02f, 1.289457400e+02f, 1.272662650e+02f, 1.255867900e+02f,
                1.428313600e+02f, 1.411467400e+02f, 1.394621200e+02f, 1.377775000e+02f, 1.360928800e+02f, 1.344082600e+02f, 1.327236400e+02f, 1.310390200e+02f, 1.293544000e+02f, 1.276697800e+02f, 1.259851600e+02f,
                1.432811800e+02f, 1.415914150e+02f, 1.399016500e+02f, 1.382118850e+02f, 1.365221200e+02f, 1.348323550e+02f, 1.331425900e+02f, 1.314528250e+02f, 1.297630600e+02f, 1.280732950e+02f, 1.263835300e+02f,
                1.437310000e+02f, 1.420360900e+02f, 1.403411800e+02f, 1.386462700e+02f, 1.369513600e+02f, 1.352564500e+02f, 1.335615400e+02f, 1.318666300e+02f, 1.301717200e+02f, 1.284768100e+02f, 1.267819000e+02f,
                1.441808200e+02f, 1.424807650e+02f, 1.407807100e+02f, 1.390806550e+02f, 1.373806000e+02f, 1.356805450e+02f, 1.339804900e+02f, 1.322804350e+02f, 1.305803800e+02f, 1.288803250e+02f, 1.271802700e+02f,
                1.446306400e+02f, 1.429254400e+02f, 1.412202400e+02f, 1.395150400e+02f, 1.378098400e+02f, 1.361046400e+02f, 1.343994400e+02f, 1.326942400e+02f, 1.309890400e+02f, 1.292838400e+02f, 1.275786400e+02f,
                1.450804600e+02f, 1.433701150e+02f, 1.416597700e+02f, 1.399494250e+02f, 1.382390800e+02f, 1.365287350e+02f, 1.348183900e+02f, 1.331080450e+02f, 1.313977000e+02f, 1.296873550e+02f, 1.279770100e+02f,
                1.455302800e+02f, 1.438147900e+02f, 1.420993000e+02f, 1.403838100e+02f, 1.386683200e+02f, 1.369528300e+02f, 1.352373400e+02f, 1.335218500e+02f, 1.318063600e+02f, 1.300908700e+02f, 1.283753800e+02f,
                1.459801000e+02f, 1.442594650e+02f, 1.425388300e+02f, 1.408181950e+02f, 1.390975600e+02f, 1.373769250e+02f, 1.356562900e+02f, 1.339356550e+02f, 1.322150200e+02f, 1.304943850e+02f, 1.287737500e+02f,
                1.464299200e+02f, 1.447041400e+02f, 1.429783600e+02f, 1.412525800e+02f, 1.395268000e+02f, 1.378010200e+02f, 1.360752400e+02f, 1.343494600e+02f, 1.326236800e+02f, 1.308979000e+02f, 1.291721200e+02f,
                1.468797400e+02f, 1.451488150e+02f, 1.434178900e+02f, 1.416869650e+02f, 1.399560400e+02f, 1.382251150e+02f, 1.364941900e+02f, 1.347632650e+02f, 1.330323400e+02f, 1.313014150e+02f, 1.295704900e+02f,
                1.482292000e+02f, 1.464828400e+02f, 1.447364800e+02f, 1.429901200e+02f, 1.412437600e+02f, 1.394974000e+02f, 1.377510400e+02f, 1.360046800e+02f, 1.342583200e+02f, 1.325119600e+02f, 1.307656000e+02f,
                1.486790200e+02f, 1.469275150e+02f, 1.451760100e+02f, 1.434245050e+02f, 1.416730000e+02f, 1.399214950e+02f, 1.381699900e+02f, 1.364184850e+02f, 1.346669800e+02f, 1.329154750e+02f, 1.311639700e+02f,
                1.491288400e+02f, 1.473721900e+02f, 1.456155400e+02f, 1.438588900e+02f, 1.421022400e+02f, 1.403455900e+02f, 1.385889400e+02f, 1.368322900e+02f, 1.350756400e+02f, 1.333189900e+02f, 1.315623400e+02f,
                1.495786600e+02f, 1.478168650e+02f, 1.460550700e+02f, 1.442932750e+02f, 1.425314800e+02f, 1.407696850e+02f, 1.390078900e+02f, 1.372460950e+02f, 1.354843000e+02f, 1.337225050e+02f, 1.319607100e+02f,
                1.500284800e+02f, 1.482615400e+02f, 1.464946000e+02f, 1.447276600e+02f, 1.429607200e+02f, 1.411937800e+02f, 1.394268400e+02f, 1.376599000e+02f, 1.358929600e+02f, 1.341260200e+02f, 1.323590800e+02f,
                1.504783000e+02f, 1.487062150e+02f, 1.469341300e+02f, 1.451620450e+02f, 1.433899600e+02f, 1.416178750e+02f, 1.398457900e+02f, 1.380737050e+02f, 1.363016200e+02f, 1.345295350e+02f, 1.327574500e+02f,
                1.509281200e+02f, 1.491508900e+02f, 1.473736600e+02f, 1.455964300e+02f, 1.438192000e+02f, 1.420419700e+02f, 1.402647400e+02f, 1.384875100e+02f, 1.367102800e+02f, 1.349330500e+02f, 1.331558200e+02f,
                1.513779400e+02f, 1.495955650e+02f, 1.478131900e+02f, 1.460308150e+02f, 1.442484400e+02f, 1.424660650e+02f, 1.406836900e+02f, 1.389013150e+02f, 1.371189400e+02f, 1.353365650e+02f, 1.335541900e+02f,
                1.518277600e+02f, 1.500402400e+02f, 1.482527200e+02f, 1.464652000e+02f, 1.446776800e+02f, 1.428901600e+02f, 1.411026400e+02f, 1.393151200e+02f, 1.375276000e+02f, 1.357400800e+02f, 1.339525600e+02f,
                1.522775800e+02f, 1.504849150e+02f, 1.486922500e+02f, 1.468995850e+02f, 1.451069200e+02f, 1.433142550e+02f, 1.415215900e+02f, 1.397289250e+02f, 1.379362600e+02f, 1.361435950e+02f, 1.343509300e+02f,
                1.527274000e+02f, 1.509295900e+02f, 1.491317800e+02f, 1.473339700e+02f, 1.455361600e+02f, 1.437383500e+02f, 1.419405400e+02f, 1.401427300e+02f, 1.383449200e+02f, 1.365471100e+02f, 1.347493000e+02f,
                1.540768600e+02f, 1.522636150e+02f, 1.504503700e+02f, 1.486371250e+02f, 1.468238800e+02f, 1.450106350e+02f, 1.431973900e+02f, 1.413841450e+02f, 1.395709000e+02f, 1.377576550e+02f, 1.359444100e+02f,
                1.545266800e+02f, 1.527082900e+02f, 1.508899000e+02f, 1.490715100e+02f, 1.472531200e+02f, 1.454347300e+02f, 1.436163400e+02f, 1.417979500e+02f, 1.399795600e+02f, 1.381611700e+02f, 1.363427800e+02f,
                1.549765000e+02f, 1.531529650e+02f, 1.513294300e+02f, 1.495058950e+02f, 1.476823600e+02f, 1.458588250e+02f, 1.440352900e+02f, 1.422117550e+02f, 1.403882200e+02f, 1.385646850e+02f, 1.367411500e+02f,
                1.554263200e+02f, 1.535976400e+02f, 1.517689600e+02f, 1.499402800e+02f, 1.481116000e+02f, 1.462829200e+02f, 1.444542400e+02f, 1.426255600e+02f, 1.407968800e+02f, 1.389682000e+02f, 1.371395200e+02f,
                1.558761400e+02f, 1.540423150e+02f, 1.522084900e+02f, 1.503746650e+02f, 1.485408400e+02f, 1.467070150e+02f, 1.448731900e+02f, 1.430393650e+02f, 1.412055400e+02f, 1.393717150e+02f, 1.375378900e+02f,
                1.563259600e+02f, 1.544869900e+02f, 1.526480200e+02f, 1.508090500e+02f, 1.489700800e+02f, 1.471311100e+02f, 1.452921400e+02f, 1.434531700e+02f, 1.416142000e+02f, 1.397752300e+02f, 1.379362600e+02f,
                1.567757800e+02f, 1.549316650e+02f, 1.530875500e+02f, 1.512434350e+02f, 1.493993200e+02f, 1.475552050e+02f, 1.457110900e+02f, 1.438669750e+02f, 1.420228600e+02f, 1.401787450e+02f, 1.383346300e+02f,
                1.572256000e+02f, 1.553763400e+02f, 1.535270800e+02f, 1.516778200e+02f, 1.498285600e+02f, 1.479793000e+02f, 1.461300400e+02f, 1.442807800e+02f, 1.424315200e+02f, 1.405822600e+02f, 1.387330000e+02f,
                1.576754200e+02f, 1.558210150e+02f, 1.539666100e+02f, 1.521122050e+02f, 1.502578000e+02f, 1.484033950e+02f, 1.465489900e+02f, 1.446945850e+02f, 1.428401800e+02f, 1.409857750e+02f, 1.391313700e+02f,
                1.581252400e+02f, 1.562656900e+02f, 1.544061400e+02f, 1.525465900e+02f, 1.506870400e+02f, 1.488274900e+02f, 1.469679400e+02f, 1.451083900e+02f, 1.432488400e+02f, 1.413892900e+02f, 1.395297400e+02f,
                1.585750600e+02f, 1.567103650e+02f, 1.548456700e+02f, 1.529809750e+02f, 1.511162800e+02f, 1.492515850e+02f, 1.473868900e+02f, 1.455221950e+02f, 1.436575000e+02f, 1.417928050e+02f, 1.399281100e+02f,
                1.599245200e+02f, 1.580443900e+02f, 1.561642600e+02f, 1.542841300e+02f, 1.524040000e+02f, 1.505238700e+02f, 1.486437400e+02f, 1.467636100e+02f, 1.448834800e+02f, 1.430033500e+02f, 1.411232200e+02f,
                1.603743400e+02f, 1.584890650e+02f, 1.566037900e+02f, 1.547185150e+02f, 1.528332400e+02f, 1.509479650e+02f, 1.490626900e+02f, 1.471774150e+02f, 1.452921400e+02f, 1.434068650e+02f, 1.415215900e+02f,
                1.608241600e+02f, 1.589337400e+02f, 1.570433200e+02f, 1.551529000e+02f, 1.532624800e+02f, 1.513720600e+02f, 1.494816400e+02f, 1.475912200e+02f, 1.457008000e+02f, 1.438103800e+02f, 1.419199600e+02f,
                1.612739800e+02f, 1.593784150e+02f, 1.574828500e+02f, 1.555872850e+02f, 1.536917200e+02f, 1.517961550e+02f, 1.499005900e+02f, 1.480050250e+02f, 1.461094600e+02f, 1.442138950e+02f, 1.423183300e+02f,
                1.617238000e+02f, 1.598230900e+02f, 1.579223800e+02f, 1.560216700e+02f, 1.541209600e+02f, 1.522202500e+02f, 1.503195400e+02f, 1.484188300e+02f, 1.465181200e+02f, 1.446174100e+02f, 1.427167000e+02f,
                1.621736200e+02f, 1.602677650e+02f, 1.583619100e+02f, 1.564560550e+02f, 1.545502000e+02f, 1.526443450e+02f, 1.507384900e+02f, 1.488326350e+02f, 1.469267800e+02f, 1.450209250e+02f, 1.431150700e+02f,
                1.626234400e+02f, 1.607124400e+02f, 1.588014400e+02f, 1.568904400e+02f, 1.549794400e+02f, 1.530684400e+02f, 1.511574400e+02f, 1.492464400e+02f, 1.473354400e+02f, 1.454244400e+02f, 1.435134400e+02f,
                1.630732600e+02f, 1.611571150e+02f, 1.592409700e+02f, 1.573248250e+02f, 1.554086800e+02f, 1.534925350e+02f, 1.515763900e+02f, 1.496602450e+02f, 1.477441000e+02f, 1.458279550e+02f, 1.439118100e+02f,
                1.635230800e+02f, 1.616017900e+02f, 1.596805000e+02f, 1.577592100e+02f, 1.558379200e+02f, 1.539166300e+02f, 1.519953400e+02f, 1.500740500e+02f, 1.481527600e+02f, 1.462314700e+02f, 1.443101800e+02f,
                1.639729000e+02f, 1.620464650e+02f, 1.601200300e+02f, 1.581935950e+02f, 1.562671600e+02f, 1.543407250e+02f, 1.524142900e+02f, 1.504878550e+02f, 1.485614200e+02f, 1.466349850e+02f, 1.447085500e+02f,
                1.644227200e+02f, 1.624911400e+02f, 1.605595600e+02f, 1.586279800e+02f, 1.566964000e+02f, 1.547648200e+02f, 1.528332400e+02f, 1.509016600e+02f, 1.489700800e+02f, 1.470385000e+02f, 1.451069200e+02f,
                1.657721800e+02f, 1.638251650e+02f, 1.618781500e+02f, 1.599311350e+02f, 1.579841200e+02f, 1.560371050e+02f, 1.540900900e+02f, 1.521430750e+02f, 1.501960600e+02f, 1.482490450e+02f, 1.463020300e+02f,
                1.662220000e+02f, 1.642698400e+02f, 1.623176800e+02f, 1.603655200e+02f, 1.584133600e+02f, 1.564612000e+02f, 1.545090400e+02f, 1.525568800e+02f, 1.506047200e+02f, 1.486525600e+02f, 1.467004000e+02f,
                1.666718200e+02f, 1.647145150e+02f, 1.627572100e+02f, 1.607999050e+02f, 1.588426000e+02f, 1.568852950e+02f, 1.549279900e+02f, 1.529706850e+02f, 1.510133800e+02f, 1.490560750e+02f, 1.470987700e+02f,
                1.671216400e+02f, 1.651591900e+02f, 1.631967400e+02f, 1.612342900e+02f, 1.592718400e+02f, 1.573093900e+02f, 1.553469400e+02f, 1.533844900e+02f, 1.514220400e+02f, 1.494595900e+02f, 1.474971400e+02f,
                1.675714600e+02f, 1.656038650e+02f, 1.636362700e+02f, 1.616686750e+02f, 1.597010800e+02f, 1.577334850e+02f, 1.557658900e+02f, 1.537982950e+02f, 1.518307000e+02f, 1.498631050e+02f, 1.478955100e+02f,
                1.680212800e+02f, 1.660485400e+02f, 1.640758000e+02f, 1.621030600e+02f, 1.601303200e+02f, 1.581575800e+02f, 1.561848400e+02f, 1.542121000e+02f, 1.522393600e+02f, 1.502666200e+02f, 1.482938800e+02f,
                1.684711000e+02f, 1.664932150e+02f, 1.645153300e+02f, 1.625374450e+02f, 1.605595600e+02f, 1.585816750e+02f, 1.566037900e+02f, 1.546259050e+02f, 1.526480200e+02f, 1.506701350e+02f, 1.486922500e+02f,
                1.689209200e+02f, 1.669378900e+02f, 1.649548600e+02f, 1.629718300e+02f, 1.609888000e+02f, 1.590057700e+02f, 1.570227400e+02f, 1.550397100e+02f, 1.530566800e+02f, 1.510736500e+02f, 1.490906200e+02f,
                1.693707400e+02f, 1.673825650e+02f, 1.653943900e+02f, 1.634062150e+02f, 1.614180400e+02f, 1.594298650e+02f, 1.574416900e+02f, 1.554535150e+02f, 1.534653400e+02f, 1.514771650e+02f, 1.494889900e+02f,
                1.698205600e+02f, 1.678272400e+02f, 1.658339200e+02f, 1.638406000e+02f, 1.618472800e+02f, 1.598539600e+02f, 1.578606400e+02f, 1.558673200e+02f, 1.538740000e+02f, 1.518806800e+02f, 1.498873600e+02f,
                1.702703800e+02f, 1.682719150e+02f, 1.662734500e+02f, 1.642749850e+02f, 1.622765200e+02f, 1.602780550e+02f, 1.582795900e+02f, 1.562811250e+02f, 1.542826600e+02f, 1.522841950e+02f, 1.502857300e+02f,
                1.716198400e+02f, 1.696059400e+02f, 1.675920400e+02f, 1.655781400e+02f, 1.635642400e+02f, 1.615503400e+02f, 1.595364400e+02f, 1.575225400e+02f, 1.555086400e+02f, 1.534947400e+02f, 1.514808400e+02f,
                1.720696600e+02f, 1.700506150e+02f, 1.680315700e+02f, 1.660125250e+02f, 1.639934800e+02f, 1.619744350e+02f, 1.599553900e+02f, 1.579363450e+02f, 1.559173000e+02f, 1.538982550e+02f, 1.518792100e+02f,
                1.725194800e+02f, 1.704952900e+02f, 1.684711000e+02f, 1.664469100e+02f, 1.644227200e+02f, 1.623985300e+02f, 1.603743400e+02f, 1.583501500e+02f, 1.563259600e+02f, 1.543017700e+02f, 1.522775800e+02f,
                1.729693000e+02f, 1.709399650e+02f, 1.689106300e+02f, 1.668812950e+02f, 1.648519600e+02f, 1.628226250e+02f, 1.607932900e+02f, 1.587639550e+02f, 1.567346200e+02f, 1.547052850e+02f, 1.526759500e+02f,
                1.734191200e+02f, 1.713846400e+02f, 1.693501600e+02f, 1.673156800e+02f, 1.652812000e+02f, 1.632467200e+02f, 1.612122400e+02f, 1.591777600e+02f, 1.571432800e+02f, 1.551088000e+02f, 1.530743200e+02f,
                1.738689400e+02f, 1.718293150e+02f, 1.697896900e+02f, 1.677500650e+02f, 1.657104400e+02f, 1.636708150e+02f, 1.616311900e+02f, 1.595915650e+02f, 1.575519400e+02f, 1.555123150e+02f, 1.534726900e+02f,
                1.743187600e+02f, 1.722739900e+02f, 1.702292200e+02f, 1.681844500e+02f, 1.661396800e+02f, 1.640949100e+02f, 1.620501400e+02f, 1.600053700e+02f, 1.579606000e+02f, 1.559158300e+02f, 1.538710600e+02f,
                1.747685800e+02f, 1.727186650e+02f, 1.706687500e+02f, 1.686188350e+02f, 1.665689200e+02f, 1.645190050e+02f, 1.624690900e+02f, 1.604191750e+02f, 1.583692600e+02f, 1.563193450e+02f, 1.542694300e+02f,
                1.752184000e+02f, 1.731633400e+02f, 1.711082800e+02f, 1.690532200e+02f, 1.669981600e+02f, 1.649431000e+02f, 1.628880400e+02f, 1.608329800e+02f, 1.587779200e+02f, 1.567228600e+02f, 1.546678000e+02f,
                1.756682200e+02f, 1.736080150e+02f, 1.715478100e+02f, 1.694876050e+02f, 1.674274000e+02f, 1.653671950e+02f, 1.633069900e+02f, 1.612467850e+02f, 1.591865800e+02f, 1.571263750e+02f, 1.550661700e+02f,
                1.761180400e+02f, 1.740526900e+02f, 1.719873400e+02f, 1.699219900e+02f, 1.678566400e+02f, 1.657912900e+02f, 1.637259400e+02f, 1.616605900e+02f, 1.595952400e+02f, 1.575298900e+02f, 1.554645400e+02f,
                1.774675000e+02f, 1.753867150e+02f, 1.733059300e+02f, 1.712251450e+02f, 1.691443600e+02f, 1.670635750e+02f, 1.649827900e+02f, 1.629020050e+02f, 1.608212200e+02f, 1.587404350e+02f, 1.566596500e+02f,
                1.779173200e+02f, 1.758313900e+02f, 1.737454600e+02f, 1.716595300e+02f, 1.695736000e+02f, 1.674876700e+02f, 1.654017400e+02f, 1.633158100e+02f, 1.612298800e+02f, 1.591439500e+02f, 1.570580200e+02f,
                1.783671400e+02f, 1.762760650e+02f, 1.741849900e+02f, 1.720939150e+02f, 1.700028400e+02f, 1.679117650e+02f, 1.658206900e+02f, 1.637296150e+02f, 1.616385400e+02f, 1.595474650e+02f, 1.574563900e+02f,
                1.788169600e+02f, 1.767207400e+02f, 1.746245200e+02f, 1.725283000e+02f, 1.704320800e+02f, 1.683358600e+02f, 1.662396400e+02f, 1.641434200e+02f, 1.620472000e+02f, 1.599509800e+02f, 1.578547600e+02f,
                1.792667800e+02f, 1.771654150e+02f, 1.750640500e+02f, 1.729626850e+02f, 1.708613200e+02f, 1.687599550e+02f, 1.666585900e+02f, 1.645572250e+02f, 1.624558600e+02f, 1.603544950e+02f, 1.582531300e+02f,
                1.797166000e+02f, 1.776100900e+02f, 1.755035800e+02f, 1.733970700e+02f, 1.712905600e+02f, 1.691840500e+02f, 1.670775400e+02f, 1.649710300e+02f, 1.628645200e+02f, 1.607580100e+02f, 1.586515000e+02f,
                1.801664200e+02f, 1.780547650e+02f, 1.759431100e+02f, 1.738314550e+02f, 1.717198000e+02f, 1.696081450e+02f, 1.674964900e+02f, 1.653848350e+02f, 1.632731800e+02f, 1.611615250e+02f, 1.590498700e+02f,
                1.806162400e+02f, 1.784994400e+02f, 1.763826400e+02f, 1.742658400e+02f, 1.721490400e+02f, 1.700322400e+02f, 1.679154400e+02f, 1.657986400e+02f, 1.636818400e+02f, 1.615650400e+02f, 1.594482400e+02f,
                1.810660600e+02f, 1.789441150e+02f, 1.768221700e+02f, 1.747002250e+02f, 1.725782800e+02f, 1.704563350e+02f, 1.683343900e+02f, 1.662124450e+02f, 1.640905000e+02f, 1.619685550e+02f, 1.598466100e+02f,
                1.815158800e+02f, 1.793887900e+02f, 1.772617000e+02f, 1.751346100e+02f, 1.730075200e+02f, 1.708804300e+02f, 1.687533400e+02f, 1.666262500e+02f, 1.644991600e+02f, 1.623720700e+02f, 1.602449800e+02f,
                1.819657000e+02f, 1.798334650e+02f, 1.777012300e+02f, 1.755689950e+02f, 1.734367600e+02f, 1.713045250e+02f, 1.691722900e+02f, 1.670400550e+02f, 1.649078200e+02f, 1.627755850e+02f, 1.606433500e+02f,
            };

            float[] y_actual = y.ToArray();

            AssertError.Tolerance(y_expect, y_actual, 1e-7f, 1e-5f, $"mismatch value {inchannels},{outchannels},{kwidth},{kheight},{inwidth},{inheight},{batch}");
        }
    }
}
