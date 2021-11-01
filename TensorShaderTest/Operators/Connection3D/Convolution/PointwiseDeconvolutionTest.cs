using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;
using TensorShader;
using TensorShader.Operators.Connection3D;
using TensorShaderCudaBackend.API;

namespace TensorShaderTest.Operators.Connection3D {
    [TestClass]
    public class PointwiseDeconvolutionTest {
        [TestMethod]
        public void ExecuteFPTest() {
            TensorShaderCudaBackend.Environment.Precision = TensorShaderCudaBackend.Environment.PrecisionMode.Float;
            TensorShaderCudaBackend.Environment.CudnnEnabled = false;

            float max_err = 0;

            foreach (int batch in new int[] { 1, 2 }) {
                foreach (int inchannels in new int[] { 1, 2, 3, 4, 5, 10, 15, 20 }) {
                    foreach (int outchannels in new int[] { 7, 13 }) {
                        foreach ((int width, int height, int depth) in new (int, int, int)[] { (13, 13, 13), (17, 17, 17), (19, 19, 19), (17, 19, 13), (13, 17, 19), (19, 13, 17) }) {
                            float[] yval = (new float[width * height * depth * outchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
                            float[] wval = (new float[inchannels * outchannels]).Select((_, idx) => (idx + 1) * 1e-3f).Reverse().ToArray();

                            Map3D y = new(outchannels, width, height, depth, batch, yval);
                            Filter3D w = new(inchannels, outchannels, 1, 1, 1, wval);

                            Map3D x = Reference(y, w);

                            OverflowCheckedTensor y_tensor = new(Shape.Map3D(outchannels, width, height, depth, batch), yval);
                            OverflowCheckedTensor w_tensor = new(Shape.Kernel0D(inchannels, outchannels), wval);

                            OverflowCheckedTensor x_tensor = new(Shape.Map3D(inchannels, width, height, depth, batch));

                            PointwiseDeconvolution ope = new(width, height, depth, outchannels, inchannels, batch);

                            ope.Execute(y_tensor, w_tensor, x_tensor);

                            float[] x_expect = x.ToArray();
                            float[] x_actual = x_tensor.State.Value;

                            CollectionAssert.AreEqual(yval, y_tensor.State.Value);
                            CollectionAssert.AreEqual(wval, w_tensor.State.Value);

                            AssertError.Tolerance(x_expect, x_actual, 1e-6f, 1e-4f, ref max_err, $"mismatch value {inchannels},{outchannels},{width},{height},{depth},{batch}");

                            Console.WriteLine($"pass: {inchannels},{outchannels},{width},{height},{depth},{batch}");
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
                foreach (int inchannels in new int[] { 1, 2, 3, 4, 5, 10, 15, 20 }) {
                    foreach (int outchannels in new int[] { 7, 13 }) {
                        foreach ((int width, int height, int depth) in new (int, int, int)[] { (13, 13, 13), (17, 17, 17), (19, 19, 19), (17, 19, 13), (13, 17, 19), (19, 13, 17) }) {
                            float[] yval = (new float[width * height * depth * outchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
                            float[] wval = (new float[inchannels * outchannels]).Select((_, idx) => (idx + 1) * 1e-3f).Reverse().ToArray();

                            Map3D y = new(outchannels, width, height, depth, batch, yval);
                            Filter3D w = new(inchannels, outchannels, 1, 1, 1, wval);

                            Map3D x = Reference(y, w);

                            OverflowCheckedTensor y_tensor = new(Shape.Map3D(outchannels, width, height, depth, batch), yval);
                            OverflowCheckedTensor w_tensor = new(Shape.Kernel0D(inchannels, outchannels), wval);

                            OverflowCheckedTensor x_tensor = new(Shape.Map3D(inchannels, width, height, depth, batch));

                            PointwiseDeconvolution ope = new(width, height, depth, outchannels, inchannels, batch);

                            ope.Execute(y_tensor, w_tensor, x_tensor);

                            float[] x_expect = x.ToArray();
                            float[] x_actual = x_tensor.State.Value;

                            CollectionAssert.AreEqual(yval, y_tensor.State.Value);
                            CollectionAssert.AreEqual(wval, w_tensor.State.Value);

                            AssertError.Tolerance(x_expect, x_actual, 1e-7f, 1e-5f, ref max_err, $"mismatch value {inchannels},{outchannels},{width},{height},{depth},{batch}");

                            Console.WriteLine($"pass: {inchannels},{outchannels},{width},{height},{depth},{batch}");
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
                foreach (int inchannels in new int[] { 1, 2, 3, 4, 5, 10, 15, 20 }) {
                    foreach (int outchannels in new int[] { 7, 13 }) {
                        foreach ((int width, int height, int depth) in new (int, int, int)[] { (13, 13, 13), (17, 17, 17), (19, 19, 19), (17, 19, 13), (13, 17, 19), (19, 13, 17) }) {
                            float[] yval = (new float[width * height * depth * outchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
                            float[] wval = (new float[inchannels * outchannels]).Select((_, idx) => (idx + 1) * 1e-3f).Reverse().ToArray();

                            Map3D y = new(outchannels, width, height, depth, batch, yval);
                            Filter3D w = new(inchannels, outchannels, 1, 1, 1, wval);

                            Map3D x = Reference(y, w);

                            OverflowCheckedTensor y_tensor = new(Shape.Map3D(outchannels, width, height, depth, batch), yval);
                            OverflowCheckedTensor w_tensor = new(Shape.Kernel0D(inchannels, outchannels), wval);

                            OverflowCheckedTensor x_tensor = new(Shape.Map3D(inchannels, width, height, depth, batch));

                            PointwiseDeconvolution ope = new(width, height, depth, outchannels, inchannels, batch);

                            ope.Execute(y_tensor, w_tensor, x_tensor);

                            float[] x_expect = x.ToArray();
                            float[] x_actual = x_tensor.State.Value;

                            CollectionAssert.AreEqual(yval, y_tensor.State.Value);
                            CollectionAssert.AreEqual(wval, w_tensor.State.Value);

                            AssertError.Tolerance(x_expect, x_actual, 1e-6f, 1e-4f, ref max_err, $"mismatch value {inchannels},{outchannels},{width},{height},{depth},{batch}");

                            Console.WriteLine($"pass: {inchannels},{outchannels},{width},{height},{depth},{batch}");
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
            int width = 128, height = 196, depth = 4;

            float[] yval = (new float[width * height * depth * outchannels * batch]).Select((_, idx) => (float)random.NextDouble() * 1e-2f).ToArray();
            float[] wval = (new float[inchannels * outchannels]).Select((_, idx) => idx * (float)random.NextDouble() * 1e-2f).ToArray();

            Map3D y = new(outchannels, width, height, depth, batch, yval);
            Filter3D w = new(inchannels, outchannels, 1, 1, 1, wval);

            Map3D x = Reference(y, w);

            OverflowCheckedTensor y_tensor = new(Shape.Map3D(outchannels, width, height, depth, batch), yval);
            OverflowCheckedTensor w_tensor = new(Shape.Kernel0D(inchannels, outchannels), wval);

            OverflowCheckedTensor x_tensor = new(Shape.Map3D(inchannels, width, height, depth, batch));

            PointwiseDeconvolution ope = new(width, height, depth, outchannels, inchannels, batch);

            ope.Execute(y_tensor, w_tensor, x_tensor);

            float[] x_expect = x.ToArray();
            float[] x_actual = x_tensor.State.Value;

            CollectionAssert.AreEqual(yval, y_tensor.State.Value);
            CollectionAssert.AreEqual(wval, w_tensor.State.Value);

            AssertError.Tolerance(x_expect, x_actual, 1e-7f, 1e-5f, ref max_err, $"mismatch value {inchannels},{outchannels},{width},{height},{depth},{batch}");

            Console.WriteLine($"pass: {inchannels},{outchannels},{width},{height},{depth},{batch}");

            Console.WriteLine($"maxerr:{max_err}");
        }

        [TestMethod]
        public void SpeedFPTest() {
            TensorShaderCudaBackend.Environment.Precision = TensorShaderCudaBackend.Environment.PrecisionMode.Float;
            TensorShaderCudaBackend.Environment.CudnnEnabled = false;

            int width = 64, height = 64, depth = 64, inchannels = 31, outchannels = 31;

            OverflowCheckedTensor y_tensor = new(Shape.Map3D(outchannels, width, height, depth));
            OverflowCheckedTensor w_tensor = new(Shape.Kernel0D(inchannels, outchannels));

            OverflowCheckedTensor x_tensor = new(Shape.Map3D(inchannels, width, height, depth));

            PointwiseDeconvolution ope = new(width, height, depth, outchannels, inchannels);

            ope.Execute(y_tensor, w_tensor, x_tensor);

            Cuda.Profiler.Initialize("../../../../profiler.nvsetting", "../../nvprofiles/ptwise_deconvolution_3d_fp.nvvp");
            Cuda.Profiler.Start();

            ope.Execute(y_tensor, w_tensor, x_tensor);

            Cuda.Profiler.Stop();
        }

        [TestMethod]
        public void SpeedFFPTest() {
            TensorShaderCudaBackend.Environment.CudnnEnabled = false;
            TensorShaderCudaBackend.Environment.Precision = TensorShaderCudaBackend.Environment.PrecisionMode.FloatFloat;

            int width = 64, height = 64, depth = 64, inchannels = 31, outchannels = 31;

            OverflowCheckedTensor y_tensor = new(Shape.Map3D(outchannels, width, height, depth));
            OverflowCheckedTensor w_tensor = new(Shape.Kernel0D(inchannels, outchannels));

            OverflowCheckedTensor x_tensor = new(Shape.Map3D(inchannels, width, height, depth));

            PointwiseDeconvolution ope = new(width, height, depth, outchannels, inchannels);

            ope.Execute(y_tensor, w_tensor, x_tensor);

            Cuda.Profiler.Initialize("../../../../profiler.nvsetting", "../../nvprofiles/ptwise_deconvolution_3d_ffp.nvvp");
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

            int width = 64, height = 64, depth = 64, inchannels = 31, outchannels = 31;

            OverflowCheckedTensor y_tensor = new(Shape.Map3D(outchannels, width, height, depth));
            OverflowCheckedTensor w_tensor = new(Shape.Kernel0D(inchannels, outchannels));

            OverflowCheckedTensor x_tensor = new(Shape.Map3D(inchannels, width, height, depth));

            PointwiseDeconvolution ope = new(width, height, depth, outchannels, inchannels);

            ope.Execute(y_tensor, w_tensor, x_tensor);

            Cuda.Profiler.Initialize("../../../../profiler.nvsetting", "../../nvprofiles/ptwise_deconvolution_3d_cudnn.nvvp");
            Cuda.Profiler.Start();

            ope.Execute(y_tensor, w_tensor, x_tensor);

            Cuda.Profiler.Stop();
        }

        public static Map3D Reference(Map3D y, Filter3D w) {
            int inchannels = w.InChannels, outchannels = w.OutChannels, batch = y.Batch;
            int inw = y.Width, inh = y.Height, ind = y.Depth;

            Map3D x = new(inchannels, inw, inh, ind, batch);

            for (int th = 0; th < batch; th++) {
                for (int ix, iy, iz = 0; iz < ind; iz++) {
                    for (iy = 0; iy < inh; iy++) {
                        for (ix = 0; ix < inw; ix++) {
                            for (int outch = 0; outch < outchannels; outch++) {
                                double v = y[outch, ix, iy, iz, th];

                                for (int inch = 0; inch < inchannels; inch++) {
                                    x[inch, ix, iy, iz, th] += v * w[inch, outch, 0, 0, 0];
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
            int inchannels = 2, outchannels = 3, inwidth = 13, inheight = 12, indepth = 11, batch = 2;

            float[] yval = (new float[inwidth * inheight * indepth * outchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
            float[] wval = (new float[outchannels * inchannels]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

            Map3D y = new(outchannels, inwidth, inheight, indepth, batch, yval);
            Filter3D w = new(inchannels, outchannels, 1, 1, 1, wval);

            Map3D x = Reference(y, w);

            float[] x_expect = {
                5.000000000e-06f,  2.000000000e-06f,  3.200000000e-05f,  2.000000000e-05f,  5.900000000e-05f,  3.800000000e-05f,
                8.600000000e-05f,  5.600000000e-05f,  1.130000000e-04f,  7.400000000e-05f,  1.400000000e-04f,  9.200000000e-05f,
                1.670000000e-04f,  1.100000000e-04f,  1.940000000e-04f,  1.280000000e-04f,  2.210000000e-04f,  1.460000000e-04f,
                2.480000000e-04f,  1.640000000e-04f,  2.750000000e-04f,  1.820000000e-04f,  3.020000000e-04f,  2.000000000e-04f,
                3.290000000e-04f,  2.180000000e-04f,  3.560000000e-04f,  2.360000000e-04f,  3.830000000e-04f,  2.540000000e-04f,
                4.100000000e-04f,  2.720000000e-04f,  4.370000000e-04f,  2.900000000e-04f,  4.640000000e-04f,  3.080000000e-04f,
                4.910000000e-04f,  3.260000000e-04f,  5.180000000e-04f,  3.440000000e-04f,  5.450000000e-04f,  3.620000000e-04f,
                5.720000000e-04f,  3.800000000e-04f,  5.990000000e-04f,  3.980000000e-04f,  6.260000000e-04f,  4.160000000e-04f,
                6.530000000e-04f,  4.340000000e-04f,  6.800000000e-04f,  4.520000000e-04f,  7.070000000e-04f,  4.700000000e-04f,
                7.340000000e-04f,  4.880000000e-04f,  7.610000000e-04f,  5.060000000e-04f,  7.880000000e-04f,  5.240000000e-04f,
                8.150000000e-04f,  5.420000000e-04f,  8.420000000e-04f,  5.600000000e-04f,  8.690000000e-04f,  5.780000000e-04f,
                8.960000000e-04f,  5.960000000e-04f,  9.230000000e-04f,  6.140000000e-04f,  9.500000000e-04f,  6.320000000e-04f,
                9.770000000e-04f,  6.500000000e-04f,  1.004000000e-03f,  6.680000000e-04f,  1.031000000e-03f,  6.860000000e-04f,
                1.058000000e-03f,  7.040000000e-04f,  1.085000000e-03f,  7.220000000e-04f,  1.112000000e-03f,  7.400000000e-04f,
                1.139000000e-03f,  7.580000000e-04f,  1.166000000e-03f,  7.760000000e-04f,  1.193000000e-03f,  7.940000000e-04f,
                1.220000000e-03f,  8.120000000e-04f,  1.247000000e-03f,  8.300000000e-04f,  1.274000000e-03f,  8.480000000e-04f,
                1.301000000e-03f,  8.660000000e-04f,  1.328000000e-03f,  8.840000000e-04f,  1.355000000e-03f,  9.020000000e-04f,
                1.382000000e-03f,  9.200000000e-04f,  1.409000000e-03f,  9.380000000e-04f,  1.436000000e-03f,  9.560000000e-04f,
                1.463000000e-03f,  9.740000000e-04f,  1.490000000e-03f,  9.920000000e-04f,  1.517000000e-03f,  1.010000000e-03f,
                1.544000000e-03f,  1.028000000e-03f,  1.571000000e-03f,  1.046000000e-03f,  1.598000000e-03f,  1.064000000e-03f,
                1.625000000e-03f,  1.082000000e-03f,  1.652000000e-03f,  1.100000000e-03f,  1.679000000e-03f,  1.118000000e-03f,
                1.706000000e-03f,  1.136000000e-03f,  1.733000000e-03f,  1.154000000e-03f,  1.760000000e-03f,  1.172000000e-03f,
                1.787000000e-03f,  1.190000000e-03f,  1.814000000e-03f,  1.208000000e-03f,  1.841000000e-03f,  1.226000000e-03f,
                1.868000000e-03f,  1.244000000e-03f,  1.895000000e-03f,  1.262000000e-03f,  1.922000000e-03f,  1.280000000e-03f,
                1.949000000e-03f,  1.298000000e-03f,  1.976000000e-03f,  1.316000000e-03f,  2.003000000e-03f,  1.334000000e-03f,
                2.030000000e-03f,  1.352000000e-03f,  2.057000000e-03f,  1.370000000e-03f,  2.084000000e-03f,  1.388000000e-03f,
                2.111000000e-03f,  1.406000000e-03f,  2.138000000e-03f,  1.424000000e-03f,  2.165000000e-03f,  1.442000000e-03f,
                2.192000000e-03f,  1.460000000e-03f,  2.219000000e-03f,  1.478000000e-03f,  2.246000000e-03f,  1.496000000e-03f,
                2.273000000e-03f,  1.514000000e-03f,  2.300000000e-03f,  1.532000000e-03f,  2.327000000e-03f,  1.550000000e-03f,
                2.354000000e-03f,  1.568000000e-03f,  2.381000000e-03f,  1.586000000e-03f,  2.408000000e-03f,  1.604000000e-03f,
                2.435000000e-03f,  1.622000000e-03f,  2.462000000e-03f,  1.640000000e-03f,  2.489000000e-03f,  1.658000000e-03f,
                2.516000000e-03f,  1.676000000e-03f,  2.543000000e-03f,  1.694000000e-03f,  2.570000000e-03f,  1.712000000e-03f,
                2.597000000e-03f,  1.730000000e-03f,  2.624000000e-03f,  1.748000000e-03f,  2.651000000e-03f,  1.766000000e-03f,
                2.678000000e-03f,  1.784000000e-03f,  2.705000000e-03f,  1.802000000e-03f,  2.732000000e-03f,  1.820000000e-03f,
                2.759000000e-03f,  1.838000000e-03f,  2.786000000e-03f,  1.856000000e-03f,  2.813000000e-03f,  1.874000000e-03f,
                2.840000000e-03f,  1.892000000e-03f,  2.867000000e-03f,  1.910000000e-03f,  2.894000000e-03f,  1.928000000e-03f,
                2.921000000e-03f,  1.946000000e-03f,  2.948000000e-03f,  1.964000000e-03f,  2.975000000e-03f,  1.982000000e-03f,
                3.002000000e-03f,  2.000000000e-03f,  3.029000000e-03f,  2.018000000e-03f,  3.056000000e-03f,  2.036000000e-03f,
                3.083000000e-03f,  2.054000000e-03f,  3.110000000e-03f,  2.072000000e-03f,  3.137000000e-03f,  2.090000000e-03f,
                3.164000000e-03f,  2.108000000e-03f,  3.191000000e-03f,  2.126000000e-03f,  3.218000000e-03f,  2.144000000e-03f,
                3.245000000e-03f,  2.162000000e-03f,  3.272000000e-03f,  2.180000000e-03f,  3.299000000e-03f,  2.198000000e-03f,
                3.326000000e-03f,  2.216000000e-03f,  3.353000000e-03f,  2.234000000e-03f,  3.380000000e-03f,  2.252000000e-03f,
                3.407000000e-03f,  2.270000000e-03f,  3.434000000e-03f,  2.288000000e-03f,  3.461000000e-03f,  2.306000000e-03f,
                3.488000000e-03f,  2.324000000e-03f,  3.515000000e-03f,  2.342000000e-03f,  3.542000000e-03f,  2.360000000e-03f,
                3.569000000e-03f,  2.378000000e-03f,  3.596000000e-03f,  2.396000000e-03f,  3.623000000e-03f,  2.414000000e-03f,
                3.650000000e-03f,  2.432000000e-03f,  3.677000000e-03f,  2.450000000e-03f,  3.704000000e-03f,  2.468000000e-03f,
                3.731000000e-03f,  2.486000000e-03f,  3.758000000e-03f,  2.504000000e-03f,  3.785000000e-03f,  2.522000000e-03f,
                3.812000000e-03f,  2.540000000e-03f,  3.839000000e-03f,  2.558000000e-03f,  3.866000000e-03f,  2.576000000e-03f,
                3.893000000e-03f,  2.594000000e-03f,  3.920000000e-03f,  2.612000000e-03f,  3.947000000e-03f,  2.630000000e-03f,
                3.974000000e-03f,  2.648000000e-03f,  4.001000000e-03f,  2.666000000e-03f,  4.028000000e-03f,  2.684000000e-03f,
                4.055000000e-03f,  2.702000000e-03f,  4.082000000e-03f,  2.720000000e-03f,  4.109000000e-03f,  2.738000000e-03f,
                4.136000000e-03f,  2.756000000e-03f,  4.163000000e-03f,  2.774000000e-03f,  4.190000000e-03f,  2.792000000e-03f,
                4.217000000e-03f,  2.810000000e-03f,  4.244000000e-03f,  2.828000000e-03f,  4.271000000e-03f,  2.846000000e-03f,
                4.298000000e-03f,  2.864000000e-03f,  4.325000000e-03f,  2.882000000e-03f,  4.352000000e-03f,  2.900000000e-03f,
                4.379000000e-03f,  2.918000000e-03f,  4.406000000e-03f,  2.936000000e-03f,  4.433000000e-03f,  2.954000000e-03f,
                4.460000000e-03f,  2.972000000e-03f,  4.487000000e-03f,  2.990000000e-03f,  4.514000000e-03f,  3.008000000e-03f,
                4.541000000e-03f,  3.026000000e-03f,  4.568000000e-03f,  3.044000000e-03f,  4.595000000e-03f,  3.062000000e-03f,
                4.622000000e-03f,  3.080000000e-03f,  4.649000000e-03f,  3.098000000e-03f,  4.676000000e-03f,  3.116000000e-03f,
                4.703000000e-03f,  3.134000000e-03f,  4.730000000e-03f,  3.152000000e-03f,  4.757000000e-03f,  3.170000000e-03f,
                4.784000000e-03f,  3.188000000e-03f,  4.811000000e-03f,  3.206000000e-03f,  4.838000000e-03f,  3.224000000e-03f,
                4.865000000e-03f,  3.242000000e-03f,  4.892000000e-03f,  3.260000000e-03f,  4.919000000e-03f,  3.278000000e-03f,
                4.946000000e-03f,  3.296000000e-03f,  4.973000000e-03f,  3.314000000e-03f,  5.000000000e-03f,  3.332000000e-03f,
                5.027000000e-03f,  3.350000000e-03f,  5.054000000e-03f,  3.368000000e-03f,  5.081000000e-03f,  3.386000000e-03f,
                5.108000000e-03f,  3.404000000e-03f,  5.135000000e-03f,  3.422000000e-03f,  5.162000000e-03f,  3.440000000e-03f,
                5.189000000e-03f,  3.458000000e-03f,  5.216000000e-03f,  3.476000000e-03f,  5.243000000e-03f,  3.494000000e-03f,
                5.270000000e-03f,  3.512000000e-03f,  5.297000000e-03f,  3.530000000e-03f,  5.324000000e-03f,  3.548000000e-03f,
                5.351000000e-03f,  3.566000000e-03f,  5.378000000e-03f,  3.584000000e-03f,  5.405000000e-03f,  3.602000000e-03f,
                5.432000000e-03f,  3.620000000e-03f,  5.459000000e-03f,  3.638000000e-03f,  5.486000000e-03f,  3.656000000e-03f,
                5.513000000e-03f,  3.674000000e-03f,  5.540000000e-03f,  3.692000000e-03f,  5.567000000e-03f,  3.710000000e-03f,
                5.594000000e-03f,  3.728000000e-03f,  5.621000000e-03f,  3.746000000e-03f,  5.648000000e-03f,  3.764000000e-03f,
                5.675000000e-03f,  3.782000000e-03f,  5.702000000e-03f,  3.800000000e-03f,  5.729000000e-03f,  3.818000000e-03f,
                5.756000000e-03f,  3.836000000e-03f,  5.783000000e-03f,  3.854000000e-03f,  5.810000000e-03f,  3.872000000e-03f,
                5.837000000e-03f,  3.890000000e-03f,  5.864000000e-03f,  3.908000000e-03f,  5.891000000e-03f,  3.926000000e-03f,
                5.918000000e-03f,  3.944000000e-03f,  5.945000000e-03f,  3.962000000e-03f,  5.972000000e-03f,  3.980000000e-03f,
                5.999000000e-03f,  3.998000000e-03f,  6.026000000e-03f,  4.016000000e-03f,  6.053000000e-03f,  4.034000000e-03f,
                6.080000000e-03f,  4.052000000e-03f,  6.107000000e-03f,  4.070000000e-03f,  6.134000000e-03f,  4.088000000e-03f,
                6.161000000e-03f,  4.106000000e-03f,  6.188000000e-03f,  4.124000000e-03f,  6.215000000e-03f,  4.142000000e-03f,
                6.242000000e-03f,  4.160000000e-03f,  6.269000000e-03f,  4.178000000e-03f,  6.296000000e-03f,  4.196000000e-03f,
                6.323000000e-03f,  4.214000000e-03f,  6.350000000e-03f,  4.232000000e-03f,  6.377000000e-03f,  4.250000000e-03f,
                6.404000000e-03f,  4.268000000e-03f,  6.431000000e-03f,  4.286000000e-03f,  6.458000000e-03f,  4.304000000e-03f,
                6.485000000e-03f,  4.322000000e-03f,  6.512000000e-03f,  4.340000000e-03f,  6.539000000e-03f,  4.358000000e-03f,
                6.566000000e-03f,  4.376000000e-03f,  6.593000000e-03f,  4.394000000e-03f,  6.620000000e-03f,  4.412000000e-03f,
                6.647000000e-03f,  4.430000000e-03f,  6.674000000e-03f,  4.448000000e-03f,  6.701000000e-03f,  4.466000000e-03f,
                6.728000000e-03f,  4.484000000e-03f,  6.755000000e-03f,  4.502000000e-03f,  6.782000000e-03f,  4.520000000e-03f,
                6.809000000e-03f,  4.538000000e-03f,  6.836000000e-03f,  4.556000000e-03f,  6.863000000e-03f,  4.574000000e-03f,
                6.890000000e-03f,  4.592000000e-03f,  6.917000000e-03f,  4.610000000e-03f,  6.944000000e-03f,  4.628000000e-03f,
                6.971000000e-03f,  4.646000000e-03f,  6.998000000e-03f,  4.664000000e-03f,  7.025000000e-03f,  4.682000000e-03f,
                7.052000000e-03f,  4.700000000e-03f,  7.079000000e-03f,  4.718000000e-03f,  7.106000000e-03f,  4.736000000e-03f,
                7.133000000e-03f,  4.754000000e-03f,  7.160000000e-03f,  4.772000000e-03f,  7.187000000e-03f,  4.790000000e-03f,
                7.214000000e-03f,  4.808000000e-03f,  7.241000000e-03f,  4.826000000e-03f,  7.268000000e-03f,  4.844000000e-03f,
                7.295000000e-03f,  4.862000000e-03f,  7.322000000e-03f,  4.880000000e-03f,  7.349000000e-03f,  4.898000000e-03f,
                7.376000000e-03f,  4.916000000e-03f,  7.403000000e-03f,  4.934000000e-03f,  7.430000000e-03f,  4.952000000e-03f,
                7.457000000e-03f,  4.970000000e-03f,  7.484000000e-03f,  4.988000000e-03f,  7.511000000e-03f,  5.006000000e-03f,
                7.538000000e-03f,  5.024000000e-03f,  7.565000000e-03f,  5.042000000e-03f,  7.592000000e-03f,  5.060000000e-03f,
                7.619000000e-03f,  5.078000000e-03f,  7.646000000e-03f,  5.096000000e-03f,  7.673000000e-03f,  5.114000000e-03f,
                7.700000000e-03f,  5.132000000e-03f,  7.727000000e-03f,  5.150000000e-03f,  7.754000000e-03f,  5.168000000e-03f,
                7.781000000e-03f,  5.186000000e-03f,  7.808000000e-03f,  5.204000000e-03f,  7.835000000e-03f,  5.222000000e-03f,
                7.862000000e-03f,  5.240000000e-03f,  7.889000000e-03f,  5.258000000e-03f,  7.916000000e-03f,  5.276000000e-03f,
                7.943000000e-03f,  5.294000000e-03f,  7.970000000e-03f,  5.312000000e-03f,  7.997000000e-03f,  5.330000000e-03f,
                8.024000000e-03f,  5.348000000e-03f,  8.051000000e-03f,  5.366000000e-03f,  8.078000000e-03f,  5.384000000e-03f,
                8.105000000e-03f,  5.402000000e-03f,  8.132000000e-03f,  5.420000000e-03f,  8.159000000e-03f,  5.438000000e-03f,
                8.186000000e-03f,  5.456000000e-03f,  8.213000000e-03f,  5.474000000e-03f,  8.240000000e-03f,  5.492000000e-03f,
                8.267000000e-03f,  5.510000000e-03f,  8.294000000e-03f,  5.528000000e-03f,  8.321000000e-03f,  5.546000000e-03f,
                8.348000000e-03f,  5.564000000e-03f,  8.375000000e-03f,  5.582000000e-03f,  8.402000000e-03f,  5.600000000e-03f,
                8.429000000e-03f,  5.618000000e-03f,  8.456000000e-03f,  5.636000000e-03f,  8.483000000e-03f,  5.654000000e-03f,
                8.510000000e-03f,  5.672000000e-03f,  8.537000000e-03f,  5.690000000e-03f,  8.564000000e-03f,  5.708000000e-03f,
                8.591000000e-03f,  5.726000000e-03f,  8.618000000e-03f,  5.744000000e-03f,  8.645000000e-03f,  5.762000000e-03f,
                8.672000000e-03f,  5.780000000e-03f,  8.699000000e-03f,  5.798000000e-03f,  8.726000000e-03f,  5.816000000e-03f,
                8.753000000e-03f,  5.834000000e-03f,  8.780000000e-03f,  5.852000000e-03f,  8.807000000e-03f,  5.870000000e-03f,
                8.834000000e-03f,  5.888000000e-03f,  8.861000000e-03f,  5.906000000e-03f,  8.888000000e-03f,  5.924000000e-03f,
                8.915000000e-03f,  5.942000000e-03f,  8.942000000e-03f,  5.960000000e-03f,  8.969000000e-03f,  5.978000000e-03f,
                8.996000000e-03f,  5.996000000e-03f,  9.023000000e-03f,  6.014000000e-03f,  9.050000000e-03f,  6.032000000e-03f,
                9.077000000e-03f,  6.050000000e-03f,  9.104000000e-03f,  6.068000000e-03f,  9.131000000e-03f,  6.086000000e-03f,
                9.158000000e-03f,  6.104000000e-03f,  9.185000000e-03f,  6.122000000e-03f,  9.212000000e-03f,  6.140000000e-03f,
                9.239000000e-03f,  6.158000000e-03f,  9.266000000e-03f,  6.176000000e-03f,  9.293000000e-03f,  6.194000000e-03f,
                9.320000000e-03f,  6.212000000e-03f,  9.347000000e-03f,  6.230000000e-03f,  9.374000000e-03f,  6.248000000e-03f,
                9.401000000e-03f,  6.266000000e-03f,  9.428000000e-03f,  6.284000000e-03f,  9.455000000e-03f,  6.302000000e-03f,
                9.482000000e-03f,  6.320000000e-03f,  9.509000000e-03f,  6.338000000e-03f,  9.536000000e-03f,  6.356000000e-03f,
                9.563000000e-03f,  6.374000000e-03f,  9.590000000e-03f,  6.392000000e-03f,  9.617000000e-03f,  6.410000000e-03f,
                9.644000000e-03f,  6.428000000e-03f,  9.671000000e-03f,  6.446000000e-03f,  9.698000000e-03f,  6.464000000e-03f,
                9.725000000e-03f,  6.482000000e-03f,  9.752000000e-03f,  6.500000000e-03f,  9.779000000e-03f,  6.518000000e-03f,
                9.806000000e-03f,  6.536000000e-03f,  9.833000000e-03f,  6.554000000e-03f,  9.860000000e-03f,  6.572000000e-03f,
                9.887000000e-03f,  6.590000000e-03f,  9.914000000e-03f,  6.608000000e-03f,  9.941000000e-03f,  6.626000000e-03f,
                9.968000000e-03f,  6.644000000e-03f,  9.995000000e-03f,  6.662000000e-03f,  1.002200000e-02f,  6.680000000e-03f,
                1.004900000e-02f,  6.698000000e-03f,  1.007600000e-02f,  6.716000000e-03f,  1.010300000e-02f,  6.734000000e-03f,
                1.013000000e-02f,  6.752000000e-03f,  1.015700000e-02f,  6.770000000e-03f,  1.018400000e-02f,  6.788000000e-03f,
                1.021100000e-02f,  6.806000000e-03f,  1.023800000e-02f,  6.824000000e-03f,  1.026500000e-02f,  6.842000000e-03f,
                1.029200000e-02f,  6.860000000e-03f,  1.031900000e-02f,  6.878000000e-03f,  1.034600000e-02f,  6.896000000e-03f,
                1.037300000e-02f,  6.914000000e-03f,  1.040000000e-02f,  6.932000000e-03f,  1.042700000e-02f,  6.950000000e-03f,
                1.045400000e-02f,  6.968000000e-03f,  1.048100000e-02f,  6.986000000e-03f,  1.050800000e-02f,  7.004000000e-03f,
                1.053500000e-02f,  7.022000000e-03f,  1.056200000e-02f,  7.040000000e-03f,  1.058900000e-02f,  7.058000000e-03f,
                1.061600000e-02f,  7.076000000e-03f,  1.064300000e-02f,  7.094000000e-03f,  1.067000000e-02f,  7.112000000e-03f,
                1.069700000e-02f,  7.130000000e-03f,  1.072400000e-02f,  7.148000000e-03f,  1.075100000e-02f,  7.166000000e-03f,
                1.077800000e-02f,  7.184000000e-03f,  1.080500000e-02f,  7.202000000e-03f,  1.083200000e-02f,  7.220000000e-03f,
                1.085900000e-02f,  7.238000000e-03f,  1.088600000e-02f,  7.256000000e-03f,  1.091300000e-02f,  7.274000000e-03f,
                1.094000000e-02f,  7.292000000e-03f,  1.096700000e-02f,  7.310000000e-03f,  1.099400000e-02f,  7.328000000e-03f,
                1.102100000e-02f,  7.346000000e-03f,  1.104800000e-02f,  7.364000000e-03f,  1.107500000e-02f,  7.382000000e-03f,
                1.110200000e-02f,  7.400000000e-03f,  1.112900000e-02f,  7.418000000e-03f,  1.115600000e-02f,  7.436000000e-03f,
                1.118300000e-02f,  7.454000000e-03f,  1.121000000e-02f,  7.472000000e-03f,  1.123700000e-02f,  7.490000000e-03f,
                1.126400000e-02f,  7.508000000e-03f,  1.129100000e-02f,  7.526000000e-03f,  1.131800000e-02f,  7.544000000e-03f,
                1.134500000e-02f,  7.562000000e-03f,  1.137200000e-02f,  7.580000000e-03f,  1.139900000e-02f,  7.598000000e-03f,
                1.142600000e-02f,  7.616000000e-03f,  1.145300000e-02f,  7.634000000e-03f,  1.148000000e-02f,  7.652000000e-03f,
                1.150700000e-02f,  7.670000000e-03f,  1.153400000e-02f,  7.688000000e-03f,  1.156100000e-02f,  7.706000000e-03f,
                1.158800000e-02f,  7.724000000e-03f,  1.161500000e-02f,  7.742000000e-03f,  1.164200000e-02f,  7.760000000e-03f,
                1.166900000e-02f,  7.778000000e-03f,  1.169600000e-02f,  7.796000000e-03f,  1.172300000e-02f,  7.814000000e-03f,
                1.175000000e-02f,  7.832000000e-03f,  1.177700000e-02f,  7.850000000e-03f,  1.180400000e-02f,  7.868000000e-03f,
                1.183100000e-02f,  7.886000000e-03f,  1.185800000e-02f,  7.904000000e-03f,  1.188500000e-02f,  7.922000000e-03f,
                1.191200000e-02f,  7.940000000e-03f,  1.193900000e-02f,  7.958000000e-03f,  1.196600000e-02f,  7.976000000e-03f,
                1.199300000e-02f,  7.994000000e-03f,  1.202000000e-02f,  8.012000000e-03f,  1.204700000e-02f,  8.030000000e-03f,
                1.207400000e-02f,  8.048000000e-03f,  1.210100000e-02f,  8.066000000e-03f,  1.212800000e-02f,  8.084000000e-03f,
                1.215500000e-02f,  8.102000000e-03f,  1.218200000e-02f,  8.120000000e-03f,  1.220900000e-02f,  8.138000000e-03f,
                1.223600000e-02f,  8.156000000e-03f,  1.226300000e-02f,  8.174000000e-03f,  1.229000000e-02f,  8.192000000e-03f,
                1.231700000e-02f,  8.210000000e-03f,  1.234400000e-02f,  8.228000000e-03f,  1.237100000e-02f,  8.246000000e-03f,
                1.239800000e-02f,  8.264000000e-03f,  1.242500000e-02f,  8.282000000e-03f,  1.245200000e-02f,  8.300000000e-03f,
                1.247900000e-02f,  8.318000000e-03f,  1.250600000e-02f,  8.336000000e-03f,  1.253300000e-02f,  8.354000000e-03f,
                1.256000000e-02f,  8.372000000e-03f,  1.258700000e-02f,  8.390000000e-03f,  1.261400000e-02f,  8.408000000e-03f,
                1.264100000e-02f,  8.426000000e-03f,  1.266800000e-02f,  8.444000000e-03f,  1.269500000e-02f,  8.462000000e-03f,
                1.272200000e-02f,  8.480000000e-03f,  1.274900000e-02f,  8.498000000e-03f,  1.277600000e-02f,  8.516000000e-03f,
                1.280300000e-02f,  8.534000000e-03f,  1.283000000e-02f,  8.552000000e-03f,  1.285700000e-02f,  8.570000000e-03f,
                1.288400000e-02f,  8.588000000e-03f,  1.291100000e-02f,  8.606000000e-03f,  1.293800000e-02f,  8.624000000e-03f,
                1.296500000e-02f,  8.642000000e-03f,  1.299200000e-02f,  8.660000000e-03f,  1.301900000e-02f,  8.678000000e-03f,
                1.304600000e-02f,  8.696000000e-03f,  1.307300000e-02f,  8.714000000e-03f,  1.310000000e-02f,  8.732000000e-03f,
                1.312700000e-02f,  8.750000000e-03f,  1.315400000e-02f,  8.768000000e-03f,  1.318100000e-02f,  8.786000000e-03f,
                1.320800000e-02f,  8.804000000e-03f,  1.323500000e-02f,  8.822000000e-03f,  1.326200000e-02f,  8.840000000e-03f,
                1.328900000e-02f,  8.858000000e-03f,  1.331600000e-02f,  8.876000000e-03f,  1.334300000e-02f,  8.894000000e-03f,
                1.337000000e-02f,  8.912000000e-03f,  1.339700000e-02f,  8.930000000e-03f,  1.342400000e-02f,  8.948000000e-03f,
                1.345100000e-02f,  8.966000000e-03f,  1.347800000e-02f,  8.984000000e-03f,  1.350500000e-02f,  9.002000000e-03f,
                1.353200000e-02f,  9.020000000e-03f,  1.355900000e-02f,  9.038000000e-03f,  1.358600000e-02f,  9.056000000e-03f,
                1.361300000e-02f,  9.074000000e-03f,  1.364000000e-02f,  9.092000000e-03f,  1.366700000e-02f,  9.110000000e-03f,
                1.369400000e-02f,  9.128000000e-03f,  1.372100000e-02f,  9.146000000e-03f,  1.374800000e-02f,  9.164000000e-03f,
                1.377500000e-02f,  9.182000000e-03f,  1.380200000e-02f,  9.200000000e-03f,  1.382900000e-02f,  9.218000000e-03f,
                1.385600000e-02f,  9.236000000e-03f,  1.388300000e-02f,  9.254000000e-03f,  1.391000000e-02f,  9.272000000e-03f,
                1.393700000e-02f,  9.290000000e-03f,  1.396400000e-02f,  9.308000000e-03f,  1.399100000e-02f,  9.326000000e-03f,
                1.401800000e-02f,  9.344000000e-03f,  1.404500000e-02f,  9.362000000e-03f,  1.407200000e-02f,  9.380000000e-03f,
                1.409900000e-02f,  9.398000000e-03f,  1.412600000e-02f,  9.416000000e-03f,  1.415300000e-02f,  9.434000000e-03f,
                1.418000000e-02f,  9.452000000e-03f,  1.420700000e-02f,  9.470000000e-03f,  1.423400000e-02f,  9.488000000e-03f,
                1.426100000e-02f,  9.506000000e-03f,  1.428800000e-02f,  9.524000000e-03f,  1.431500000e-02f,  9.542000000e-03f,
                1.434200000e-02f,  9.560000000e-03f,  1.436900000e-02f,  9.578000000e-03f,  1.439600000e-02f,  9.596000000e-03f,
                1.442300000e-02f,  9.614000000e-03f,  1.445000000e-02f,  9.632000000e-03f,  1.447700000e-02f,  9.650000000e-03f,
                1.450400000e-02f,  9.668000000e-03f,  1.453100000e-02f,  9.686000000e-03f,  1.455800000e-02f,  9.704000000e-03f,
                1.458500000e-02f,  9.722000000e-03f,  1.461200000e-02f,  9.740000000e-03f,  1.463900000e-02f,  9.758000000e-03f,
                1.466600000e-02f,  9.776000000e-03f,  1.469300000e-02f,  9.794000000e-03f,  1.472000000e-02f,  9.812000000e-03f,
                1.474700000e-02f,  9.830000000e-03f,  1.477400000e-02f,  9.848000000e-03f,  1.480100000e-02f,  9.866000000e-03f,
                1.482800000e-02f,  9.884000000e-03f,  1.485500000e-02f,  9.902000000e-03f,  1.488200000e-02f,  9.920000000e-03f,
                1.490900000e-02f,  9.938000000e-03f,  1.493600000e-02f,  9.956000000e-03f,  1.496300000e-02f,  9.974000000e-03f,
                1.499000000e-02f,  9.992000000e-03f,  1.501700000e-02f,  1.001000000e-02f,  1.504400000e-02f,  1.002800000e-02f,
                1.507100000e-02f,  1.004600000e-02f,  1.509800000e-02f,  1.006400000e-02f,  1.512500000e-02f,  1.008200000e-02f,
                1.515200000e-02f,  1.010000000e-02f,  1.517900000e-02f,  1.011800000e-02f,  1.520600000e-02f,  1.013600000e-02f,
                1.523300000e-02f,  1.015400000e-02f,  1.526000000e-02f,  1.017200000e-02f,  1.528700000e-02f,  1.019000000e-02f,
                1.531400000e-02f,  1.020800000e-02f,  1.534100000e-02f,  1.022600000e-02f,  1.536800000e-02f,  1.024400000e-02f,
                1.539500000e-02f,  1.026200000e-02f,  1.542200000e-02f,  1.028000000e-02f,  1.544900000e-02f,  1.029800000e-02f,
                1.547600000e-02f,  1.031600000e-02f,  1.550300000e-02f,  1.033400000e-02f,  1.553000000e-02f,  1.035200000e-02f,
                1.555700000e-02f,  1.037000000e-02f,  1.558400000e-02f,  1.038800000e-02f,  1.561100000e-02f,  1.040600000e-02f,
                1.563800000e-02f,  1.042400000e-02f,  1.566500000e-02f,  1.044200000e-02f,  1.569200000e-02f,  1.046000000e-02f,
                1.571900000e-02f,  1.047800000e-02f,  1.574600000e-02f,  1.049600000e-02f,  1.577300000e-02f,  1.051400000e-02f,
                1.580000000e-02f,  1.053200000e-02f,  1.582700000e-02f,  1.055000000e-02f,  1.585400000e-02f,  1.056800000e-02f,
                1.588100000e-02f,  1.058600000e-02f,  1.590800000e-02f,  1.060400000e-02f,  1.593500000e-02f,  1.062200000e-02f,
                1.596200000e-02f,  1.064000000e-02f,  1.598900000e-02f,  1.065800000e-02f,  1.601600000e-02f,  1.067600000e-02f,
                1.604300000e-02f,  1.069400000e-02f,  1.607000000e-02f,  1.071200000e-02f,  1.609700000e-02f,  1.073000000e-02f,
                1.612400000e-02f,  1.074800000e-02f,  1.615100000e-02f,  1.076600000e-02f,  1.617800000e-02f,  1.078400000e-02f,
                1.620500000e-02f,  1.080200000e-02f,  1.623200000e-02f,  1.082000000e-02f,  1.625900000e-02f,  1.083800000e-02f,
                1.628600000e-02f,  1.085600000e-02f,  1.631300000e-02f,  1.087400000e-02f,  1.634000000e-02f,  1.089200000e-02f,
                1.636700000e-02f,  1.091000000e-02f,  1.639400000e-02f,  1.092800000e-02f,  1.642100000e-02f,  1.094600000e-02f,
                1.644800000e-02f,  1.096400000e-02f,  1.647500000e-02f,  1.098200000e-02f,  1.650200000e-02f,  1.100000000e-02f,
                1.652900000e-02f,  1.101800000e-02f,  1.655600000e-02f,  1.103600000e-02f,  1.658300000e-02f,  1.105400000e-02f,
                1.661000000e-02f,  1.107200000e-02f,  1.663700000e-02f,  1.109000000e-02f,  1.666400000e-02f,  1.110800000e-02f,
                1.669100000e-02f,  1.112600000e-02f,  1.671800000e-02f,  1.114400000e-02f,  1.674500000e-02f,  1.116200000e-02f,
                1.677200000e-02f,  1.118000000e-02f,  1.679900000e-02f,  1.119800000e-02f,  1.682600000e-02f,  1.121600000e-02f,
                1.685300000e-02f,  1.123400000e-02f,  1.688000000e-02f,  1.125200000e-02f,  1.690700000e-02f,  1.127000000e-02f,
                1.693400000e-02f,  1.128800000e-02f,  1.696100000e-02f,  1.130600000e-02f,  1.698800000e-02f,  1.132400000e-02f,
                1.701500000e-02f,  1.134200000e-02f,  1.704200000e-02f,  1.136000000e-02f,  1.706900000e-02f,  1.137800000e-02f,
                1.709600000e-02f,  1.139600000e-02f,  1.712300000e-02f,  1.141400000e-02f,  1.715000000e-02f,  1.143200000e-02f,
                1.717700000e-02f,  1.145000000e-02f,  1.720400000e-02f,  1.146800000e-02f,  1.723100000e-02f,  1.148600000e-02f,
                1.725800000e-02f,  1.150400000e-02f,  1.728500000e-02f,  1.152200000e-02f,  1.731200000e-02f,  1.154000000e-02f,
                1.733900000e-02f,  1.155800000e-02f,  1.736600000e-02f,  1.157600000e-02f,  1.739300000e-02f,  1.159400000e-02f,
                1.742000000e-02f,  1.161200000e-02f,  1.744700000e-02f,  1.163000000e-02f,  1.747400000e-02f,  1.164800000e-02f,
                1.750100000e-02f,  1.166600000e-02f,  1.752800000e-02f,  1.168400000e-02f,  1.755500000e-02f,  1.170200000e-02f,
                1.758200000e-02f,  1.172000000e-02f,  1.760900000e-02f,  1.173800000e-02f,  1.763600000e-02f,  1.175600000e-02f,
                1.766300000e-02f,  1.177400000e-02f,  1.769000000e-02f,  1.179200000e-02f,  1.771700000e-02f,  1.181000000e-02f,
                1.774400000e-02f,  1.182800000e-02f,  1.777100000e-02f,  1.184600000e-02f,  1.779800000e-02f,  1.186400000e-02f,
                1.782500000e-02f,  1.188200000e-02f,  1.785200000e-02f,  1.190000000e-02f,  1.787900000e-02f,  1.191800000e-02f,
                1.790600000e-02f,  1.193600000e-02f,  1.793300000e-02f,  1.195400000e-02f,  1.796000000e-02f,  1.197200000e-02f,
                1.798700000e-02f,  1.199000000e-02f,  1.801400000e-02f,  1.200800000e-02f,  1.804100000e-02f,  1.202600000e-02f,
                1.806800000e-02f,  1.204400000e-02f,  1.809500000e-02f,  1.206200000e-02f,  1.812200000e-02f,  1.208000000e-02f,
                1.814900000e-02f,  1.209800000e-02f,  1.817600000e-02f,  1.211600000e-02f,  1.820300000e-02f,  1.213400000e-02f,
                1.823000000e-02f,  1.215200000e-02f,  1.825700000e-02f,  1.217000000e-02f,  1.828400000e-02f,  1.218800000e-02f,
                1.831100000e-02f,  1.220600000e-02f,  1.833800000e-02f,  1.222400000e-02f,  1.836500000e-02f,  1.224200000e-02f,
                1.839200000e-02f,  1.226000000e-02f,  1.841900000e-02f,  1.227800000e-02f,  1.844600000e-02f,  1.229600000e-02f,
                1.847300000e-02f,  1.231400000e-02f,  1.850000000e-02f,  1.233200000e-02f,  1.852700000e-02f,  1.235000000e-02f,
                1.855400000e-02f,  1.236800000e-02f,  1.858100000e-02f,  1.238600000e-02f,  1.860800000e-02f,  1.240400000e-02f,
                1.863500000e-02f,  1.242200000e-02f,  1.866200000e-02f,  1.244000000e-02f,  1.868900000e-02f,  1.245800000e-02f,
                1.871600000e-02f,  1.247600000e-02f,  1.874300000e-02f,  1.249400000e-02f,  1.877000000e-02f,  1.251200000e-02f,
                1.879700000e-02f,  1.253000000e-02f,  1.882400000e-02f,  1.254800000e-02f,  1.885100000e-02f,  1.256600000e-02f,
                1.887800000e-02f,  1.258400000e-02f,  1.890500000e-02f,  1.260200000e-02f,  1.893200000e-02f,  1.262000000e-02f,
                1.895900000e-02f,  1.263800000e-02f,  1.898600000e-02f,  1.265600000e-02f,  1.901300000e-02f,  1.267400000e-02f,
                1.904000000e-02f,  1.269200000e-02f,  1.906700000e-02f,  1.271000000e-02f,  1.909400000e-02f,  1.272800000e-02f,
                1.912100000e-02f,  1.274600000e-02f,  1.914800000e-02f,  1.276400000e-02f,  1.917500000e-02f,  1.278200000e-02f,
                1.920200000e-02f,  1.280000000e-02f,  1.922900000e-02f,  1.281800000e-02f,  1.925600000e-02f,  1.283600000e-02f,
                1.928300000e-02f,  1.285400000e-02f,  1.931000000e-02f,  1.287200000e-02f,  1.933700000e-02f,  1.289000000e-02f,
                1.936400000e-02f,  1.290800000e-02f,  1.939100000e-02f,  1.292600000e-02f,  1.941800000e-02f,  1.294400000e-02f,
                1.944500000e-02f,  1.296200000e-02f,  1.947200000e-02f,  1.298000000e-02f,  1.949900000e-02f,  1.299800000e-02f,
                1.952600000e-02f,  1.301600000e-02f,  1.955300000e-02f,  1.303400000e-02f,  1.958000000e-02f,  1.305200000e-02f,
                1.960700000e-02f,  1.307000000e-02f,  1.963400000e-02f,  1.308800000e-02f,  1.966100000e-02f,  1.310600000e-02f,
                1.968800000e-02f,  1.312400000e-02f,  1.971500000e-02f,  1.314200000e-02f,  1.974200000e-02f,  1.316000000e-02f,
                1.976900000e-02f,  1.317800000e-02f,  1.979600000e-02f,  1.319600000e-02f,  1.982300000e-02f,  1.321400000e-02f,
                1.985000000e-02f,  1.323200000e-02f,  1.987700000e-02f,  1.325000000e-02f,  1.990400000e-02f,  1.326800000e-02f,
                1.993100000e-02f,  1.328600000e-02f,  1.995800000e-02f,  1.330400000e-02f,  1.998500000e-02f,  1.332200000e-02f,
                2.001200000e-02f,  1.334000000e-02f,  2.003900000e-02f,  1.335800000e-02f,  2.006600000e-02f,  1.337600000e-02f,
                2.009300000e-02f,  1.339400000e-02f,  2.012000000e-02f,  1.341200000e-02f,  2.014700000e-02f,  1.343000000e-02f,
                2.017400000e-02f,  1.344800000e-02f,  2.020100000e-02f,  1.346600000e-02f,  2.022800000e-02f,  1.348400000e-02f,
                2.025500000e-02f,  1.350200000e-02f,  2.028200000e-02f,  1.352000000e-02f,  2.030900000e-02f,  1.353800000e-02f,
                2.033600000e-02f,  1.355600000e-02f,  2.036300000e-02f,  1.357400000e-02f,  2.039000000e-02f,  1.359200000e-02f,
                2.041700000e-02f,  1.361000000e-02f,  2.044400000e-02f,  1.362800000e-02f,  2.047100000e-02f,  1.364600000e-02f,
                2.049800000e-02f,  1.366400000e-02f,  2.052500000e-02f,  1.368200000e-02f,  2.055200000e-02f,  1.370000000e-02f,
                2.057900000e-02f,  1.371800000e-02f,  2.060600000e-02f,  1.373600000e-02f,  2.063300000e-02f,  1.375400000e-02f,
                2.066000000e-02f,  1.377200000e-02f,  2.068700000e-02f,  1.379000000e-02f,  2.071400000e-02f,  1.380800000e-02f,
                2.074100000e-02f,  1.382600000e-02f,  2.076800000e-02f,  1.384400000e-02f,  2.079500000e-02f,  1.386200000e-02f,
                2.082200000e-02f,  1.388000000e-02f,  2.084900000e-02f,  1.389800000e-02f,  2.087600000e-02f,  1.391600000e-02f,
                2.090300000e-02f,  1.393400000e-02f,  2.093000000e-02f,  1.395200000e-02f,  2.095700000e-02f,  1.397000000e-02f,
                2.098400000e-02f,  1.398800000e-02f,  2.101100000e-02f,  1.400600000e-02f,  2.103800000e-02f,  1.402400000e-02f,
                2.106500000e-02f,  1.404200000e-02f,  2.109200000e-02f,  1.406000000e-02f,  2.111900000e-02f,  1.407800000e-02f,
                2.114600000e-02f,  1.409600000e-02f,  2.117300000e-02f,  1.411400000e-02f,  2.120000000e-02f,  1.413200000e-02f,
                2.122700000e-02f,  1.415000000e-02f,  2.125400000e-02f,  1.416800000e-02f,  2.128100000e-02f,  1.418600000e-02f,
                2.130800000e-02f,  1.420400000e-02f,  2.133500000e-02f,  1.422200000e-02f,  2.136200000e-02f,  1.424000000e-02f,
                2.138900000e-02f,  1.425800000e-02f,  2.141600000e-02f,  1.427600000e-02f,  2.144300000e-02f,  1.429400000e-02f,
                2.147000000e-02f,  1.431200000e-02f,  2.149700000e-02f,  1.433000000e-02f,  2.152400000e-02f,  1.434800000e-02f,
                2.155100000e-02f,  1.436600000e-02f,  2.157800000e-02f,  1.438400000e-02f,  2.160500000e-02f,  1.440200000e-02f,
                2.163200000e-02f,  1.442000000e-02f,  2.165900000e-02f,  1.443800000e-02f,  2.168600000e-02f,  1.445600000e-02f,
                2.171300000e-02f,  1.447400000e-02f,  2.174000000e-02f,  1.449200000e-02f,  2.176700000e-02f,  1.451000000e-02f,
                2.179400000e-02f,  1.452800000e-02f,  2.182100000e-02f,  1.454600000e-02f,  2.184800000e-02f,  1.456400000e-02f,
                2.187500000e-02f,  1.458200000e-02f,  2.190200000e-02f,  1.460000000e-02f,  2.192900000e-02f,  1.461800000e-02f,
                2.195600000e-02f,  1.463600000e-02f,  2.198300000e-02f,  1.465400000e-02f,  2.201000000e-02f,  1.467200000e-02f,
                2.203700000e-02f,  1.469000000e-02f,  2.206400000e-02f,  1.470800000e-02f,  2.209100000e-02f,  1.472600000e-02f,
                2.211800000e-02f,  1.474400000e-02f,  2.214500000e-02f,  1.476200000e-02f,  2.217200000e-02f,  1.478000000e-02f,
                2.219900000e-02f,  1.479800000e-02f,  2.222600000e-02f,  1.481600000e-02f,  2.225300000e-02f,  1.483400000e-02f,
                2.228000000e-02f,  1.485200000e-02f,  2.230700000e-02f,  1.487000000e-02f,  2.233400000e-02f,  1.488800000e-02f,
                2.236100000e-02f,  1.490600000e-02f,  2.238800000e-02f,  1.492400000e-02f,  2.241500000e-02f,  1.494200000e-02f,
                2.244200000e-02f,  1.496000000e-02f,  2.246900000e-02f,  1.497800000e-02f,  2.249600000e-02f,  1.499600000e-02f,
                2.252300000e-02f,  1.501400000e-02f,  2.255000000e-02f,  1.503200000e-02f,  2.257700000e-02f,  1.505000000e-02f,
                2.260400000e-02f,  1.506800000e-02f,  2.263100000e-02f,  1.508600000e-02f,  2.265800000e-02f,  1.510400000e-02f,
                2.268500000e-02f,  1.512200000e-02f,  2.271200000e-02f,  1.514000000e-02f,  2.273900000e-02f,  1.515800000e-02f,
                2.276600000e-02f,  1.517600000e-02f,  2.279300000e-02f,  1.519400000e-02f,  2.282000000e-02f,  1.521200000e-02f,
                2.284700000e-02f,  1.523000000e-02f,  2.287400000e-02f,  1.524800000e-02f,  2.290100000e-02f,  1.526600000e-02f,
                2.292800000e-02f,  1.528400000e-02f,  2.295500000e-02f,  1.530200000e-02f,  2.298200000e-02f,  1.532000000e-02f,
                2.300900000e-02f,  1.533800000e-02f,  2.303600000e-02f,  1.535600000e-02f,  2.306300000e-02f,  1.537400000e-02f,
                2.309000000e-02f,  1.539200000e-02f,  2.311700000e-02f,  1.541000000e-02f,  2.314400000e-02f,  1.542800000e-02f,
                2.317100000e-02f,  1.544600000e-02f,  2.319800000e-02f,  1.546400000e-02f,  2.322500000e-02f,  1.548200000e-02f,
                2.325200000e-02f,  1.550000000e-02f,  2.327900000e-02f,  1.551800000e-02f,  2.330600000e-02f,  1.553600000e-02f,
                2.333300000e-02f,  1.555400000e-02f,  2.336000000e-02f,  1.557200000e-02f,  2.338700000e-02f,  1.559000000e-02f,
                2.341400000e-02f,  1.560800000e-02f,  2.344100000e-02f,  1.562600000e-02f,  2.346800000e-02f,  1.564400000e-02f,
                2.349500000e-02f,  1.566200000e-02f,  2.352200000e-02f,  1.568000000e-02f,  2.354900000e-02f,  1.569800000e-02f,
                2.357600000e-02f,  1.571600000e-02f,  2.360300000e-02f,  1.573400000e-02f,  2.363000000e-02f,  1.575200000e-02f,
                2.365700000e-02f,  1.577000000e-02f,  2.368400000e-02f,  1.578800000e-02f,  2.371100000e-02f,  1.580600000e-02f,
                2.373800000e-02f,  1.582400000e-02f,  2.376500000e-02f,  1.584200000e-02f,  2.379200000e-02f,  1.586000000e-02f,
                2.381900000e-02f,  1.587800000e-02f,  2.384600000e-02f,  1.589600000e-02f,  2.387300000e-02f,  1.591400000e-02f,
                2.390000000e-02f,  1.593200000e-02f,  2.392700000e-02f,  1.595000000e-02f,  2.395400000e-02f,  1.596800000e-02f,
                2.398100000e-02f,  1.598600000e-02f,  2.400800000e-02f,  1.600400000e-02f,  2.403500000e-02f,  1.602200000e-02f,
                2.406200000e-02f,  1.604000000e-02f,  2.408900000e-02f,  1.605800000e-02f,  2.411600000e-02f,  1.607600000e-02f,
                2.414300000e-02f,  1.609400000e-02f,  2.417000000e-02f,  1.611200000e-02f,  2.419700000e-02f,  1.613000000e-02f,
                2.422400000e-02f,  1.614800000e-02f,  2.425100000e-02f,  1.616600000e-02f,  2.427800000e-02f,  1.618400000e-02f,
                2.430500000e-02f,  1.620200000e-02f,  2.433200000e-02f,  1.622000000e-02f,  2.435900000e-02f,  1.623800000e-02f,
                2.438600000e-02f,  1.625600000e-02f,  2.441300000e-02f,  1.627400000e-02f,  2.444000000e-02f,  1.629200000e-02f,
                2.446700000e-02f,  1.631000000e-02f,  2.449400000e-02f,  1.632800000e-02f,  2.452100000e-02f,  1.634600000e-02f,
                2.454800000e-02f,  1.636400000e-02f,  2.457500000e-02f,  1.638200000e-02f,  2.460200000e-02f,  1.640000000e-02f,
                2.462900000e-02f,  1.641800000e-02f,  2.465600000e-02f,  1.643600000e-02f,  2.468300000e-02f,  1.645400000e-02f,
                2.471000000e-02f,  1.647200000e-02f,  2.473700000e-02f,  1.649000000e-02f,  2.476400000e-02f,  1.650800000e-02f,
                2.479100000e-02f,  1.652600000e-02f,  2.481800000e-02f,  1.654400000e-02f,  2.484500000e-02f,  1.656200000e-02f,
                2.487200000e-02f,  1.658000000e-02f,  2.489900000e-02f,  1.659800000e-02f,  2.492600000e-02f,  1.661600000e-02f,
                2.495300000e-02f,  1.663400000e-02f,  2.498000000e-02f,  1.665200000e-02f,  2.500700000e-02f,  1.667000000e-02f,
                2.503400000e-02f,  1.668800000e-02f,  2.506100000e-02f,  1.670600000e-02f,  2.508800000e-02f,  1.672400000e-02f,
                2.511500000e-02f,  1.674200000e-02f,  2.514200000e-02f,  1.676000000e-02f,  2.516900000e-02f,  1.677800000e-02f,
                2.519600000e-02f,  1.679600000e-02f,  2.522300000e-02f,  1.681400000e-02f,  2.525000000e-02f,  1.683200000e-02f,
                2.527700000e-02f,  1.685000000e-02f,  2.530400000e-02f,  1.686800000e-02f,  2.533100000e-02f,  1.688600000e-02f,
                2.535800000e-02f,  1.690400000e-02f,  2.538500000e-02f,  1.692200000e-02f,  2.541200000e-02f,  1.694000000e-02f,
                2.543900000e-02f,  1.695800000e-02f,  2.546600000e-02f,  1.697600000e-02f,  2.549300000e-02f,  1.699400000e-02f,
                2.552000000e-02f,  1.701200000e-02f,  2.554700000e-02f,  1.703000000e-02f,  2.557400000e-02f,  1.704800000e-02f,
                2.560100000e-02f,  1.706600000e-02f,  2.562800000e-02f,  1.708400000e-02f,  2.565500000e-02f,  1.710200000e-02f,
                2.568200000e-02f,  1.712000000e-02f,  2.570900000e-02f,  1.713800000e-02f,  2.573600000e-02f,  1.715600000e-02f,
                2.576300000e-02f,  1.717400000e-02f,  2.579000000e-02f,  1.719200000e-02f,  2.581700000e-02f,  1.721000000e-02f,
                2.584400000e-02f,  1.722800000e-02f,  2.587100000e-02f,  1.724600000e-02f,  2.589800000e-02f,  1.726400000e-02f,
                2.592500000e-02f,  1.728200000e-02f,  2.595200000e-02f,  1.730000000e-02f,  2.597900000e-02f,  1.731800000e-02f,
                2.600600000e-02f,  1.733600000e-02f,  2.603300000e-02f,  1.735400000e-02f,  2.606000000e-02f,  1.737200000e-02f,
                2.608700000e-02f,  1.739000000e-02f,  2.611400000e-02f,  1.740800000e-02f,  2.614100000e-02f,  1.742600000e-02f,
                2.616800000e-02f,  1.744400000e-02f,  2.619500000e-02f,  1.746200000e-02f,  2.622200000e-02f,  1.748000000e-02f,
                2.624900000e-02f,  1.749800000e-02f,  2.627600000e-02f,  1.751600000e-02f,  2.630300000e-02f,  1.753400000e-02f,
                2.633000000e-02f,  1.755200000e-02f,  2.635700000e-02f,  1.757000000e-02f,  2.638400000e-02f,  1.758800000e-02f,
                2.641100000e-02f,  1.760600000e-02f,  2.643800000e-02f,  1.762400000e-02f,  2.646500000e-02f,  1.764200000e-02f,
                2.649200000e-02f,  1.766000000e-02f,  2.651900000e-02f,  1.767800000e-02f,  2.654600000e-02f,  1.769600000e-02f,
                2.657300000e-02f,  1.771400000e-02f,  2.660000000e-02f,  1.773200000e-02f,  2.662700000e-02f,  1.775000000e-02f,
                2.665400000e-02f,  1.776800000e-02f,  2.668100000e-02f,  1.778600000e-02f,  2.670800000e-02f,  1.780400000e-02f,
                2.673500000e-02f,  1.782200000e-02f,  2.676200000e-02f,  1.784000000e-02f,  2.678900000e-02f,  1.785800000e-02f,
                2.681600000e-02f,  1.787600000e-02f,  2.684300000e-02f,  1.789400000e-02f,  2.687000000e-02f,  1.791200000e-02f,
                2.689700000e-02f,  1.793000000e-02f,  2.692400000e-02f,  1.794800000e-02f,  2.695100000e-02f,  1.796600000e-02f,
                2.697800000e-02f,  1.798400000e-02f,  2.700500000e-02f,  1.800200000e-02f,  2.703200000e-02f,  1.802000000e-02f,
                2.705900000e-02f,  1.803800000e-02f,  2.708600000e-02f,  1.805600000e-02f,  2.711300000e-02f,  1.807400000e-02f,
                2.714000000e-02f,  1.809200000e-02f,  2.716700000e-02f,  1.811000000e-02f,  2.719400000e-02f,  1.812800000e-02f,
                2.722100000e-02f,  1.814600000e-02f,  2.724800000e-02f,  1.816400000e-02f,  2.727500000e-02f,  1.818200000e-02f,
                2.730200000e-02f,  1.820000000e-02f,  2.732900000e-02f,  1.821800000e-02f,  2.735600000e-02f,  1.823600000e-02f,
                2.738300000e-02f,  1.825400000e-02f,  2.741000000e-02f,  1.827200000e-02f,  2.743700000e-02f,  1.829000000e-02f,
                2.746400000e-02f,  1.830800000e-02f,  2.749100000e-02f,  1.832600000e-02f,  2.751800000e-02f,  1.834400000e-02f,
                2.754500000e-02f,  1.836200000e-02f,  2.757200000e-02f,  1.838000000e-02f,  2.759900000e-02f,  1.839800000e-02f,
                2.762600000e-02f,  1.841600000e-02f,  2.765300000e-02f,  1.843400000e-02f,  2.768000000e-02f,  1.845200000e-02f,
                2.770700000e-02f,  1.847000000e-02f,  2.773400000e-02f,  1.848800000e-02f,  2.776100000e-02f,  1.850600000e-02f,
                2.778800000e-02f,  1.852400000e-02f,  2.781500000e-02f,  1.854200000e-02f,  2.784200000e-02f,  1.856000000e-02f,
                2.786900000e-02f,  1.857800000e-02f,  2.789600000e-02f,  1.859600000e-02f,  2.792300000e-02f,  1.861400000e-02f,
                2.795000000e-02f,  1.863200000e-02f,  2.797700000e-02f,  1.865000000e-02f,  2.800400000e-02f,  1.866800000e-02f,
                2.803100000e-02f,  1.868600000e-02f,  2.805800000e-02f,  1.870400000e-02f,  2.808500000e-02f,  1.872200000e-02f,
                2.811200000e-02f,  1.874000000e-02f,  2.813900000e-02f,  1.875800000e-02f,  2.816600000e-02f,  1.877600000e-02f,
                2.819300000e-02f,  1.879400000e-02f,  2.822000000e-02f,  1.881200000e-02f,  2.824700000e-02f,  1.883000000e-02f,
                2.827400000e-02f,  1.884800000e-02f,  2.830100000e-02f,  1.886600000e-02f,  2.832800000e-02f,  1.888400000e-02f,
                2.835500000e-02f,  1.890200000e-02f,  2.838200000e-02f,  1.892000000e-02f,  2.840900000e-02f,  1.893800000e-02f,
                2.843600000e-02f,  1.895600000e-02f,  2.846300000e-02f,  1.897400000e-02f,  2.849000000e-02f,  1.899200000e-02f,
                2.851700000e-02f,  1.901000000e-02f,  2.854400000e-02f,  1.902800000e-02f,  2.857100000e-02f,  1.904600000e-02f,
                2.859800000e-02f,  1.906400000e-02f,  2.862500000e-02f,  1.908200000e-02f,  2.865200000e-02f,  1.910000000e-02f,
                2.867900000e-02f,  1.911800000e-02f,  2.870600000e-02f,  1.913600000e-02f,  2.873300000e-02f,  1.915400000e-02f,
                2.876000000e-02f,  1.917200000e-02f,  2.878700000e-02f,  1.919000000e-02f,  2.881400000e-02f,  1.920800000e-02f,
                2.884100000e-02f,  1.922600000e-02f,  2.886800000e-02f,  1.924400000e-02f,  2.889500000e-02f,  1.926200000e-02f,
                2.892200000e-02f,  1.928000000e-02f,  2.894900000e-02f,  1.929800000e-02f,  2.897600000e-02f,  1.931600000e-02f,
                2.900300000e-02f,  1.933400000e-02f,  2.903000000e-02f,  1.935200000e-02f,  2.905700000e-02f,  1.937000000e-02f,
                2.908400000e-02f,  1.938800000e-02f,  2.911100000e-02f,  1.940600000e-02f,  2.913800000e-02f,  1.942400000e-02f,
                2.916500000e-02f,  1.944200000e-02f,  2.919200000e-02f,  1.946000000e-02f,  2.921900000e-02f,  1.947800000e-02f,
                2.924600000e-02f,  1.949600000e-02f,  2.927300000e-02f,  1.951400000e-02f,  2.930000000e-02f,  1.953200000e-02f,
                2.932700000e-02f,  1.955000000e-02f,  2.935400000e-02f,  1.956800000e-02f,  2.938100000e-02f,  1.958600000e-02f,
                2.940800000e-02f,  1.960400000e-02f,  2.943500000e-02f,  1.962200000e-02f,  2.946200000e-02f,  1.964000000e-02f,
                2.948900000e-02f,  1.965800000e-02f,  2.951600000e-02f,  1.967600000e-02f,  2.954300000e-02f,  1.969400000e-02f,
                2.957000000e-02f,  1.971200000e-02f,  2.959700000e-02f,  1.973000000e-02f,  2.962400000e-02f,  1.974800000e-02f,
                2.965100000e-02f,  1.976600000e-02f,  2.967800000e-02f,  1.978400000e-02f,  2.970500000e-02f,  1.980200000e-02f,
                2.973200000e-02f,  1.982000000e-02f,  2.975900000e-02f,  1.983800000e-02f,  2.978600000e-02f,  1.985600000e-02f,
                2.981300000e-02f,  1.987400000e-02f,  2.984000000e-02f,  1.989200000e-02f,  2.986700000e-02f,  1.991000000e-02f,
                2.989400000e-02f,  1.992800000e-02f,  2.992100000e-02f,  1.994600000e-02f,  2.994800000e-02f,  1.996400000e-02f,
                2.997500000e-02f,  1.998200000e-02f,  3.000200000e-02f,  2.000000000e-02f,  3.002900000e-02f,  2.001800000e-02f,
                3.005600000e-02f,  2.003600000e-02f,  3.008300000e-02f,  2.005400000e-02f,  3.011000000e-02f,  2.007200000e-02f,
                3.013700000e-02f,  2.009000000e-02f,  3.016400000e-02f,  2.010800000e-02f,  3.019100000e-02f,  2.012600000e-02f,
                3.021800000e-02f,  2.014400000e-02f,  3.024500000e-02f,  2.016200000e-02f,  3.027200000e-02f,  2.018000000e-02f,
                3.029900000e-02f,  2.019800000e-02f,  3.032600000e-02f,  2.021600000e-02f,  3.035300000e-02f,  2.023400000e-02f,
                3.038000000e-02f,  2.025200000e-02f,  3.040700000e-02f,  2.027000000e-02f,  3.043400000e-02f,  2.028800000e-02f,
                3.046100000e-02f,  2.030600000e-02f,  3.048800000e-02f,  2.032400000e-02f,  3.051500000e-02f,  2.034200000e-02f,
                3.054200000e-02f,  2.036000000e-02f,  3.056900000e-02f,  2.037800000e-02f,  3.059600000e-02f,  2.039600000e-02f,
                3.062300000e-02f,  2.041400000e-02f,  3.065000000e-02f,  2.043200000e-02f,  3.067700000e-02f,  2.045000000e-02f,
                3.070400000e-02f,  2.046800000e-02f,  3.073100000e-02f,  2.048600000e-02f,  3.075800000e-02f,  2.050400000e-02f,
                3.078500000e-02f,  2.052200000e-02f,  3.081200000e-02f,  2.054000000e-02f,  3.083900000e-02f,  2.055800000e-02f,
                3.086600000e-02f,  2.057600000e-02f,  3.089300000e-02f,  2.059400000e-02f,  3.092000000e-02f,  2.061200000e-02f,
                3.094700000e-02f,  2.063000000e-02f,  3.097400000e-02f,  2.064800000e-02f,  3.100100000e-02f,  2.066600000e-02f,
                3.102800000e-02f,  2.068400000e-02f,  3.105500000e-02f,  2.070200000e-02f,  3.108200000e-02f,  2.072000000e-02f,
                3.110900000e-02f,  2.073800000e-02f,  3.113600000e-02f,  2.075600000e-02f,  3.116300000e-02f,  2.077400000e-02f,
                3.119000000e-02f,  2.079200000e-02f,  3.121700000e-02f,  2.081000000e-02f,  3.124400000e-02f,  2.082800000e-02f,
                3.127100000e-02f,  2.084600000e-02f,  3.129800000e-02f,  2.086400000e-02f,  3.132500000e-02f,  2.088200000e-02f,
                3.135200000e-02f,  2.090000000e-02f,  3.137900000e-02f,  2.091800000e-02f,  3.140600000e-02f,  2.093600000e-02f,
                3.143300000e-02f,  2.095400000e-02f,  3.146000000e-02f,  2.097200000e-02f,  3.148700000e-02f,  2.099000000e-02f,
                3.151400000e-02f,  2.100800000e-02f,  3.154100000e-02f,  2.102600000e-02f,  3.156800000e-02f,  2.104400000e-02f,
                3.159500000e-02f,  2.106200000e-02f,  3.162200000e-02f,  2.108000000e-02f,  3.164900000e-02f,  2.109800000e-02f,
                3.167600000e-02f,  2.111600000e-02f,  3.170300000e-02f,  2.113400000e-02f,  3.173000000e-02f,  2.115200000e-02f,
                3.175700000e-02f,  2.117000000e-02f,  3.178400000e-02f,  2.118800000e-02f,  3.181100000e-02f,  2.120600000e-02f,
                3.183800000e-02f,  2.122400000e-02f,  3.186500000e-02f,  2.124200000e-02f,  3.189200000e-02f,  2.126000000e-02f,
                3.191900000e-02f,  2.127800000e-02f,  3.194600000e-02f,  2.129600000e-02f,  3.197300000e-02f,  2.131400000e-02f,
                3.200000000e-02f,  2.133200000e-02f,  3.202700000e-02f,  2.135000000e-02f,  3.205400000e-02f,  2.136800000e-02f,
                3.208100000e-02f,  2.138600000e-02f,  3.210800000e-02f,  2.140400000e-02f,  3.213500000e-02f,  2.142200000e-02f,
                3.216200000e-02f,  2.144000000e-02f,  3.218900000e-02f,  2.145800000e-02f,  3.221600000e-02f,  2.147600000e-02f,
                3.224300000e-02f,  2.149400000e-02f,  3.227000000e-02f,  2.151200000e-02f,  3.229700000e-02f,  2.153000000e-02f,
                3.232400000e-02f,  2.154800000e-02f,  3.235100000e-02f,  2.156600000e-02f,  3.237800000e-02f,  2.158400000e-02f,
                3.240500000e-02f,  2.160200000e-02f,  3.243200000e-02f,  2.162000000e-02f,  3.245900000e-02f,  2.163800000e-02f,
                3.248600000e-02f,  2.165600000e-02f,  3.251300000e-02f,  2.167400000e-02f,  3.254000000e-02f,  2.169200000e-02f,
                3.256700000e-02f,  2.171000000e-02f,  3.259400000e-02f,  2.172800000e-02f,  3.262100000e-02f,  2.174600000e-02f,
                3.264800000e-02f,  2.176400000e-02f,  3.267500000e-02f,  2.178200000e-02f,  3.270200000e-02f,  2.180000000e-02f,
                3.272900000e-02f,  2.181800000e-02f,  3.275600000e-02f,  2.183600000e-02f,  3.278300000e-02f,  2.185400000e-02f,
                3.281000000e-02f,  2.187200000e-02f,  3.283700000e-02f,  2.189000000e-02f,  3.286400000e-02f,  2.190800000e-02f,
                3.289100000e-02f,  2.192600000e-02f,  3.291800000e-02f,  2.194400000e-02f,  3.294500000e-02f,  2.196200000e-02f,
                3.297200000e-02f,  2.198000000e-02f,  3.299900000e-02f,  2.199800000e-02f,  3.302600000e-02f,  2.201600000e-02f,
                3.305300000e-02f,  2.203400000e-02f,  3.308000000e-02f,  2.205200000e-02f,  3.310700000e-02f,  2.207000000e-02f,
                3.313400000e-02f,  2.208800000e-02f,  3.316100000e-02f,  2.210600000e-02f,  3.318800000e-02f,  2.212400000e-02f,
                3.321500000e-02f,  2.214200000e-02f,  3.324200000e-02f,  2.216000000e-02f,  3.326900000e-02f,  2.217800000e-02f,
                3.329600000e-02f,  2.219600000e-02f,  3.332300000e-02f,  2.221400000e-02f,  3.335000000e-02f,  2.223200000e-02f,
                3.337700000e-02f,  2.225000000e-02f,  3.340400000e-02f,  2.226800000e-02f,  3.343100000e-02f,  2.228600000e-02f,
                3.345800000e-02f,  2.230400000e-02f,  3.348500000e-02f,  2.232200000e-02f,  3.351200000e-02f,  2.234000000e-02f,
                3.353900000e-02f,  2.235800000e-02f,  3.356600000e-02f,  2.237600000e-02f,  3.359300000e-02f,  2.239400000e-02f,
                3.362000000e-02f,  2.241200000e-02f,  3.364700000e-02f,  2.243000000e-02f,  3.367400000e-02f,  2.244800000e-02f,
                3.370100000e-02f,  2.246600000e-02f,  3.372800000e-02f,  2.248400000e-02f,  3.375500000e-02f,  2.250200000e-02f,
                3.378200000e-02f,  2.252000000e-02f,  3.380900000e-02f,  2.253800000e-02f,  3.383600000e-02f,  2.255600000e-02f,
                3.386300000e-02f,  2.257400000e-02f,  3.389000000e-02f,  2.259200000e-02f,  3.391700000e-02f,  2.261000000e-02f,
                3.394400000e-02f,  2.262800000e-02f,  3.397100000e-02f,  2.264600000e-02f,  3.399800000e-02f,  2.266400000e-02f,
                3.402500000e-02f,  2.268200000e-02f,  3.405200000e-02f,  2.270000000e-02f,  3.407900000e-02f,  2.271800000e-02f,
                3.410600000e-02f,  2.273600000e-02f,  3.413300000e-02f,  2.275400000e-02f,  3.416000000e-02f,  2.277200000e-02f,
                3.418700000e-02f,  2.279000000e-02f,  3.421400000e-02f,  2.280800000e-02f,  3.424100000e-02f,  2.282600000e-02f,
                3.426800000e-02f,  2.284400000e-02f,  3.429500000e-02f,  2.286200000e-02f,  3.432200000e-02f,  2.288000000e-02f,
                3.434900000e-02f,  2.289800000e-02f,  3.437600000e-02f,  2.291600000e-02f,  3.440300000e-02f,  2.293400000e-02f,
                3.443000000e-02f,  2.295200000e-02f,  3.445700000e-02f,  2.297000000e-02f,  3.448400000e-02f,  2.298800000e-02f,
                3.451100000e-02f,  2.300600000e-02f,  3.453800000e-02f,  2.302400000e-02f,  3.456500000e-02f,  2.304200000e-02f,
                3.459200000e-02f,  2.306000000e-02f,  3.461900000e-02f,  2.307800000e-02f,  3.464600000e-02f,  2.309600000e-02f,
                3.467300000e-02f,  2.311400000e-02f,  3.470000000e-02f,  2.313200000e-02f,  3.472700000e-02f,  2.315000000e-02f,
                3.475400000e-02f,  2.316800000e-02f,  3.478100000e-02f,  2.318600000e-02f,  3.480800000e-02f,  2.320400000e-02f,
                3.483500000e-02f,  2.322200000e-02f,  3.486200000e-02f,  2.324000000e-02f,  3.488900000e-02f,  2.325800000e-02f,
                3.491600000e-02f,  2.327600000e-02f,  3.494300000e-02f,  2.329400000e-02f,  3.497000000e-02f,  2.331200000e-02f,
                3.499700000e-02f,  2.333000000e-02f,  3.502400000e-02f,  2.334800000e-02f,  3.505100000e-02f,  2.336600000e-02f,
                3.507800000e-02f,  2.338400000e-02f,  3.510500000e-02f,  2.340200000e-02f,  3.513200000e-02f,  2.342000000e-02f,
                3.515900000e-02f,  2.343800000e-02f,  3.518600000e-02f,  2.345600000e-02f,  3.521300000e-02f,  2.347400000e-02f,
                3.524000000e-02f,  2.349200000e-02f,  3.526700000e-02f,  2.351000000e-02f,  3.529400000e-02f,  2.352800000e-02f,
                3.532100000e-02f,  2.354600000e-02f,  3.534800000e-02f,  2.356400000e-02f,  3.537500000e-02f,  2.358200000e-02f,
                3.540200000e-02f,  2.360000000e-02f,  3.542900000e-02f,  2.361800000e-02f,  3.545600000e-02f,  2.363600000e-02f,
                3.548300000e-02f,  2.365400000e-02f,  3.551000000e-02f,  2.367200000e-02f,  3.553700000e-02f,  2.369000000e-02f,
                3.556400000e-02f,  2.370800000e-02f,  3.559100000e-02f,  2.372600000e-02f,  3.561800000e-02f,  2.374400000e-02f,
                3.564500000e-02f,  2.376200000e-02f,  3.567200000e-02f,  2.378000000e-02f,  3.569900000e-02f,  2.379800000e-02f,
                3.572600000e-02f,  2.381600000e-02f,  3.575300000e-02f,  2.383400000e-02f,  3.578000000e-02f,  2.385200000e-02f,
                3.580700000e-02f,  2.387000000e-02f,  3.583400000e-02f,  2.388800000e-02f,  3.586100000e-02f,  2.390600000e-02f,
                3.588800000e-02f,  2.392400000e-02f,  3.591500000e-02f,  2.394200000e-02f,  3.594200000e-02f,  2.396000000e-02f,
                3.596900000e-02f,  2.397800000e-02f,  3.599600000e-02f,  2.399600000e-02f,  3.602300000e-02f,  2.401400000e-02f,
                3.605000000e-02f,  2.403200000e-02f,  3.607700000e-02f,  2.405000000e-02f,  3.610400000e-02f,  2.406800000e-02f,
                3.613100000e-02f,  2.408600000e-02f,  3.615800000e-02f,  2.410400000e-02f,  3.618500000e-02f,  2.412200000e-02f,
                3.621200000e-02f,  2.414000000e-02f,  3.623900000e-02f,  2.415800000e-02f,  3.626600000e-02f,  2.417600000e-02f,
                3.629300000e-02f,  2.419400000e-02f,  3.632000000e-02f,  2.421200000e-02f,  3.634700000e-02f,  2.423000000e-02f,
                3.637400000e-02f,  2.424800000e-02f,  3.640100000e-02f,  2.426600000e-02f,  3.642800000e-02f,  2.428400000e-02f,
                3.645500000e-02f,  2.430200000e-02f,  3.648200000e-02f,  2.432000000e-02f,  3.650900000e-02f,  2.433800000e-02f,
                3.653600000e-02f,  2.435600000e-02f,  3.656300000e-02f,  2.437400000e-02f,  3.659000000e-02f,  2.439200000e-02f,
                3.661700000e-02f,  2.441000000e-02f,  3.664400000e-02f,  2.442800000e-02f,  3.667100000e-02f,  2.444600000e-02f,
                3.669800000e-02f,  2.446400000e-02f,  3.672500000e-02f,  2.448200000e-02f,  3.675200000e-02f,  2.450000000e-02f,
                3.677900000e-02f,  2.451800000e-02f,  3.680600000e-02f,  2.453600000e-02f,  3.683300000e-02f,  2.455400000e-02f,
                3.686000000e-02f,  2.457200000e-02f,  3.688700000e-02f,  2.459000000e-02f,  3.691400000e-02f,  2.460800000e-02f,
                3.694100000e-02f,  2.462600000e-02f,  3.696800000e-02f,  2.464400000e-02f,  3.699500000e-02f,  2.466200000e-02f,
                3.702200000e-02f,  2.468000000e-02f,  3.704900000e-02f,  2.469800000e-02f,  3.707600000e-02f,  2.471600000e-02f,
                3.710300000e-02f,  2.473400000e-02f,  3.713000000e-02f,  2.475200000e-02f,  3.715700000e-02f,  2.477000000e-02f,
                3.718400000e-02f,  2.478800000e-02f,  3.721100000e-02f,  2.480600000e-02f,  3.723800000e-02f,  2.482400000e-02f,
                3.726500000e-02f,  2.484200000e-02f,  3.729200000e-02f,  2.486000000e-02f,  3.731900000e-02f,  2.487800000e-02f,
                3.734600000e-02f,  2.489600000e-02f,  3.737300000e-02f,  2.491400000e-02f,  3.740000000e-02f,  2.493200000e-02f,
                3.742700000e-02f,  2.495000000e-02f,  3.745400000e-02f,  2.496800000e-02f,  3.748100000e-02f,  2.498600000e-02f,
                3.750800000e-02f,  2.500400000e-02f,  3.753500000e-02f,  2.502200000e-02f,  3.756200000e-02f,  2.504000000e-02f,
                3.758900000e-02f,  2.505800000e-02f,  3.761600000e-02f,  2.507600000e-02f,  3.764300000e-02f,  2.509400000e-02f,
                3.767000000e-02f,  2.511200000e-02f,  3.769700000e-02f,  2.513000000e-02f,  3.772400000e-02f,  2.514800000e-02f,
                3.775100000e-02f,  2.516600000e-02f,  3.777800000e-02f,  2.518400000e-02f,  3.780500000e-02f,  2.520200000e-02f,
                3.783200000e-02f,  2.522000000e-02f,  3.785900000e-02f,  2.523800000e-02f,  3.788600000e-02f,  2.525600000e-02f,
                3.791300000e-02f,  2.527400000e-02f,  3.794000000e-02f,  2.529200000e-02f,  3.796700000e-02f,  2.531000000e-02f,
                3.799400000e-02f,  2.532800000e-02f,  3.802100000e-02f,  2.534600000e-02f,  3.804800000e-02f,  2.536400000e-02f,
                3.807500000e-02f,  2.538200000e-02f,  3.810200000e-02f,  2.540000000e-02f,  3.812900000e-02f,  2.541800000e-02f,
                3.815600000e-02f,  2.543600000e-02f,  3.818300000e-02f,  2.545400000e-02f,  3.821000000e-02f,  2.547200000e-02f,
                3.823700000e-02f,  2.549000000e-02f,  3.826400000e-02f,  2.550800000e-02f,  3.829100000e-02f,  2.552600000e-02f,
                3.831800000e-02f,  2.554400000e-02f,  3.834500000e-02f,  2.556200000e-02f,  3.837200000e-02f,  2.558000000e-02f,
                3.839900000e-02f,  2.559800000e-02f,  3.842600000e-02f,  2.561600000e-02f,  3.845300000e-02f,  2.563400000e-02f,
                3.848000000e-02f,  2.565200000e-02f,  3.850700000e-02f,  2.567000000e-02f,  3.853400000e-02f,  2.568800000e-02f,
                3.856100000e-02f,  2.570600000e-02f,  3.858800000e-02f,  2.572400000e-02f,  3.861500000e-02f,  2.574200000e-02f,
                3.864200000e-02f,  2.576000000e-02f,  3.866900000e-02f,  2.577800000e-02f,  3.869600000e-02f,  2.579600000e-02f,
                3.872300000e-02f,  2.581400000e-02f,  3.875000000e-02f,  2.583200000e-02f,  3.877700000e-02f,  2.585000000e-02f,
                3.880400000e-02f,  2.586800000e-02f,  3.883100000e-02f,  2.588600000e-02f,  3.885800000e-02f,  2.590400000e-02f,
                3.888500000e-02f,  2.592200000e-02f,  3.891200000e-02f,  2.594000000e-02f,  3.893900000e-02f,  2.595800000e-02f,
                3.896600000e-02f,  2.597600000e-02f,  3.899300000e-02f,  2.599400000e-02f,  3.902000000e-02f,  2.601200000e-02f,
                3.904700000e-02f,  2.603000000e-02f,  3.907400000e-02f,  2.604800000e-02f,  3.910100000e-02f,  2.606600000e-02f,
                3.912800000e-02f,  2.608400000e-02f,  3.915500000e-02f,  2.610200000e-02f,  3.918200000e-02f,  2.612000000e-02f,
                3.920900000e-02f,  2.613800000e-02f,  3.923600000e-02f,  2.615600000e-02f,  3.926300000e-02f,  2.617400000e-02f,
                3.929000000e-02f,  2.619200000e-02f,  3.931700000e-02f,  2.621000000e-02f,  3.934400000e-02f,  2.622800000e-02f,
                3.937100000e-02f,  2.624600000e-02f,  3.939800000e-02f,  2.626400000e-02f,  3.942500000e-02f,  2.628200000e-02f,
                3.945200000e-02f,  2.630000000e-02f,  3.947900000e-02f,  2.631800000e-02f,  3.950600000e-02f,  2.633600000e-02f,
                3.953300000e-02f,  2.635400000e-02f,  3.956000000e-02f,  2.637200000e-02f,  3.958700000e-02f,  2.639000000e-02f,
                3.961400000e-02f,  2.640800000e-02f,  3.964100000e-02f,  2.642600000e-02f,  3.966800000e-02f,  2.644400000e-02f,
                3.969500000e-02f,  2.646200000e-02f,  3.972200000e-02f,  2.648000000e-02f,  3.974900000e-02f,  2.649800000e-02f,
                3.977600000e-02f,  2.651600000e-02f,  3.980300000e-02f,  2.653400000e-02f,  3.983000000e-02f,  2.655200000e-02f,
                3.985700000e-02f,  2.657000000e-02f,  3.988400000e-02f,  2.658800000e-02f,  3.991100000e-02f,  2.660600000e-02f,
                3.993800000e-02f,  2.662400000e-02f,  3.996500000e-02f,  2.664200000e-02f,  3.999200000e-02f,  2.666000000e-02f,
                4.001900000e-02f,  2.667800000e-02f,  4.004600000e-02f,  2.669600000e-02f,  4.007300000e-02f,  2.671400000e-02f,
                4.010000000e-02f,  2.673200000e-02f,  4.012700000e-02f,  2.675000000e-02f,  4.015400000e-02f,  2.676800000e-02f,
                4.018100000e-02f,  2.678600000e-02f,  4.020800000e-02f,  2.680400000e-02f,  4.023500000e-02f,  2.682200000e-02f,
                4.026200000e-02f,  2.684000000e-02f,  4.028900000e-02f,  2.685800000e-02f,  4.031600000e-02f,  2.687600000e-02f,
                4.034300000e-02f,  2.689400000e-02f,  4.037000000e-02f,  2.691200000e-02f,  4.039700000e-02f,  2.693000000e-02f,
                4.042400000e-02f,  2.694800000e-02f,  4.045100000e-02f,  2.696600000e-02f,  4.047800000e-02f,  2.698400000e-02f,
                4.050500000e-02f,  2.700200000e-02f,  4.053200000e-02f,  2.702000000e-02f,  4.055900000e-02f,  2.703800000e-02f,
                4.058600000e-02f,  2.705600000e-02f,  4.061300000e-02f,  2.707400000e-02f,  4.064000000e-02f,  2.709200000e-02f,
                4.066700000e-02f,  2.711000000e-02f,  4.069400000e-02f,  2.712800000e-02f,  4.072100000e-02f,  2.714600000e-02f,
                4.074800000e-02f,  2.716400000e-02f,  4.077500000e-02f,  2.718200000e-02f,  4.080200000e-02f,  2.720000000e-02f,
                4.082900000e-02f,  2.721800000e-02f,  4.085600000e-02f,  2.723600000e-02f,  4.088300000e-02f,  2.725400000e-02f,
                4.091000000e-02f,  2.727200000e-02f,  4.093700000e-02f,  2.729000000e-02f,  4.096400000e-02f,  2.730800000e-02f,
                4.099100000e-02f,  2.732600000e-02f,  4.101800000e-02f,  2.734400000e-02f,  4.104500000e-02f,  2.736200000e-02f,
                4.107200000e-02f,  2.738000000e-02f,  4.109900000e-02f,  2.739800000e-02f,  4.112600000e-02f,  2.741600000e-02f,
                4.115300000e-02f,  2.743400000e-02f,  4.118000000e-02f,  2.745200000e-02f,  4.120700000e-02f,  2.747000000e-02f,
                4.123400000e-02f,  2.748800000e-02f,  4.126100000e-02f,  2.750600000e-02f,  4.128800000e-02f,  2.752400000e-02f,
                4.131500000e-02f,  2.754200000e-02f,  4.134200000e-02f,  2.756000000e-02f,  4.136900000e-02f,  2.757800000e-02f,
                4.139600000e-02f,  2.759600000e-02f,  4.142300000e-02f,  2.761400000e-02f,  4.145000000e-02f,  2.763200000e-02f,
                4.147700000e-02f,  2.765000000e-02f,  4.150400000e-02f,  2.766800000e-02f,  4.153100000e-02f,  2.768600000e-02f,
                4.155800000e-02f,  2.770400000e-02f,  4.158500000e-02f,  2.772200000e-02f,  4.161200000e-02f,  2.774000000e-02f,
                4.163900000e-02f,  2.775800000e-02f,  4.166600000e-02f,  2.777600000e-02f,  4.169300000e-02f,  2.779400000e-02f,
                4.172000000e-02f,  2.781200000e-02f,  4.174700000e-02f,  2.783000000e-02f,  4.177400000e-02f,  2.784800000e-02f,
                4.180100000e-02f,  2.786600000e-02f,  4.182800000e-02f,  2.788400000e-02f,  4.185500000e-02f,  2.790200000e-02f,
                4.188200000e-02f,  2.792000000e-02f,  4.190900000e-02f,  2.793800000e-02f,  4.193600000e-02f,  2.795600000e-02f,
                4.196300000e-02f,  2.797400000e-02f,  4.199000000e-02f,  2.799200000e-02f,  4.201700000e-02f,  2.801000000e-02f,
                4.204400000e-02f,  2.802800000e-02f,  4.207100000e-02f,  2.804600000e-02f,  4.209800000e-02f,  2.806400000e-02f,
                4.212500000e-02f,  2.808200000e-02f,  4.215200000e-02f,  2.810000000e-02f,  4.217900000e-02f,  2.811800000e-02f,
                4.220600000e-02f,  2.813600000e-02f,  4.223300000e-02f,  2.815400000e-02f,  4.226000000e-02f,  2.817200000e-02f,
                4.228700000e-02f,  2.819000000e-02f,  4.231400000e-02f,  2.820800000e-02f,  4.234100000e-02f,  2.822600000e-02f,
                4.236800000e-02f,  2.824400000e-02f,  4.239500000e-02f,  2.826200000e-02f,  4.242200000e-02f,  2.828000000e-02f,
                4.244900000e-02f,  2.829800000e-02f,  4.247600000e-02f,  2.831600000e-02f,  4.250300000e-02f,  2.833400000e-02f,
                4.253000000e-02f,  2.835200000e-02f,  4.255700000e-02f,  2.837000000e-02f,  4.258400000e-02f,  2.838800000e-02f,
                4.261100000e-02f,  2.840600000e-02f,  4.263800000e-02f,  2.842400000e-02f,  4.266500000e-02f,  2.844200000e-02f,
                4.269200000e-02f,  2.846000000e-02f,  4.271900000e-02f,  2.847800000e-02f,  4.274600000e-02f,  2.849600000e-02f,
                4.277300000e-02f,  2.851400000e-02f,  4.280000000e-02f,  2.853200000e-02f,  4.282700000e-02f,  2.855000000e-02f,
                4.285400000e-02f,  2.856800000e-02f,  4.288100000e-02f,  2.858600000e-02f,  4.290800000e-02f,  2.860400000e-02f,
                4.293500000e-02f,  2.862200000e-02f,  4.296200000e-02f,  2.864000000e-02f,  4.298900000e-02f,  2.865800000e-02f,
                4.301600000e-02f,  2.867600000e-02f,  4.304300000e-02f,  2.869400000e-02f,  4.307000000e-02f,  2.871200000e-02f,
                4.309700000e-02f,  2.873000000e-02f,  4.312400000e-02f,  2.874800000e-02f,  4.315100000e-02f,  2.876600000e-02f,
                4.317800000e-02f,  2.878400000e-02f,  4.320500000e-02f,  2.880200000e-02f,  4.323200000e-02f,  2.882000000e-02f,
                4.325900000e-02f,  2.883800000e-02f,  4.328600000e-02f,  2.885600000e-02f,  4.331300000e-02f,  2.887400000e-02f,
                4.334000000e-02f,  2.889200000e-02f,  4.336700000e-02f,  2.891000000e-02f,  4.339400000e-02f,  2.892800000e-02f,
                4.342100000e-02f,  2.894600000e-02f,  4.344800000e-02f,  2.896400000e-02f,  4.347500000e-02f,  2.898200000e-02f,
                4.350200000e-02f,  2.900000000e-02f,  4.352900000e-02f,  2.901800000e-02f,  4.355600000e-02f,  2.903600000e-02f,
                4.358300000e-02f,  2.905400000e-02f,  4.361000000e-02f,  2.907200000e-02f,  4.363700000e-02f,  2.909000000e-02f,
                4.366400000e-02f,  2.910800000e-02f,  4.369100000e-02f,  2.912600000e-02f,  4.371800000e-02f,  2.914400000e-02f,
                4.374500000e-02f,  2.916200000e-02f,  4.377200000e-02f,  2.918000000e-02f,  4.379900000e-02f,  2.919800000e-02f,
                4.382600000e-02f,  2.921600000e-02f,  4.385300000e-02f,  2.923400000e-02f,  4.388000000e-02f,  2.925200000e-02f,
                4.390700000e-02f,  2.927000000e-02f,  4.393400000e-02f,  2.928800000e-02f,  4.396100000e-02f,  2.930600000e-02f,
                4.398800000e-02f,  2.932400000e-02f,  4.401500000e-02f,  2.934200000e-02f,  4.404200000e-02f,  2.936000000e-02f,
                4.406900000e-02f,  2.937800000e-02f,  4.409600000e-02f,  2.939600000e-02f,  4.412300000e-02f,  2.941400000e-02f,
                4.415000000e-02f,  2.943200000e-02f,  4.417700000e-02f,  2.945000000e-02f,  4.420400000e-02f,  2.946800000e-02f,
                4.423100000e-02f,  2.948600000e-02f,  4.425800000e-02f,  2.950400000e-02f,  4.428500000e-02f,  2.952200000e-02f,
                4.431200000e-02f,  2.954000000e-02f,  4.433900000e-02f,  2.955800000e-02f,  4.436600000e-02f,  2.957600000e-02f,
                4.439300000e-02f,  2.959400000e-02f,  4.442000000e-02f,  2.961200000e-02f,  4.444700000e-02f,  2.963000000e-02f,
                4.447400000e-02f,  2.964800000e-02f,  4.450100000e-02f,  2.966600000e-02f,  4.452800000e-02f,  2.968400000e-02f,
                4.455500000e-02f,  2.970200000e-02f,  4.458200000e-02f,  2.972000000e-02f,  4.460900000e-02f,  2.973800000e-02f,
                4.463600000e-02f,  2.975600000e-02f,  4.466300000e-02f,  2.977400000e-02f,  4.469000000e-02f,  2.979200000e-02f,
                4.471700000e-02f,  2.981000000e-02f,  4.474400000e-02f,  2.982800000e-02f,  4.477100000e-02f,  2.984600000e-02f,
                4.479800000e-02f,  2.986400000e-02f,  4.482500000e-02f,  2.988200000e-02f,  4.485200000e-02f,  2.990000000e-02f,
                4.487900000e-02f,  2.991800000e-02f,  4.490600000e-02f,  2.993600000e-02f,  4.493300000e-02f,  2.995400000e-02f,
                4.496000000e-02f,  2.997200000e-02f,  4.498700000e-02f,  2.999000000e-02f,  4.501400000e-02f,  3.000800000e-02f,
                4.504100000e-02f,  3.002600000e-02f,  4.506800000e-02f,  3.004400000e-02f,  4.509500000e-02f,  3.006200000e-02f,
                4.512200000e-02f,  3.008000000e-02f,  4.514900000e-02f,  3.009800000e-02f,  4.517600000e-02f,  3.011600000e-02f,
                4.520300000e-02f,  3.013400000e-02f,  4.523000000e-02f,  3.015200000e-02f,  4.525700000e-02f,  3.017000000e-02f,
                4.528400000e-02f,  3.018800000e-02f,  4.531100000e-02f,  3.020600000e-02f,  4.533800000e-02f,  3.022400000e-02f,
                4.536500000e-02f,  3.024200000e-02f,  4.539200000e-02f,  3.026000000e-02f,  4.541900000e-02f,  3.027800000e-02f,
                4.544600000e-02f,  3.029600000e-02f,  4.547300000e-02f,  3.031400000e-02f,  4.550000000e-02f,  3.033200000e-02f,
                4.552700000e-02f,  3.035000000e-02f,  4.555400000e-02f,  3.036800000e-02f,  4.558100000e-02f,  3.038600000e-02f,
                4.560800000e-02f,  3.040400000e-02f,  4.563500000e-02f,  3.042200000e-02f,  4.566200000e-02f,  3.044000000e-02f,
                4.568900000e-02f,  3.045800000e-02f,  4.571600000e-02f,  3.047600000e-02f,  4.574300000e-02f,  3.049400000e-02f,
                4.577000000e-02f,  3.051200000e-02f,  4.579700000e-02f,  3.053000000e-02f,  4.582400000e-02f,  3.054800000e-02f,
                4.585100000e-02f,  3.056600000e-02f,  4.587800000e-02f,  3.058400000e-02f,  4.590500000e-02f,  3.060200000e-02f,
                4.593200000e-02f,  3.062000000e-02f,  4.595900000e-02f,  3.063800000e-02f,  4.598600000e-02f,  3.065600000e-02f,
                4.601300000e-02f,  3.067400000e-02f,  4.604000000e-02f,  3.069200000e-02f,  4.606700000e-02f,  3.071000000e-02f,
                4.609400000e-02f,  3.072800000e-02f,  4.612100000e-02f,  3.074600000e-02f,  4.614800000e-02f,  3.076400000e-02f,
                4.617500000e-02f,  3.078200000e-02f,  4.620200000e-02f,  3.080000000e-02f,  4.622900000e-02f,  3.081800000e-02f,
                4.625600000e-02f,  3.083600000e-02f,  4.628300000e-02f,  3.085400000e-02f,  4.631000000e-02f,  3.087200000e-02f,
                4.633700000e-02f,  3.089000000e-02f,  4.636400000e-02f,  3.090800000e-02f,  4.639100000e-02f,  3.092600000e-02f,
                4.641800000e-02f,  3.094400000e-02f,  4.644500000e-02f,  3.096200000e-02f,  4.647200000e-02f,  3.098000000e-02f,
                4.649900000e-02f,  3.099800000e-02f,  4.652600000e-02f,  3.101600000e-02f,  4.655300000e-02f,  3.103400000e-02f,
                4.658000000e-02f,  3.105200000e-02f,  4.660700000e-02f,  3.107000000e-02f,  4.663400000e-02f,  3.108800000e-02f,
                4.666100000e-02f,  3.110600000e-02f,  4.668800000e-02f,  3.112400000e-02f,  4.671500000e-02f,  3.114200000e-02f,
                4.674200000e-02f,  3.116000000e-02f,  4.676900000e-02f,  3.117800000e-02f,  4.679600000e-02f,  3.119600000e-02f,
                4.682300000e-02f,  3.121400000e-02f,  4.685000000e-02f,  3.123200000e-02f,  4.687700000e-02f,  3.125000000e-02f,
                4.690400000e-02f,  3.126800000e-02f,  4.693100000e-02f,  3.128600000e-02f,  4.695800000e-02f,  3.130400000e-02f,
                4.698500000e-02f,  3.132200000e-02f,  4.701200000e-02f,  3.134000000e-02f,  4.703900000e-02f,  3.135800000e-02f,
                4.706600000e-02f,  3.137600000e-02f,  4.709300000e-02f,  3.139400000e-02f,  4.712000000e-02f,  3.141200000e-02f,
                4.714700000e-02f,  3.143000000e-02f,  4.717400000e-02f,  3.144800000e-02f,  4.720100000e-02f,  3.146600000e-02f,
                4.722800000e-02f,  3.148400000e-02f,  4.725500000e-02f,  3.150200000e-02f,  4.728200000e-02f,  3.152000000e-02f,
                4.730900000e-02f,  3.153800000e-02f,  4.733600000e-02f,  3.155600000e-02f,  4.736300000e-02f,  3.157400000e-02f,
                4.739000000e-02f,  3.159200000e-02f,  4.741700000e-02f,  3.161000000e-02f,  4.744400000e-02f,  3.162800000e-02f,
                4.747100000e-02f,  3.164600000e-02f,  4.749800000e-02f,  3.166400000e-02f,  4.752500000e-02f,  3.168200000e-02f,
                4.755200000e-02f,  3.170000000e-02f,  4.757900000e-02f,  3.171800000e-02f,  4.760600000e-02f,  3.173600000e-02f,
                4.763300000e-02f,  3.175400000e-02f,  4.766000000e-02f,  3.177200000e-02f,  4.768700000e-02f,  3.179000000e-02f,
                4.771400000e-02f,  3.180800000e-02f,  4.774100000e-02f,  3.182600000e-02f,  4.776800000e-02f,  3.184400000e-02f,
                4.779500000e-02f,  3.186200000e-02f,  4.782200000e-02f,  3.188000000e-02f,  4.784900000e-02f,  3.189800000e-02f,
                4.787600000e-02f,  3.191600000e-02f,  4.790300000e-02f,  3.193400000e-02f,  4.793000000e-02f,  3.195200000e-02f,
                4.795700000e-02f,  3.197000000e-02f,  4.798400000e-02f,  3.198800000e-02f,  4.801100000e-02f,  3.200600000e-02f,
                4.803800000e-02f,  3.202400000e-02f,  4.806500000e-02f,  3.204200000e-02f,  4.809200000e-02f,  3.206000000e-02f,
                4.811900000e-02f,  3.207800000e-02f,  4.814600000e-02f,  3.209600000e-02f,  4.817300000e-02f,  3.211400000e-02f,
                4.820000000e-02f,  3.213200000e-02f,  4.822700000e-02f,  3.215000000e-02f,  4.825400000e-02f,  3.216800000e-02f,
                4.828100000e-02f,  3.218600000e-02f,  4.830800000e-02f,  3.220400000e-02f,  4.833500000e-02f,  3.222200000e-02f,
                4.836200000e-02f,  3.224000000e-02f,  4.838900000e-02f,  3.225800000e-02f,  4.841600000e-02f,  3.227600000e-02f,
                4.844300000e-02f,  3.229400000e-02f,  4.847000000e-02f,  3.231200000e-02f,  4.849700000e-02f,  3.233000000e-02f,
                4.852400000e-02f,  3.234800000e-02f,  4.855100000e-02f,  3.236600000e-02f,  4.857800000e-02f,  3.238400000e-02f,
                4.860500000e-02f,  3.240200000e-02f,  4.863200000e-02f,  3.242000000e-02f,  4.865900000e-02f,  3.243800000e-02f,
                4.868600000e-02f,  3.245600000e-02f,  4.871300000e-02f,  3.247400000e-02f,  4.874000000e-02f,  3.249200000e-02f,
                4.876700000e-02f,  3.251000000e-02f,  4.879400000e-02f,  3.252800000e-02f,  4.882100000e-02f,  3.254600000e-02f,
                4.884800000e-02f,  3.256400000e-02f,  4.887500000e-02f,  3.258200000e-02f,  4.890200000e-02f,  3.260000000e-02f,
                4.892900000e-02f,  3.261800000e-02f,  4.895600000e-02f,  3.263600000e-02f,  4.898300000e-02f,  3.265400000e-02f,
                4.901000000e-02f,  3.267200000e-02f,  4.903700000e-02f,  3.269000000e-02f,  4.906400000e-02f,  3.270800000e-02f,
                4.909100000e-02f,  3.272600000e-02f,  4.911800000e-02f,  3.274400000e-02f,  4.914500000e-02f,  3.276200000e-02f,
                4.917200000e-02f,  3.278000000e-02f,  4.919900000e-02f,  3.279800000e-02f,  4.922600000e-02f,  3.281600000e-02f,
                4.925300000e-02f,  3.283400000e-02f,  4.928000000e-02f,  3.285200000e-02f,  4.930700000e-02f,  3.287000000e-02f,
                4.933400000e-02f,  3.288800000e-02f,  4.936100000e-02f,  3.290600000e-02f,  4.938800000e-02f,  3.292400000e-02f,
                4.941500000e-02f,  3.294200000e-02f,  4.944200000e-02f,  3.296000000e-02f,  4.946900000e-02f,  3.297800000e-02f,
                4.949600000e-02f,  3.299600000e-02f,  4.952300000e-02f,  3.301400000e-02f,  4.955000000e-02f,  3.303200000e-02f,
                4.957700000e-02f,  3.305000000e-02f,  4.960400000e-02f,  3.306800000e-02f,  4.963100000e-02f,  3.308600000e-02f,
                4.965800000e-02f,  3.310400000e-02f,  4.968500000e-02f,  3.312200000e-02f,  4.971200000e-02f,  3.314000000e-02f,
                4.973900000e-02f,  3.315800000e-02f,  4.976600000e-02f,  3.317600000e-02f,  4.979300000e-02f,  3.319400000e-02f,
                4.982000000e-02f,  3.321200000e-02f,  4.984700000e-02f,  3.323000000e-02f,  4.987400000e-02f,  3.324800000e-02f,
                4.990100000e-02f,  3.326600000e-02f,  4.992800000e-02f,  3.328400000e-02f,  4.995500000e-02f,  3.330200000e-02f,
                4.998200000e-02f,  3.332000000e-02f,  5.000900000e-02f,  3.333800000e-02f,  5.003600000e-02f,  3.335600000e-02f,
                5.006300000e-02f,  3.337400000e-02f,  5.009000000e-02f,  3.339200000e-02f,  5.011700000e-02f,  3.341000000e-02f,
                5.014400000e-02f,  3.342800000e-02f,  5.017100000e-02f,  3.344600000e-02f,  5.019800000e-02f,  3.346400000e-02f,
                5.022500000e-02f,  3.348200000e-02f,  5.025200000e-02f,  3.350000000e-02f,  5.027900000e-02f,  3.351800000e-02f,
                5.030600000e-02f,  3.353600000e-02f,  5.033300000e-02f,  3.355400000e-02f,  5.036000000e-02f,  3.357200000e-02f,
                5.038700000e-02f,  3.359000000e-02f,  5.041400000e-02f,  3.360800000e-02f,  5.044100000e-02f,  3.362600000e-02f,
                5.046800000e-02f,  3.364400000e-02f,  5.049500000e-02f,  3.366200000e-02f,  5.052200000e-02f,  3.368000000e-02f,
                5.054900000e-02f,  3.369800000e-02f,  5.057600000e-02f,  3.371600000e-02f,  5.060300000e-02f,  3.373400000e-02f,
                5.063000000e-02f,  3.375200000e-02f,  5.065700000e-02f,  3.377000000e-02f,  5.068400000e-02f,  3.378800000e-02f,
                5.071100000e-02f,  3.380600000e-02f,  5.073800000e-02f,  3.382400000e-02f,  5.076500000e-02f,  3.384200000e-02f,
                5.079200000e-02f,  3.386000000e-02f,  5.081900000e-02f,  3.387800000e-02f,  5.084600000e-02f,  3.389600000e-02f,
                5.087300000e-02f,  3.391400000e-02f,  5.090000000e-02f,  3.393200000e-02f,  5.092700000e-02f,  3.395000000e-02f,
                5.095400000e-02f,  3.396800000e-02f,  5.098100000e-02f,  3.398600000e-02f,  5.100800000e-02f,  3.400400000e-02f,
                5.103500000e-02f,  3.402200000e-02f,  5.106200000e-02f,  3.404000000e-02f,  5.108900000e-02f,  3.405800000e-02f,
                5.111600000e-02f,  3.407600000e-02f,  5.114300000e-02f,  3.409400000e-02f,  5.117000000e-02f,  3.411200000e-02f,
                5.119700000e-02f,  3.413000000e-02f,  5.122400000e-02f,  3.414800000e-02f,  5.125100000e-02f,  3.416600000e-02f,
                5.127800000e-02f,  3.418400000e-02f,  5.130500000e-02f,  3.420200000e-02f,  5.133200000e-02f,  3.422000000e-02f,
                5.135900000e-02f,  3.423800000e-02f,  5.138600000e-02f,  3.425600000e-02f,  5.141300000e-02f,  3.427400000e-02f,
                5.144000000e-02f,  3.429200000e-02f,  5.146700000e-02f,  3.431000000e-02f,  5.149400000e-02f,  3.432800000e-02f,
                5.152100000e-02f,  3.434600000e-02f,  5.154800000e-02f,  3.436400000e-02f,  5.157500000e-02f,  3.438200000e-02f,
                5.160200000e-02f,  3.440000000e-02f,  5.162900000e-02f,  3.441800000e-02f,  5.165600000e-02f,  3.443600000e-02f,
                5.168300000e-02f,  3.445400000e-02f,  5.171000000e-02f,  3.447200000e-02f,  5.173700000e-02f,  3.449000000e-02f,
                5.176400000e-02f,  3.450800000e-02f,  5.179100000e-02f,  3.452600000e-02f,  5.181800000e-02f,  3.454400000e-02f,
                5.184500000e-02f,  3.456200000e-02f,  5.187200000e-02f,  3.458000000e-02f,  5.189900000e-02f,  3.459800000e-02f,
                5.192600000e-02f,  3.461600000e-02f,  5.195300000e-02f,  3.463400000e-02f,  5.198000000e-02f,  3.465200000e-02f,
                5.200700000e-02f,  3.467000000e-02f,  5.203400000e-02f,  3.468800000e-02f,  5.206100000e-02f,  3.470600000e-02f,
                5.208800000e-02f,  3.472400000e-02f,  5.211500000e-02f,  3.474200000e-02f,  5.214200000e-02f,  3.476000000e-02f,
                5.216900000e-02f,  3.477800000e-02f,  5.219600000e-02f,  3.479600000e-02f,  5.222300000e-02f,  3.481400000e-02f,
                5.225000000e-02f,  3.483200000e-02f,  5.227700000e-02f,  3.485000000e-02f,  5.230400000e-02f,  3.486800000e-02f,
                5.233100000e-02f,  3.488600000e-02f,  5.235800000e-02f,  3.490400000e-02f,  5.238500000e-02f,  3.492200000e-02f,
                5.241200000e-02f,  3.494000000e-02f,  5.243900000e-02f,  3.495800000e-02f,  5.246600000e-02f,  3.497600000e-02f,
                5.249300000e-02f,  3.499400000e-02f,  5.252000000e-02f,  3.501200000e-02f,  5.254700000e-02f,  3.503000000e-02f,
                5.257400000e-02f,  3.504800000e-02f,  5.260100000e-02f,  3.506600000e-02f,  5.262800000e-02f,  3.508400000e-02f,
                5.265500000e-02f,  3.510200000e-02f,  5.268200000e-02f,  3.512000000e-02f,  5.270900000e-02f,  3.513800000e-02f,
                5.273600000e-02f,  3.515600000e-02f,  5.276300000e-02f,  3.517400000e-02f,  5.279000000e-02f,  3.519200000e-02f,
                5.281700000e-02f,  3.521000000e-02f,  5.284400000e-02f,  3.522800000e-02f,  5.287100000e-02f,  3.524600000e-02f,
                5.289800000e-02f,  3.526400000e-02f,  5.292500000e-02f,  3.528200000e-02f,  5.295200000e-02f,  3.530000000e-02f,
                5.297900000e-02f,  3.531800000e-02f,  5.300600000e-02f,  3.533600000e-02f,  5.303300000e-02f,  3.535400000e-02f,
                5.306000000e-02f,  3.537200000e-02f,  5.308700000e-02f,  3.539000000e-02f,  5.311400000e-02f,  3.540800000e-02f,
                5.314100000e-02f,  3.542600000e-02f,  5.316800000e-02f,  3.544400000e-02f,  5.319500000e-02f,  3.546200000e-02f,
                5.322200000e-02f,  3.548000000e-02f,  5.324900000e-02f,  3.549800000e-02f,  5.327600000e-02f,  3.551600000e-02f,
                5.330300000e-02f,  3.553400000e-02f,  5.333000000e-02f,  3.555200000e-02f,  5.335700000e-02f,  3.557000000e-02f,
                5.338400000e-02f,  3.558800000e-02f,  5.341100000e-02f,  3.560600000e-02f,  5.343800000e-02f,  3.562400000e-02f,
                5.346500000e-02f,  3.564200000e-02f,  5.349200000e-02f,  3.566000000e-02f,  5.351900000e-02f,  3.567800000e-02f,
                5.354600000e-02f,  3.569600000e-02f,  5.357300000e-02f,  3.571400000e-02f,  5.360000000e-02f,  3.573200000e-02f,
                5.362700000e-02f,  3.575000000e-02f,  5.365400000e-02f,  3.576800000e-02f,  5.368100000e-02f,  3.578600000e-02f,
                5.370800000e-02f,  3.580400000e-02f,  5.373500000e-02f,  3.582200000e-02f,  5.376200000e-02f,  3.584000000e-02f,
                5.378900000e-02f,  3.585800000e-02f,  5.381600000e-02f,  3.587600000e-02f,  5.384300000e-02f,  3.589400000e-02f,
                5.387000000e-02f,  3.591200000e-02f,  5.389700000e-02f,  3.593000000e-02f,  5.392400000e-02f,  3.594800000e-02f,
                5.395100000e-02f,  3.596600000e-02f,  5.397800000e-02f,  3.598400000e-02f,  5.400500000e-02f,  3.600200000e-02f,
                5.403200000e-02f,  3.602000000e-02f,  5.405900000e-02f,  3.603800000e-02f,  5.408600000e-02f,  3.605600000e-02f,
                5.411300000e-02f,  3.607400000e-02f,  5.414000000e-02f,  3.609200000e-02f,  5.416700000e-02f,  3.611000000e-02f,
                5.419400000e-02f,  3.612800000e-02f,  5.422100000e-02f,  3.614600000e-02f,  5.424800000e-02f,  3.616400000e-02f,
                5.427500000e-02f,  3.618200000e-02f,  5.430200000e-02f,  3.620000000e-02f,  5.432900000e-02f,  3.621800000e-02f,
                5.435600000e-02f,  3.623600000e-02f,  5.438300000e-02f,  3.625400000e-02f,  5.441000000e-02f,  3.627200000e-02f,
                5.443700000e-02f,  3.629000000e-02f,  5.446400000e-02f,  3.630800000e-02f,  5.449100000e-02f,  3.632600000e-02f,
                5.451800000e-02f,  3.634400000e-02f,  5.454500000e-02f,  3.636200000e-02f,  5.457200000e-02f,  3.638000000e-02f,
                5.459900000e-02f,  3.639800000e-02f,  5.462600000e-02f,  3.641600000e-02f,  5.465300000e-02f,  3.643400000e-02f,
                5.468000000e-02f,  3.645200000e-02f,  5.470700000e-02f,  3.647000000e-02f,  5.473400000e-02f,  3.648800000e-02f,
                5.476100000e-02f,  3.650600000e-02f,  5.478800000e-02f,  3.652400000e-02f,  5.481500000e-02f,  3.654200000e-02f,
                5.484200000e-02f,  3.656000000e-02f,  5.486900000e-02f,  3.657800000e-02f,  5.489600000e-02f,  3.659600000e-02f,
                5.492300000e-02f,  3.661400000e-02f,  5.495000000e-02f,  3.663200000e-02f,  5.497700000e-02f,  3.665000000e-02f,
                5.500400000e-02f,  3.666800000e-02f,  5.503100000e-02f,  3.668600000e-02f,  5.505800000e-02f,  3.670400000e-02f,
                5.508500000e-02f,  3.672200000e-02f,  5.511200000e-02f,  3.674000000e-02f,  5.513900000e-02f,  3.675800000e-02f,
                5.516600000e-02f,  3.677600000e-02f,  5.519300000e-02f,  3.679400000e-02f,  5.522000000e-02f,  3.681200000e-02f,
                5.524700000e-02f,  3.683000000e-02f,  5.527400000e-02f,  3.684800000e-02f,  5.530100000e-02f,  3.686600000e-02f,
                5.532800000e-02f,  3.688400000e-02f,  5.535500000e-02f,  3.690200000e-02f,  5.538200000e-02f,  3.692000000e-02f,
                5.540900000e-02f,  3.693800000e-02f,  5.543600000e-02f,  3.695600000e-02f,  5.546300000e-02f,  3.697400000e-02f,
                5.549000000e-02f,  3.699200000e-02f,  5.551700000e-02f,  3.701000000e-02f,  5.554400000e-02f,  3.702800000e-02f,
                5.557100000e-02f,  3.704600000e-02f,  5.559800000e-02f,  3.706400000e-02f,  5.562500000e-02f,  3.708200000e-02f,
                5.565200000e-02f,  3.710000000e-02f,  5.567900000e-02f,  3.711800000e-02f,  5.570600000e-02f,  3.713600000e-02f,
                5.573300000e-02f,  3.715400000e-02f,  5.576000000e-02f,  3.717200000e-02f,  5.578700000e-02f,  3.719000000e-02f,
                5.581400000e-02f,  3.720800000e-02f,  5.584100000e-02f,  3.722600000e-02f,  5.586800000e-02f,  3.724400000e-02f,
                5.589500000e-02f,  3.726200000e-02f,  5.592200000e-02f,  3.728000000e-02f,  5.594900000e-02f,  3.729800000e-02f,
                5.597600000e-02f,  3.731600000e-02f,  5.600300000e-02f,  3.733400000e-02f,  5.603000000e-02f,  3.735200000e-02f,
                5.605700000e-02f,  3.737000000e-02f,  5.608400000e-02f,  3.738800000e-02f,  5.611100000e-02f,  3.740600000e-02f,
                5.613800000e-02f,  3.742400000e-02f,  5.616500000e-02f,  3.744200000e-02f,  5.619200000e-02f,  3.746000000e-02f,
                5.621900000e-02f,  3.747800000e-02f,  5.624600000e-02f,  3.749600000e-02f,  5.627300000e-02f,  3.751400000e-02f,
                5.630000000e-02f,  3.753200000e-02f,  5.632700000e-02f,  3.755000000e-02f,  5.635400000e-02f,  3.756800000e-02f,
                5.638100000e-02f,  3.758600000e-02f,  5.640800000e-02f,  3.760400000e-02f,  5.643500000e-02f,  3.762200000e-02f,
                5.646200000e-02f,  3.764000000e-02f,  5.648900000e-02f,  3.765800000e-02f,  5.651600000e-02f,  3.767600000e-02f,
                5.654300000e-02f,  3.769400000e-02f,  5.657000000e-02f,  3.771200000e-02f,  5.659700000e-02f,  3.773000000e-02f,
                5.662400000e-02f,  3.774800000e-02f,  5.665100000e-02f,  3.776600000e-02f,  5.667800000e-02f,  3.778400000e-02f,
                5.670500000e-02f,  3.780200000e-02f,  5.673200000e-02f,  3.782000000e-02f,  5.675900000e-02f,  3.783800000e-02f,
                5.678600000e-02f,  3.785600000e-02f,  5.681300000e-02f,  3.787400000e-02f,  5.684000000e-02f,  3.789200000e-02f,
                5.686700000e-02f,  3.791000000e-02f,  5.689400000e-02f,  3.792800000e-02f,  5.692100000e-02f,  3.794600000e-02f,
                5.694800000e-02f,  3.796400000e-02f,  5.697500000e-02f,  3.798200000e-02f,  5.700200000e-02f,  3.800000000e-02f,
                5.702900000e-02f,  3.801800000e-02f,  5.705600000e-02f,  3.803600000e-02f,  5.708300000e-02f,  3.805400000e-02f,
                5.711000000e-02f,  3.807200000e-02f,  5.713700000e-02f,  3.809000000e-02f,  5.716400000e-02f,  3.810800000e-02f,
                5.719100000e-02f,  3.812600000e-02f,  5.721800000e-02f,  3.814400000e-02f,  5.724500000e-02f,  3.816200000e-02f,
                5.727200000e-02f,  3.818000000e-02f,  5.729900000e-02f,  3.819800000e-02f,  5.732600000e-02f,  3.821600000e-02f,
                5.735300000e-02f,  3.823400000e-02f,  5.738000000e-02f,  3.825200000e-02f,  5.740700000e-02f,  3.827000000e-02f,
                5.743400000e-02f,  3.828800000e-02f,  5.746100000e-02f,  3.830600000e-02f,  5.748800000e-02f,  3.832400000e-02f,
                5.751500000e-02f,  3.834200000e-02f,  5.754200000e-02f,  3.836000000e-02f,  5.756900000e-02f,  3.837800000e-02f,
                5.759600000e-02f,  3.839600000e-02f,  5.762300000e-02f,  3.841400000e-02f,  5.765000000e-02f,  3.843200000e-02f,
                5.767700000e-02f,  3.845000000e-02f,  5.770400000e-02f,  3.846800000e-02f,  5.773100000e-02f,  3.848600000e-02f,
                5.775800000e-02f,  3.850400000e-02f,  5.778500000e-02f,  3.852200000e-02f,  5.781200000e-02f,  3.854000000e-02f,
                5.783900000e-02f,  3.855800000e-02f,  5.786600000e-02f,  3.857600000e-02f,  5.789300000e-02f,  3.859400000e-02f,
                5.792000000e-02f,  3.861200000e-02f,  5.794700000e-02f,  3.863000000e-02f,  5.797400000e-02f,  3.864800000e-02f,
                5.800100000e-02f,  3.866600000e-02f,  5.802800000e-02f,  3.868400000e-02f,  5.805500000e-02f,  3.870200000e-02f,
                5.808200000e-02f,  3.872000000e-02f,  5.810900000e-02f,  3.873800000e-02f,  5.813600000e-02f,  3.875600000e-02f,
                5.816300000e-02f,  3.877400000e-02f,  5.819000000e-02f,  3.879200000e-02f,  5.821700000e-02f,  3.881000000e-02f,
                5.824400000e-02f,  3.882800000e-02f,  5.827100000e-02f,  3.884600000e-02f,  5.829800000e-02f,  3.886400000e-02f,
                5.832500000e-02f,  3.888200000e-02f,  5.835200000e-02f,  3.890000000e-02f,  5.837900000e-02f,  3.891800000e-02f,
                5.840600000e-02f,  3.893600000e-02f,  5.843300000e-02f,  3.895400000e-02f,  5.846000000e-02f,  3.897200000e-02f,
                5.848700000e-02f,  3.899000000e-02f,  5.851400000e-02f,  3.900800000e-02f,  5.854100000e-02f,  3.902600000e-02f,
                5.856800000e-02f,  3.904400000e-02f,  5.859500000e-02f,  3.906200000e-02f,  5.862200000e-02f,  3.908000000e-02f,
                5.864900000e-02f,  3.909800000e-02f,  5.867600000e-02f,  3.911600000e-02f,  5.870300000e-02f,  3.913400000e-02f,
                5.873000000e-02f,  3.915200000e-02f,  5.875700000e-02f,  3.917000000e-02f,  5.878400000e-02f,  3.918800000e-02f,
                5.881100000e-02f,  3.920600000e-02f,  5.883800000e-02f,  3.922400000e-02f,  5.886500000e-02f,  3.924200000e-02f,
                5.889200000e-02f,  3.926000000e-02f,  5.891900000e-02f,  3.927800000e-02f,  5.894600000e-02f,  3.929600000e-02f,
                5.897300000e-02f,  3.931400000e-02f,  5.900000000e-02f,  3.933200000e-02f,  5.902700000e-02f,  3.935000000e-02f,
                5.905400000e-02f,  3.936800000e-02f,  5.908100000e-02f,  3.938600000e-02f,  5.910800000e-02f,  3.940400000e-02f,
                5.913500000e-02f,  3.942200000e-02f,  5.916200000e-02f,  3.944000000e-02f,  5.918900000e-02f,  3.945800000e-02f,
                5.921600000e-02f,  3.947600000e-02f,  5.924300000e-02f,  3.949400000e-02f,  5.927000000e-02f,  3.951200000e-02f,
                5.929700000e-02f,  3.953000000e-02f,  5.932400000e-02f,  3.954800000e-02f,  5.935100000e-02f,  3.956600000e-02f,
                5.937800000e-02f,  3.958400000e-02f,  5.940500000e-02f,  3.960200000e-02f,  5.943200000e-02f,  3.962000000e-02f,
                5.945900000e-02f,  3.963800000e-02f,  5.948600000e-02f,  3.965600000e-02f,  5.951300000e-02f,  3.967400000e-02f,
                5.954000000e-02f,  3.969200000e-02f,  5.956700000e-02f,  3.971000000e-02f,  5.959400000e-02f,  3.972800000e-02f,
                5.962100000e-02f,  3.974600000e-02f,  5.964800000e-02f,  3.976400000e-02f,  5.967500000e-02f,  3.978200000e-02f,
                5.970200000e-02f,  3.980000000e-02f,  5.972900000e-02f,  3.981800000e-02f,  5.975600000e-02f,  3.983600000e-02f,
                5.978300000e-02f,  3.985400000e-02f,  5.981000000e-02f,  3.987200000e-02f,  5.983700000e-02f,  3.989000000e-02f,
                5.986400000e-02f,  3.990800000e-02f,  5.989100000e-02f,  3.992600000e-02f,  5.991800000e-02f,  3.994400000e-02f,
                5.994500000e-02f,  3.996200000e-02f,  5.997200000e-02f,  3.998000000e-02f,  5.999900000e-02f,  3.999800000e-02f,
                6.002600000e-02f,  4.001600000e-02f,  6.005300000e-02f,  4.003400000e-02f,  6.008000000e-02f,  4.005200000e-02f,
                6.010700000e-02f,  4.007000000e-02f,  6.013400000e-02f,  4.008800000e-02f,  6.016100000e-02f,  4.010600000e-02f,
                6.018800000e-02f,  4.012400000e-02f,  6.021500000e-02f,  4.014200000e-02f,  6.024200000e-02f,  4.016000000e-02f,
                6.026900000e-02f,  4.017800000e-02f,  6.029600000e-02f,  4.019600000e-02f,  6.032300000e-02f,  4.021400000e-02f,
                6.035000000e-02f,  4.023200000e-02f,  6.037700000e-02f,  4.025000000e-02f,  6.040400000e-02f,  4.026800000e-02f,
                6.043100000e-02f,  4.028600000e-02f,  6.045800000e-02f,  4.030400000e-02f,  6.048500000e-02f,  4.032200000e-02f,
                6.051200000e-02f,  4.034000000e-02f,  6.053900000e-02f,  4.035800000e-02f,  6.056600000e-02f,  4.037600000e-02f,
                6.059300000e-02f,  4.039400000e-02f,  6.062000000e-02f,  4.041200000e-02f,  6.064700000e-02f,  4.043000000e-02f,
                6.067400000e-02f,  4.044800000e-02f,  6.070100000e-02f,  4.046600000e-02f,  6.072800000e-02f,  4.048400000e-02f,
                6.075500000e-02f,  4.050200000e-02f,  6.078200000e-02f,  4.052000000e-02f,  6.080900000e-02f,  4.053800000e-02f,
                6.083600000e-02f,  4.055600000e-02f,  6.086300000e-02f,  4.057400000e-02f,  6.089000000e-02f,  4.059200000e-02f,
                6.091700000e-02f,  4.061000000e-02f,  6.094400000e-02f,  4.062800000e-02f,  6.097100000e-02f,  4.064600000e-02f,
                6.099800000e-02f,  4.066400000e-02f,  6.102500000e-02f,  4.068200000e-02f,  6.105200000e-02f,  4.070000000e-02f,
                6.107900000e-02f,  4.071800000e-02f,  6.110600000e-02f,  4.073600000e-02f,  6.113300000e-02f,  4.075400000e-02f,
                6.116000000e-02f,  4.077200000e-02f,  6.118700000e-02f,  4.079000000e-02f,  6.121400000e-02f,  4.080800000e-02f,
                6.124100000e-02f,  4.082600000e-02f,  6.126800000e-02f,  4.084400000e-02f,  6.129500000e-02f,  4.086200000e-02f,
                6.132200000e-02f,  4.088000000e-02f,  6.134900000e-02f,  4.089800000e-02f,  6.137600000e-02f,  4.091600000e-02f,
                6.140300000e-02f,  4.093400000e-02f,  6.143000000e-02f,  4.095200000e-02f,  6.145700000e-02f,  4.097000000e-02f,
                6.148400000e-02f,  4.098800000e-02f,  6.151100000e-02f,  4.100600000e-02f,  6.153800000e-02f,  4.102400000e-02f,
                6.156500000e-02f,  4.104200000e-02f,  6.159200000e-02f,  4.106000000e-02f,  6.161900000e-02f,  4.107800000e-02f,
                6.164600000e-02f,  4.109600000e-02f,  6.167300000e-02f,  4.111400000e-02f,  6.170000000e-02f,  4.113200000e-02f,
                6.172700000e-02f,  4.115000000e-02f,  6.175400000e-02f,  4.116800000e-02f,  6.178100000e-02f,  4.118600000e-02f,
                6.180800000e-02f,  4.120400000e-02f,  6.183500000e-02f,  4.122200000e-02f,  6.186200000e-02f,  4.124000000e-02f,
                6.188900000e-02f,  4.125800000e-02f,  6.191600000e-02f,  4.127600000e-02f,  6.194300000e-02f,  4.129400000e-02f,
                6.197000000e-02f,  4.131200000e-02f,  6.199700000e-02f,  4.133000000e-02f,  6.202400000e-02f,  4.134800000e-02f,
                6.205100000e-02f,  4.136600000e-02f,  6.207800000e-02f,  4.138400000e-02f,  6.210500000e-02f,  4.140200000e-02f,
                6.213200000e-02f,  4.142000000e-02f,  6.215900000e-02f,  4.143800000e-02f,  6.218600000e-02f,  4.145600000e-02f,
                6.221300000e-02f,  4.147400000e-02f,  6.224000000e-02f,  4.149200000e-02f,  6.226700000e-02f,  4.151000000e-02f,
                6.229400000e-02f,  4.152800000e-02f,  6.232100000e-02f,  4.154600000e-02f,  6.234800000e-02f,  4.156400000e-02f,
                6.237500000e-02f,  4.158200000e-02f,  6.240200000e-02f,  4.160000000e-02f,  6.242900000e-02f,  4.161800000e-02f,
                6.245600000e-02f,  4.163600000e-02f,  6.248300000e-02f,  4.165400000e-02f,  6.251000000e-02f,  4.167200000e-02f,
                6.253700000e-02f,  4.169000000e-02f,  6.256400000e-02f,  4.170800000e-02f,  6.259100000e-02f,  4.172600000e-02f,
                6.261800000e-02f,  4.174400000e-02f,  6.264500000e-02f,  4.176200000e-02f,  6.267200000e-02f,  4.178000000e-02f,
                6.269900000e-02f,  4.179800000e-02f,  6.272600000e-02f,  4.181600000e-02f,  6.275300000e-02f,  4.183400000e-02f,
                6.278000000e-02f,  4.185200000e-02f,  6.280700000e-02f,  4.187000000e-02f,  6.283400000e-02f,  4.188800000e-02f,
                6.286100000e-02f,  4.190600000e-02f,  6.288800000e-02f,  4.192400000e-02f,  6.291500000e-02f,  4.194200000e-02f,
                6.294200000e-02f,  4.196000000e-02f,  6.296900000e-02f,  4.197800000e-02f,  6.299600000e-02f,  4.199600000e-02f,
                6.302300000e-02f,  4.201400000e-02f,  6.305000000e-02f,  4.203200000e-02f,  6.307700000e-02f,  4.205000000e-02f,
                6.310400000e-02f,  4.206800000e-02f,  6.313100000e-02f,  4.208600000e-02f,  6.315800000e-02f,  4.210400000e-02f,
                6.318500000e-02f,  4.212200000e-02f,  6.321200000e-02f,  4.214000000e-02f,  6.323900000e-02f,  4.215800000e-02f,
                6.326600000e-02f,  4.217600000e-02f,  6.329300000e-02f,  4.219400000e-02f,  6.332000000e-02f,  4.221200000e-02f,
                6.334700000e-02f,  4.223000000e-02f,  6.337400000e-02f,  4.224800000e-02f,  6.340100000e-02f,  4.226600000e-02f,
                6.342800000e-02f,  4.228400000e-02f,  6.345500000e-02f,  4.230200000e-02f,  6.348200000e-02f,  4.232000000e-02f,
                6.350900000e-02f,  4.233800000e-02f,  6.353600000e-02f,  4.235600000e-02f,  6.356300000e-02f,  4.237400000e-02f,
                6.359000000e-02f,  4.239200000e-02f,  6.361700000e-02f,  4.241000000e-02f,  6.364400000e-02f,  4.242800000e-02f,
                6.367100000e-02f,  4.244600000e-02f,  6.369800000e-02f,  4.246400000e-02f,  6.372500000e-02f,  4.248200000e-02f,
                6.375200000e-02f,  4.250000000e-02f,  6.377900000e-02f,  4.251800000e-02f,  6.380600000e-02f,  4.253600000e-02f,
                6.383300000e-02f,  4.255400000e-02f,  6.386000000e-02f,  4.257200000e-02f,  6.388700000e-02f,  4.259000000e-02f,
                6.391400000e-02f,  4.260800000e-02f,  6.394100000e-02f,  4.262600000e-02f,  6.396800000e-02f,  4.264400000e-02f,
                6.399500000e-02f,  4.266200000e-02f,  6.402200000e-02f,  4.268000000e-02f,  6.404900000e-02f,  4.269800000e-02f,
                6.407600000e-02f,  4.271600000e-02f,  6.410300000e-02f,  4.273400000e-02f,  6.413000000e-02f,  4.275200000e-02f,
                6.415700000e-02f,  4.277000000e-02f,  6.418400000e-02f,  4.278800000e-02f,  6.421100000e-02f,  4.280600000e-02f,
                6.423800000e-02f,  4.282400000e-02f,  6.426500000e-02f,  4.284200000e-02f,  6.429200000e-02f,  4.286000000e-02f,
                6.431900000e-02f,  4.287800000e-02f,  6.434600000e-02f,  4.289600000e-02f,  6.437300000e-02f,  4.291400000e-02f,
                6.440000000e-02f,  4.293200000e-02f,  6.442700000e-02f,  4.295000000e-02f,  6.445400000e-02f,  4.296800000e-02f,
                6.448100000e-02f,  4.298600000e-02f,  6.450800000e-02f,  4.300400000e-02f,  6.453500000e-02f,  4.302200000e-02f,
                6.456200000e-02f,  4.304000000e-02f,  6.458900000e-02f,  4.305800000e-02f,  6.461600000e-02f,  4.307600000e-02f,
                6.464300000e-02f,  4.309400000e-02f,  6.467000000e-02f,  4.311200000e-02f,  6.469700000e-02f,  4.313000000e-02f,
                6.472400000e-02f,  4.314800000e-02f,  6.475100000e-02f,  4.316600000e-02f,  6.477800000e-02f,  4.318400000e-02f,
                6.480500000e-02f,  4.320200000e-02f,  6.483200000e-02f,  4.322000000e-02f,  6.485900000e-02f,  4.323800000e-02f,
                6.488600000e-02f,  4.325600000e-02f,  6.491300000e-02f,  4.327400000e-02f,  6.494000000e-02f,  4.329200000e-02f,
                6.496700000e-02f,  4.331000000e-02f,  6.499400000e-02f,  4.332800000e-02f,  6.502100000e-02f,  4.334600000e-02f,
                6.504800000e-02f,  4.336400000e-02f,  6.507500000e-02f,  4.338200000e-02f,  6.510200000e-02f,  4.340000000e-02f,
                6.512900000e-02f,  4.341800000e-02f,  6.515600000e-02f,  4.343600000e-02f,  6.518300000e-02f,  4.345400000e-02f,
                6.521000000e-02f,  4.347200000e-02f,  6.523700000e-02f,  4.349000000e-02f,  6.526400000e-02f,  4.350800000e-02f,
                6.529100000e-02f,  4.352600000e-02f,  6.531800000e-02f,  4.354400000e-02f,  6.534500000e-02f,  4.356200000e-02f,
                6.537200000e-02f,  4.358000000e-02f,  6.539900000e-02f,  4.359800000e-02f,  6.542600000e-02f,  4.361600000e-02f,
                6.545300000e-02f,  4.363400000e-02f,  6.548000000e-02f,  4.365200000e-02f,  6.550700000e-02f,  4.367000000e-02f,
                6.553400000e-02f,  4.368800000e-02f,  6.556100000e-02f,  4.370600000e-02f,  6.558800000e-02f,  4.372400000e-02f,
                6.561500000e-02f,  4.374200000e-02f,  6.564200000e-02f,  4.376000000e-02f,  6.566900000e-02f,  4.377800000e-02f,
                6.569600000e-02f,  4.379600000e-02f,  6.572300000e-02f,  4.381400000e-02f,  6.575000000e-02f,  4.383200000e-02f,
                6.577700000e-02f,  4.385000000e-02f,  6.580400000e-02f,  4.386800000e-02f,  6.583100000e-02f,  4.388600000e-02f,
                6.585800000e-02f,  4.390400000e-02f,  6.588500000e-02f,  4.392200000e-02f,  6.591200000e-02f,  4.394000000e-02f,
                6.593900000e-02f,  4.395800000e-02f,  6.596600000e-02f,  4.397600000e-02f,  6.599300000e-02f,  4.399400000e-02f,
                6.602000000e-02f,  4.401200000e-02f,  6.604700000e-02f,  4.403000000e-02f,  6.607400000e-02f,  4.404800000e-02f,
                6.610100000e-02f,  4.406600000e-02f,  6.612800000e-02f,  4.408400000e-02f,  6.615500000e-02f,  4.410200000e-02f,
                6.618200000e-02f,  4.412000000e-02f,  6.620900000e-02f,  4.413800000e-02f,  6.623600000e-02f,  4.415600000e-02f,
                6.626300000e-02f,  4.417400000e-02f,  6.629000000e-02f,  4.419200000e-02f,  6.631700000e-02f,  4.421000000e-02f,
                6.634400000e-02f,  4.422800000e-02f,  6.637100000e-02f,  4.424600000e-02f,  6.639800000e-02f,  4.426400000e-02f,
                6.642500000e-02f,  4.428200000e-02f,  6.645200000e-02f,  4.430000000e-02f,  6.647900000e-02f,  4.431800000e-02f,
                6.650600000e-02f,  4.433600000e-02f,  6.653300000e-02f,  4.435400000e-02f,  6.656000000e-02f,  4.437200000e-02f,
                6.658700000e-02f,  4.439000000e-02f,  6.661400000e-02f,  4.440800000e-02f,  6.664100000e-02f,  4.442600000e-02f,
                6.666800000e-02f,  4.444400000e-02f,  6.669500000e-02f,  4.446200000e-02f,  6.672200000e-02f,  4.448000000e-02f,
                6.674900000e-02f,  4.449800000e-02f,  6.677600000e-02f,  4.451600000e-02f,  6.680300000e-02f,  4.453400000e-02f,
                6.683000000e-02f,  4.455200000e-02f,  6.685700000e-02f,  4.457000000e-02f,  6.688400000e-02f,  4.458800000e-02f,
                6.691100000e-02f,  4.460600000e-02f,  6.693800000e-02f,  4.462400000e-02f,  6.696500000e-02f,  4.464200000e-02f,
                6.699200000e-02f,  4.466000000e-02f,  6.701900000e-02f,  4.467800000e-02f,  6.704600000e-02f,  4.469600000e-02f,
                6.707300000e-02f,  4.471400000e-02f,  6.710000000e-02f,  4.473200000e-02f,  6.712700000e-02f,  4.475000000e-02f,
                6.715400000e-02f,  4.476800000e-02f,  6.718100000e-02f,  4.478600000e-02f,  6.720800000e-02f,  4.480400000e-02f,
                6.723500000e-02f,  4.482200000e-02f,  6.726200000e-02f,  4.484000000e-02f,  6.728900000e-02f,  4.485800000e-02f,
                6.731600000e-02f,  4.487600000e-02f,  6.734300000e-02f,  4.489400000e-02f,  6.737000000e-02f,  4.491200000e-02f,
                6.739700000e-02f,  4.493000000e-02f,  6.742400000e-02f,  4.494800000e-02f,  6.745100000e-02f,  4.496600000e-02f,
                6.747800000e-02f,  4.498400000e-02f,  6.750500000e-02f,  4.500200000e-02f,  6.753200000e-02f,  4.502000000e-02f,
                6.755900000e-02f,  4.503800000e-02f,  6.758600000e-02f,  4.505600000e-02f,  6.761300000e-02f,  4.507400000e-02f,
                6.764000000e-02f,  4.509200000e-02f,  6.766700000e-02f,  4.511000000e-02f,  6.769400000e-02f,  4.512800000e-02f,
                6.772100000e-02f,  4.514600000e-02f,  6.774800000e-02f,  4.516400000e-02f,  6.777500000e-02f,  4.518200000e-02f,
                6.780200000e-02f,  4.520000000e-02f,  6.782900000e-02f,  4.521800000e-02f,  6.785600000e-02f,  4.523600000e-02f,
                6.788300000e-02f,  4.525400000e-02f,  6.791000000e-02f,  4.527200000e-02f,  6.793700000e-02f,  4.529000000e-02f,
                6.796400000e-02f,  4.530800000e-02f,  6.799100000e-02f,  4.532600000e-02f,  6.801800000e-02f,  4.534400000e-02f,
                6.804500000e-02f,  4.536200000e-02f,  6.807200000e-02f,  4.538000000e-02f,  6.809900000e-02f,  4.539800000e-02f,
                6.812600000e-02f,  4.541600000e-02f,  6.815300000e-02f,  4.543400000e-02f,  6.818000000e-02f,  4.545200000e-02f,
                6.820700000e-02f,  4.547000000e-02f,  6.823400000e-02f,  4.548800000e-02f,  6.826100000e-02f,  4.550600000e-02f,
                6.828800000e-02f,  4.552400000e-02f,  6.831500000e-02f,  4.554200000e-02f,  6.834200000e-02f,  4.556000000e-02f,
                6.836900000e-02f,  4.557800000e-02f,  6.839600000e-02f,  4.559600000e-02f,  6.842300000e-02f,  4.561400000e-02f,
                6.845000000e-02f,  4.563200000e-02f,  6.847700000e-02f,  4.565000000e-02f,  6.850400000e-02f,  4.566800000e-02f,
                6.853100000e-02f,  4.568600000e-02f,  6.855800000e-02f,  4.570400000e-02f,  6.858500000e-02f,  4.572200000e-02f,
                6.861200000e-02f,  4.574000000e-02f,  6.863900000e-02f,  4.575800000e-02f,  6.866600000e-02f,  4.577600000e-02f,
                6.869300000e-02f,  4.579400000e-02f,  6.872000000e-02f,  4.581200000e-02f,  6.874700000e-02f,  4.583000000e-02f,
                6.877400000e-02f,  4.584800000e-02f,  6.880100000e-02f,  4.586600000e-02f,  6.882800000e-02f,  4.588400000e-02f,
                6.885500000e-02f,  4.590200000e-02f,  6.888200000e-02f,  4.592000000e-02f,  6.890900000e-02f,  4.593800000e-02f,
                6.893600000e-02f,  4.595600000e-02f,  6.896300000e-02f,  4.597400000e-02f,  6.899000000e-02f,  4.599200000e-02f,
                6.901700000e-02f,  4.601000000e-02f,  6.904400000e-02f,  4.602800000e-02f,  6.907100000e-02f,  4.604600000e-02f,
                6.909800000e-02f,  4.606400000e-02f,  6.912500000e-02f,  4.608200000e-02f,  6.915200000e-02f,  4.610000000e-02f,
                6.917900000e-02f,  4.611800000e-02f,  6.920600000e-02f,  4.613600000e-02f,  6.923300000e-02f,  4.615400000e-02f,
                6.926000000e-02f,  4.617200000e-02f,  6.928700000e-02f,  4.619000000e-02f,  6.931400000e-02f,  4.620800000e-02f,
                6.934100000e-02f,  4.622600000e-02f,  6.936800000e-02f,  4.624400000e-02f,  6.939500000e-02f,  4.626200000e-02f,
                6.942200000e-02f,  4.628000000e-02f,  6.944900000e-02f,  4.629800000e-02f,  6.947600000e-02f,  4.631600000e-02f,
                6.950300000e-02f,  4.633400000e-02f,  6.953000000e-02f,  4.635200000e-02f,  6.955700000e-02f,  4.637000000e-02f,
                6.958400000e-02f,  4.638800000e-02f,  6.961100000e-02f,  4.640600000e-02f,  6.963800000e-02f,  4.642400000e-02f,
                6.966500000e-02f,  4.644200000e-02f,  6.969200000e-02f,  4.646000000e-02f,  6.971900000e-02f,  4.647800000e-02f,
                6.974600000e-02f,  4.649600000e-02f,  6.977300000e-02f,  4.651400000e-02f,  6.980000000e-02f,  4.653200000e-02f,
                6.982700000e-02f,  4.655000000e-02f,  6.985400000e-02f,  4.656800000e-02f,  6.988100000e-02f,  4.658600000e-02f,
                6.990800000e-02f,  4.660400000e-02f,  6.993500000e-02f,  4.662200000e-02f,  6.996200000e-02f,  4.664000000e-02f,
                6.998900000e-02f,  4.665800000e-02f,  7.001600000e-02f,  4.667600000e-02f,  7.004300000e-02f,  4.669400000e-02f,
                7.007000000e-02f,  4.671200000e-02f,  7.009700000e-02f,  4.673000000e-02f,  7.012400000e-02f,  4.674800000e-02f,
                7.015100000e-02f,  4.676600000e-02f,  7.017800000e-02f,  4.678400000e-02f,  7.020500000e-02f,  4.680200000e-02f,
                7.023200000e-02f,  4.682000000e-02f,  7.025900000e-02f,  4.683800000e-02f,  7.028600000e-02f,  4.685600000e-02f,
                7.031300000e-02f,  4.687400000e-02f,  7.034000000e-02f,  4.689200000e-02f,  7.036700000e-02f,  4.691000000e-02f,
                7.039400000e-02f,  4.692800000e-02f,  7.042100000e-02f,  4.694600000e-02f,  7.044800000e-02f,  4.696400000e-02f,
                7.047500000e-02f,  4.698200000e-02f,  7.050200000e-02f,  4.700000000e-02f,  7.052900000e-02f,  4.701800000e-02f,
                7.055600000e-02f,  4.703600000e-02f,  7.058300000e-02f,  4.705400000e-02f,  7.061000000e-02f,  4.707200000e-02f,
                7.063700000e-02f,  4.709000000e-02f,  7.066400000e-02f,  4.710800000e-02f,  7.069100000e-02f,  4.712600000e-02f,
                7.071800000e-02f,  4.714400000e-02f,  7.074500000e-02f,  4.716200000e-02f,  7.077200000e-02f,  4.718000000e-02f,
                7.079900000e-02f,  4.719800000e-02f,  7.082600000e-02f,  4.721600000e-02f,  7.085300000e-02f,  4.723400000e-02f,
                7.088000000e-02f,  4.725200000e-02f,  7.090700000e-02f,  4.727000000e-02f,  7.093400000e-02f,  4.728800000e-02f,
                7.096100000e-02f,  4.730600000e-02f,  7.098800000e-02f,  4.732400000e-02f,  7.101500000e-02f,  4.734200000e-02f,
                7.104200000e-02f,  4.736000000e-02f,  7.106900000e-02f,  4.737800000e-02f,  7.109600000e-02f,  4.739600000e-02f,
                7.112300000e-02f,  4.741400000e-02f,  7.115000000e-02f,  4.743200000e-02f,  7.117700000e-02f,  4.745000000e-02f,
                7.120400000e-02f,  4.746800000e-02f,  7.123100000e-02f,  4.748600000e-02f,  7.125800000e-02f,  4.750400000e-02f,
                7.128500000e-02f,  4.752200000e-02f,  7.131200000e-02f,  4.754000000e-02f,  7.133900000e-02f,  4.755800000e-02f,
                7.136600000e-02f,  4.757600000e-02f,  7.139300000e-02f,  4.759400000e-02f,  7.142000000e-02f,  4.761200000e-02f,
                7.144700000e-02f,  4.763000000e-02f,  7.147400000e-02f,  4.764800000e-02f,  7.150100000e-02f,  4.766600000e-02f,
                7.152800000e-02f,  4.768400000e-02f,  7.155500000e-02f,  4.770200000e-02f,  7.158200000e-02f,  4.772000000e-02f,
                7.160900000e-02f,  4.773800000e-02f,  7.163600000e-02f,  4.775600000e-02f,  7.166300000e-02f,  4.777400000e-02f,
                7.169000000e-02f,  4.779200000e-02f,  7.171700000e-02f,  4.781000000e-02f,  7.174400000e-02f,  4.782800000e-02f,
                7.177100000e-02f,  4.784600000e-02f,  7.179800000e-02f,  4.786400000e-02f,  7.182500000e-02f,  4.788200000e-02f,
                7.185200000e-02f,  4.790000000e-02f,  7.187900000e-02f,  4.791800000e-02f,  7.190600000e-02f,  4.793600000e-02f,
                7.193300000e-02f,  4.795400000e-02f,  7.196000000e-02f,  4.797200000e-02f,  7.198700000e-02f,  4.799000000e-02f,
                7.201400000e-02f,  4.800800000e-02f,  7.204100000e-02f,  4.802600000e-02f,  7.206800000e-02f,  4.804400000e-02f,
                7.209500000e-02f,  4.806200000e-02f,  7.212200000e-02f,  4.808000000e-02f,  7.214900000e-02f,  4.809800000e-02f,
                7.217600000e-02f,  4.811600000e-02f,  7.220300000e-02f,  4.813400000e-02f,  7.223000000e-02f,  4.815200000e-02f,
                7.225700000e-02f,  4.817000000e-02f,  7.228400000e-02f,  4.818800000e-02f,  7.231100000e-02f,  4.820600000e-02f,
                7.233800000e-02f,  4.822400000e-02f,  7.236500000e-02f,  4.824200000e-02f,  7.239200000e-02f,  4.826000000e-02f,
                7.241900000e-02f,  4.827800000e-02f,  7.244600000e-02f,  4.829600000e-02f,  7.247300000e-02f,  4.831400000e-02f,
                7.250000000e-02f,  4.833200000e-02f,  7.252700000e-02f,  4.835000000e-02f,  7.255400000e-02f,  4.836800000e-02f,
                7.258100000e-02f,  4.838600000e-02f,  7.260800000e-02f,  4.840400000e-02f,  7.263500000e-02f,  4.842200000e-02f,
                7.266200000e-02f,  4.844000000e-02f,  7.268900000e-02f,  4.845800000e-02f,  7.271600000e-02f,  4.847600000e-02f,
                7.274300000e-02f,  4.849400000e-02f,  7.277000000e-02f,  4.851200000e-02f,  7.279700000e-02f,  4.853000000e-02f,
                7.282400000e-02f,  4.854800000e-02f,  7.285100000e-02f,  4.856600000e-02f,  7.287800000e-02f,  4.858400000e-02f,
                7.290500000e-02f,  4.860200000e-02f,  7.293200000e-02f,  4.862000000e-02f,  7.295900000e-02f,  4.863800000e-02f,
                7.298600000e-02f,  4.865600000e-02f,  7.301300000e-02f,  4.867400000e-02f,  7.304000000e-02f,  4.869200000e-02f,
                7.306700000e-02f,  4.871000000e-02f,  7.309400000e-02f,  4.872800000e-02f,  7.312100000e-02f,  4.874600000e-02f,
                7.314800000e-02f,  4.876400000e-02f,  7.317500000e-02f,  4.878200000e-02f,  7.320200000e-02f,  4.880000000e-02f,
                7.322900000e-02f,  4.881800000e-02f,  7.325600000e-02f,  4.883600000e-02f,  7.328300000e-02f,  4.885400000e-02f,
                7.331000000e-02f,  4.887200000e-02f,  7.333700000e-02f,  4.889000000e-02f,  7.336400000e-02f,  4.890800000e-02f,
                7.339100000e-02f,  4.892600000e-02f,  7.341800000e-02f,  4.894400000e-02f,  7.344500000e-02f,  4.896200000e-02f,
                7.347200000e-02f,  4.898000000e-02f,  7.349900000e-02f,  4.899800000e-02f,  7.352600000e-02f,  4.901600000e-02f,
                7.355300000e-02f,  4.903400000e-02f,  7.358000000e-02f,  4.905200000e-02f,  7.360700000e-02f,  4.907000000e-02f,
                7.363400000e-02f,  4.908800000e-02f,  7.366100000e-02f,  4.910600000e-02f,  7.368800000e-02f,  4.912400000e-02f,
                7.371500000e-02f,  4.914200000e-02f,  7.374200000e-02f,  4.916000000e-02f,  7.376900000e-02f,  4.917800000e-02f,
                7.379600000e-02f,  4.919600000e-02f,  7.382300000e-02f,  4.921400000e-02f,  7.385000000e-02f,  4.923200000e-02f,
                7.387700000e-02f,  4.925000000e-02f,  7.390400000e-02f,  4.926800000e-02f,  7.393100000e-02f,  4.928600000e-02f,
                7.395800000e-02f,  4.930400000e-02f,  7.398500000e-02f,  4.932200000e-02f,  7.401200000e-02f,  4.934000000e-02f,
                7.403900000e-02f,  4.935800000e-02f,  7.406600000e-02f,  4.937600000e-02f,  7.409300000e-02f,  4.939400000e-02f,
                7.412000000e-02f,  4.941200000e-02f,  7.414700000e-02f,  4.943000000e-02f,  7.417400000e-02f,  4.944800000e-02f,
                7.420100000e-02f,  4.946600000e-02f,  7.422800000e-02f,  4.948400000e-02f,  7.425500000e-02f,  4.950200000e-02f,
                7.428200000e-02f,  4.952000000e-02f,  7.430900000e-02f,  4.953800000e-02f,  7.433600000e-02f,  4.955600000e-02f,
                7.436300000e-02f,  4.957400000e-02f,  7.439000000e-02f,  4.959200000e-02f,  7.441700000e-02f,  4.961000000e-02f,
                7.444400000e-02f,  4.962800000e-02f,  7.447100000e-02f,  4.964600000e-02f,  7.449800000e-02f,  4.966400000e-02f,
                7.452500000e-02f,  4.968200000e-02f,  7.455200000e-02f,  4.970000000e-02f,  7.457900000e-02f,  4.971800000e-02f,
                7.460600000e-02f,  4.973600000e-02f,  7.463300000e-02f,  4.975400000e-02f,  7.466000000e-02f,  4.977200000e-02f,
                7.468700000e-02f,  4.979000000e-02f,  7.471400000e-02f,  4.980800000e-02f,  7.474100000e-02f,  4.982600000e-02f,
                7.476800000e-02f,  4.984400000e-02f,  7.479500000e-02f,  4.986200000e-02f,  7.482200000e-02f,  4.988000000e-02f,
                7.484900000e-02f,  4.989800000e-02f,  7.487600000e-02f,  4.991600000e-02f,  7.490300000e-02f,  4.993400000e-02f,
                7.493000000e-02f,  4.995200000e-02f,  7.495700000e-02f,  4.997000000e-02f,  7.498400000e-02f,  4.998800000e-02f,
                7.501100000e-02f,  5.000600000e-02f,  7.503800000e-02f,  5.002400000e-02f,  7.506500000e-02f,  5.004200000e-02f,
                7.509200000e-02f,  5.006000000e-02f,  7.511900000e-02f,  5.007800000e-02f,  7.514600000e-02f,  5.009600000e-02f,
                7.517300000e-02f,  5.011400000e-02f,  7.520000000e-02f,  5.013200000e-02f,  7.522700000e-02f,  5.015000000e-02f,
                7.525400000e-02f,  5.016800000e-02f,  7.528100000e-02f,  5.018600000e-02f,  7.530800000e-02f,  5.020400000e-02f,
                7.533500000e-02f,  5.022200000e-02f,  7.536200000e-02f,  5.024000000e-02f,  7.538900000e-02f,  5.025800000e-02f,
                7.541600000e-02f,  5.027600000e-02f,  7.544300000e-02f,  5.029400000e-02f,  7.547000000e-02f,  5.031200000e-02f,
                7.549700000e-02f,  5.033000000e-02f,  7.552400000e-02f,  5.034800000e-02f,  7.555100000e-02f,  5.036600000e-02f,
                7.557800000e-02f,  5.038400000e-02f,  7.560500000e-02f,  5.040200000e-02f,  7.563200000e-02f,  5.042000000e-02f,
                7.565900000e-02f,  5.043800000e-02f,  7.568600000e-02f,  5.045600000e-02f,  7.571300000e-02f,  5.047400000e-02f,
                7.574000000e-02f,  5.049200000e-02f,  7.576700000e-02f,  5.051000000e-02f,  7.579400000e-02f,  5.052800000e-02f,
                7.582100000e-02f,  5.054600000e-02f,  7.584800000e-02f,  5.056400000e-02f,  7.587500000e-02f,  5.058200000e-02f,
                7.590200000e-02f,  5.060000000e-02f,  7.592900000e-02f,  5.061800000e-02f,  7.595600000e-02f,  5.063600000e-02f,
                7.598300000e-02f,  5.065400000e-02f,  7.601000000e-02f,  5.067200000e-02f,  7.603700000e-02f,  5.069000000e-02f,
                7.606400000e-02f,  5.070800000e-02f,  7.609100000e-02f,  5.072600000e-02f,  7.611800000e-02f,  5.074400000e-02f,
                7.614500000e-02f,  5.076200000e-02f,  7.617200000e-02f,  5.078000000e-02f,  7.619900000e-02f,  5.079800000e-02f,
                7.622600000e-02f,  5.081600000e-02f,  7.625300000e-02f,  5.083400000e-02f,  7.628000000e-02f,  5.085200000e-02f,
                7.630700000e-02f,  5.087000000e-02f,  7.633400000e-02f,  5.088800000e-02f,  7.636100000e-02f,  5.090600000e-02f,
                7.638800000e-02f,  5.092400000e-02f,  7.641500000e-02f,  5.094200000e-02f,  7.644200000e-02f,  5.096000000e-02f,
                7.646900000e-02f,  5.097800000e-02f,  7.649600000e-02f,  5.099600000e-02f,  7.652300000e-02f,  5.101400000e-02f,
                7.655000000e-02f,  5.103200000e-02f,  7.657700000e-02f,  5.105000000e-02f,  7.660400000e-02f,  5.106800000e-02f,
                7.663100000e-02f,  5.108600000e-02f,  7.665800000e-02f,  5.110400000e-02f,  7.668500000e-02f,  5.112200000e-02f,
                7.671200000e-02f,  5.114000000e-02f,  7.673900000e-02f,  5.115800000e-02f,  7.676600000e-02f,  5.117600000e-02f,
                7.679300000e-02f,  5.119400000e-02f,  7.682000000e-02f,  5.121200000e-02f,  7.684700000e-02f,  5.123000000e-02f,
                7.687400000e-02f,  5.124800000e-02f,  7.690100000e-02f,  5.126600000e-02f,  7.692800000e-02f,  5.128400000e-02f,
                7.695500000e-02f,  5.130200000e-02f,  7.698200000e-02f,  5.132000000e-02f,  7.700900000e-02f,  5.133800000e-02f,
                7.703600000e-02f,  5.135600000e-02f,  7.706300000e-02f,  5.137400000e-02f,  7.709000000e-02f,  5.139200000e-02f,
                7.711700000e-02f,  5.141000000e-02f,  7.714400000e-02f,  5.142800000e-02f,  7.717100000e-02f,  5.144600000e-02f,
                7.719800000e-02f,  5.146400000e-02f,  7.722500000e-02f,  5.148200000e-02f,  7.725200000e-02f,  5.150000000e-02f,
                7.727900000e-02f,  5.151800000e-02f,  7.730600000e-02f,  5.153600000e-02f,  7.733300000e-02f,  5.155400000e-02f,
                7.736000000e-02f,  5.157200000e-02f,  7.738700000e-02f,  5.159000000e-02f,  7.741400000e-02f,  5.160800000e-02f,
                7.744100000e-02f,  5.162600000e-02f,  7.746800000e-02f,  5.164400000e-02f,  7.749500000e-02f,  5.166200000e-02f,
                7.752200000e-02f,  5.168000000e-02f,  7.754900000e-02f,  5.169800000e-02f,  7.757600000e-02f,  5.171600000e-02f,
                7.760300000e-02f,  5.173400000e-02f,  7.763000000e-02f,  5.175200000e-02f,  7.765700000e-02f,  5.177000000e-02f,
                7.768400000e-02f,  5.178800000e-02f,  7.771100000e-02f,  5.180600000e-02f,  7.773800000e-02f,  5.182400000e-02f,
                7.776500000e-02f,  5.184200000e-02f,  7.779200000e-02f,  5.186000000e-02f,  7.781900000e-02f,  5.187800000e-02f,
                7.784600000e-02f,  5.189600000e-02f,  7.787300000e-02f,  5.191400000e-02f,  7.790000000e-02f,  5.193200000e-02f,
                7.792700000e-02f,  5.195000000e-02f,  7.795400000e-02f,  5.196800000e-02f,  7.798100000e-02f,  5.198600000e-02f,
                7.800800000e-02f,  5.200400000e-02f,  7.803500000e-02f,  5.202200000e-02f,  7.806200000e-02f,  5.204000000e-02f,
                7.808900000e-02f,  5.205800000e-02f,  7.811600000e-02f,  5.207600000e-02f,  7.814300000e-02f,  5.209400000e-02f,
                7.817000000e-02f,  5.211200000e-02f,  7.819700000e-02f,  5.213000000e-02f,  7.822400000e-02f,  5.214800000e-02f,
                7.825100000e-02f,  5.216600000e-02f,  7.827800000e-02f,  5.218400000e-02f,  7.830500000e-02f,  5.220200000e-02f,
                7.833200000e-02f,  5.222000000e-02f,  7.835900000e-02f,  5.223800000e-02f,  7.838600000e-02f,  5.225600000e-02f,
                7.841300000e-02f,  5.227400000e-02f,  7.844000000e-02f,  5.229200000e-02f,  7.846700000e-02f,  5.231000000e-02f,
                7.849400000e-02f,  5.232800000e-02f,  7.852100000e-02f,  5.234600000e-02f,  7.854800000e-02f,  5.236400000e-02f,
                7.857500000e-02f,  5.238200000e-02f,  7.860200000e-02f,  5.240000000e-02f,  7.862900000e-02f,  5.241800000e-02f,
                7.865600000e-02f,  5.243600000e-02f,  7.868300000e-02f,  5.245400000e-02f,  7.871000000e-02f,  5.247200000e-02f,
                7.873700000e-02f,  5.249000000e-02f,  7.876400000e-02f,  5.250800000e-02f,  7.879100000e-02f,  5.252600000e-02f,
                7.881800000e-02f,  5.254400000e-02f,  7.884500000e-02f,  5.256200000e-02f,  7.887200000e-02f,  5.258000000e-02f,
                7.889900000e-02f,  5.259800000e-02f,  7.892600000e-02f,  5.261600000e-02f,  7.895300000e-02f,  5.263400000e-02f,
                7.898000000e-02f,  5.265200000e-02f,  7.900700000e-02f,  5.267000000e-02f,  7.903400000e-02f,  5.268800000e-02f,
                7.906100000e-02f,  5.270600000e-02f,  7.908800000e-02f,  5.272400000e-02f,  7.911500000e-02f,  5.274200000e-02f,
                7.914200000e-02f,  5.276000000e-02f,  7.916900000e-02f,  5.277800000e-02f,  7.919600000e-02f,  5.279600000e-02f,
                7.922300000e-02f,  5.281400000e-02f,  7.925000000e-02f,  5.283200000e-02f,  7.927700000e-02f,  5.285000000e-02f,
                7.930400000e-02f,  5.286800000e-02f,  7.933100000e-02f,  5.288600000e-02f,  7.935800000e-02f,  5.290400000e-02f,
                7.938500000e-02f,  5.292200000e-02f,  7.941200000e-02f,  5.294000000e-02f,  7.943900000e-02f,  5.295800000e-02f,
                7.946600000e-02f,  5.297600000e-02f,  7.949300000e-02f,  5.299400000e-02f,  7.952000000e-02f,  5.301200000e-02f,
                7.954700000e-02f,  5.303000000e-02f,  7.957400000e-02f,  5.304800000e-02f,  7.960100000e-02f,  5.306600000e-02f,
                7.962800000e-02f,  5.308400000e-02f,  7.965500000e-02f,  5.310200000e-02f,  7.968200000e-02f,  5.312000000e-02f,
                7.970900000e-02f,  5.313800000e-02f,  7.973600000e-02f,  5.315600000e-02f,  7.976300000e-02f,  5.317400000e-02f,
                7.979000000e-02f,  5.319200000e-02f,  7.981700000e-02f,  5.321000000e-02f,  7.984400000e-02f,  5.322800000e-02f,
                7.987100000e-02f,  5.324600000e-02f,  7.989800000e-02f,  5.326400000e-02f,  7.992500000e-02f,  5.328200000e-02f,
                7.995200000e-02f,  5.330000000e-02f,  7.997900000e-02f,  5.331800000e-02f,  8.000600000e-02f,  5.333600000e-02f,
                8.003300000e-02f,  5.335400000e-02f,  8.006000000e-02f,  5.337200000e-02f,  8.008700000e-02f,  5.339000000e-02f,
                8.011400000e-02f,  5.340800000e-02f,  8.014100000e-02f,  5.342600000e-02f,  8.016800000e-02f,  5.344400000e-02f,
                8.019500000e-02f,  5.346200000e-02f,  8.022200000e-02f,  5.348000000e-02f,  8.024900000e-02f,  5.349800000e-02f,
                8.027600000e-02f,  5.351600000e-02f,  8.030300000e-02f,  5.353400000e-02f,  8.033000000e-02f,  5.355200000e-02f,
                8.035700000e-02f,  5.357000000e-02f,  8.038400000e-02f,  5.358800000e-02f,  8.041100000e-02f,  5.360600000e-02f,
                8.043800000e-02f,  5.362400000e-02f,  8.046500000e-02f,  5.364200000e-02f,  8.049200000e-02f,  5.366000000e-02f,
                8.051900000e-02f,  5.367800000e-02f,  8.054600000e-02f,  5.369600000e-02f,  8.057300000e-02f,  5.371400000e-02f,
                8.060000000e-02f,  5.373200000e-02f,  8.062700000e-02f,  5.375000000e-02f,  8.065400000e-02f,  5.376800000e-02f,
                8.068100000e-02f,  5.378600000e-02f,  8.070800000e-02f,  5.380400000e-02f,  8.073500000e-02f,  5.382200000e-02f,
                8.076200000e-02f,  5.384000000e-02f,  8.078900000e-02f,  5.385800000e-02f,  8.081600000e-02f,  5.387600000e-02f,
                8.084300000e-02f,  5.389400000e-02f,  8.087000000e-02f,  5.391200000e-02f,  8.089700000e-02f,  5.393000000e-02f,
                8.092400000e-02f,  5.394800000e-02f,  8.095100000e-02f,  5.396600000e-02f,  8.097800000e-02f,  5.398400000e-02f,
                8.100500000e-02f,  5.400200000e-02f,  8.103200000e-02f,  5.402000000e-02f,  8.105900000e-02f,  5.403800000e-02f,
                8.108600000e-02f,  5.405600000e-02f,  8.111300000e-02f,  5.407400000e-02f,  8.114000000e-02f,  5.409200000e-02f,
                8.116700000e-02f,  5.411000000e-02f,  8.119400000e-02f,  5.412800000e-02f,  8.122100000e-02f,  5.414600000e-02f,
                8.124800000e-02f,  5.416400000e-02f,  8.127500000e-02f,  5.418200000e-02f,  8.130200000e-02f,  5.420000000e-02f,
                8.132900000e-02f,  5.421800000e-02f,  8.135600000e-02f,  5.423600000e-02f,  8.138300000e-02f,  5.425400000e-02f,
                8.141000000e-02f,  5.427200000e-02f,  8.143700000e-02f,  5.429000000e-02f,  8.146400000e-02f,  5.430800000e-02f,
                8.149100000e-02f,  5.432600000e-02f,  8.151800000e-02f,  5.434400000e-02f,  8.154500000e-02f,  5.436200000e-02f,
                8.157200000e-02f,  5.438000000e-02f,  8.159900000e-02f,  5.439800000e-02f,  8.162600000e-02f,  5.441600000e-02f,
                8.165300000e-02f,  5.443400000e-02f,  8.168000000e-02f,  5.445200000e-02f,  8.170700000e-02f,  5.447000000e-02f,
                8.173400000e-02f,  5.448800000e-02f,  8.176100000e-02f,  5.450600000e-02f,  8.178800000e-02f,  5.452400000e-02f,
                8.181500000e-02f,  5.454200000e-02f,  8.184200000e-02f,  5.456000000e-02f,  8.186900000e-02f,  5.457800000e-02f,
                8.189600000e-02f,  5.459600000e-02f,  8.192300000e-02f,  5.461400000e-02f,  8.195000000e-02f,  5.463200000e-02f,
                8.197700000e-02f,  5.465000000e-02f,  8.200400000e-02f,  5.466800000e-02f,  8.203100000e-02f,  5.468600000e-02f,
                8.205800000e-02f,  5.470400000e-02f,  8.208500000e-02f,  5.472200000e-02f,  8.211200000e-02f,  5.474000000e-02f,
                8.213900000e-02f,  5.475800000e-02f,  8.216600000e-02f,  5.477600000e-02f,  8.219300000e-02f,  5.479400000e-02f,
                8.222000000e-02f,  5.481200000e-02f,  8.224700000e-02f,  5.483000000e-02f,  8.227400000e-02f,  5.484800000e-02f,
                8.230100000e-02f,  5.486600000e-02f,  8.232800000e-02f,  5.488400000e-02f,  8.235500000e-02f,  5.490200000e-02f,
                8.238200000e-02f,  5.492000000e-02f,  8.240900000e-02f,  5.493800000e-02f,  8.243600000e-02f,  5.495600000e-02f,
                8.246300000e-02f,  5.497400000e-02f,  8.249000000e-02f,  5.499200000e-02f,  8.251700000e-02f,  5.501000000e-02f,
                8.254400000e-02f,  5.502800000e-02f,  8.257100000e-02f,  5.504600000e-02f,  8.259800000e-02f,  5.506400000e-02f,
                8.262500000e-02f,  5.508200000e-02f,  8.265200000e-02f,  5.510000000e-02f,  8.267900000e-02f,  5.511800000e-02f,
                8.270600000e-02f,  5.513600000e-02f,  8.273300000e-02f,  5.515400000e-02f,  8.276000000e-02f,  5.517200000e-02f,
                8.278700000e-02f,  5.519000000e-02f,  8.281400000e-02f,  5.520800000e-02f,  8.284100000e-02f,  5.522600000e-02f,
                8.286800000e-02f,  5.524400000e-02f,  8.289500000e-02f,  5.526200000e-02f,  8.292200000e-02f,  5.528000000e-02f,
                8.294900000e-02f,  5.529800000e-02f,  8.297600000e-02f,  5.531600000e-02f,  8.300300000e-02f,  5.533400000e-02f,
                8.303000000e-02f,  5.535200000e-02f,  8.305700000e-02f,  5.537000000e-02f,  8.308400000e-02f,  5.538800000e-02f,
                8.311100000e-02f,  5.540600000e-02f,  8.313800000e-02f,  5.542400000e-02f,  8.316500000e-02f,  5.544200000e-02f,
                8.319200000e-02f,  5.546000000e-02f,  8.321900000e-02f,  5.547800000e-02f,  8.324600000e-02f,  5.549600000e-02f,
                8.327300000e-02f,  5.551400000e-02f,  8.330000000e-02f,  5.553200000e-02f,  8.332700000e-02f,  5.555000000e-02f,
                8.335400000e-02f,  5.556800000e-02f,  8.338100000e-02f,  5.558600000e-02f,  8.340800000e-02f,  5.560400000e-02f,
                8.343500000e-02f,  5.562200000e-02f,  8.346200000e-02f,  5.564000000e-02f,  8.348900000e-02f,  5.565800000e-02f,
                8.351600000e-02f,  5.567600000e-02f,  8.354300000e-02f,  5.569400000e-02f,  8.357000000e-02f,  5.571200000e-02f,
                8.359700000e-02f,  5.573000000e-02f,  8.362400000e-02f,  5.574800000e-02f,  8.365100000e-02f,  5.576600000e-02f,
                8.367800000e-02f,  5.578400000e-02f,  8.370500000e-02f,  5.580200000e-02f,  8.373200000e-02f,  5.582000000e-02f,
                8.375900000e-02f,  5.583800000e-02f,  8.378600000e-02f,  5.585600000e-02f,  8.381300000e-02f,  5.587400000e-02f,
                8.384000000e-02f,  5.589200000e-02f,  8.386700000e-02f,  5.591000000e-02f,  8.389400000e-02f,  5.592800000e-02f,
                8.392100000e-02f,  5.594600000e-02f,  8.394800000e-02f,  5.596400000e-02f,  8.397500000e-02f,  5.598200000e-02f,
                8.400200000e-02f,  5.600000000e-02f,  8.402900000e-02f,  5.601800000e-02f,  8.405600000e-02f,  5.603600000e-02f,
                8.408300000e-02f,  5.605400000e-02f,  8.411000000e-02f,  5.607200000e-02f,  8.413700000e-02f,  5.609000000e-02f,
                8.416400000e-02f,  5.610800000e-02f,  8.419100000e-02f,  5.612600000e-02f,  8.421800000e-02f,  5.614400000e-02f,
                8.424500000e-02f,  5.616200000e-02f,  8.427200000e-02f,  5.618000000e-02f,  8.429900000e-02f,  5.619800000e-02f,
                8.432600000e-02f,  5.621600000e-02f,  8.435300000e-02f,  5.623400000e-02f,  8.438000000e-02f,  5.625200000e-02f,
                8.440700000e-02f,  5.627000000e-02f,  8.443400000e-02f,  5.628800000e-02f,  8.446100000e-02f,  5.630600000e-02f,
                8.448800000e-02f,  5.632400000e-02f,  8.451500000e-02f,  5.634200000e-02f,  8.454200000e-02f,  5.636000000e-02f,
                8.456900000e-02f,  5.637800000e-02f,  8.459600000e-02f,  5.639600000e-02f,  8.462300000e-02f,  5.641400000e-02f,
                8.465000000e-02f,  5.643200000e-02f,  8.467700000e-02f,  5.645000000e-02f,  8.470400000e-02f,  5.646800000e-02f,
                8.473100000e-02f,  5.648600000e-02f,  8.475800000e-02f,  5.650400000e-02f,  8.478500000e-02f,  5.652200000e-02f,
                8.481200000e-02f,  5.654000000e-02f,  8.483900000e-02f,  5.655800000e-02f,  8.486600000e-02f,  5.657600000e-02f,
                8.489300000e-02f,  5.659400000e-02f,  8.492000000e-02f,  5.661200000e-02f,  8.494700000e-02f,  5.663000000e-02f,
                8.497400000e-02f,  5.664800000e-02f,  8.500100000e-02f,  5.666600000e-02f,  8.502800000e-02f,  5.668400000e-02f,
                8.505500000e-02f,  5.670200000e-02f,  8.508200000e-02f,  5.672000000e-02f,  8.510900000e-02f,  5.673800000e-02f,
                8.513600000e-02f,  5.675600000e-02f,  8.516300000e-02f,  5.677400000e-02f,  8.519000000e-02f,  5.679200000e-02f,
                8.521700000e-02f,  5.681000000e-02f,  8.524400000e-02f,  5.682800000e-02f,  8.527100000e-02f,  5.684600000e-02f,
                8.529800000e-02f,  5.686400000e-02f,  8.532500000e-02f,  5.688200000e-02f,  8.535200000e-02f,  5.690000000e-02f,
                8.537900000e-02f,  5.691800000e-02f,  8.540600000e-02f,  5.693600000e-02f,  8.543300000e-02f,  5.695400000e-02f,
                8.546000000e-02f,  5.697200000e-02f,  8.548700000e-02f,  5.699000000e-02f,  8.551400000e-02f,  5.700800000e-02f,
                8.554100000e-02f,  5.702600000e-02f,  8.556800000e-02f,  5.704400000e-02f,  8.559500000e-02f,  5.706200000e-02f,
                8.562200000e-02f,  5.708000000e-02f,  8.564900000e-02f,  5.709800000e-02f,  8.567600000e-02f,  5.711600000e-02f,
                8.570300000e-02f,  5.713400000e-02f,  8.573000000e-02f,  5.715200000e-02f,  8.575700000e-02f,  5.717000000e-02f,
                8.578400000e-02f,  5.718800000e-02f,  8.581100000e-02f,  5.720600000e-02f,  8.583800000e-02f,  5.722400000e-02f,
                8.586500000e-02f,  5.724200000e-02f,  8.589200000e-02f,  5.726000000e-02f,  8.591900000e-02f,  5.727800000e-02f,
                8.594600000e-02f,  5.729600000e-02f,  8.597300000e-02f,  5.731400000e-02f,  8.600000000e-02f,  5.733200000e-02f,
                8.602700000e-02f,  5.735000000e-02f,  8.605400000e-02f,  5.736800000e-02f,  8.608100000e-02f,  5.738600000e-02f,
                8.610800000e-02f,  5.740400000e-02f,  8.613500000e-02f,  5.742200000e-02f,  8.616200000e-02f,  5.744000000e-02f,
                8.618900000e-02f,  5.745800000e-02f,  8.621600000e-02f,  5.747600000e-02f,  8.624300000e-02f,  5.749400000e-02f,
                8.627000000e-02f,  5.751200000e-02f,  8.629700000e-02f,  5.753000000e-02f,  8.632400000e-02f,  5.754800000e-02f,
                8.635100000e-02f,  5.756600000e-02f,  8.637800000e-02f,  5.758400000e-02f,  8.640500000e-02f,  5.760200000e-02f,
                8.643200000e-02f,  5.762000000e-02f,  8.645900000e-02f,  5.763800000e-02f,  8.648600000e-02f,  5.765600000e-02f,
                8.651300000e-02f,  5.767400000e-02f,  8.654000000e-02f,  5.769200000e-02f,  8.656700000e-02f,  5.771000000e-02f,
                8.659400000e-02f,  5.772800000e-02f,  8.662100000e-02f,  5.774600000e-02f,  8.664800000e-02f,  5.776400000e-02f,
                8.667500000e-02f,  5.778200000e-02f,  8.670200000e-02f,  5.780000000e-02f,  8.672900000e-02f,  5.781800000e-02f,
                8.675600000e-02f,  5.783600000e-02f,  8.678300000e-02f,  5.785400000e-02f,  8.681000000e-02f,  5.787200000e-02f,
                8.683700000e-02f,  5.789000000e-02f,  8.686400000e-02f,  5.790800000e-02f,  8.689100000e-02f,  5.792600000e-02f,
                8.691800000e-02f,  5.794400000e-02f,  8.694500000e-02f,  5.796200000e-02f,  8.697200000e-02f,  5.798000000e-02f,
                8.699900000e-02f,  5.799800000e-02f,  8.702600000e-02f,  5.801600000e-02f,  8.705300000e-02f,  5.803400000e-02f,
                8.708000000e-02f,  5.805200000e-02f,  8.710700000e-02f,  5.807000000e-02f,  8.713400000e-02f,  5.808800000e-02f,
                8.716100000e-02f,  5.810600000e-02f,  8.718800000e-02f,  5.812400000e-02f,  8.721500000e-02f,  5.814200000e-02f,
                8.724200000e-02f,  5.816000000e-02f,  8.726900000e-02f,  5.817800000e-02f,  8.729600000e-02f,  5.819600000e-02f,
                8.732300000e-02f,  5.821400000e-02f,  8.735000000e-02f,  5.823200000e-02f,  8.737700000e-02f,  5.825000000e-02f,
                8.740400000e-02f,  5.826800000e-02f,  8.743100000e-02f,  5.828600000e-02f,  8.745800000e-02f,  5.830400000e-02f,
                8.748500000e-02f,  5.832200000e-02f,  8.751200000e-02f,  5.834000000e-02f,  8.753900000e-02f,  5.835800000e-02f,
                8.756600000e-02f,  5.837600000e-02f,  8.759300000e-02f,  5.839400000e-02f,  8.762000000e-02f,  5.841200000e-02f,
                8.764700000e-02f,  5.843000000e-02f,  8.767400000e-02f,  5.844800000e-02f,  8.770100000e-02f,  5.846600000e-02f,
                8.772800000e-02f,  5.848400000e-02f,  8.775500000e-02f,  5.850200000e-02f,  8.778200000e-02f,  5.852000000e-02f,
                8.780900000e-02f,  5.853800000e-02f,  8.783600000e-02f,  5.855600000e-02f,  8.786300000e-02f,  5.857400000e-02f,
                8.789000000e-02f,  5.859200000e-02f,  8.791700000e-02f,  5.861000000e-02f,  8.794400000e-02f,  5.862800000e-02f,
                8.797100000e-02f,  5.864600000e-02f,  8.799800000e-02f,  5.866400000e-02f,  8.802500000e-02f,  5.868200000e-02f,
                8.805200000e-02f,  5.870000000e-02f,  8.807900000e-02f,  5.871800000e-02f,  8.810600000e-02f,  5.873600000e-02f,
                8.813300000e-02f,  5.875400000e-02f,  8.816000000e-02f,  5.877200000e-02f,  8.818700000e-02f,  5.879000000e-02f,
                8.821400000e-02f,  5.880800000e-02f,  8.824100000e-02f,  5.882600000e-02f,  8.826800000e-02f,  5.884400000e-02f,
                8.829500000e-02f,  5.886200000e-02f,  8.832200000e-02f,  5.888000000e-02f,  8.834900000e-02f,  5.889800000e-02f,
                8.837600000e-02f,  5.891600000e-02f,  8.840300000e-02f,  5.893400000e-02f,  8.843000000e-02f,  5.895200000e-02f,
                8.845700000e-02f,  5.897000000e-02f,  8.848400000e-02f,  5.898800000e-02f,  8.851100000e-02f,  5.900600000e-02f,
                8.853800000e-02f,  5.902400000e-02f,  8.856500000e-02f,  5.904200000e-02f,  8.859200000e-02f,  5.906000000e-02f,
                8.861900000e-02f,  5.907800000e-02f,  8.864600000e-02f,  5.909600000e-02f,  8.867300000e-02f,  5.911400000e-02f,
                8.870000000e-02f,  5.913200000e-02f,  8.872700000e-02f,  5.915000000e-02f,  8.875400000e-02f,  5.916800000e-02f,
                8.878100000e-02f,  5.918600000e-02f,  8.880800000e-02f,  5.920400000e-02f,  8.883500000e-02f,  5.922200000e-02f,
                8.886200000e-02f,  5.924000000e-02f,  8.888900000e-02f,  5.925800000e-02f,  8.891600000e-02f,  5.927600000e-02f,
                8.894300000e-02f,  5.929400000e-02f,  8.897000000e-02f,  5.931200000e-02f,  8.899700000e-02f,  5.933000000e-02f,
                8.902400000e-02f,  5.934800000e-02f,  8.905100000e-02f,  5.936600000e-02f,  8.907800000e-02f,  5.938400000e-02f,
                8.910500000e-02f,  5.940200000e-02f,  8.913200000e-02f,  5.942000000e-02f,  8.915900000e-02f,  5.943800000e-02f,
                8.918600000e-02f,  5.945600000e-02f,  8.921300000e-02f,  5.947400000e-02f,  8.924000000e-02f,  5.949200000e-02f,
                8.926700000e-02f,  5.951000000e-02f,  8.929400000e-02f,  5.952800000e-02f,  8.932100000e-02f,  5.954600000e-02f,
                8.934800000e-02f,  5.956400000e-02f,  8.937500000e-02f,  5.958200000e-02f,  8.940200000e-02f,  5.960000000e-02f,
                8.942900000e-02f,  5.961800000e-02f,  8.945600000e-02f,  5.963600000e-02f,  8.948300000e-02f,  5.965400000e-02f,
                8.951000000e-02f,  5.967200000e-02f,  8.953700000e-02f,  5.969000000e-02f,  8.956400000e-02f,  5.970800000e-02f,
                8.959100000e-02f,  5.972600000e-02f,  8.961800000e-02f,  5.974400000e-02f,  8.964500000e-02f,  5.976200000e-02f,
                8.967200000e-02f,  5.978000000e-02f,  8.969900000e-02f,  5.979800000e-02f,  8.972600000e-02f,  5.981600000e-02f,
                8.975300000e-02f,  5.983400000e-02f,  8.978000000e-02f,  5.985200000e-02f,  8.980700000e-02f,  5.987000000e-02f,
                8.983400000e-02f,  5.988800000e-02f,  8.986100000e-02f,  5.990600000e-02f,  8.988800000e-02f,  5.992400000e-02f,
                8.991500000e-02f,  5.994200000e-02f,  8.994200000e-02f,  5.996000000e-02f,  8.996900000e-02f,  5.997800000e-02f,
                8.999600000e-02f,  5.999600000e-02f,  9.002300000e-02f,  6.001400000e-02f,  9.005000000e-02f,  6.003200000e-02f,
                9.007700000e-02f,  6.005000000e-02f,  9.010400000e-02f,  6.006800000e-02f,  9.013100000e-02f,  6.008600000e-02f,
                9.015800000e-02f,  6.010400000e-02f,  9.018500000e-02f,  6.012200000e-02f,  9.021200000e-02f,  6.014000000e-02f,
                9.023900000e-02f,  6.015800000e-02f,  9.026600000e-02f,  6.017600000e-02f,  9.029300000e-02f,  6.019400000e-02f,
                9.032000000e-02f,  6.021200000e-02f,  9.034700000e-02f,  6.023000000e-02f,  9.037400000e-02f,  6.024800000e-02f,
                9.040100000e-02f,  6.026600000e-02f,  9.042800000e-02f,  6.028400000e-02f,  9.045500000e-02f,  6.030200000e-02f,
                9.048200000e-02f,  6.032000000e-02f,  9.050900000e-02f,  6.033800000e-02f,  9.053600000e-02f,  6.035600000e-02f,
                9.056300000e-02f,  6.037400000e-02f,  9.059000000e-02f,  6.039200000e-02f,  9.061700000e-02f,  6.041000000e-02f,
                9.064400000e-02f,  6.042800000e-02f,  9.067100000e-02f,  6.044600000e-02f,  9.069800000e-02f,  6.046400000e-02f,
                9.072500000e-02f,  6.048200000e-02f,  9.075200000e-02f,  6.050000000e-02f,  9.077900000e-02f,  6.051800000e-02f,
                9.080600000e-02f,  6.053600000e-02f,  9.083300000e-02f,  6.055400000e-02f,  9.086000000e-02f,  6.057200000e-02f,
                9.088700000e-02f,  6.059000000e-02f,  9.091400000e-02f,  6.060800000e-02f,  9.094100000e-02f,  6.062600000e-02f,
                9.096800000e-02f,  6.064400000e-02f,  9.099500000e-02f,  6.066200000e-02f,  9.102200000e-02f,  6.068000000e-02f,
                9.104900000e-02f,  6.069800000e-02f,  9.107600000e-02f,  6.071600000e-02f,  9.110300000e-02f,  6.073400000e-02f,
                9.113000000e-02f,  6.075200000e-02f,  9.115700000e-02f,  6.077000000e-02f,  9.118400000e-02f,  6.078800000e-02f,
                9.121100000e-02f,  6.080600000e-02f,  9.123800000e-02f,  6.082400000e-02f,  9.126500000e-02f,  6.084200000e-02f,
                9.129200000e-02f,  6.086000000e-02f,  9.131900000e-02f,  6.087800000e-02f,  9.134600000e-02f,  6.089600000e-02f,
                9.137300000e-02f,  6.091400000e-02f,  9.140000000e-02f,  6.093200000e-02f,  9.142700000e-02f,  6.095000000e-02f,
                9.145400000e-02f,  6.096800000e-02f,  9.148100000e-02f,  6.098600000e-02f,  9.150800000e-02f,  6.100400000e-02f,
                9.153500000e-02f,  6.102200000e-02f,  9.156200000e-02f,  6.104000000e-02f,  9.158900000e-02f,  6.105800000e-02f,
                9.161600000e-02f,  6.107600000e-02f,  9.164300000e-02f,  6.109400000e-02f,  9.167000000e-02f,  6.111200000e-02f,
                9.169700000e-02f,  6.113000000e-02f,  9.172400000e-02f,  6.114800000e-02f,  9.175100000e-02f,  6.116600000e-02f,
                9.177800000e-02f,  6.118400000e-02f,  9.180500000e-02f,  6.120200000e-02f,  9.183200000e-02f,  6.122000000e-02f,
                9.185900000e-02f,  6.123800000e-02f,  9.188600000e-02f,  6.125600000e-02f,  9.191300000e-02f,  6.127400000e-02f,
                9.194000000e-02f,  6.129200000e-02f,  9.196700000e-02f,  6.131000000e-02f,  9.199400000e-02f,  6.132800000e-02f,
                9.202100000e-02f,  6.134600000e-02f,  9.204800000e-02f,  6.136400000e-02f,  9.207500000e-02f,  6.138200000e-02f,
                9.210200000e-02f,  6.140000000e-02f,  9.212900000e-02f,  6.141800000e-02f,  9.215600000e-02f,  6.143600000e-02f,
                9.218300000e-02f,  6.145400000e-02f,  9.221000000e-02f,  6.147200000e-02f,  9.223700000e-02f,  6.149000000e-02f,
                9.226400000e-02f,  6.150800000e-02f,  9.229100000e-02f,  6.152600000e-02f,  9.231800000e-02f,  6.154400000e-02f,
                9.234500000e-02f,  6.156200000e-02f,  9.237200000e-02f,  6.158000000e-02f,  9.239900000e-02f,  6.159800000e-02f,
                9.242600000e-02f,  6.161600000e-02f,  9.245300000e-02f,  6.163400000e-02f,  9.248000000e-02f,  6.165200000e-02f,
                9.250700000e-02f,  6.167000000e-02f,  9.253400000e-02f,  6.168800000e-02f,  9.256100000e-02f,  6.170600000e-02f,
                9.258800000e-02f,  6.172400000e-02f,  9.261500000e-02f,  6.174200000e-02f,  9.264200000e-02f,  6.176000000e-02f,
            };

            float[] x_actual = x.ToArray();

            AssertError.Tolerance(x_expect, x_actual, 1e-7f, 1e-5f, $"mismatch value {inchannels},{outchannels},{inwidth},{inheight},{indepth},{batch}");
        }
    }
}
