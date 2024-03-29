using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;
using TensorShader;
using TensorShader.Operators.Connection3D;
using TensorShaderCudaBackend.API;

namespace TensorShaderTest.Operators.Connection3D {
    [TestClass]
    public class DeconvolutionTest {
        [TestMethod]
        public void ExecuteFPTest() {
            TensorShaderCudaBackend.Environment.Precision = TensorShaderCudaBackend.Environment.PrecisionMode.Float;
            TensorShaderCudaBackend.Environment.CudnnEnabled = false;

            float max_err = 0;

            foreach (int batch in new int[] { 1, 2 }) {
                foreach ((int inchannels, int outchannels) in new (int, int)[] { (1, 1), (2, 1), (1, 2), (2, 3), (3, 1), (3, 4), (4, 5), (5, 3), (5, 10), (10, 15), (15, 5), (15, 20), (20, 32), (32, 15), (32, 33), (33, 33) }) {
                    foreach ((int kwidth, int kheight, int kdepth) in new (int, int, int)[] { (1, 1, 1), (3, 3, 3), (5, 5, 5), (1, 3, 5), (3, 5, 1), (5, 1, 3) }) {
                        foreach ((int inwidth, int inheight, int indepth) in new (int, int, int)[] { (kwidth, kheight, kdepth), (kwidth * 2, kheight * 2, kdepth * 2), (13, 13, 13), (17, 17, 17), (19, 19, 19), (17, 19, 13), (13, 17, 19), (19, 13, 17) }) {

                            int outwidth = inwidth - kwidth + 1, outheight = inheight - kheight + 1, outdepth = indepth - kdepth + 1;

                            float[] yval = (new float[outwidth * outheight * outdepth * outchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
                            float[] wval = (new float[kwidth * kheight * kdepth * inchannels * outchannels]).Select((_, idx) => (idx + 1) * 1e-3f).Reverse().ToArray();

                            Map3D y = new(outchannels, outwidth, outheight, outdepth, batch, yval);
                            Filter3D w = new(inchannels, outchannels, kwidth, kheight, kdepth, wval);

                            Map3D x = Reference(y, w, inwidth, inheight, indepth, kwidth, kheight, kdepth);

                            OverflowCheckedTensor y_tensor = new(Shape.Map3D(outchannels, outwidth, outheight, outdepth, batch), yval);
                            OverflowCheckedTensor w_tensor = new(Shape.Kernel3D(inchannels, outchannels, kwidth, kheight, kdepth), wval);

                            OverflowCheckedTensor x_tensor = new(Shape.Map3D(inchannels, inwidth, inheight, indepth, batch));

                            Deconvolution ope = new(outwidth, outheight, outdepth, outchannels, inchannels, kwidth, kheight, kdepth, batch);

                            ope.Execute(y_tensor, w_tensor, x_tensor);

                            float[] x_expect = x.ToArray();
                            float[] x_actual = x_tensor.State.Value;

                            CollectionAssert.AreEqual(yval, y_tensor.State.Value);
                            CollectionAssert.AreEqual(wval, w_tensor.State.Value);

                            AssertError.Tolerance(x_expect, x_actual, 1e-6f, 1e-4f, ref max_err, $"mismatch value {inchannels},{outchannels},{kwidth},{kheight},{kdepth},{inwidth},{inheight},{indepth},{batch}");

                            Console.WriteLine($"pass: {inchannels},{outchannels},{kwidth},{kheight},{kdepth},{inwidth},{inheight},{indepth},{batch}");
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
                    foreach ((int kwidth, int kheight, int kdepth) in new (int, int, int)[] { (1, 1, 1), (3, 3, 3), (5, 5, 5), (1, 3, 5), (3, 5, 1), (5, 1, 3) }) {
                        foreach ((int inwidth, int inheight, int indepth) in new (int, int, int)[] { (kwidth, kheight, kdepth), (kwidth * 2, kheight * 2, kdepth * 2), (13, 13, 13), (17, 17, 17), (19, 19, 19), (17, 19, 13), (13, 17, 19), (19, 13, 17) }) {

                            int outwidth = inwidth - kwidth + 1, outheight = inheight - kheight + 1, outdepth = indepth - kdepth + 1;

                            float[] yval = (new float[outwidth * outheight * outdepth * outchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
                            float[] wval = (new float[kwidth * kheight * kdepth * inchannels * outchannels]).Select((_, idx) => (idx + 1) * 1e-3f).Reverse().ToArray();

                            Map3D y = new(outchannels, outwidth, outheight, outdepth, batch, yval);
                            Filter3D w = new(inchannels, outchannels, kwidth, kheight, kdepth, wval);

                            Map3D x = Reference(y, w, inwidth, inheight, indepth, kwidth, kheight, kdepth);

                            OverflowCheckedTensor y_tensor = new(Shape.Map3D(outchannels, outwidth, outheight, outdepth, batch), yval);
                            OverflowCheckedTensor w_tensor = new(Shape.Kernel3D(inchannels, outchannels, kwidth, kheight, kdepth), wval);

                            OverflowCheckedTensor x_tensor = new(Shape.Map3D(inchannels, inwidth, inheight, indepth, batch));

                            Deconvolution ope = new(outwidth, outheight, outdepth, outchannels, inchannels, kwidth, kheight, kdepth, batch);

                            ope.Execute(y_tensor, w_tensor, x_tensor);

                            float[] x_expect = x.ToArray();
                            float[] x_actual = x_tensor.State.Value;

                            CollectionAssert.AreEqual(yval, y_tensor.State.Value);
                            CollectionAssert.AreEqual(wval, w_tensor.State.Value);

                            AssertError.Tolerance(x_expect, x_actual, 1e-7f, 1e-5f, ref max_err, $"mismatch value {inchannels},{outchannels},{kwidth},{kheight},{kdepth},{inwidth},{inheight},{indepth},{batch}");

                            Console.WriteLine($"pass: {inchannels},{outchannels},{kwidth},{kheight},{kdepth},{inwidth},{inheight},{indepth},{batch}");
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
                    foreach ((int kwidth, int kheight, int kdepth) in new (int, int, int)[] { (1, 1, 1), (3, 3, 3), (5, 5, 5), (1, 3, 5), (3, 5, 1), (5, 1, 3) }) {
                        foreach ((int inwidth, int inheight, int indepth) in new (int, int, int)[] { (kwidth, kheight, kdepth), (kwidth * 2, kheight * 2, kdepth * 2), (13, 13, 13), (17, 17, 17), (19, 19, 19), (17, 19, 13), (13, 17, 19), (19, 13, 17) }) {

                            int outwidth = inwidth - kwidth + 1, outheight = inheight - kheight + 1, outdepth = indepth - kdepth + 1;

                            float[] yval = (new float[outwidth * outheight * outdepth * outchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
                            float[] wval = (new float[kwidth * kheight * kdepth * inchannels * outchannels]).Select((_, idx) => (idx + 1) * 1e-3f).Reverse().ToArray();

                            Map3D y = new(outchannels, outwidth, outheight, outdepth, batch, yval);
                            Filter3D w = new(inchannels, outchannels, kwidth, kheight, kdepth, wval);

                            Map3D x = Reference(y, w, inwidth, inheight, indepth, kwidth, kheight, kdepth);

                            OverflowCheckedTensor y_tensor = new(Shape.Map3D(outchannels, outwidth, outheight, outdepth, batch), yval);
                            OverflowCheckedTensor w_tensor = new(Shape.Kernel3D(inchannels, outchannels, kwidth, kheight, kdepth), wval);

                            OverflowCheckedTensor x_tensor = new(Shape.Map3D(inchannels, inwidth, inheight, indepth, batch));

                            Deconvolution ope = new(outwidth, outheight, outdepth, outchannels, inchannels, kwidth, kheight, kdepth, batch);

                            ope.Execute(y_tensor, w_tensor, x_tensor);

                            float[] x_expect = x.ToArray();
                            float[] x_actual = x_tensor.State.Value;

                            CollectionAssert.AreEqual(yval, y_tensor.State.Value);
                            CollectionAssert.AreEqual(wval, w_tensor.State.Value);

                            AssertError.Tolerance(x_expect, x_actual, 1e-6f, 1e-4f, ref max_err, $"mismatch value {inchannels},{outchannels},{kwidth},{kheight},{kdepth},{inwidth},{inheight},{indepth},{batch}");

                            Console.WriteLine($"pass: {inchannels},{outchannels},{kwidth},{kheight},{kdepth},{inwidth},{inheight},{indepth},{batch}");
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
            int kwidth = 7, kheight = 5, kdepth = 3;
            int inwidth = 125, inheight = 196, indepth = 4;
            int outwidth = inwidth - kwidth + 1, outheight = inheight - kheight + 1, outdepth = indepth - kdepth + 1;

            float[] yval = (new float[outwidth * outheight * outdepth * outchannels * batch]).Select((_, idx) => (float)random.NextDouble() * 1e-2f).ToArray();
            float[] wval = (new float[kwidth * kheight * kdepth * inchannels * outchannels]).Select((_, idx) => (float)random.NextDouble() * 1e-2f).ToArray();

            Map3D y = new(outchannels, outwidth, outheight, outdepth, batch, yval);
            Filter3D w = new(inchannels, outchannels, kwidth, kheight, kdepth, wval);

            Map3D x = Reference(y, w, inwidth, inheight, indepth, kwidth, kheight, kdepth);

            OverflowCheckedTensor y_tensor = new(Shape.Map3D(outchannels, outwidth, outheight, outdepth, batch), yval);
            OverflowCheckedTensor w_tensor = new(Shape.Kernel3D(inchannels, outchannels, kwidth, kheight, kdepth), wval);

            OverflowCheckedTensor x_tensor = new(Shape.Map3D(inchannels, inwidth, inheight, indepth, batch));

            Deconvolution ope = new(outwidth, outheight, outdepth, outchannels, inchannels, kwidth, kheight, kdepth, batch);

            ope.Execute(y_tensor, w_tensor, x_tensor);

            float[] x_expect = x.ToArray();
            float[] x_actual = x_tensor.State.Value;

            CollectionAssert.AreEqual(yval, y_tensor.State.Value);
            CollectionAssert.AreEqual(wval, w_tensor.State.Value);

            AssertError.Tolerance(x_expect, x_actual, 1e-7f, 1e-5f, ref max_err, $"mismatch value {inchannels},{outchannels},{kwidth},{kheight},{kdepth},{inwidth},{inheight},{indepth},{batch}");

            Console.WriteLine($"maxerr:{max_err}");
        }

        [TestMethod]
        public void SpeedFPTest() {
            TensorShaderCudaBackend.Environment.Precision = TensorShaderCudaBackend.Environment.PrecisionMode.Float;
            TensorShaderCudaBackend.Environment.CudnnEnabled = false;

            int inwidth = 64, inheight = 64, indepth = 64, inchannels = 31, outchannels = 31, ksize = 3;
            int outwidth = inwidth - ksize + 1, outheight = inheight - ksize + 1, outdepth = indepth - ksize + 1;

            OverflowCheckedTensor y_tensor = new(Shape.Map3D(outchannels, outwidth, outheight, outdepth));
            OverflowCheckedTensor w_tensor = new(Shape.Kernel3D(inchannels, outchannels, ksize, ksize, ksize));

            OverflowCheckedTensor x_tensor = new(Shape.Map3D(inchannels, inwidth, inheight, indepth));

            Deconvolution ope = new(outwidth, outheight, outdepth, outchannels, inchannels, ksize, ksize, ksize);

            ope.Execute(y_tensor, w_tensor, x_tensor);

            Cuda.Profiler.Initialize("../../../../profiler.nvsetting", "../../nvprofiles/deconvolution_3d_fp.nvvp");
            Cuda.Profiler.Start();

            ope.Execute(y_tensor, w_tensor, x_tensor);

            Cuda.Profiler.Stop();
        }

        [TestMethod]
        public void SpeedFFPTest() {
            TensorShaderCudaBackend.Environment.CudnnEnabled = false;
            TensorShaderCudaBackend.Environment.Precision = TensorShaderCudaBackend.Environment.PrecisionMode.FloatFloat;

            int inwidth = 64, inheight = 64, indepth = 64, inchannels = 31, outchannels = 31, ksize = 3;
            int outwidth = inwidth - ksize + 1, outheight = inheight - ksize + 1, outdepth = indepth - ksize + 1;

            OverflowCheckedTensor y_tensor = new(Shape.Map3D(outchannels, outwidth, outheight, outdepth));
            OverflowCheckedTensor w_tensor = new(Shape.Kernel3D(inchannels, outchannels, ksize, ksize, ksize));

            OverflowCheckedTensor x_tensor = new(Shape.Map3D(inchannels, inwidth, inheight, indepth));

            Deconvolution ope = new(outwidth, outheight, outdepth, outchannels, inchannels, ksize, ksize, ksize);

            ope.Execute(y_tensor, w_tensor, x_tensor);

            Cuda.Profiler.Initialize("../../../../profiler.nvsetting", "../../nvprofiles/deconvolution_3d_ffp.nvvp");
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

            int inwidth = 64, inheight = 64, indepth = 64, inchannels = 31, outchannels = 31, ksize = 3;
            int outwidth = inwidth - ksize + 1, outheight = inheight - ksize + 1, outdepth = indepth - ksize + 1;

            OverflowCheckedTensor y_tensor = new(Shape.Map3D(outchannels, outwidth, outheight, outdepth));
            OverflowCheckedTensor w_tensor = new(Shape.Kernel3D(inchannels, outchannels, ksize, ksize, ksize));

            OverflowCheckedTensor x_tensor = new(Shape.Map3D(inchannels, inwidth, inheight, indepth));

            Deconvolution ope = new(outwidth, outheight, outdepth, outchannels, inchannels, ksize, ksize, ksize);

            ope.Execute(y_tensor, w_tensor, x_tensor);

            Cuda.Profiler.Initialize("../../../../profiler.nvsetting", "../../nvprofiles/deconvolution_3d_cudnn.nvvp");
            Cuda.Profiler.Start();

            ope.Execute(y_tensor, w_tensor, x_tensor);

            Cuda.Profiler.Stop();
        }

        public static Map3D Reference(Map3D y, Filter3D w, int inw, int inh, int ind, int kwidth, int kheight, int kdepth) {
            int inchannels = w.InChannels, outchannels = w.OutChannels, batch = y.Batch;
            int outw = inw - kwidth + 1, outh = inh - kheight + 1, outd = ind - kdepth + 1;

            if (y.Width != outw || y.Height != outh || y.Depth != outd) {
                throw new ArgumentException("mismatch shape");
            }

            Map3D x = new(inchannels, inw, inh, ind, batch);

            for (int kx, ky, kz = 0; kz < kdepth; kz++) {
                for (ky = 0; ky < kheight; ky++) {
                    for (kx = 0; kx < kwidth; kx++) {
                        for (int th = 0; th < batch; th++) {
                            for (int ox, oy, oz = 0; oz < outd; oz++) {
                                for (oy = 0; oy < outh; oy++) {
                                    for (ox = 0; ox < outw; ox++) {
                                        for (int outch = 0; outch < outchannels; outch++) {
                                            double v = y[outch, ox, oy, oz, th];

                                            for (int inch = 0; inch < inchannels; inch++) {
                                                x[inch, kx + ox, ky + oy, kz + oz, th] += v * w[inch, outch, kx, ky, kz];
                                            }
                                        }
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
            int inchannels = 2, outchannels = 3, kwidth = 3, kheight = 5, kdepth = 7, inwidth = 13, inheight = 12, indepth = 11, batch = 2;
            int outwidth = inwidth - kwidth + 1, outheight = inheight - kheight + 1, outdepth = indepth - kdepth + 1;

            float[] yval = (new float[outwidth * outheight * outdepth * outchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
            float[] wval = (new float[kwidth * kheight * kdepth * outchannels * inchannels]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

            Map3D y = new(outchannels, outwidth, outheight, outdepth, batch, yval);
            Filter3D w = new(inchannels, outchannels, kwidth, kheight, kdepth, wval);

            Map3D x = Reference(y, w, inwidth, inheight, indepth, kwidth, kheight, kdepth);

            float[] x_expect = {
                1.877000000e-03f,  1.874000000e-03f,  9.379000000e-03f,  9.364000000e-03f,  2.245200000e-02f,  2.241600000e-02f,
                3.921900000e-02f,  3.915600000e-02f,  5.598600000e-02f,  5.589600000e-02f,  7.275300000e-02f,  7.263600000e-02f,
                8.952000000e-02f,  8.937600000e-02f,  1.062870000e-01f,  1.061160000e-01f,  1.230540000e-01f,  1.228560000e-01f,
                1.398210000e-01f,  1.395960000e-01f,  1.565880000e-01f,  1.563360000e-01f,  1.094050000e-01f,  1.092280000e-01f,
                5.719100000e-02f,  5.709800000e-02f,  6.577300000e-02f,  6.566800000e-02f,  1.420400000e-01f,  1.418120000e-01f,
                2.286930000e-01f,  2.283240000e-01f,  2.617410000e-01f,  2.613180000e-01f,  2.947890000e-01f,  2.943120000e-01f,
                3.278370000e-01f,  3.273060000e-01f,  3.608850000e-01f,  3.603000000e-01f,  3.939330000e-01f,  3.932940000e-01f,
                4.269810000e-01f,  4.262880000e-01f,  4.600290000e-01f,  4.592820000e-01f,  4.930770000e-01f,  4.922760000e-01f,
                3.379880000e-01f,  3.374360000e-01f,  1.735930000e-01f,  1.733080000e-01f,  1.899060000e-01f,  1.896000000e-01f,
                3.944190000e-01f,  3.937800000e-01f,  6.133770000e-01f,  6.123780000e-01f,  6.622200000e-01f,  6.611400000e-01f,
                7.110630000e-01f,  7.099020000e-01f,  7.599060000e-01f,  7.586640000e-01f,  8.087490000e-01f,  8.074260000e-01f,
                8.575920000e-01f,  8.561880000e-01f,  9.064350000e-01f,  9.049500000e-01f,  9.552780000e-01f,  9.537120000e-01f,
                1.004121000e+00f,  1.002474000e+00f,  6.821850000e-01f,  6.810600000e-01f,  3.474240000e-01f,  3.468480000e-01f,
                3.724940000e-01f,  3.718880000e-01f,  7.629520000e-01f,  7.617040000e-01f,  1.171158000e+00f,  1.169232000e+00f,
                1.235310000e+00f,  1.233276000e+00f,  1.299462000e+00f,  1.297320000e+00f,  1.363614000e+00f,  1.361364000e+00f,
                1.427766000e+00f,  1.425408000e+00f,  1.491918000e+00f,  1.489452000e+00f,  1.556070000e+00f,  1.553496000e+00f,
                1.620222000e+00f,  1.617540000e+00f,  1.684374000e+00f,  1.681584000e+00f,  1.138432000e+00f,  1.136536000e+00f,
                5.769020000e-01f,  5.759360000e-01f,  6.117550000e-01f,  6.107500000e-01f,  1.244075000e+00f,  1.242020000e+00f,
                1.896690000e+00f,  1.893540000e+00f,  1.975665000e+00f,  1.972380000e+00f,  2.054640000e+00f,  2.051220000e+00f,
                2.133615000e+00f,  2.130060000e+00f,  2.212590000e+00f,  2.208900000e+00f,  2.291565000e+00f,  2.287740000e+00f,
                2.370540000e+00f,  2.366580000e+00f,  2.449515000e+00f,  2.445420000e+00f,  2.528490000e+00f,  2.524260000e+00f,
                1.703165000e+00f,  1.700300000e+00f,  8.602450000e-01f,  8.587900000e-01f,  9.043000000e-01f,  9.028000000e-01f,
                1.826195000e+00f,  1.823150000e+00f,  2.765415000e+00f,  2.760780000e+00f,  2.844390000e+00f,  2.839620000e+00f,
                2.923365000e+00f,  2.918460000e+00f,  3.002340000e+00f,  2.997300000e+00f,  3.081315000e+00f,  3.076140000e+00f,
                3.160290000e+00f,  3.154980000e+00f,  3.239265000e+00f,  3.233820000e+00f,  3.318240000e+00f,  3.312660000e+00f,
                3.397215000e+00f,  3.391500000e+00f,  2.279345000e+00f,  2.275490000e+00f,  1.146850000e+00f,  1.144900000e+00f,
                1.196845000e+00f,  1.194850000e+00f,  2.408315000e+00f,  2.404280000e+00f,  3.634140000e+00f,  3.628020000e+00f,
                3.713115000e+00f,  3.706860000e+00f,  3.792090000e+00f,  3.785700000e+00f,  3.871065000e+00f,  3.864540000e+00f,
                3.950040000e+00f,  3.943380000e+00f,  4.029015000e+00f,  4.022220000e+00f,  4.107990000e+00f,  4.101060000e+00f,
                4.186965000e+00f,  4.179900000e+00f,  4.265940000e+00f,  4.258740000e+00f,  2.855525000e+00f,  2.850680000e+00f,
                1.433455000e+00f,  1.431010000e+00f,  1.489390000e+00f,  1.486900000e+00f,  2.990435000e+00f,  2.985410000e+00f,
                4.502865000e+00f,  4.495260000e+00f,  4.581840000e+00f,  4.574100000e+00f,  4.660815000e+00f,  4.652940000e+00f,
                4.739790000e+00f,  4.731780000e+00f,  4.818765000e+00f,  4.810620000e+00f,  4.897740000e+00f,  4.889460000e+00f,
                4.976715000e+00f,  4.968300000e+00f,  5.055690000e+00f,  5.047140000e+00f,  5.134665000e+00f,  5.125980000e+00f,
                3.431705000e+00f,  3.425870000e+00f,  1.720060000e+00f,  1.717120000e+00f,  1.283474000e+00f,  1.281284000e+00f,
                2.574760000e+00f,  2.570344000e+00f,  3.873642000e+00f,  3.866964000e+00f,  3.935850000e+00f,  3.929064000e+00f,
                3.998058000e+00f,  3.991164000e+00f,  4.060266000e+00f,  4.053264000e+00f,  4.122474000e+00f,  4.115364000e+00f,
                4.184682000e+00f,  4.177464000e+00f,  4.246890000e+00f,  4.239564000e+00f,  4.309098000e+00f,  4.301664000e+00f,
                4.371306000e+00f,  4.363764000e+00f,  2.919568000e+00f,  2.914504000e+00f,  1.462394000e+00f,  1.459844000e+00f,
                1.029795000e+00f,  1.028004000e+00f,  2.064315000e+00f,  2.060706000e+00f,  3.103398000e+00f,  3.097944000e+00f,
                3.149325000e+00f,  3.143790000e+00f,  3.195252000e+00f,  3.189636000e+00f,  3.241179000e+00f,  3.235482000e+00f,
                3.287106000e+00f,  3.281328000e+00f,  3.333033000e+00f,  3.327174000e+00f,  3.378960000e+00f,  3.373020000e+00f,
                3.424887000e+00f,  3.418866000e+00f,  3.470814000e+00f,  3.464712000e+00f,  2.316765000e+00f,  2.312670000e+00f,
                1.159773000e+00f,  1.157712000e+00f,  7.301350000e-01f,  7.288420000e-01f,  1.462664000e+00f,  1.460060000e+00f,
                2.197479000e+00f,  2.193546000e+00f,  2.227611000e+00f,  2.223624000e+00f,  2.257743000e+00f,  2.253702000e+00f,
                2.287875000e+00f,  2.283780000e+00f,  2.318007000e+00f,  2.313858000e+00f,  2.348139000e+00f,  2.343936000e+00f,
                2.378271000e+00f,  2.374014000e+00f,  2.408403000e+00f,  2.404092000e+00f,  2.438535000e+00f,  2.434170000e+00f,
                1.626860000e+00f,  1.623932000e+00f,  8.139790000e-01f,  8.125060000e-01f,  3.862760000e-01f,  3.855800000e-01f,
                7.733710000e-01f,  7.719700000e-01f,  1.161231000e+00f,  1.159116000e+00f,  1.176054000e+00f,  1.173912000e+00f,
                1.190877000e+00f,  1.188708000e+00f,  1.205700000e+00f,  1.203504000e+00f,  1.220523000e+00f,  1.218300000e+00f,
                1.235346000e+00f,  1.233096000e+00f,  1.250169000e+00f,  1.247892000e+00f,  1.264992000e+00f,  1.262688000e+00f,
                1.279815000e+00f,  1.277484000e+00f,  8.534170000e-01f,  8.518540000e-01f,  4.267940000e-01f,  4.260080000e-01f,
                5.000680000e-01f,  4.992700000e-01f,  1.005824000e+00f,  1.004210000e+00f,  1.517160000e+00f,  1.514712000e+00f,
                1.548264000e+00f,  1.545762000e+00f,  1.579368000e+00f,  1.576812000e+00f,  1.610472000e+00f,  1.607862000e+00f,
                1.641576000e+00f,  1.638912000e+00f,  1.672680000e+00f,  1.669962000e+00f,  1.703784000e+00f,  1.701012000e+00f,
                1.734888000e+00f,  1.732062000e+00f,  1.765992000e+00f,  1.763112000e+00f,  1.181792000e+00f,  1.179854000e+00f,
                5.930920000e-01f,  5.921140000e-01f,  1.101008000e+00f,  1.099214000e+00f,  2.211880000e+00f,  2.208256000e+00f,
                3.332400000e+00f,  3.326910000e+00f,  3.393636000e+00f,  3.388038000e+00f,  3.454872000e+00f,  3.449166000e+00f,
                3.516108000e+00f,  3.510294000e+00f,  3.577344000e+00f,  3.571422000e+00f,  3.638580000e+00f,  3.632550000e+00f,
                3.699816000e+00f,  3.693678000e+00f,  3.761052000e+00f,  3.754806000e+00f,  3.822288000e+00f,  3.815934000e+00f,
                2.555608000e+00f,  2.551336000e+00f,  1.281440000e+00f,  1.279286000e+00f,  1.799256000e+00f,  1.796268000e+00f,
                3.611040000e+00f,  3.605010000e+00f,  5.435028000e+00f,  5.425902000e+00f,  5.525424000e+00f,  5.516136000e+00f,
                5.615820000e+00f,  5.606370000e+00f,  5.706216000e+00f,  5.696604000e+00f,  5.796612000e+00f,  5.786838000e+00f,
                5.887008000e+00f,  5.877072000e+00f,  5.977404000e+00f,  5.967306000e+00f,  6.067800000e+00f,  6.057540000e+00f,
                6.158196000e+00f,  6.147774000e+00f,  4.114320000e+00f,  4.107318000e+00f,  2.061480000e+00f,  2.057952000e+00f,
                2.591248000e+00f,  2.586868000e+00f,  5.196176000e+00f,  5.187344000e+00f,  7.814352000e+00f,  7.800996000e+00f,
                7.932936000e+00f,  7.919364000e+00f,  8.051520000e+00f,  8.037732000e+00f,  8.170104000e+00f,  8.156100000e+00f,
                8.288688000e+00f,  8.274468000e+00f,  8.407272000e+00f,  8.392836000e+00f,  8.525856000e+00f,  8.511204000e+00f,
                8.644440000e+00f,  8.629572000e+00f,  8.763024000e+00f,  8.747940000e+00f,  5.850800000e+00f,  5.840672000e+00f,
                2.929648000e+00f,  2.924548000e+00f,  3.473420000e+00f,  3.467450000e+00f,  6.960160000e+00f,  6.948130000e+00f,
                1.045968000e+01f,  1.044150000e+01f,  1.060548000e+01f,  1.058703000e+01f,  1.075128000e+01f,  1.073256000e+01f,
                1.089708000e+01f,  1.087809000e+01f,  1.104288000e+01f,  1.102362000e+01f,  1.118868000e+01f,  1.116915000e+01f,
                1.133448000e+01f,  1.131468000e+01f,  1.148028000e+01f,  1.146021000e+01f,  1.162608000e+01f,  1.160574000e+01f,
                7.757920000e+00f,  7.744270000e+00f,  3.882380000e+00f,  3.875510000e+00f,  4.013960000e+00f,  4.007000000e+00f,
                8.035300000e+00f,  8.021290000e+00f,  1.206348000e+01f,  1.204233000e+01f,  1.220928000e+01f,  1.218786000e+01f,
                1.235508000e+01f,  1.233339000e+01f,  1.250088000e+01f,  1.247892000e+01f,  1.264668000e+01f,  1.262445000e+01f,
                1.279248000e+01f,  1.276998000e+01f,  1.293828000e+01f,  1.291551000e+01f,  1.308408000e+01f,  1.306104000e+01f,
                1.322988000e+01f,  1.320657000e+01f,  8.821180000e+00f,  8.805550000e+00f,  4.411040000e+00f,  4.403180000e+00f,
                4.554500000e+00f,  4.546550000e+00f,  9.110440000e+00f,  9.094450000e+00f,  1.366728000e+01f,  1.364316000e+01f,
                1.381308000e+01f,  1.378869000e+01f,  1.395888000e+01f,  1.393422000e+01f,  1.410468000e+01f,  1.407975000e+01f,
                1.425048000e+01f,  1.422528000e+01f,  1.439628000e+01f,  1.437081000e+01f,  1.454208000e+01f,  1.451634000e+01f,
                1.468788000e+01f,  1.466187000e+01f,  1.483368000e+01f,  1.480740000e+01f,  9.884440000e+00f,  9.866830000e+00f,
                4.939700000e+00f,  4.930850000e+00f,  5.095040000e+00f,  5.086100000e+00f,  1.018558000e+01f,  1.016761000e+01f,
                1.527108000e+01f,  1.524399000e+01f,  1.541688000e+01f,  1.538952000e+01f,  1.556268000e+01f,  1.553505000e+01f,
                1.570848000e+01f,  1.568058000e+01f,  1.585428000e+01f,  1.582611000e+01f,  1.600008000e+01f,  1.597164000e+01f,
                1.614588000e+01f,  1.611717000e+01f,  1.629168000e+01f,  1.626270000e+01f,  1.643748000e+01f,  1.640823000e+01f,
                1.094770000e+01f,  1.092811000e+01f,  5.468360000e+00f,  5.458520000e+00f,  4.213624000e+00f,  4.206076000e+00f,
                8.420624000e+00f,  8.405456000e+00f,  1.262056800e+01f,  1.259770800e+01f,  1.273526400e+01f,  1.271218800e+01f,
                1.284996000e+01f,  1.282666800e+01f,  1.296465600e+01f,  1.294114800e+01f,  1.307935200e+01f,  1.305562800e+01f,
                1.319404800e+01f,  1.317010800e+01f,  1.330874400e+01f,  1.328458800e+01f,  1.342344000e+01f,  1.339906800e+01f,
                1.353813600e+01f,  1.351354800e+01f,  9.013904000e+00f,  8.997440000e+00f,  4.501048000e+00f,  4.492780000e+00f,
                3.259848000e+00f,  3.253890000e+00f,  6.512460000e+00f,  6.500490000e+00f,  9.757512000e+00f,  9.739476000e+00f,
                9.842076000e+00f,  9.823878000e+00f,  9.926640000e+00f,  9.908280000e+00f,  1.001120400e+01f,  9.992682000e+00f,
                1.009576800e+01f,  1.007708400e+01f,  1.018033200e+01f,  1.016148600e+01f,  1.026489600e+01f,  1.024588800e+01f,
                1.034946000e+01f,  1.033029000e+01f,  1.043402400e+01f,  1.041469200e+01f,  6.945108000e+00f,  6.932166000e+00f,
                3.466992000e+00f,  3.460494000e+00f,  2.237276000e+00f,  2.233106000e+00f,  4.468216000e+00f,  4.459840000e+00f,
                6.692604000e+00f,  6.679986000e+00f,  6.748008000e+00f,  6.735282000e+00f,  6.803412000e+00f,  6.790578000e+00f,
                6.858816000e+00f,  6.845874000e+00f,  6.914220000e+00f,  6.901170000e+00f,  6.969624000e+00f,  6.956466000e+00f,
                7.025028000e+00f,  7.011762000e+00f,  7.080432000e+00f,  7.067058000e+00f,  7.135836000e+00f,  7.122354000e+00f,
                4.748440000e+00f,  4.739416000e+00f,  2.369756000e+00f,  2.365226000e+00f,  1.149472000e+00f,  1.147288000e+00f,
                2.295020000e+00f,  2.290634000e+00f,  3.436536000e+00f,  3.429930000e+00f,  3.463752000e+00f,  3.457092000e+00f,
                3.490968000e+00f,  3.484254000e+00f,  3.518184000e+00f,  3.511416000e+00f,  3.545400000e+00f,  3.538578000e+00f,
                3.572616000e+00f,  3.565740000e+00f,  3.599832000e+00f,  3.592902000e+00f,  3.627048000e+00f,  3.620064000e+00f,
                3.654264000e+00f,  3.647226000e+00f,  2.431028000e+00f,  2.426318000e+00f,  1.212904000e+00f,  1.210540000e+00f,
                1.423293000e+00f,  1.420908000e+00f,  2.846775000e+00f,  2.841978000e+00f,  4.270284000e+00f,  4.263048000e+00f,
                4.313295000e+00f,  4.305978000e+00f,  4.356306000e+00f,  4.348908000e+00f,  4.399317000e+00f,  4.391838000e+00f,
                4.442328000e+00f,  4.434768000e+00f,  4.485339000e+00f,  4.477698000e+00f,  4.528350000e+00f,  4.520628000e+00f,
                4.571361000e+00f,  4.563558000e+00f,  4.614372000e+00f,  4.606488000e+00f,  3.074601000e+00f,  3.069318000e+00f,
                1.536423000e+00f,  1.533768000e+00f,  2.963145000e+00f,  2.958078000e+00f,  5.924400000e+00f,  5.914212000e+00f,
                8.883441000e+00f,  8.868078000e+00f,  8.968005000e+00f,  8.952480000e+00f,  9.052569000e+00f,  9.036882000e+00f,
                9.137133000e+00f,  9.121284000e+00f,  9.221697000e+00f,  9.205686000e+00f,  9.306261000e+00f,  9.290088000e+00f,
                9.390825000e+00f,  9.374490000e+00f,  9.475389000e+00f,  9.458892000e+00f,  9.559953000e+00f,  9.543294000e+00f,
                6.367740000e+00f,  6.356580000e+00f,  3.180981000e+00f,  3.175374000e+00f,  4.614210000e+00f,  4.606164000e+00f,
                9.222183000e+00f,  9.206010000e+00f,  1.382343300e+01f,  1.379905200e+01f,  1.394809200e+01f,  1.392346800e+01f,
                1.407275100e+01f,  1.404788400e+01f,  1.419741000e+01f,  1.417230000e+01f,  1.432206900e+01f,  1.429671600e+01f,
                1.444672800e+01f,  1.442113200e+01f,  1.457138700e+01f,  1.454554800e+01f,  1.469604600e+01f,  1.466996400e+01f,
                1.482070500e+01f,  1.479438000e+01f,  9.868725000e+00f,  9.851094000e+00f,  4.928328000e+00f,  4.919472000e+00f,
                6.371142000e+00f,  6.359820000e+00f,  1.272943200e+01f,  1.270668000e+01f,  1.907422200e+01f,  1.903993200e+01f,
                1.923751800e+01f,  1.920290400e+01f,  1.940081400e+01f,  1.936587600e+01f,  1.956411000e+01f,  1.952884800e+01f,
                1.972740600e+01f,  1.969182000e+01f,  1.989070200e+01f,  1.985479200e+01f,  2.005399800e+01f,  2.001776400e+01f,
                2.021729400e+01f,  2.018073600e+01f,  2.038059000e+01f,  2.034370800e+01f,  1.356686400e+01f,  1.354216800e+01f,
                6.773118000e+00f,  6.760716000e+00f,  8.228595000e+00f,  8.213700000e+00f,  1.643545500e+01f,  1.640553000e+01f,
                2.461977000e+01f,  2.457468000e+01f,  2.482024500e+01f,  2.477475000e+01f,  2.502072000e+01f,  2.497482000e+01f,
                2.522119500e+01f,  2.517489000e+01f,  2.542167000e+01f,  2.537496000e+01f,  2.562214500e+01f,  2.557503000e+01f,
                2.582262000e+01f,  2.577510000e+01f,  2.602309500e+01f,  2.597517000e+01f,  2.622357000e+01f,  2.617524000e+01f,
                1.745146500e+01f,  1.741911000e+01f,  8.710005000e+00f,  8.693760000e+00f,  8.972580000e+00f,  8.956200000e+00f,
                1.791451500e+01f,  1.788162000e+01f,  2.682499500e+01f,  2.677545000e+01f,  2.702547000e+01f,  2.697552000e+01f,
                2.722594500e+01f,  2.717559000e+01f,  2.742642000e+01f,  2.737566000e+01f,  2.762689500e+01f,  2.757573000e+01f,
                2.782737000e+01f,  2.777580000e+01f,  2.802784500e+01f,  2.797587000e+01f,  2.822832000e+01f,  2.817594000e+01f,
                2.842879500e+01f,  2.837601000e+01f,  1.891270500e+01f,  1.887738000e+01f,  9.436170000e+00f,  9.418440000e+00f,
                9.716565000e+00f,  9.698700000e+00f,  1.939357500e+01f,  1.935771000e+01f,  2.903022000e+01f,  2.897622000e+01f,
                2.923069500e+01f,  2.917629000e+01f,  2.943117000e+01f,  2.937636000e+01f,  2.963164500e+01f,  2.957643000e+01f,
                2.983212000e+01f,  2.977650000e+01f,  3.003259500e+01f,  2.997657000e+01f,  3.023307000e+01f,  3.017664000e+01f,
                3.043354500e+01f,  3.037671000e+01f,  3.063402000e+01f,  3.057678000e+01f,  2.037394500e+01f,  2.033565000e+01f,
                1.016233500e+01f,  1.014312000e+01f,  1.046055000e+01f,  1.044120000e+01f,  2.087263500e+01f,  2.083380000e+01f,
                3.123544500e+01f,  3.117699000e+01f,  3.143592000e+01f,  3.137706000e+01f,  3.163639500e+01f,  3.157713000e+01f,
                3.183687000e+01f,  3.177720000e+01f,  3.203734500e+01f,  3.197727000e+01f,  3.223782000e+01f,  3.217734000e+01f,
                3.243829500e+01f,  3.237741000e+01f,  3.263877000e+01f,  3.257748000e+01f,  3.283924500e+01f,  3.277755000e+01f,
                2.183518500e+01f,  2.179392000e+01f,  1.088850000e+01f,  1.086780000e+01f,  8.505330000e+00f,  8.489256000e+00f,
                1.696735200e+01f,  1.693509600e+01f,  2.538541800e+01f,  2.533687200e+01f,  2.554288200e+01f,  2.549401200e+01f,
                2.570034600e+01f,  2.565115200e+01f,  2.585781000e+01f,  2.580829200e+01f,  2.601527400e+01f,  2.596543200e+01f,
                2.617273800e+01f,  2.612257200e+01f,  2.633020200e+01f,  2.627971200e+01f,  2.648766600e+01f,  2.643685200e+01f,
                2.664513000e+01f,  2.659399200e+01f,  1.771276800e+01f,  1.767856800e+01f,  8.830842000e+00f,  8.813688000e+00f,
                6.476319000e+00f,  6.463818000e+00f,  1.291675500e+01f,  1.289167200e+01f,  1.932082200e+01f,  1.928307600e+01f,
                1.943673300e+01f,  1.939874400e+01f,  1.955264400e+01f,  1.951441200e+01f,  1.966855500e+01f,  1.963008000e+01f,
                1.978446600e+01f,  1.974574800e+01f,  1.990037700e+01f,  1.986141600e+01f,  2.001628800e+01f,  1.997708400e+01f,
                2.013219900e+01f,  2.009275200e+01f,  2.024811000e+01f,  2.020842000e+01f,  1.345734900e+01f,  1.343080800e+01f,
                6.707817000e+00f,  6.694506000e+00f,  4.378863000e+00f,  4.370232000e+00f,  8.731536000e+00f,  8.714220000e+00f,
                1.305769500e+01f,  1.303164000e+01f,  1.313351100e+01f,  1.310729400e+01f,  1.320932700e+01f,  1.318294800e+01f,
                1.328514300e+01f,  1.325860200e+01f,  1.336095900e+01f,  1.333425600e+01f,  1.343677500e+01f,  1.340991000e+01f,
                1.351259100e+01f,  1.348556400e+01f,  1.358840700e+01f,  1.356121800e+01f,  1.366422300e+01f,  1.363687200e+01f,
                9.079620000e+00f,  9.061332000e+00f,  4.524771000e+00f,  4.515600000e+00f,  2.218308000e+00f,  2.213844000e+00f,
                4.422387000e+00f,  4.413432000e+00f,  6.612075000e+00f,  6.598602000e+00f,  6.649254000e+00f,  6.635700000e+00f,
                6.686433000e+00f,  6.672798000e+00f,  6.723612000e+00f,  6.709896000e+00f,  6.760791000e+00f,  6.746994000e+00f,
                6.797970000e+00f,  6.784092000e+00f,  6.835149000e+00f,  6.821190000e+00f,  6.872328000e+00f,  6.858288000e+00f,
                6.909507000e+00f,  6.895386000e+00f,  4.590273000e+00f,  4.580832000e+00f,  2.287050000e+00f,  2.282316000e+00f,
                2.700272000e+00f,  2.695508000e+00f,  5.389672000e+00f,  5.380108000e+00f,  8.067984000e+00f,  8.053584000e+00f,
                8.120472000e+00f,  8.105964000e+00f,  8.172960000e+00f,  8.158344000e+00f,  8.225448000e+00f,  8.210724000e+00f,
                8.277936000e+00f,  8.263104000e+00f,  8.330424000e+00f,  8.315484000e+00f,  8.382912000e+00f,  8.367864000e+00f,
                8.435400000e+00f,  8.420244000e+00f,  8.487888000e+00f,  8.472624000e+00f,  5.645272000e+00f,  5.635060000e+00f,
                2.815904000e+00f,  2.810780000e+00f,  5.509624000e+00f,  5.499700000e+00f,  1.099448000e+01f,  1.097456000e+01f,
                1.645413600e+01f,  1.642414800e+01f,  1.655716800e+01f,  1.652696400e+01f,  1.666020000e+01f,  1.662978000e+01f,
                1.676323200e+01f,  1.673259600e+01f,  1.686626400e+01f,  1.683541200e+01f,  1.696929600e+01f,  1.693822800e+01f,
                1.707232800e+01f,  1.704104400e+01f,  1.717536000e+01f,  1.714386000e+01f,  1.727839200e+01f,  1.724667600e+01f,
                1.148926400e+01f,  1.146804800e+01f,  5.729656000e+00f,  5.719012000e+00f,  8.420928000e+00f,  8.405448000e+00f,
                1.680016800e+01f,  1.676910000e+01f,  2.513707200e+01f,  2.509030800e+01f,  2.528870400e+01f,  2.524161600e+01f,
                2.544033600e+01f,  2.539292400e+01f,  2.559196800e+01f,  2.554423200e+01f,  2.574360000e+01f,  2.569554000e+01f,
                2.589523200e+01f,  2.584684800e+01f,  2.604686400e+01f,  2.599815600e+01f,  2.619849600e+01f,  2.614946400e+01f,
                2.635012800e+01f,  2.630077200e+01f,  1.751772000e+01f,  1.748470800e+01f,  8.734128000e+00f,  8.717568000e+00f,
                1.142705600e+01f,  1.140562400e+01f,  2.279248000e+01f,  2.274947200e+01f,  3.409540800e+01f,  3.403068000e+01f,
                3.429369600e+01f,  3.422853600e+01f,  3.449198400e+01f,  3.442639200e+01f,  3.469027200e+01f,  3.462424800e+01f,
                3.488856000e+01f,  3.482210400e+01f,  3.508684800e+01f,  3.501996000e+01f,  3.528513600e+01f,  3.521781600e+01f,
                3.548342400e+01f,  3.541567200e+01f,  3.568171200e+01f,  3.561352800e+01f,  2.371638400e+01f,  2.367078400e+01f,
                1.182219200e+01f,  1.179932000e+01f,  1.452088000e+01f,  1.449310000e+01f,  2.895716000e+01f,  2.890142000e+01f,
                4.330776000e+01f,  4.322388000e+01f,  4.355076000e+01f,  4.346634000e+01f,  4.379376000e+01f,  4.370880000e+01f,
                4.403676000e+01f,  4.395126000e+01f,  4.427976000e+01f,  4.419372000e+01f,  4.452276000e+01f,  4.443618000e+01f,
                4.476576000e+01f,  4.467864000e+01f,  4.500876000e+01f,  4.492110000e+01f,  4.525176000e+01f,  4.516356000e+01f,
                3.007100000e+01f,  3.001202000e+01f,  1.498672000e+01f,  1.495714000e+01f,  1.542376000e+01f,  1.539400000e+01f,
                3.075104000e+01f,  3.069134000e+01f,  4.598076000e+01f,  4.589094000e+01f,  4.622376000e+01f,  4.613340000e+01f,
                4.646676000e+01f,  4.637586000e+01f,  4.670976000e+01f,  4.661832000e+01f,  4.695276000e+01f,  4.686078000e+01f,
                4.719576000e+01f,  4.710324000e+01f,  4.743876000e+01f,  4.734570000e+01f,  4.768176000e+01f,  4.758816000e+01f,
                4.792476000e+01f,  4.783062000e+01f,  3.184112000e+01f,  3.177818000e+01f,  1.586584000e+01f,  1.583428000e+01f,
                1.632664000e+01f,  1.629490000e+01f,  3.254492000e+01f,  3.248126000e+01f,  4.865376000e+01f,  4.855800000e+01f,
                4.889676000e+01f,  4.880046000e+01f,  4.913976000e+01f,  4.904292000e+01f,  4.938276000e+01f,  4.928538000e+01f,
                4.962576000e+01f,  4.952784000e+01f,  4.986876000e+01f,  4.977030000e+01f,  5.011176000e+01f,  5.001276000e+01f,
                5.035476000e+01f,  5.025522000e+01f,  5.059776000e+01f,  5.049768000e+01f,  3.361124000e+01f,  3.354434000e+01f,
                1.674496000e+01f,  1.671142000e+01f,  1.722952000e+01f,  1.719580000e+01f,  3.433880000e+01f,  3.427118000e+01f,
                5.132676000e+01f,  5.122506000e+01f,  5.156976000e+01f,  5.146752000e+01f,  5.181276000e+01f,  5.170998000e+01f,
                5.205576000e+01f,  5.195244000e+01f,  5.229876000e+01f,  5.219490000e+01f,  5.254176000e+01f,  5.243736000e+01f,
                5.278476000e+01f,  5.267982000e+01f,  5.302776000e+01f,  5.292228000e+01f,  5.327076000e+01f,  5.316474000e+01f,
                3.538136000e+01f,  3.531050000e+01f,  1.762408000e+01f,  1.758856000e+01f,  1.387347200e+01f,  1.384570400e+01f,
                2.764470400e+01f,  2.758902400e+01f,  4.131283200e+01f,  4.122909600e+01f,  4.150334400e+01f,  4.141917600e+01f,
                4.169385600e+01f,  4.160925600e+01f,  4.188436800e+01f,  4.179933600e+01f,  4.207488000e+01f,  4.198941600e+01f,
                4.226539200e+01f,  4.217949600e+01f,  4.245590400e+01f,  4.236957600e+01f,  4.264641600e+01f,  4.255965600e+01f,
                4.283692800e+01f,  4.274973600e+01f,  2.844592000e+01f,  2.838764800e+01f,  1.416665600e+01f,  1.413744800e+01f,
                1.046536800e+01f,  1.044394800e+01f,  2.084952000e+01f,  2.080657200e+01f,  3.115180800e+01f,  3.108722400e+01f,
                3.129177600e+01f,  3.122686800e+01f,  3.143174400e+01f,  3.136651200e+01f,  3.157171200e+01f,  3.150615600e+01f,
                3.171168000e+01f,  3.164580000e+01f,  3.185164800e+01f,  3.178544400e+01f,  3.199161600e+01f,  3.192508800e+01f,
                3.213158400e+01f,  3.206473200e+01f,  3.227155200e+01f,  3.220437600e+01f,  2.142580800e+01f,  2.138091600e+01f,
                1.066840800e+01f,  1.064590800e+01f,  7.012336000e+00f,  6.997660000e+00f,  1.396750400e+01f,  1.393808000e+01f,
                2.086507200e+01f,  2.082082800e+01f,  2.095644000e+01f,  2.091198000e+01f,  2.104780800e+01f,  2.100313200e+01f,
                2.113917600e+01f,  2.109428400e+01f,  2.123054400e+01f,  2.118543600e+01f,  2.132191200e+01f,  2.127658800e+01f,
                2.141328000e+01f,  2.136774000e+01f,  2.150464800e+01f,  2.145889200e+01f,  2.159601600e+01f,  2.155004400e+01f,
                1.433528000e+01f,  1.430456000e+01f,  7.136464000e+00f,  7.121068000e+00f,  3.521504000e+00f,  3.513968000e+00f,
                7.012912000e+00f,  6.997804000e+00f,  1.047400800e+01f,  1.045129200e+01f,  1.051872000e+01f,  1.049589600e+01f,
                1.056343200e+01f,  1.054050000e+01f,  1.060814400e+01f,  1.058510400e+01f,  1.065285600e+01f,  1.062970800e+01f,
                1.069756800e+01f,  1.067431200e+01f,  1.074228000e+01f,  1.071891600e+01f,  1.078699200e+01f,  1.076352000e+01f,
                1.083170400e+01f,  1.080812400e+01f,  7.188592000e+00f,  7.172836000e+00f,  3.577952000e+00f,  3.570056000e+00f,
                4.259725000e+00f,  4.251790000e+00f,  8.491955000e+00f,  8.476040000e+00f,  1.269642000e+01f,  1.267248000e+01f,
                1.275595500e+01f,  1.273188000e+01f,  1.281549000e+01f,  1.279128000e+01f,  1.287502500e+01f,  1.285068000e+01f,
                1.293456000e+01f,  1.291008000e+01f,  1.299409500e+01f,  1.296948000e+01f,  1.305363000e+01f,  1.302888000e+01f,
                1.311316500e+01f,  1.308828000e+01f,  1.317270000e+01f,  1.314768000e+01f,  8.751245000e+00f,  8.734520000e+00f,
                4.360255000e+00f,  4.351870000e+00f,  8.597885000e+00f,  8.581520000e+00f,  1.713700000e+01f,  1.710418000e+01f,
                2.561680500e+01f,  2.556744000e+01f,  2.573344500e+01f,  2.568381000e+01f,  2.585008500e+01f,  2.580018000e+01f,
                2.596672500e+01f,  2.591655000e+01f,  2.608336500e+01f,  2.603292000e+01f,  2.620000500e+01f,  2.614929000e+01f,
                2.631664500e+01f,  2.626566000e+01f,  2.643328500e+01f,  2.638203000e+01f,  2.654992500e+01f,  2.649840000e+01f,
                1.763506000e+01f,  1.760062000e+01f,  8.784905000e+00f,  8.767640000e+00f,  1.300557000e+01f,  1.298028000e+01f,
                2.591731500e+01f,  2.586660000e+01f,  3.873442500e+01f,  3.865815000e+01f,  3.890574000e+01f,  3.882906000e+01f,
                3.907705500e+01f,  3.899997000e+01f,  3.924837000e+01f,  3.917088000e+01f,  3.941968500e+01f,  3.934179000e+01f,
                3.959100000e+01f,  3.951270000e+01f,  3.976231500e+01f,  3.968361000e+01f,  3.993363000e+01f,  3.985452000e+01f,
                4.010494500e+01f,  4.002543000e+01f,  2.663362500e+01f,  2.658048000e+01f,  1.326504000e+01f,  1.323840000e+01f,
                1.747387000e+01f,  1.743916000e+01f,  3.481508000e+01f,  3.474548000e+01f,  5.202255000e+01f,  5.191788000e+01f,
                5.224611000e+01f,  5.214090000e+01f,  5.246967000e+01f,  5.236392000e+01f,  5.269323000e+01f,  5.258694000e+01f,
                5.291679000e+01f,  5.280996000e+01f,  5.314035000e+01f,  5.303298000e+01f,  5.336391000e+01f,  5.325600000e+01f,
                5.358747000e+01f,  5.347902000e+01f,  5.381103000e+01f,  5.370204000e+01f,  3.572912000e+01f,  3.565628000e+01f,
                1.779175000e+01f,  1.775524000e+01f,  2.199387500e+01f,  2.194925000e+01f,  4.381247500e+01f,  4.372300000e+01f,
                6.545445000e+01f,  6.531990000e+01f,  6.572782500e+01f,  6.559260000e+01f,  6.600120000e+01f,  6.586530000e+01f,
                6.627457500e+01f,  6.613800000e+01f,  6.654795000e+01f,  6.641070000e+01f,  6.682132500e+01f,  6.668340000e+01f,
                6.709470000e+01f,  6.695610000e+01f,  6.736807500e+01f,  6.722880000e+01f,  6.764145000e+01f,  6.750150000e+01f,
                4.490372500e+01f,  4.481020000e+01f,  2.235612500e+01f,  2.230925000e+01f,  2.301110000e+01f,  2.296400000e+01f,
                4.583207500e+01f,  4.573765000e+01f,  6.846157500e+01f,  6.831960000e+01f,  6.873495000e+01f,  6.859230000e+01f,
                6.900832500e+01f,  6.886500000e+01f,  6.928170000e+01f,  6.913770000e+01f,  6.955507500e+01f,  6.941040000e+01f,
                6.982845000e+01f,  6.968310000e+01f,  7.010182500e+01f,  6.995580000e+01f,  7.037520000e+01f,  7.022850000e+01f,
                7.064857500e+01f,  7.050120000e+01f,  4.689362500e+01f,  4.679515000e+01f,  2.334365000e+01f,  2.329430000e+01f,
                2.402832500e+01f,  2.397875000e+01f,  4.785167500e+01f,  4.775230000e+01f,  7.146870000e+01f,  7.131930000e+01f,
                7.174207500e+01f,  7.159200000e+01f,  7.201545000e+01f,  7.186470000e+01f,  7.228882500e+01f,  7.213740000e+01f,
                7.256220000e+01f,  7.241010000e+01f,  7.283557500e+01f,  7.268280000e+01f,  7.310895000e+01f,  7.295550000e+01f,
                7.338232500e+01f,  7.322820000e+01f,  7.365570000e+01f,  7.350090000e+01f,  4.888352500e+01f,  4.878010000e+01f,
                2.433117500e+01f,  2.427935000e+01f,  2.504555000e+01f,  2.499350000e+01f,  4.987127500e+01f,  4.976695000e+01f,
                7.447582500e+01f,  7.431900000e+01f,  7.474920000e+01f,  7.459170000e+01f,  7.502257500e+01f,  7.486440000e+01f,
                7.529595000e+01f,  7.513710000e+01f,  7.556932500e+01f,  7.540980000e+01f,  7.584270000e+01f,  7.568250000e+01f,
                7.611607500e+01f,  7.595520000e+01f,  7.638945000e+01f,  7.622790000e+01f,  7.666282500e+01f,  7.650060000e+01f,
                5.087342500e+01f,  5.076505000e+01f,  2.531870000e+01f,  2.526440000e+01f,  2.003293000e+01f,  1.999030000e+01f,
                3.988244000e+01f,  3.979700000e+01f,  5.954745000e+01f,  5.941902000e+01f,  5.976129000e+01f,  5.963232000e+01f,
                5.997513000e+01f,  5.984562000e+01f,  6.018897000e+01f,  6.005892000e+01f,  6.040281000e+01f,  6.027222000e+01f,
                6.061665000e+01f,  6.048552000e+01f,  6.083049000e+01f,  6.069882000e+01f,  6.104433000e+01f,  6.091212000e+01f,
                6.125817000e+01f,  6.112542000e+01f,  4.064312000e+01f,  4.055444000e+01f,  2.022337000e+01f,  2.017894000e+01f,
                1.501315500e+01f,  1.498044000e+01f,  2.988307500e+01f,  2.981751000e+01f,  4.460895000e+01f,  4.451040000e+01f,
                4.476568500e+01f,  4.466673000e+01f,  4.492242000e+01f,  4.482306000e+01f,  4.507915500e+01f,  4.497939000e+01f,
                4.523589000e+01f,  4.513572000e+01f,  4.539262500e+01f,  4.529205000e+01f,  4.554936000e+01f,  4.544838000e+01f,
                4.570609500e+01f,  4.560471000e+01f,  4.586283000e+01f,  4.576104000e+01f,  3.042280500e+01f,  3.035481000e+01f,
                1.513492500e+01f,  1.510086000e+01f,  9.995135000e+00f,  9.972830000e+00f,  1.989100000e+01f,  1.984630000e+01f,
                2.968705500e+01f,  2.961987000e+01f,  2.978911500e+01f,  2.972166000e+01f,  2.989117500e+01f,  2.982345000e+01f,
                2.999323500e+01f,  2.992524000e+01f,  3.009529500e+01f,  3.002703000e+01f,  3.019735500e+01f,  3.012882000e+01f,
                3.029941500e+01f,  3.023061000e+01f,  3.040147500e+01f,  3.033240000e+01f,  3.050353500e+01f,  3.043419000e+01f,
                2.023030000e+01f,  2.018398000e+01f,  1.006227500e+01f,  1.003907000e+01f,  4.987780000e+00f,  4.976380000e+00f,
                9.924035000e+00f,  9.901190000e+00f,  1.480849500e+01f,  1.477416000e+01f,  1.485831000e+01f,  1.482384000e+01f,
                1.490812500e+01f,  1.487352000e+01f,  1.495794000e+01f,  1.492320000e+01f,  1.500775500e+01f,  1.497288000e+01f,
                1.505757000e+01f,  1.502256000e+01f,  1.510738500e+01f,  1.507224000e+01f,  1.515720000e+01f,  1.512192000e+01f,
                1.520701500e+01f,  1.517160000e+01f,  1.008342500e+01f,  1.005977000e+01f,  5.014330000e+00f,  5.002480000e+00f,
                3.545575000e+00f,  3.537640000e+00f,  7.059605000e+00f,  7.043690000e+00f,  1.054182000e+01f,  1.051788000e+01f,
                1.058920500e+01f,  1.056513000e+01f,  1.063659000e+01f,  1.061238000e+01f,  1.068397500e+01f,  1.065963000e+01f,
                1.073136000e+01f,  1.070688000e+01f,  1.077874500e+01f,  1.075413000e+01f,  1.082613000e+01f,  1.080138000e+01f,
                1.087351500e+01f,  1.084863000e+01f,  1.092090000e+01f,  1.089588000e+01f,  7.245995000e+00f,  7.229270000e+00f,
                3.605605000e+00f,  3.597220000e+00f,  7.125035000e+00f,  7.108670000e+00f,  1.418320000e+01f,  1.415038000e+01f,
                2.117395500e+01f,  2.112459000e+01f,  2.126629500e+01f,  2.121666000e+01f,  2.135863500e+01f,  2.130873000e+01f,
                2.145097500e+01f,  2.140080000e+01f,  2.154331500e+01f,  2.149287000e+01f,  2.163565500e+01f,  2.158494000e+01f,
                2.172799500e+01f,  2.167701000e+01f,  2.182033500e+01f,  2.176908000e+01f,  2.191267500e+01f,  2.186115000e+01f,
                1.453546000e+01f,  1.450102000e+01f,  7.231055000e+00f,  7.213790000e+00f,  1.072947000e+01f,  1.070418000e+01f,
                2.135296500e+01f,  2.130225000e+01f,  3.186967500e+01f,  3.179340000e+01f,  3.200454000e+01f,  3.192786000e+01f,
                3.213940500e+01f,  3.206232000e+01f,  3.227427000e+01f,  3.219678000e+01f,  3.240913500e+01f,  3.233124000e+01f,
                3.254400000e+01f,  3.246570000e+01f,  3.267886500e+01f,  3.260016000e+01f,  3.281373000e+01f,  3.273462000e+01f,
                3.294859500e+01f,  3.286908000e+01f,  2.185057500e+01f,  2.179743000e+01f,  1.086744000e+01f,  1.084080000e+01f,
                1.434997000e+01f,  1.431526000e+01f,  2.855108000e+01f,  2.848148000e+01f,  4.260225000e+01f,  4.249758000e+01f,
                4.277721000e+01f,  4.267200000e+01f,  4.295217000e+01f,  4.284642000e+01f,  4.312713000e+01f,  4.302084000e+01f,
                4.330209000e+01f,  4.319526000e+01f,  4.347705000e+01f,  4.336968000e+01f,  4.365201000e+01f,  4.354410000e+01f,
                4.382697000e+01f,  4.371852000e+01f,  4.400193000e+01f,  4.389294000e+01f,  2.917352000e+01f,  2.910068000e+01f,
                1.450585000e+01f,  1.446934000e+01f,  1.797762500e+01f,  1.793300000e+01f,  3.575972500e+01f,  3.567025000e+01f,
                5.334495000e+01f,  5.321040000e+01f,  5.355757500e+01f,  5.342235000e+01f,  5.377020000e+01f,  5.363430000e+01f,
                5.398282500e+01f,  5.384625000e+01f,  5.419545000e+01f,  5.405820000e+01f,  5.440807500e+01f,  5.427015000e+01f,
                5.462070000e+01f,  5.448210000e+01f,  5.483332500e+01f,  5.469405000e+01f,  5.504595000e+01f,  5.490600000e+01f,
                3.648647500e+01f,  3.639295000e+01f,  1.813737500e+01f,  1.809050000e+01f,  1.877210000e+01f,  1.872500000e+01f,
                3.733382500e+01f,  3.723940000e+01f,  5.568382500e+01f,  5.554185000e+01f,  5.589645000e+01f,  5.575380000e+01f,
                5.610907500e+01f,  5.596575000e+01f,  5.632170000e+01f,  5.617770000e+01f,  5.653432500e+01f,  5.638965000e+01f,
                5.674695000e+01f,  5.660160000e+01f,  5.695957500e+01f,  5.681355000e+01f,  5.717220000e+01f,  5.702550000e+01f,
                5.738482500e+01f,  5.723745000e+01f,  3.803087500e+01f,  3.793240000e+01f,  1.890215000e+01f,  1.885280000e+01f,
                1.956657500e+01f,  1.951700000e+01f,  3.890792500e+01f,  3.880855000e+01f,  5.802270000e+01f,  5.787330000e+01f,
                5.823532500e+01f,  5.808525000e+01f,  5.844795000e+01f,  5.829720000e+01f,  5.866057500e+01f,  5.850915000e+01f,
                5.887320000e+01f,  5.872110000e+01f,  5.908582500e+01f,  5.893305000e+01f,  5.929845000e+01f,  5.914500000e+01f,
                5.951107500e+01f,  5.935695000e+01f,  5.972370000e+01f,  5.956890000e+01f,  3.957527500e+01f,  3.947185000e+01f,
                1.966692500e+01f,  1.961510000e+01f,  2.036105000e+01f,  2.030900000e+01f,  4.048202500e+01f,  4.037770000e+01f,
                6.036157500e+01f,  6.020475000e+01f,  6.057420000e+01f,  6.041670000e+01f,  6.078682500e+01f,  6.062865000e+01f,
                6.099945000e+01f,  6.084060000e+01f,  6.121207500e+01f,  6.105255000e+01f,  6.142470000e+01f,  6.126450000e+01f,
                6.163732500e+01f,  6.147645000e+01f,  6.184995000e+01f,  6.168840000e+01f,  6.206257500e+01f,  6.190035000e+01f,
                4.111967500e+01f,  4.101130000e+01f,  2.043170000e+01f,  2.037740000e+01f,  1.619623000e+01f,  1.615360000e+01f,
                3.219284000e+01f,  3.210740000e+01f,  4.798875000e+01f,  4.786032000e+01f,  4.815399000e+01f,  4.802502000e+01f,
                4.831923000e+01f,  4.818972000e+01f,  4.848447000e+01f,  4.835442000e+01f,  4.864971000e+01f,  4.851912000e+01f,
                4.881495000e+01f,  4.868382000e+01f,  4.898019000e+01f,  4.884852000e+01f,  4.914543000e+01f,  4.901322000e+01f,
                4.931067000e+01f,  4.917792000e+01f,  3.266192000e+01f,  3.257324000e+01f,  1.622467000e+01f,  1.618024000e+01f,
                1.206880500e+01f,  1.203609000e+01f,  2.398222500e+01f,  2.391666000e+01f,  3.573945000e+01f,  3.564090000e+01f,
                3.585973500e+01f,  3.576078000e+01f,  3.598002000e+01f,  3.588066000e+01f,  3.610030500e+01f,  3.600054000e+01f,
                3.622059000e+01f,  3.612042000e+01f,  3.634087500e+01f,  3.624030000e+01f,  3.646116000e+01f,  3.636018000e+01f,
                3.658144500e+01f,  3.648006000e+01f,  3.670173000e+01f,  3.659994000e+01f,  2.430325500e+01f,  2.423526000e+01f,
                1.206907500e+01f,  1.203501000e+01f,  7.987685000e+00f,  7.965380000e+00f,  1.586800000e+01f,  1.582330000e+01f,
                2.364040500e+01f,  2.357322000e+01f,  2.371816500e+01f,  2.365071000e+01f,  2.379592500e+01f,  2.372820000e+01f,
                2.387368500e+01f,  2.380569000e+01f,  2.395144500e+01f,  2.388318000e+01f,  2.402920500e+01f,  2.396067000e+01f,
                2.410696500e+01f,  2.403816000e+01f,  2.418472500e+01f,  2.411565000e+01f,  2.426248500e+01f,  2.419314000e+01f,
                1.606150000e+01f,  1.601518000e+01f,  7.973825000e+00f,  7.950620000e+00f,  3.961780000e+00f,  3.950380000e+00f,
                7.867985000e+00f,  7.845140000e+00f,  1.171834500e+01f,  1.168401000e+01f,  1.175601000e+01f,  1.172154000e+01f,
                1.179367500e+01f,  1.175907000e+01f,  1.183134000e+01f,  1.179660000e+01f,  1.186900500e+01f,  1.183413000e+01f,
                1.190667000e+01f,  1.187166000e+01f,  1.194433500e+01f,  1.190919000e+01f,  1.198200000e+01f,  1.194672000e+01f,
                1.201966500e+01f,  1.198425000e+01f,  7.954475000e+00f,  7.930820000e+00f,  3.947830000e+00f,  3.935980000e+00f,
                2.831425000e+00f,  2.823490000e+00f,  5.627255000e+00f,  5.611340000e+00f,  8.387220000e+00f,  8.363280000e+00f,
                8.422455000e+00f,  8.398380000e+00f,  8.457690000e+00f,  8.433480000e+00f,  8.492925000e+00f,  8.468580000e+00f,
                8.528160000e+00f,  8.503680000e+00f,  8.563395000e+00f,  8.538780000e+00f,  8.598630000e+00f,  8.573880000e+00f,
                8.633865000e+00f,  8.608980000e+00f,  8.669100000e+00f,  8.644080000e+00f,  5.740745000e+00f,  5.724020000e+00f,
                2.850955000e+00f,  2.842570000e+00f,  5.652185000e+00f,  5.635820000e+00f,  1.122940000e+01f,  1.119658000e+01f,
                1.673110500e+01f,  1.668174000e+01f,  1.679914500e+01f,  1.674951000e+01f,  1.686718500e+01f,  1.681728000e+01f,
                1.693522500e+01f,  1.688505000e+01f,  1.700326500e+01f,  1.695282000e+01f,  1.707130500e+01f,  1.702059000e+01f,
                1.713934500e+01f,  1.708836000e+01f,  1.720738500e+01f,  1.715613000e+01f,  1.727542500e+01f,  1.722390000e+01f,
                1.143586000e+01f,  1.140142000e+01f,  5.677205000e+00f,  5.659940000e+00f,  8.453370000e+00f,  8.428080000e+00f,
                1.678861500e+01f,  1.673790000e+01f,  2.500492500e+01f,  2.492865000e+01f,  2.510334000e+01f,  2.502666000e+01f,
                2.520175500e+01f,  2.512467000e+01f,  2.530017000e+01f,  2.522268000e+01f,  2.539858500e+01f,  2.532069000e+01f,
                2.549700000e+01f,  2.541870000e+01f,  2.559541500e+01f,  2.551671000e+01f,  2.569383000e+01f,  2.561472000e+01f,
                2.579224500e+01f,  2.571273000e+01f,  1.706752500e+01f,  1.701438000e+01f,  8.469840000e+00f,  8.443200000e+00f,
                1.122607000e+01f,  1.119136000e+01f,  2.228708000e+01f,  2.221748000e+01f,  3.318195000e+01f,  3.307728000e+01f,
                3.330831000e+01f,  3.320310000e+01f,  3.343467000e+01f,  3.332892000e+01f,  3.356103000e+01f,  3.345474000e+01f,
                3.368739000e+01f,  3.358056000e+01f,  3.381375000e+01f,  3.370638000e+01f,  3.394011000e+01f,  3.383220000e+01f,
                3.406647000e+01f,  3.395802000e+01f,  3.419283000e+01f,  3.408384000e+01f,  2.261792000e+01f,  2.254508000e+01f,
                1.121995000e+01f,  1.118344000e+01f,  1.396137500e+01f,  1.391675000e+01f,  2.770697500e+01f,  2.761750000e+01f,
                4.123545000e+01f,  4.110090000e+01f,  4.138732500e+01f,  4.125210000e+01f,  4.153920000e+01f,  4.140330000e+01f,
                4.169107500e+01f,  4.155450000e+01f,  4.184295000e+01f,  4.170570000e+01f,  4.199482500e+01f,  4.185690000e+01f,
                4.214670000e+01f,  4.200810000e+01f,  4.229857500e+01f,  4.215930000e+01f,  4.245045000e+01f,  4.231050000e+01f,
                2.806922500e+01f,  2.797570000e+01f,  1.391862500e+01f,  1.387175000e+01f,  1.453310000e+01f,  1.448600000e+01f,
                2.883557500e+01f,  2.874115000e+01f,  4.290607500e+01f,  4.276410000e+01f,  4.305795000e+01f,  4.291530000e+01f,
                4.320982500e+01f,  4.306650000e+01f,  4.336170000e+01f,  4.321770000e+01f,  4.351357500e+01f,  4.336890000e+01f,
                4.366545000e+01f,  4.352010000e+01f,  4.381732500e+01f,  4.367130000e+01f,  4.396920000e+01f,  4.382250000e+01f,
                4.412107500e+01f,  4.397370000e+01f,  2.916812500e+01f,  2.906965000e+01f,  1.446065000e+01f,  1.441130000e+01f,
                1.510482500e+01f,  1.505525000e+01f,  2.996417500e+01f,  2.986480000e+01f,  4.457670000e+01f,  4.442730000e+01f,
                4.472857500e+01f,  4.457850000e+01f,  4.488045000e+01f,  4.472970000e+01f,  4.503232500e+01f,  4.488090000e+01f,
                4.518420000e+01f,  4.503210000e+01f,  4.533607500e+01f,  4.518330000e+01f,  4.548795000e+01f,  4.533450000e+01f,
                4.563982500e+01f,  4.548570000e+01f,  4.579170000e+01f,  4.563690000e+01f,  3.026702500e+01f,  3.016360000e+01f,
                1.500267500e+01f,  1.495085000e+01f,  1.567655000e+01f,  1.562450000e+01f,  3.109277500e+01f,  3.098845000e+01f,
                4.624732500e+01f,  4.609050000e+01f,  4.639920000e+01f,  4.624170000e+01f,  4.655107500e+01f,  4.639290000e+01f,
                4.670295000e+01f,  4.654410000e+01f,  4.685482500e+01f,  4.669530000e+01f,  4.700670000e+01f,  4.684650000e+01f,
                4.715857500e+01f,  4.699770000e+01f,  4.731045000e+01f,  4.714890000e+01f,  4.746232500e+01f,  4.730010000e+01f,
                3.136592500e+01f,  3.125755000e+01f,  1.554470000e+01f,  1.549040000e+01f,  1.235953000e+01f,  1.231690000e+01f,
                2.450324000e+01f,  2.441780000e+01f,  3.643005000e+01f,  3.630162000e+01f,  3.654669000e+01f,  3.641772000e+01f,
                3.666333000e+01f,  3.653382000e+01f,  3.677997000e+01f,  3.664992000e+01f,  3.689661000e+01f,  3.676602000e+01f,
                3.701325000e+01f,  3.688212000e+01f,  3.712989000e+01f,  3.699822000e+01f,  3.724653000e+01f,  3.711432000e+01f,
                3.736317000e+01f,  3.723042000e+01f,  2.468072000e+01f,  2.459204000e+01f,  1.222597000e+01f,  1.218154000e+01f,
                9.124455000e+00f,  9.091740000e+00f,  1.808137500e+01f,  1.801581000e+01f,  2.686995000e+01f,  2.677140000e+01f,
                2.695378500e+01f,  2.685483000e+01f,  2.703762000e+01f,  2.693826000e+01f,  2.712145500e+01f,  2.702169000e+01f,
                2.720529000e+01f,  2.710512000e+01f,  2.728912500e+01f,  2.718855000e+01f,  2.737296000e+01f,  2.727198000e+01f,
                2.745679500e+01f,  2.735541000e+01f,  2.754063000e+01f,  2.743884000e+01f,  1.818370500e+01f,  1.811571000e+01f,
                9.003225000e+00f,  8.969160000e+00f,  5.980235000e+00f,  5.957930000e+00f,  1.184500000e+01f,  1.180030000e+01f,
                1.759375500e+01f,  1.752657000e+01f,  1.764721500e+01f,  1.757976000e+01f,  1.770067500e+01f,  1.763295000e+01f,
                1.775413500e+01f,  1.768614000e+01f,  1.780759500e+01f,  1.773933000e+01f,  1.786105500e+01f,  1.779252000e+01f,
                1.791451500e+01f,  1.784571000e+01f,  1.796797500e+01f,  1.789890000e+01f,  1.802143500e+01f,  1.795209000e+01f,
                1.189270000e+01f,  1.184638000e+01f,  5.885375000e+00f,  5.862170000e+00f,  2.935780000e+00f,  2.924380000e+00f,
                5.811935000e+00f,  5.789090000e+00f,  8.628195000e+00f,  8.593860000e+00f,  8.653710000e+00f,  8.619240000e+00f,
                8.679225000e+00f,  8.644620000e+00f,  8.704740000e+00f,  8.670000000e+00f,  8.730255000e+00f,  8.695380000e+00f,
                8.755770000e+00f,  8.720760000e+00f,  8.781285000e+00f,  8.746140000e+00f,  8.806800000e+00f,  8.771520000e+00f,
                8.832315000e+00f,  8.796900000e+00f,  5.825525000e+00f,  5.801870000e+00f,  2.881330000e+00f,  2.869480000e+00f,
                2.117288000e+00f,  2.109356000e+00f,  4.194976000e+00f,  4.179076000e+00f,  6.232848000e+00f,  6.208944000e+00f,
                6.256176000e+00f,  6.232164000e+00f,  6.279504000e+00f,  6.255384000e+00f,  6.302832000e+00f,  6.278604000e+00f,
                6.326160000e+00f,  6.301824000e+00f,  6.349488000e+00f,  6.325044000e+00f,  6.372816000e+00f,  6.348264000e+00f,
                6.396144000e+00f,  6.371484000e+00f,  6.419472000e+00f,  6.394704000e+00f,  4.237600000e+00f,  4.221052000e+00f,
                2.097704000e+00f,  2.089412000e+00f,  4.179712000e+00f,  4.163452000e+00f,  8.277200000e+00f,  8.244608000e+00f,
                1.229203200e+01f,  1.224303600e+01f,  1.233674400e+01f,  1.228753200e+01f,  1.238145600e+01f,  1.233202800e+01f,
                1.242616800e+01f,  1.237652400e+01f,  1.247088000e+01f,  1.242102000e+01f,  1.251559200e+01f,  1.246551600e+01f,
                1.256030400e+01f,  1.251001200e+01f,  1.260501600e+01f,  1.255450800e+01f,  1.264972800e+01f,  1.259900400e+01f,
                8.346032000e+00f,  8.312144000e+00f,  4.129312000e+00f,  4.112332000e+00f,  6.180144000e+00f,  6.155160000e+00f,
                1.223241600e+01f,  1.218234000e+01f,  1.815616800e+01f,  1.808089200e+01f,  1.822032000e+01f,  1.814472000e+01f,
                1.828447200e+01f,  1.820854800e+01f,  1.834862400e+01f,  1.827237600e+01f,  1.841277600e+01f,  1.833620400e+01f,
                1.847692800e+01f,  1.840003200e+01f,  1.854108000e+01f,  1.846386000e+01f,  1.860523200e+01f,  1.852768800e+01f,
                1.866938400e+01f,  1.859151600e+01f,  1.231104000e+01f,  1.225902000e+01f,  6.087696000e+00f,  6.061632000e+00f,
                8.111456000e+00f,  8.077352000e+00f,  1.604636800e+01f,  1.597801600e+01f,  2.380387200e+01f,  2.370112800e+01f,
                2.388552000e+01f,  2.378234400e+01f,  2.396716800e+01f,  2.386356000e+01f,  2.404881600e+01f,  2.394477600e+01f,
                2.413046400e+01f,  2.402599200e+01f,  2.421211200e+01f,  2.410720800e+01f,  2.429376000e+01f,  2.418842400e+01f,
                2.437540800e+01f,  2.426964000e+01f,  2.445705600e+01f,  2.435085600e+01f,  1.611836800e+01f,  1.604742400e+01f,
                7.965728000e+00f,  7.930184000e+00f,  9.966520000e+00f,  9.922900000e+00f,  1.970480000e+01f,  1.961738000e+01f,
                2.921376000e+01f,  2.908236000e+01f,  2.931096000e+01f,  2.917902000e+01f,  2.940816000e+01f,  2.927568000e+01f,
                2.950536000e+01f,  2.937234000e+01f,  2.960256000e+01f,  2.946900000e+01f,  2.969976000e+01f,  2.956566000e+01f,
                2.979696000e+01f,  2.966232000e+01f,  2.989416000e+01f,  2.975898000e+01f,  2.999136000e+01f,  2.985564000e+01f,
                1.975376000e+01f,  1.966310000e+01f,  9.756280000e+00f,  9.710860000e+00f,  1.033480000e+01f,  1.028920000e+01f,
                2.042948000e+01f,  2.033810000e+01f,  3.028296000e+01f,  3.014562000e+01f,  3.038016000e+01f,  3.024228000e+01f,
                3.047736000e+01f,  3.033894000e+01f,  3.057456000e+01f,  3.043560000e+01f,  3.067176000e+01f,  3.053226000e+01f,
                3.076896000e+01f,  3.062892000e+01f,  3.086616000e+01f,  3.072558000e+01f,  3.096336000e+01f,  3.082224000e+01f,
                3.106056000e+01f,  3.091890000e+01f,  2.045468000e+01f,  2.036006000e+01f,  1.010080000e+01f,  1.005340000e+01f,
                1.070308000e+01f,  1.065550000e+01f,  2.115416000e+01f,  2.105882000e+01f,  3.135216000e+01f,  3.120888000e+01f,
                3.144936000e+01f,  3.130554000e+01f,  3.154656000e+01f,  3.140220000e+01f,  3.164376000e+01f,  3.149886000e+01f,
                3.174096000e+01f,  3.159552000e+01f,  3.183816000e+01f,  3.169218000e+01f,  3.193536000e+01f,  3.178884000e+01f,
                3.203256000e+01f,  3.188550000e+01f,  3.212976000e+01f,  3.198216000e+01f,  2.115560000e+01f,  2.105702000e+01f,
                1.044532000e+01f,  1.039594000e+01f,  1.107136000e+01f,  1.102180000e+01f,  2.187884000e+01f,  2.177954000e+01f,
                3.242136000e+01f,  3.227214000e+01f,  3.251856000e+01f,  3.236880000e+01f,  3.261576000e+01f,  3.246546000e+01f,
                3.271296000e+01f,  3.256212000e+01f,  3.281016000e+01f,  3.265878000e+01f,  3.290736000e+01f,  3.275544000e+01f,
                3.300456000e+01f,  3.285210000e+01f,  3.310176000e+01f,  3.294876000e+01f,  3.319896000e+01f,  3.304542000e+01f,
                2.185652000e+01f,  2.175398000e+01f,  1.078984000e+01f,  1.073848000e+01f,  8.619056000e+00f,  8.578616000e+00f,
                1.702096000e+01f,  1.693993600e+01f,  2.520484800e+01f,  2.508309600e+01f,  2.527872000e+01f,  2.515653600e+01f,
                2.535259200e+01f,  2.522997600e+01f,  2.542646400e+01f,  2.530341600e+01f,  2.550033600e+01f,  2.537685600e+01f,
                2.557420800e+01f,  2.545029600e+01f,  2.564808000e+01f,  2.552373600e+01f,  2.572195200e+01f,  2.559717600e+01f,
                2.579582400e+01f,  2.567061600e+01f,  1.697027200e+01f,  1.688665600e+01f,  8.371376000e+00f,  8.329496000e+00f,
                6.278640000e+00f,  6.247716000e+00f,  1.238988000e+01f,  1.232792400e+01f,  1.833307200e+01f,  1.823997600e+01f,
                1.838556000e+01f,  1.829214000e+01f,  1.843804800e+01f,  1.834430400e+01f,  1.849053600e+01f,  1.839646800e+01f,
                1.854302400e+01f,  1.844863200e+01f,  1.859551200e+01f,  1.850079600e+01f,  1.864800000e+01f,  1.855296000e+01f,
                1.870048800e+01f,  1.860512400e+01f,  1.875297600e+01f,  1.865728800e+01f,  1.232724000e+01f,  1.226334000e+01f,
                6.076032000e+00f,  6.044028000e+00f,  4.057240000e+00f,  4.036228000e+00f,  7.999856000e+00f,  7.957760000e+00f,
                1.182741600e+01f,  1.176416400e+01f,  1.186046400e+01f,  1.179699600e+01f,  1.189351200e+01f,  1.182982800e+01f,
                1.192656000e+01f,  1.186266000e+01f,  1.195960800e+01f,  1.189549200e+01f,  1.199265600e+01f,  1.192832400e+01f,
                1.202570400e+01f,  1.196115600e+01f,  1.205875200e+01f,  1.199398800e+01f,  1.209180000e+01f,  1.202682000e+01f,
                7.941680000e+00f,  7.898288000e+00f,  3.910936000e+00f,  3.889204000e+00f,  1.961984000e+00f,  1.951280000e+00f,
                3.865144000e+00f,  3.843700000e+00f,  5.709264000e+00f,  5.677044000e+00f,  5.724816000e+00f,  5.692488000e+00f,
                5.740368000e+00f,  5.707932000e+00f,  5.755920000e+00f,  5.723376000e+00f,  5.771472000e+00f,  5.738820000e+00f,
                5.787024000e+00f,  5.754264000e+00f,  5.802576000e+00f,  5.769708000e+00f,  5.818128000e+00f,  5.785152000e+00f,
                5.833680000e+00f,  5.800596000e+00f,  3.827848000e+00f,  3.805756000e+00f,  1.883216000e+00f,  1.872152000e+00f,
                1.405797000e+00f,  1.398660000e+00f,  2.773551000e+00f,  2.759250000e+00f,  4.103100000e+00f,  4.081608000e+00f,
                4.116951000e+00f,  4.095378000e+00f,  4.130802000e+00f,  4.109148000e+00f,  4.144653000e+00f,  4.122918000e+00f,
                4.158504000e+00f,  4.136688000e+00f,  4.172355000e+00f,  4.150458000e+00f,  4.186206000e+00f,  4.164228000e+00f,
                4.200057000e+00f,  4.177998000e+00f,  4.213908000e+00f,  4.191768000e+00f,  2.769393000e+00f,  2.754606000e+00f,
                1.364703000e+00f,  1.357296000e+00f,  2.735697000e+00f,  2.721126000e+00f,  5.393040000e+00f,  5.363844000e+00f,
                7.971705000e+00f,  7.927830000e+00f,  7.997949000e+00f,  7.953912000e+00f,  8.024193000e+00f,  7.979994000e+00f,
                8.050437000e+00f,  8.006076000e+00f,  8.076681000e+00f,  8.032158000e+00f,  8.102925000e+00f,  8.058240000e+00f,
                8.129169000e+00f,  8.084322000e+00f,  8.155413000e+00f,  8.110404000e+00f,  8.181657000e+00f,  8.136486000e+00f,
                5.372412000e+00f,  5.342244000e+00f,  2.645085000e+00f,  2.629974000e+00f,  3.984354000e+00f,  3.962052000e+00f,
                7.847775000e+00f,  7.803090000e+00f,  1.158977700e+01f,  1.152262800e+01f,  1.162695600e+01f,  1.155956400e+01f,
                1.166413500e+01f,  1.159650000e+01f,  1.170131400e+01f,  1.163343600e+01f,  1.173849300e+01f,  1.167037200e+01f,
                1.177567200e+01f,  1.170730800e+01f,  1.181285100e+01f,  1.174424400e+01f,  1.185003000e+01f,  1.178118000e+01f,
                1.188720900e+01f,  1.181811600e+01f,  7.798365000e+00f,  7.752222000e+00f,  3.835800000e+00f,  3.812688000e+00f,
                5.146422000e+00f,  5.116092000e+00f,  1.012706400e+01f,  1.006629600e+01f,  1.494127800e+01f,  1.484996400e+01f,
                1.498793400e+01f,  1.489629600e+01f,  1.503459000e+01f,  1.494262800e+01f,  1.508124600e+01f,  1.498896000e+01f,
                1.512790200e+01f,  1.503529200e+01f,  1.517455800e+01f,  1.508162400e+01f,  1.522121400e+01f,  1.512795600e+01f,
                1.526787000e+01f,  1.517428800e+01f,  1.531452600e+01f,  1.522062000e+01f,  1.003656000e+01f,  9.973848000e+00f,
                4.931502000e+00f,  4.900092000e+00f,  6.216555000e+00f,  6.177900000e+00f,  1.222021500e+01f,  1.214277000e+01f,
                1.801017000e+01f,  1.789380000e+01f,  1.806484500e+01f,  1.794807000e+01f,  1.811952000e+01f,  1.800234000e+01f,
                1.817419500e+01f,  1.805661000e+01f,  1.822887000e+01f,  1.811088000e+01f,  1.828354500e+01f,  1.816515000e+01f,
                1.833822000e+01f,  1.821942000e+01f,  1.839289500e+01f,  1.827369000e+01f,  1.844757000e+01f,  1.832796000e+01f,
                1.207630500e+01f,  1.199643000e+01f,  5.926845000e+00f,  5.886840000e+00f,  6.425940000e+00f,  6.385800000e+00f,
                1.263007500e+01f,  1.254966000e+01f,  1.861159500e+01f,  1.849077000e+01f,  1.866627000e+01f,  1.854504000e+01f,
                1.872094500e+01f,  1.859931000e+01f,  1.877562000e+01f,  1.865358000e+01f,  1.883029500e+01f,  1.870785000e+01f,
                1.888497000e+01f,  1.876212000e+01f,  1.893964500e+01f,  1.881639000e+01f,  1.899432000e+01f,  1.887066000e+01f,
                1.904899500e+01f,  1.892493000e+01f,  1.246834500e+01f,  1.238550000e+01f,  6.118410000e+00f,  6.076920000e+00f,
                6.635325000e+00f,  6.593700000e+00f,  1.303993500e+01f,  1.295655000e+01f,  1.921302000e+01f,  1.908774000e+01f,
                1.926769500e+01f,  1.914201000e+01f,  1.932237000e+01f,  1.919628000e+01f,  1.937704500e+01f,  1.925055000e+01f,
                1.943172000e+01f,  1.930482000e+01f,  1.948639500e+01f,  1.935909000e+01f,  1.954107000e+01f,  1.941336000e+01f,
                1.959574500e+01f,  1.946763000e+01f,  1.965042000e+01f,  1.952190000e+01f,  1.286038500e+01f,  1.277457000e+01f,
                6.309975000e+00f,  6.267000000e+00f,  6.844710000e+00f,  6.801600000e+00f,  1.344979500e+01f,  1.336344000e+01f,
                1.981444500e+01f,  1.968471000e+01f,  1.986912000e+01f,  1.973898000e+01f,  1.992379500e+01f,  1.979325000e+01f,
                1.997847000e+01f,  1.984752000e+01f,  2.003314500e+01f,  1.990179000e+01f,  2.008782000e+01f,  1.995606000e+01f,
                2.014249500e+01f,  2.001033000e+01f,  2.019717000e+01f,  2.006460000e+01f,  2.025184500e+01f,  2.011887000e+01f,
                1.325242500e+01f,  1.316364000e+01f,  6.501540000e+00f,  6.457080000e+00f,  5.227746000e+00f,  5.192664000e+00f,
                1.025925600e+01f,  1.018898400e+01f,  1.509388200e+01f,  1.498831200e+01f,  1.513470600e+01f,  1.502881200e+01f,
                1.517553000e+01f,  1.506931200e+01f,  1.521635400e+01f,  1.510981200e+01f,  1.525717800e+01f,  1.515031200e+01f,
                1.529800200e+01f,  1.519081200e+01f,  1.533882600e+01f,  1.523131200e+01f,  1.537965000e+01f,  1.527181200e+01f,
                1.542047400e+01f,  1.531231200e+01f,  1.007673600e+01f,  1.000452000e+01f,  4.936362000e+00f,  4.900200000e+00f,
                3.729447000e+00f,  3.702690000e+00f,  7.308315000e+00f,  7.254720000e+00f,  1.073611800e+01f,  1.065560400e+01f,
                1.076454900e+01f,  1.068379200e+01f,  1.079298000e+01f,  1.071198000e+01f,  1.082141100e+01f,  1.074016800e+01f,
                1.084984200e+01f,  1.076835600e+01f,  1.087827300e+01f,  1.079654400e+01f,  1.090670400e+01f,  1.082473200e+01f,
                1.093513500e+01f,  1.085292000e+01f,  1.096356600e+01f,  1.088110800e+01f,  7.152957000e+00f,  7.097904000e+00f,
                3.498273000e+00f,  3.470706000e+00f,  2.355159000e+00f,  2.337024000e+00f,  4.607664000e+00f,  4.571340000e+00f,
                6.757191000e+00f,  6.702624000e+00f,  6.774687000e+00f,  6.719958000e+00f,  6.792183000e+00f,  6.737292000e+00f,
                6.809679000e+00f,  6.754626000e+00f,  6.827175000e+00f,  6.771960000e+00f,  6.844671000e+00f,  6.789294000e+00f,
                6.862167000e+00f,  6.806628000e+00f,  6.879663000e+00f,  6.823962000e+00f,  6.897159000e+00f,  6.841296000e+00f,
                4.491780000e+00f,  4.454484000e+00f,  2.192619000e+00f,  2.173944000e+00f,  1.110228000e+00f,  1.101012000e+00f,
                2.167995000e+00f,  2.149536000e+00f,  3.173139000e+00f,  3.145410000e+00f,  3.181158000e+00f,  3.153348000e+00f,
                3.189177000e+00f,  3.161286000e+00f,  3.197196000e+00f,  3.169224000e+00f,  3.205215000e+00f,  3.177162000e+00f,
                3.213234000e+00f,  3.185100000e+00f,  3.221253000e+00f,  3.193038000e+00f,  3.229272000e+00f,  3.200976000e+00f,
                3.237291000e+00f,  3.208914000e+00f,  2.103897000e+00f,  2.084952000e+00f,  1.024746000e+00f,  1.015260000e+00f,
                7.682320000e-01f,  7.626820000e-01f,  1.505540000e+00f,  1.494422000e+00f,  2.211816000e+00f,  2.195112000e+00f,
                2.218620000e+00f,  2.201862000e+00f,  2.225424000e+00f,  2.208612000e+00f,  2.232228000e+00f,  2.215362000e+00f,
                2.239032000e+00f,  2.222112000e+00f,  2.245836000e+00f,  2.228862000e+00f,  2.252640000e+00f,  2.235612000e+00f,
                2.259444000e+00f,  2.242362000e+00f,  2.266248000e+00f,  2.249112000e+00f,  1.478684000e+00f,  1.467242000e+00f,
                7.232320000e-01f,  7.175020000e-01f,  1.462700000e+00f,  1.451402000e+00f,  2.862040000e+00f,  2.839408000e+00f,
                4.197804000e+00f,  4.163802000e+00f,  4.210440000e+00f,  4.176330000e+00f,  4.223076000e+00f,  4.188858000e+00f,
                4.235712000e+00f,  4.201386000e+00f,  4.248348000e+00f,  4.213914000e+00f,  4.260984000e+00f,  4.226442000e+00f,
                4.273620000e+00f,  4.238970000e+00f,  4.286256000e+00f,  4.251498000e+00f,  4.298892000e+00f,  4.264026000e+00f,
                2.800120000e+00f,  2.776840000e+00f,  1.367084000e+00f,  1.355426000e+00f,  2.079840000e+00f,  2.062596000e+00f,
                4.062372000e+00f,  4.027830000e+00f,  5.947272000e+00f,  5.895378000e+00f,  5.964768000e+00f,  5.912712000e+00f,
                5.982264000e+00f,  5.930046000e+00f,  5.999760000e+00f,  5.947380000e+00f,  6.017256000e+00f,  5.964714000e+00f,
                6.034752000e+00f,  5.982048000e+00f,  6.052248000e+00f,  5.999382000e+00f,  6.069744000e+00f,  6.016716000e+00f,
                6.087240000e+00f,  6.034050000e+00f,  3.957180000e+00f,  3.921666000e+00f,  1.927992000e+00f,  1.910208000e+00f,
                2.616088000e+00f,  2.592700000e+00f,  5.099408000e+00f,  5.052560000e+00f,  7.449528000e+00f,  7.379148000e+00f,
                7.470912000e+00f,  7.400316000e+00f,  7.492296000e+00f,  7.421484000e+00f,  7.513680000e+00f,  7.442652000e+00f,
                7.535064000e+00f,  7.463820000e+00f,  7.556448000e+00f,  7.484988000e+00f,  7.577832000e+00f,  7.506156000e+00f,
                7.599216000e+00f,  7.527324000e+00f,  7.620600000e+00f,  7.548492000e+00f,  4.942736000e+00f,  4.894592000e+00f,
                2.402392000e+00f,  2.378284000e+00f,  3.067880000e+00f,  3.038150000e+00f,  5.966020000e+00f,  5.906470000e+00f,
                8.693880000e+00f,  8.604420000e+00f,  8.718180000e+00f,  8.628450000e+00f,  8.742480000e+00f,  8.652480000e+00f,
                8.766780000e+00f,  8.676510000e+00f,  8.791080000e+00f,  8.700540000e+00f,  8.815380000e+00f,  8.724570000e+00f,
                8.839680000e+00f,  8.748600000e+00f,  8.863980000e+00f,  8.772630000e+00f,  8.888280000e+00f,  8.796660000e+00f,
                5.749660000e+00f,  5.688490000e+00f,  2.786720000e+00f,  2.756090000e+00f,  3.162920000e+00f,  3.132200000e+00f,
                6.150160000e+00f,  6.088630000e+00f,  8.961180000e+00f,  8.868750000e+00f,  8.985480000e+00f,  8.892780000e+00f,
                9.009780000e+00f,  8.916810000e+00f,  9.034080000e+00f,  8.940840000e+00f,  9.058380000e+00f,  8.964870000e+00f,
                9.082680000e+00f,  8.988900000e+00f,  9.106980000e+00f,  9.012930000e+00f,  9.131280000e+00f,  9.036960000e+00f,
                9.155580000e+00f,  9.060990000e+00f,  5.921920000e+00f,  5.858770000e+00f,  2.869880000e+00f,  2.838260000e+00f,
                3.257960000e+00f,  3.226250000e+00f,  6.334300000e+00f,  6.270790000e+00f,  9.228480000e+00f,  9.133080000e+00f,
                9.252780000e+00f,  9.157110000e+00f,  9.277080000e+00f,  9.181140000e+00f,  9.301380000e+00f,  9.205170000e+00f,
                9.325680000e+00f,  9.229200000e+00f,  9.349980000e+00f,  9.253230000e+00f,  9.374280000e+00f,  9.277260000e+00f,
                9.398580000e+00f,  9.301290000e+00f,  9.422880000e+00f,  9.325320000e+00f,  6.094180000e+00f,  6.029050000e+00f,
                2.953040000e+00f,  2.920430000e+00f,  3.353000000e+00f,  3.320300000e+00f,  6.518440000e+00f,  6.452950000e+00f,
                9.495780000e+00f,  9.397410000e+00f,  9.520080000e+00f,  9.421440000e+00f,  9.544380000e+00f,  9.445470000e+00f,
                9.568680000e+00f,  9.469500000e+00f,  9.592980000e+00f,  9.493530000e+00f,  9.617280000e+00f,  9.517560000e+00f,
                9.641580000e+00f,  9.541590000e+00f,  9.665880000e+00f,  9.565620000e+00f,  9.690180000e+00f,  9.589650000e+00f,
                6.266440000e+00f,  6.199330000e+00f,  3.036200000e+00f,  3.002600000e+00f,  2.470720000e+00f,  2.444164000e+00f,
                4.788368000e+00f,  4.735184000e+00f,  6.952512000e+00f,  6.872628000e+00f,  6.970008000e+00f,  6.889908000e+00f,
                6.987504000e+00f,  6.907188000e+00f,  7.005000000e+00f,  6.924468000e+00f,  7.022496000e+00f,  6.941748000e+00f,
                7.039992000e+00f,  6.959028000e+00f,  7.057488000e+00f,  6.976308000e+00f,  7.074984000e+00f,  6.993588000e+00f,
                7.092480000e+00f,  7.010868000e+00f,  4.570352000e+00f,  4.515872000e+00f,  2.206048000e+00f,  2.178772000e+00f,
                1.690716000e+00f,  1.670502000e+00f,  3.264360000e+00f,  3.223878000e+00f,  4.720608000e+00f,  4.659804000e+00f,
                4.732272000e+00f,  4.671306000e+00f,  4.743936000e+00f,  4.682808000e+00f,  4.755600000e+00f,  4.694310000e+00f,
                4.767264000e+00f,  4.705812000e+00f,  4.778928000e+00f,  4.717314000e+00f,  4.790592000e+00f,  4.728816000e+00f,
                4.802256000e+00f,  4.740318000e+00f,  4.813920000e+00f,  4.751820000e+00f,  3.088536000e+00f,  3.047082000e+00f,
                1.483788000e+00f,  1.463034000e+00f,  1.016552000e+00f,  1.002878000e+00f,  1.953544000e+00f,  1.926160000e+00f,
                2.810760000e+00f,  2.769630000e+00f,  2.817564000e+00f,  2.776326000e+00f,  2.824368000e+00f,  2.783022000e+00f,
                2.831172000e+00f,  2.789718000e+00f,  2.837976000e+00f,  2.796414000e+00f,  2.844780000e+00f,  2.803110000e+00f,
                2.851584000e+00f,  2.809806000e+00f,  2.858388000e+00f,  2.816502000e+00f,  2.865192000e+00f,  2.823198000e+00f,
                1.828120000e+00f,  1.800088000e+00f,  8.729840000e-01f,  8.589500000e-01f,  4.517920000e-01f,  4.448560000e-01f,
                8.630480000e-01f,  8.491580000e-01f,  1.233660000e+00f,  1.212798000e+00f,  1.236576000e+00f,  1.215660000e+00f,
                1.239492000e+00f,  1.218522000e+00f,  1.242408000e+00f,  1.221384000e+00f,  1.245324000e+00f,  1.224246000e+00f,
                1.248240000e+00f,  1.227108000e+00f,  1.251156000e+00f,  1.229970000e+00f,  1.254072000e+00f,  1.232832000e+00f,
                1.256988000e+00f,  1.235694000e+00f,  7.962320000e-01f,  7.820180000e-01f,  3.772000000e-01f,  3.700840000e-01f,
                2.758730000e-01f,  2.727020000e-01f,  5.335030000e-01f,  5.271520000e-01f,  7.728360000e-01f,  7.632960000e-01f,
                7.750230000e-01f,  7.654560000e-01f,  7.772100000e-01f,  7.676160000e-01f,  7.793970000e-01f,  7.697760000e-01f,
                7.815840000e-01f,  7.719360000e-01f,  7.837710000e-01f,  7.740960000e-01f,  7.859580000e-01f,  7.762560000e-01f,
                7.881450000e-01f,  7.784160000e-01f,  7.903320000e-01f,  7.805760000e-01f,  5.080330000e-01f,  5.015200000e-01f,
                2.445710000e-01f,  2.413100000e-01f,  5.032810000e-01f,  4.968400000e-01f,  9.693200000e-01f,  9.564200000e-01f,
                1.398009000e+00f,  1.378632000e+00f,  1.401897000e+00f,  1.382466000e+00f,  1.405785000e+00f,  1.386300000e+00f,
                1.409673000e+00f,  1.390134000e+00f,  1.413561000e+00f,  1.393968000e+00f,  1.417449000e+00f,  1.397802000e+00f,
                1.421337000e+00f,  1.401636000e+00f,  1.425225000e+00f,  1.405470000e+00f,  1.429113000e+00f,  1.409304000e+00f,
                9.142760000e-01f,  9.010520000e-01f,  4.378690000e-01f,  4.312480000e-01f,  6.804420000e-01f,  6.706320000e-01f,
                1.303887000e+00f,  1.284240000e+00f,  1.870173000e+00f,  1.840662000e+00f,  1.875276000e+00f,  1.845684000e+00f,
                1.880379000e+00f,  1.850706000e+00f,  1.885482000e+00f,  1.855728000e+00f,  1.890585000e+00f,  1.860750000e+00f,
                1.895688000e+00f,  1.865772000e+00f,  1.900791000e+00f,  1.870794000e+00f,  1.905894000e+00f,  1.875816000e+00f,
                1.910997000e+00f,  1.880838000e+00f,  1.215165000e+00f,  1.195032000e+00f,  5.781120000e-01f,  5.680320000e-01f,
                8.055740000e-01f,  7.922960000e-01f,  1.533640000e+00f,  1.507048000e+00f,  2.183982000e+00f,  2.144040000e+00f,
                2.189814000e+00f,  2.149764000e+00f,  2.195646000e+00f,  2.155488000e+00f,  2.201478000e+00f,  2.161212000e+00f,
                2.207310000e+00f,  2.166936000e+00f,  2.213142000e+00f,  2.172660000e+00f,  2.218974000e+00f,  2.178384000e+00f,
                2.224806000e+00f,  2.184108000e+00f,  2.230638000e+00f,  2.189832000e+00f,  1.407136000e+00f,  1.379896000e+00f,
                6.635180000e-01f,  6.498800000e-01f,  8.768950000e-01f,  8.600500000e-01f,  1.655015000e+00f,  1.621280000e+00f,
                2.334090000e+00f,  2.283420000e+00f,  2.340165000e+00f,  2.289360000e+00f,  2.346240000e+00f,  2.295300000e+00f,
                2.352315000e+00f,  2.301240000e+00f,  2.358390000e+00f,  2.307180000e+00f,  2.364465000e+00f,  2.313120000e+00f,
                2.370540000e+00f,  2.319060000e+00f,  2.376615000e+00f,  2.325000000e+00f,  2.382690000e+00f,  2.330940000e+00f,
                1.486625000e+00f,  1.452080000e+00f,  6.923050000e-01f,  6.750100000e-01f,  9.021400000e-01f,  8.848000000e-01f,
                1.702535000e+00f,  1.667810000e+00f,  2.400915000e+00f,  2.348760000e+00f,  2.406990000e+00f,  2.354700000e+00f,
                2.413065000e+00f,  2.360640000e+00f,  2.419140000e+00f,  2.366580000e+00f,  2.425215000e+00f,  2.372520000e+00f,
                2.431290000e+00f,  2.378460000e+00f,  2.437365000e+00f,  2.384400000e+00f,  2.443440000e+00f,  2.390340000e+00f,
                2.449515000e+00f,  2.396280000e+00f,  1.528205000e+00f,  1.492670000e+00f,  7.116100000e-01f,  6.938200000e-01f,
                9.273850000e-01f,  9.095500000e-01f,  1.750055000e+00f,  1.714340000e+00f,  2.467740000e+00f,  2.414100000e+00f,
                2.473815000e+00f,  2.420040000e+00f,  2.479890000e+00f,  2.425980000e+00f,  2.485965000e+00f,  2.431920000e+00f,
                2.492040000e+00f,  2.437860000e+00f,  2.498115000e+00f,  2.443800000e+00f,  2.504190000e+00f,  2.449740000e+00f,
                2.510265000e+00f,  2.455680000e+00f,  2.516340000e+00f,  2.461620000e+00f,  1.569785000e+00f,  1.533260000e+00f,
                7.309150000e-01f,  7.126300000e-01f,  9.526300000e-01f,  9.343000000e-01f,  1.797575000e+00f,  1.760870000e+00f,
                2.534565000e+00f,  2.479440000e+00f,  2.540640000e+00f,  2.485380000e+00f,  2.546715000e+00f,  2.491320000e+00f,
                2.552790000e+00f,  2.497260000e+00f,  2.558865000e+00f,  2.503200000e+00f,  2.564940000e+00f,  2.509140000e+00f,
                2.571015000e+00f,  2.515080000e+00f,  2.577090000e+00f,  2.521020000e+00f,  2.583165000e+00f,  2.526960000e+00f,
                1.611365000e+00f,  1.573850000e+00f,  7.502200000e-01f,  7.314400000e-01f,  6.330980000e-01f,  6.182360000e-01f,
                1.178536000e+00f,  1.148776000e+00f,  1.636098000e+00f,  1.591404000e+00f,  1.639986000e+00f,  1.595184000e+00f,
                1.643874000e+00f,  1.598964000e+00f,  1.647762000e+00f,  1.602744000e+00f,  1.651650000e+00f,  1.606524000e+00f,
                1.655538000e+00f,  1.610304000e+00f,  1.659426000e+00f,  1.614084000e+00f,  1.663314000e+00f,  1.617864000e+00f,
                1.667202000e+00f,  1.621644000e+00f,  1.021360000e+00f,  9.909520000e-01f,  4.655540000e-01f,  4.503320000e-01f,
                3.762870000e-01f,  3.649920000e-01f,  6.856950000e-01f,  6.630780000e-01f,  9.280620000e-01f,  8.940960000e-01f,
                9.302490000e-01f,  8.962020000e-01f,  9.324360000e-01f,  8.983080000e-01f,  9.346230000e-01f,  9.004140000e-01f,
                9.368100000e-01f,  9.025200000e-01f,  9.389970000e-01f,  9.046260000e-01f,  9.411840000e-01f,  9.067320000e-01f,
                9.433710000e-01f,  9.088380000e-01f,  9.455580000e-01f,  9.109440000e-01f,  5.616570000e-01f,  5.385540000e-01f,
                2.464170000e-01f,  2.348520000e-01f,  1.839790000e-01f,  1.763500000e-01f,  3.226160000e-01f,  3.073400000e-01f,
                4.158030000e-01f,  3.928620000e-01f,  4.167750000e-01f,  3.937800000e-01f,  4.177470000e-01f,  3.946980000e-01f,
                4.187190000e-01f,  3.956160000e-01f,  4.196910000e-01f,  3.965340000e-01f,  4.206630000e-01f,  3.974520000e-01f,
                4.216350000e-01f,  3.983700000e-01f,  4.226070000e-01f,  3.992880000e-01f,  4.235790000e-01f,  4.002060000e-01f,
                2.358200000e-01f,  2.202200000e-01f,  9.459100000e-02f,  8.678200000e-02f,  5.795600000e-02f,  5.409200000e-02f,
                9.286300000e-02f,  8.512600000e-02f,  1.046670000e-01f,  9.304800000e-02f,  1.049100000e-01f,  9.326400000e-02f,
                1.051530000e-01f,  9.348000000e-02f,  1.053960000e-01f,  9.369600000e-02f,  1.056390000e-01f,  9.391200000e-02f,
                1.058820000e-01f,  9.412800000e-02f,  1.061250000e-01f,  9.434400000e-02f,  1.063680000e-01f,  9.456000000e-02f,
                1.066110000e-01f,  9.477600000e-02f,  4.741300000e-02f,  3.951400000e-02f,  1.185800000e-02f,  7.904000000e-03f,
                2.484797000e+00f,  2.480834000e+00f,  4.951459000e+00f,  4.943524000e+00f,  7.399932000e+00f,  7.388016000e+00f,
                7.416699000e+00f,  7.404756000e+00f,  7.433466000e+00f,  7.421496000e+00f,  7.450233000e+00f,  7.438236000e+00f,
                7.467000000e+00f,  7.454976000e+00f,  7.483767000e+00f,  7.471716000e+00f,  7.500534000e+00f,  7.488456000e+00f,
                7.517301000e+00f,  7.505196000e+00f,  7.534068000e+00f,  7.521936000e+00f,  5.003965000e+00f,  4.995868000e+00f,
                2.492591000e+00f,  2.488538000e+00f,  4.960333000e+00f,  4.952308000e+00f,  9.883640000e+00f,  9.867572000e+00f,
                1.476981300e+01f,  1.474568400e+01f,  1.480286100e+01f,  1.477867800e+01f,  1.483590900e+01f,  1.481167200e+01f,
                1.486895700e+01f,  1.484466600e+01f,  1.490200500e+01f,  1.487766000e+01f,  1.493505300e+01f,  1.491065400e+01f,
                1.496810100e+01f,  1.494364800e+01f,  1.500114900e+01f,  1.497664200e+01f,  1.503419700e+01f,  1.500963600e+01f,
                9.984548000e+00f,  9.968156000e+00f,  4.973113000e+00f,  4.964908000e+00f,  7.424826000e+00f,  7.412640000e+00f,
                1.479297900e+01f,  1.476858000e+01f,  2.210429700e+01f,  2.206765800e+01f,  2.215314000e+01f,  2.211642000e+01f,
                2.220198300e+01f,  2.216518200e+01f,  2.225082600e+01f,  2.221394400e+01f,  2.229966900e+01f,  2.226270600e+01f,
                2.234851200e+01f,  2.231146800e+01f,  2.239735500e+01f,  2.236023000e+01f,  2.244619800e+01f,  2.240899200e+01f,
                2.249504100e+01f,  2.245775400e+01f,  1.493818500e+01f,  1.491330000e+01f,  7.439784000e+00f,  7.427328000e+00f,
                9.876494000e+00f,  9.860048000e+00f,  1.967591200e+01f,  1.964298400e+01f,  2.939803800e+01f,  2.934859200e+01f,
                2.946219000e+01f,  2.941263600e+01f,  2.952634200e+01f,  2.947668000e+01f,  2.959049400e+01f,  2.954072400e+01f,
                2.965464600e+01f,  2.960476800e+01f,  2.971879800e+01f,  2.966881200e+01f,  2.978295000e+01f,  2.973285600e+01f,
                2.984710200e+01f,  2.979690000e+01f,  2.991125400e+01f,  2.986094400e+01f,  1.986131200e+01f,  1.982773600e+01f,
                9.890822000e+00f,  9.874016000e+00f,  1.231355500e+01f,  1.229275000e+01f,  2.452887500e+01f,  2.448722000e+01f,
                3.664569000e+01f,  3.658314000e+01f,  3.672466500e+01f,  3.666198000e+01f,  3.680364000e+01f,  3.674082000e+01f,
                3.688261500e+01f,  3.681966000e+01f,  3.696159000e+01f,  3.689850000e+01f,  3.704056500e+01f,  3.697734000e+01f,
                3.711954000e+01f,  3.705618000e+01f,  3.719851500e+01f,  3.713502000e+01f,  3.727749000e+01f,  3.721386000e+01f,
                2.475036500e+01f,  2.470790000e+01f,  1.232444500e+01f,  1.230319000e+01f,  1.260610000e+01f,  1.258480000e+01f,
                2.511099500e+01f,  2.506835000e+01f,  3.751441500e+01f,  3.745038000e+01f,  3.759339000e+01f,  3.752922000e+01f,
                3.767236500e+01f,  3.760806000e+01f,  3.775134000e+01f,  3.768690000e+01f,  3.783031500e+01f,  3.776574000e+01f,
                3.790929000e+01f,  3.784458000e+01f,  3.798826500e+01f,  3.792342000e+01f,  3.806724000e+01f,  3.800226000e+01f,
                3.814621500e+01f,  3.808110000e+01f,  2.532654500e+01f,  2.528309000e+01f,  1.261105000e+01f,  1.258930000e+01f,
                1.289864500e+01f,  1.287685000e+01f,  2.569311500e+01f,  2.564948000e+01f,  3.838314000e+01f,  3.831762000e+01f,
                3.846211500e+01f,  3.839646000e+01f,  3.854109000e+01f,  3.847530000e+01f,  3.862006500e+01f,  3.855414000e+01f,
                3.869904000e+01f,  3.863298000e+01f,  3.877801500e+01f,  3.871182000e+01f,  3.885699000e+01f,  3.879066000e+01f,
                3.893596500e+01f,  3.886950000e+01f,  3.901494000e+01f,  3.894834000e+01f,  2.590272500e+01f,  2.585828000e+01f,
                1.289765500e+01f,  1.287541000e+01f,  1.319119000e+01f,  1.316890000e+01f,  2.627523500e+01f,  2.623061000e+01f,
                3.925186500e+01f,  3.918486000e+01f,  3.933084000e+01f,  3.926370000e+01f,  3.940981500e+01f,  3.934254000e+01f,
                3.948879000e+01f,  3.942138000e+01f,  3.956776500e+01f,  3.950022000e+01f,  3.964674000e+01f,  3.957906000e+01f,
                3.972571500e+01f,  3.965790000e+01f,  3.980469000e+01f,  3.973674000e+01f,  3.988366500e+01f,  3.981558000e+01f,
                2.647890500e+01f,  2.643347000e+01f,  1.318426000e+01f,  1.316152000e+01f,  1.050235400e+01f,  1.048432400e+01f,
                2.091748000e+01f,  2.088138400e+01f,  3.124516200e+01f,  3.119096400e+01f,  3.130737000e+01f,  3.125306400e+01f,
                3.136957800e+01f,  3.131516400e+01f,  3.143178600e+01f,  3.137726400e+01f,  3.149399400e+01f,  3.143936400e+01f,
                3.155620200e+01f,  3.150146400e+01f,  3.161841000e+01f,  3.156356400e+01f,  3.168061800e+01f,  3.162566400e+01f,
                3.174282600e+01f,  3.168776400e+01f,  2.107220800e+01f,  2.103546400e+01f,  1.049119400e+01f,  1.047280400e+01f,
                7.837035000e+00f,  7.823364000e+00f,  1.560751500e+01f,  1.558014600e+01f,  2.331127800e+01f,  2.327018400e+01f,
                2.335720500e+01f,  2.331603000e+01f,  2.340313200e+01f,  2.336187600e+01f,  2.344905900e+01f,  2.340772200e+01f,
                2.349498600e+01f,  2.345356800e+01f,  2.354091300e+01f,  2.349941400e+01f,  2.358684000e+01f,  2.354526000e+01f,
                2.363276700e+01f,  2.359110600e+01f,  2.367869400e+01f,  2.363695200e+01f,  1.571740500e+01f,  1.568955000e+01f,
                7.824453000e+00f,  7.810512000e+00f,  5.197015000e+00f,  5.187802000e+00f,  1.034890400e+01f,  1.033046000e+01f,
                1.545555900e+01f,  1.542786600e+01f,  1.548569100e+01f,  1.545794400e+01f,  1.551582300e+01f,  1.548802200e+01f,
                1.554595500e+01f,  1.551810000e+01f,  1.557608700e+01f,  1.554817800e+01f,  1.560621900e+01f,  1.557825600e+01f,
                1.563635100e+01f,  1.560833400e+01f,  1.566648300e+01f,  1.563841200e+01f,  1.569661500e+01f,  1.566849000e+01f,
                1.041806000e+01f,  1.039929200e+01f,  5.185819000e+00f,  5.176426000e+00f,  2.584076000e+00f,  2.579420000e+00f,
                5.145211000e+00f,  5.135890000e+00f,  7.683351000e+00f,  7.669356000e+00f,  7.698174000e+00f,  7.684152000e+00f,
                7.712997000e+00f,  7.698948000e+00f,  7.727820000e+00f,  7.713744000e+00f,  7.742643000e+00f,  7.728540000e+00f,
                7.757466000e+00f,  7.743336000e+00f,  7.772289000e+00f,  7.758132000e+00f,  7.787112000e+00f,  7.772928000e+00f,
                7.801935000e+00f,  7.787724000e+00f,  5.177737000e+00f,  5.168254000e+00f,  2.577074000e+00f,  2.572328000e+00f,
                5.109508000e+00f,  5.100790000e+00f,  1.017718400e+01f,  1.015973000e+01f,  1.520292000e+01f,  1.517671200e+01f,
                1.523402400e+01f,  1.520776200e+01f,  1.526512800e+01f,  1.523881200e+01f,  1.529623200e+01f,  1.526986200e+01f,
                1.532733600e+01f,  1.530091200e+01f,  1.535844000e+01f,  1.533196200e+01f,  1.538954400e+01f,  1.536301200e+01f,
                1.542064800e+01f,  1.539406200e+01f,  1.545175200e+01f,  1.542511200e+01f,  1.025811200e+01f,  1.024033400e+01f,
                5.107492000e+00f,  5.098594000e+00f,  1.017732800e+01f,  1.015969400e+01f,  2.026948000e+01f,  2.023417600e+01f,
                3.027624000e+01f,  3.022323000e+01f,  3.033747600e+01f,  3.028435800e+01f,  3.039871200e+01f,  3.034548600e+01f,
                3.045994800e+01f,  3.040661400e+01f,  3.052118400e+01f,  3.046774200e+01f,  3.058242000e+01f,  3.052887000e+01f,
                3.064365600e+01f,  3.058999800e+01f,  3.070489200e+01f,  3.065112600e+01f,  3.076612800e+01f,  3.071225400e+01f,
                2.042312800e+01f,  2.038717600e+01f,  1.016768000e+01f,  1.014968600e+01f,  1.519989600e+01f,  1.517314800e+01f,
                3.026976000e+01f,  3.021621000e+01f,  4.520926800e+01f,  4.512886200e+01f,  4.529966400e+01f,  4.521909600e+01f,
                4.539006000e+01f,  4.530933000e+01f,  4.548045600e+01f,  4.539956400e+01f,  4.557085200e+01f,  4.548979800e+01f,
                4.566124800e+01f,  4.558003200e+01f,  4.575164400e+01f,  4.567026600e+01f,  4.584204000e+01f,  4.576050000e+01f,
                4.593243600e+01f,  4.585073400e+01f,  3.048792000e+01f,  3.043339800e+01f,  1.517700000e+01f,  1.514971200e+01f,
                2.017364800e+01f,  2.013758800e+01f,  4.017089600e+01f,  4.009870400e+01f,  5.999131200e+01f,  5.988291600e+01f,
                6.010989600e+01f,  6.000128400e+01f,  6.022848000e+01f,  6.011965200e+01f,  6.034706400e+01f,  6.023802000e+01f,
                6.046564800e+01f,  6.035638800e+01f,  6.058423200e+01f,  6.047475600e+01f,  6.070281600e+01f,  6.059312400e+01f,
                6.082140000e+01f,  6.071149200e+01f,  6.093998400e+01f,  6.082986000e+01f,  4.044536000e+01f,  4.037187200e+01f,
                2.013188800e+01f,  2.009510800e+01f,  2.509502000e+01f,  2.504945000e+01f,  4.996576000e+01f,  4.987453000e+01f,
                7.461168000e+01f,  7.447470000e+01f,  7.475748000e+01f,  7.462023000e+01f,  7.490328000e+01f,  7.476576000e+01f,
                7.504908000e+01f,  7.491129000e+01f,  7.519488000e+01f,  7.505682000e+01f,  7.534068000e+01f,  7.520235000e+01f,
                7.548648000e+01f,  7.534788000e+01f,  7.563228000e+01f,  7.549341000e+01f,  7.577808000e+01f,  7.563894000e+01f,
                5.028832000e+01f,  5.019547000e+01f,  2.502878000e+01f,  2.498231000e+01f,  2.563556000e+01f,  2.558900000e+01f,
                5.104090000e+01f,  5.094769000e+01f,  7.621548000e+01f,  7.607553000e+01f,  7.636128000e+01f,  7.622106000e+01f,
                7.650708000e+01f,  7.636659000e+01f,  7.665288000e+01f,  7.651212000e+01f,  7.679868000e+01f,  7.665765000e+01f,
                7.694448000e+01f,  7.680318000e+01f,  7.709028000e+01f,  7.694871000e+01f,  7.723608000e+01f,  7.709424000e+01f,
                7.738188000e+01f,  7.723977000e+01f,  5.135158000e+01f,  5.125675000e+01f,  2.555744000e+01f,  2.550998000e+01f,
                2.617610000e+01f,  2.612855000e+01f,  5.211604000e+01f,  5.202085000e+01f,  7.781928000e+01f,  7.767636000e+01f,
                7.796508000e+01f,  7.782189000e+01f,  7.811088000e+01f,  7.796742000e+01f,  7.825668000e+01f,  7.811295000e+01f,
                7.840248000e+01f,  7.825848000e+01f,  7.854828000e+01f,  7.840401000e+01f,  7.869408000e+01f,  7.854954000e+01f,
                7.883988000e+01f,  7.869507000e+01f,  7.898568000e+01f,  7.884060000e+01f,  5.241484000e+01f,  5.231803000e+01f,
                2.608610000e+01f,  2.603765000e+01f,  2.671664000e+01f,  2.666810000e+01f,  5.319118000e+01f,  5.309401000e+01f,
                7.942308000e+01f,  7.927719000e+01f,  7.956888000e+01f,  7.942272000e+01f,  7.971468000e+01f,  7.956825000e+01f,
                7.986048000e+01f,  7.971378000e+01f,  8.000628000e+01f,  7.985931000e+01f,  8.015208000e+01f,  8.000484000e+01f,
                8.029788000e+01f,  8.015037000e+01f,  8.044368000e+01f,  8.029590000e+01f,  8.058948000e+01f,  8.044143000e+01f,
                5.347810000e+01f,  5.337931000e+01f,  2.661476000e+01f,  2.656532000e+01f,  2.122578400e+01f,  2.118655600e+01f,
                4.225486400e+01f,  4.217633600e+01f,  6.308680800e+01f,  6.296890800e+01f,  6.320150400e+01f,  6.308338800e+01f,
                6.331620000e+01f,  6.319786800e+01f,  6.343089600e+01f,  6.331234800e+01f,  6.354559200e+01f,  6.342682800e+01f,
                6.366028800e+01f,  6.354130800e+01f,  6.377498400e+01f,  6.365578800e+01f,  6.388968000e+01f,  6.377026800e+01f,
                6.400437600e+01f,  6.388474800e+01f,  4.246798400e+01f,  4.238816000e+01f,  2.113304800e+01f,  2.109310000e+01f,
                1.580512800e+01f,  1.577541000e+01f,  3.146046000e+01f,  3.140097000e+01f,  4.696567200e+01f,  4.687635600e+01f,
                4.705023600e+01f,  4.696075800e+01f,  4.713480000e+01f,  4.704516000e+01f,  4.721936400e+01f,  4.712956200e+01f,
                4.730392800e+01f,  4.721396400e+01f,  4.738849200e+01f,  4.729836600e+01f,  4.747305600e+01f,  4.738276800e+01f,
                4.755762000e+01f,  4.746717000e+01f,  4.764218400e+01f,  4.755157200e+01f,  3.160798800e+01f,  3.154752600e+01f,
                1.572715200e+01f,  1.569689400e+01f,  1.045823600e+01f,  1.043822600e+01f,  2.081509600e+01f,  2.077504000e+01f,
                3.107036400e+01f,  3.101022600e+01f,  3.112576800e+01f,  3.106552200e+01f,  3.118117200e+01f,  3.112081800e+01f,
                3.123657600e+01f,  3.117611400e+01f,  3.129198000e+01f,  3.123141000e+01f,  3.134738400e+01f,  3.128670600e+01f,
                3.140278800e+01f,  3.134200200e+01f,  3.145819200e+01f,  3.139729800e+01f,  3.151359600e+01f,  3.145259400e+01f,
                2.090524000e+01f,  2.086453600e+01f,  1.040063600e+01f,  1.038026600e+01f,  5.188672000e+00f,  5.178568000e+00f,
                1.032590000e+01f,  1.030567400e+01f,  1.541157600e+01f,  1.538121000e+01f,  1.543879200e+01f,  1.540837200e+01f,
                1.546600800e+01f,  1.543553400e+01f,  1.549322400e+01f,  1.546269600e+01f,  1.552044000e+01f,  1.548985800e+01f,
                1.554765600e+01f,  1.551702000e+01f,  1.557487200e+01f,  1.554418200e+01f,  1.560208800e+01f,  1.557134400e+01f,
                1.562930400e+01f,  1.559850600e+01f,  1.036686800e+01f,  1.034631800e+01f,  5.157064000e+00f,  5.146780000e+00f,
                7.802853000e+00f,  7.788588000e+00f,  1.553461500e+01f,  1.550605800e+01f,  2.319512400e+01f,  2.315224800e+01f,
                2.323813500e+01f,  2.319517800e+01f,  2.328114600e+01f,  2.323810800e+01f,  2.332415700e+01f,  2.328103800e+01f,
                2.336716800e+01f,  2.332396800e+01f,  2.341017900e+01f,  2.336689800e+01f,  2.345319000e+01f,  2.340982800e+01f,
                2.349620100e+01f,  2.345275800e+01f,  2.353921200e+01f,  2.349568800e+01f,  1.561988100e+01f,  1.559083800e+01f,
                7.773423000e+00f,  7.758888000e+00f,  1.550842500e+01f,  1.547959800e+01f,  3.087240000e+01f,  3.081469200e+01f,
                4.609160100e+01f,  4.600495800e+01f,  4.617616500e+01f,  4.608936000e+01f,  4.626072900e+01f,  4.617376200e+01f,
                4.634529300e+01f,  4.625816400e+01f,  4.642985700e+01f,  4.634256600e+01f,  4.651442100e+01f,  4.642696800e+01f,
                4.659898500e+01f,  4.651137000e+01f,  4.668354900e+01f,  4.659577200e+01f,  4.676811300e+01f,  4.668017400e+01f,
                3.103062000e+01f,  3.097194000e+01f,  1.544114100e+01f,  1.541177400e+01f,  2.311137000e+01f,  2.306768400e+01f,
                4.600266300e+01f,  4.591521000e+01f,  6.867339300e+01f,  6.854209200e+01f,  6.879805200e+01f,  6.866650800e+01f,
                6.892271100e+01f,  6.879092400e+01f,  6.904737000e+01f,  6.891534000e+01f,  6.917202900e+01f,  6.903975600e+01f,
                6.929668800e+01f,  6.916417200e+01f,  6.942134700e+01f,  6.928858800e+01f,  6.954600600e+01f,  6.941300400e+01f,
                6.967066500e+01f,  6.953742000e+01f,  4.622152500e+01f,  4.613261400e+01f,  2.299780800e+01f,  2.295331200e+01f,
                3.060634200e+01f,  3.054750000e+01f,  6.091471200e+01f,  6.079692000e+01f,  9.092446200e+01f,  9.074761200e+01f,
                9.108775800e+01f,  9.091058400e+01f,  9.125105400e+01f,  9.107355600e+01f,  9.141435000e+01f,  9.123652800e+01f,
                9.157764600e+01f,  9.139950000e+01f,  9.174094200e+01f,  9.156247200e+01f,  9.190423800e+01f,  9.172544400e+01f,
                9.206753400e+01f,  9.188841600e+01f,  9.223083000e+01f,  9.205138800e+01f,  6.118190400e+01f,  6.106216800e+01f,
                3.043807800e+01f,  3.037815600e+01f,  3.798799500e+01f,  3.791370000e+01f,  7.559785500e+01f,  7.544913000e+01f,
                1.128287700e+02f,  1.126054800e+02f,  1.130292450e+02f,  1.128055500e+02f,  1.132297200e+02f,  1.130056200e+02f,
                1.134301950e+02f,  1.132056900e+02f,  1.136306700e+02f,  1.134057600e+02f,  1.138311450e+02f,  1.136058300e+02f,
                1.140316200e+02f,  1.138059000e+02f,  1.142320950e+02f,  1.140059700e+02f,  1.144325700e+02f,  1.142060400e+02f,
                7.590106500e+01f,  7.574991000e+01f,  3.775660500e+01f,  3.768096000e+01f,  3.873198000e+01f,  3.865620000e+01f,
                7.707691500e+01f,  7.692522000e+01f,  1.150339950e+02f,  1.148062500e+02f,  1.152344700e+02f,  1.150063200e+02f,
                1.154349450e+02f,  1.152063900e+02f,  1.156354200e+02f,  1.154064600e+02f,  1.158358950e+02f,  1.156065300e+02f,
                1.160363700e+02f,  1.158066000e+02f,  1.162368450e+02f,  1.160066700e+02f,  1.164373200e+02f,  1.162067400e+02f,
                1.166377950e+02f,  1.164068100e+02f,  7.736230500e+01f,  7.720818000e+01f,  3.848277000e+01f,  3.840564000e+01f,
                3.947596500e+01f,  3.939870000e+01f,  7.855597500e+01f,  7.840131000e+01f,  1.172392200e+02f,  1.170070200e+02f,
                1.174396950e+02f,  1.172070900e+02f,  1.176401700e+02f,  1.174071600e+02f,  1.178406450e+02f,  1.176072300e+02f,
                1.180411200e+02f,  1.178073000e+02f,  1.182415950e+02f,  1.180073700e+02f,  1.184420700e+02f,  1.182074400e+02f,
                1.186425450e+02f,  1.184075100e+02f,  1.188430200e+02f,  1.186075800e+02f,  7.882354500e+01f,  7.866645000e+01f,
                3.920893500e+01f,  3.913032000e+01f,  4.021995000e+01f,  4.014120000e+01f,  8.003503500e+01f,  7.987740000e+01f,
                1.194444450e+02f,  1.192077900e+02f,  1.196449200e+02f,  1.194078600e+02f,  1.198453950e+02f,  1.196079300e+02f,
                1.200458700e+02f,  1.198080000e+02f,  1.202463450e+02f,  1.200080700e+02f,  1.204468200e+02f,  1.202081400e+02f,
                1.206472950e+02f,  1.204082100e+02f,  1.208477700e+02f,  1.206082800e+02f,  1.210482450e+02f,  1.208083500e+02f,
                8.028478500e+01f,  8.012472000e+01f,  3.993510000e+01f,  3.985500000e+01f,  3.188517000e+01f,  3.182157600e+01f,
                6.344191200e+01f,  6.331461600e+01f,  9.466957800e+01f,  9.447847200e+01f,  9.482704200e+01f,  9.463561200e+01f,
                9.498450600e+01f,  9.479275200e+01f,  9.514197000e+01f,  9.494989200e+01f,  9.529943400e+01f,  9.510703200e+01f,
                9.545689800e+01f,  9.526417200e+01f,  9.561436200e+01f,  9.542131200e+01f,  9.577182600e+01f,  9.557845200e+01f,
                9.592929000e+01f,  9.573559200e+01f,  6.361708800e+01f,  6.348784800e+01f,  3.164044200e+01f,  3.157576800e+01f,
                2.369043900e+01f,  2.364229800e+01f,  4.713115500e+01f,  4.703479200e+01f,  7.032166200e+01f,  7.017699600e+01f,
                7.043757300e+01f,  7.029266400e+01f,  7.055348400e+01f,  7.040833200e+01f,  7.066939500e+01f,  7.052400000e+01f,
                7.078530600e+01f,  7.063966800e+01f,  7.090121700e+01f,  7.075533600e+01f,  7.101712800e+01f,  7.087100400e+01f,
                7.113303900e+01f,  7.098667200e+01f,  7.124895000e+01f,  7.110234000e+01f,  4.724406900e+01f,  4.714624800e+01f,
                2.349425700e+01f,  2.344530600e+01f,  1.564110300e+01f,  1.560871200e+01f,  3.111345600e+01f,  3.104862000e+01f,
                4.641673500e+01f,  4.631940000e+01f,  4.649255100e+01f,  4.639505400e+01f,  4.656836700e+01f,  4.647070800e+01f,
                4.664418300e+01f,  4.654636200e+01f,  4.671999900e+01f,  4.662201600e+01f,  4.679581500e+01f,  4.669767000e+01f,
                4.687163100e+01f,  4.677332400e+01f,  4.694744700e+01f,  4.684897800e+01f,  4.702326300e+01f,  4.692463200e+01f,
                3.117642000e+01f,  3.111061200e+01f,  1.550189100e+01f,  1.546896000e+01f,  7.742508000e+00f,  7.726164000e+00f,
                1.539950700e+01f,  1.536679200e+01f,  2.297083500e+01f,  2.292172200e+01f,  2.300801400e+01f,  2.295882000e+01f,
                2.304519300e+01f,  2.299591800e+01f,  2.308237200e+01f,  2.303301600e+01f,  2.311955100e+01f,  2.307011400e+01f,
                2.315673000e+01f,  2.310721200e+01f,  2.319390900e+01f,  2.314431000e+01f,  2.323108800e+01f,  2.318140800e+01f,
                2.326826700e+01f,  2.321850600e+01f,  1.542483300e+01f,  1.539163200e+01f,  7.668690000e+00f,  7.652076000e+00f,
                1.049355200e+01f,  1.047294800e+01f,  2.088119200e+01f,  2.083994800e+01f,  3.116270400e+01f,  3.110078400e+01f,
                3.121519200e+01f,  3.115316400e+01f,  3.126768000e+01f,  3.120554400e+01f,  3.132016800e+01f,  3.125792400e+01f,
                3.137265600e+01f,  3.131030400e+01f,  3.142514400e+01f,  3.136268400e+01f,  3.147763200e+01f,  3.141506400e+01f,
                3.153012000e+01f,  3.146744400e+01f,  3.158260800e+01f,  3.151982400e+01f,  2.094671200e+01f,  2.090482000e+01f,
                1.041910400e+01f,  1.039814000e+01f,  2.081106400e+01f,  2.076946000e+01f,  4.140728000e+01f,  4.132400000e+01f,
                6.178821600e+01f,  6.166318800e+01f,  6.189124800e+01f,  6.176600400e+01f,  6.199428000e+01f,  6.186882000e+01f,
                6.209731200e+01f,  6.197163600e+01f,  6.220034400e+01f,  6.207445200e+01f,  6.230337600e+01f,  6.217726800e+01f,
                6.240640800e+01f,  6.228008400e+01f,  6.250944000e+01f,  6.238290000e+01f,  6.261247200e+01f,  6.248571600e+01f,
                4.152190400e+01f,  4.143732800e+01f,  2.065093600e+01f,  2.060861200e+01f,  3.094540800e+01f,  3.088240800e+01f,
                6.156400800e+01f,  6.143790000e+01f,  9.185515200e+01f,  9.166582800e+01f,  9.200678400e+01f,  9.181713600e+01f,
                9.215841600e+01f,  9.196844400e+01f,  9.231004800e+01f,  9.211975200e+01f,  9.246168000e+01f,  9.227106000e+01f,
                9.261331200e+01f,  9.242236800e+01f,  9.276494400e+01f,  9.257367600e+01f,  9.291657600e+01f,  9.272498400e+01f,
                9.306820800e+01f,  9.287629200e+01f,  6.171132000e+01f,  6.158326800e+01f,  3.068836800e+01f,  3.062428800e+01f,
                4.088945600e+01f,  4.080466400e+01f,  8.133712000e+01f,  8.116739200e+01f,  1.213421280e+02f,  1.210873200e+02f,
                1.215404160e+02f,  1.212851760e+02f,  1.217387040e+02f,  1.214830320e+02f,  1.219369920e+02f,  1.216808880e+02f,
                1.221352800e+02f,  1.218787440e+02f,  1.223335680e+02f,  1.220766000e+02f,  1.225318560e+02f,  1.222744560e+02f,
                1.227301440e+02f,  1.224723120e+02f,  1.229284320e+02f,  1.226701680e+02f,  8.150070400e+01f,  8.132838400e+01f,
                4.052427200e+01f,  4.043804000e+01f,  5.063608000e+01f,  5.052910000e+01f,  1.007123600e+02f,  1.004982200e+02f,
                1.502277600e+02f,  1.499062800e+02f,  1.504707600e+02f,  1.501487400e+02f,  1.507137600e+02f,  1.503912000e+02f,
                1.509567600e+02f,  1.506336600e+02f,  1.511997600e+02f,  1.508761200e+02f,  1.514427600e+02f,  1.511185800e+02f,
                1.516857600e+02f,  1.513610400e+02f,  1.519287600e+02f,  1.516035000e+02f,  1.521717600e+02f,  1.518459600e+02f,
                1.008758000e+02f,  1.006584200e+02f,  5.015152000e+01f,  5.004274000e+01f,  5.153896000e+01f,  5.143000000e+01f,
                1.025062400e+02f,  1.022881400e+02f,  1.529007600e+02f,  1.525733400e+02f,  1.531437600e+02f,  1.528158000e+02f,
                1.533867600e+02f,  1.530582600e+02f,  1.536297600e+02f,  1.533007200e+02f,  1.538727600e+02f,  1.535431800e+02f,
                1.541157600e+02f,  1.537856400e+02f,  1.543587600e+02f,  1.540281000e+02f,  1.546017600e+02f,  1.542705600e+02f,
                1.548447600e+02f,  1.545130200e+02f,  1.026459200e+02f,  1.024245800e+02f,  5.103064000e+01f,  5.091988000e+01f,
                5.244184000e+01f,  5.233090000e+01f,  1.043001200e+02f,  1.040780600e+02f,  1.555737600e+02f,  1.552404000e+02f,
                1.558167600e+02f,  1.554828600e+02f,  1.560597600e+02f,  1.557253200e+02f,  1.563027600e+02f,  1.559677800e+02f,
                1.565457600e+02f,  1.562102400e+02f,  1.567887600e+02f,  1.564527000e+02f,  1.570317600e+02f,  1.566951600e+02f,
                1.572747600e+02f,  1.569376200e+02f,  1.575177600e+02f,  1.571800800e+02f,  1.044160400e+02f,  1.041907400e+02f,
                5.190976000e+01f,  5.179702000e+01f,  5.334472000e+01f,  5.323180000e+01f,  1.060940000e+02f,  1.058679800e+02f,
                1.582467600e+02f,  1.579074600e+02f,  1.584897600e+02f,  1.581499200e+02f,  1.587327600e+02f,  1.583923800e+02f,
                1.589757600e+02f,  1.586348400e+02f,  1.592187600e+02f,  1.588773000e+02f,  1.594617600e+02f,  1.591197600e+02f,
                1.597047600e+02f,  1.593622200e+02f,  1.599477600e+02f,  1.596046800e+02f,  1.601907600e+02f,  1.598471400e+02f,
                1.061861600e+02f,  1.059569000e+02f,  5.278888000e+01f,  5.267416000e+01f,  4.219539200e+01f,  4.210426400e+01f,
                8.390838400e+01f,  8.372598400e+01f,  1.251381120e+02f,  1.248642960e+02f,  1.253286240e+02f,  1.250543760e+02f,
                1.255191360e+02f,  1.252444560e+02f,  1.257096480e+02f,  1.254345360e+02f,  1.259001600e+02f,  1.256246160e+02f,
                1.260906720e+02f,  1.258146960e+02f,  1.262811840e+02f,  1.260047760e+02f,  1.264716960e+02f,  1.261948560e+02f,
                1.266622080e+02f,  1.263849360e+02f,  8.394928000e+01f,  8.376428800e+01f,  4.172825600e+01f,  4.163568800e+01f,
                3.127912800e+01f,  3.121018800e+01f,  6.219192000e+01f,  6.205393200e+01f,  9.273772800e+01f,  9.253058400e+01f,
                9.287769600e+01f,  9.267022800e+01f,  9.301766400e+01f,  9.280987200e+01f,  9.315763200e+01f,  9.294951600e+01f,
                9.329760000e+01f,  9.308916000e+01f,  9.343756800e+01f,  9.322880400e+01f,  9.357753600e+01f,  9.336844800e+01f,
                9.371750400e+01f,  9.350809200e+01f,  9.385747200e+01f,  9.364773600e+01f,  6.219796800e+01f,  6.205803600e+01f,
                3.091192800e+01f,  3.084190800e+01f,  2.060305600e+01f,  2.055670000e+01f,  4.095886400e+01f,  4.086608000e+01f,
                6.106699200e+01f,  6.092770800e+01f,  6.115836000e+01f,  6.101886000e+01f,  6.124972800e+01f,  6.111001200e+01f,
                6.134109600e+01f,  6.120116400e+01f,  6.143246400e+01f,  6.129231600e+01f,  6.152383200e+01f,  6.138346800e+01f,
                6.161520000e+01f,  6.147462000e+01f,  6.170656800e+01f,  6.156577200e+01f,  6.179793600e+01f,  6.165692400e+01f,
                4.094648000e+01f,  4.085240000e+01f,  2.034702400e+01f,  2.029994800e+01f,  1.017430400e+01f,  1.015092800e+01f,
                2.022347200e+01f,  2.017668400e+01f,  3.014728800e+01f,  3.007705200e+01f,  3.019200000e+01f,  3.012165600e+01f,
                3.023671200e+01f,  3.016626000e+01f,  3.028142400e+01f,  3.021086400e+01f,  3.032613600e+01f,  3.025546800e+01f,
                3.037084800e+01f,  3.030007200e+01f,  3.041556000e+01f,  3.034467600e+01f,  3.046027200e+01f,  3.038928000e+01f,
                3.050498400e+01f,  3.043388400e+01f,  2.020907200e+01f,  2.016163600e+01f,  1.004067200e+01f,  1.001693600e+01f,
                1.311032500e+01f,  1.308259000e+01f,  2.607435500e+01f,  2.601884000e+01f,  3.889182000e+01f,  3.880848000e+01f,
                3.895135500e+01f,  3.886788000e+01f,  3.901089000e+01f,  3.892728000e+01f,  3.907042500e+01f,  3.898668000e+01f,
                3.912996000e+01f,  3.904608000e+01f,  3.918949500e+01f,  3.910548000e+01f,  3.924903000e+01f,  3.916488000e+01f,
                3.930856500e+01f,  3.922428000e+01f,  3.936810000e+01f,  3.928368000e+01f,  2.609604500e+01f,  2.603972000e+01f,
                1.297325500e+01f,  1.294507000e+01f,  2.594268500e+01f,  2.588672000e+01f,  5.158900000e+01f,  5.147698000e+01f,
                7.693840500e+01f,  7.677024000e+01f,  7.705504500e+01f,  7.688661000e+01f,  7.717168500e+01f,  7.700298000e+01f,
                7.728832500e+01f,  7.711935000e+01f,  7.740496500e+01f,  7.723572000e+01f,  7.752160500e+01f,  7.735209000e+01f,
                7.763824500e+01f,  7.746846000e+01f,  7.775488500e+01f,  7.758483000e+01f,  7.787152500e+01f,  7.770120000e+01f,
                5.161186000e+01f,  5.149822000e+01f,  2.565450500e+01f,  2.559764000e+01f,  3.848817000e+01f,  3.840348000e+01f,
                7.652611500e+01f,  7.635660000e+01f,  1.141130250e+02f,  1.138585500e+02f,  1.142843400e+02f,  1.140294600e+02f,
                1.144556550e+02f,  1.142003700e+02f,  1.146269700e+02f,  1.143712800e+02f,  1.147982850e+02f,  1.145421900e+02f,
                1.149696000e+02f,  1.147131000e+02f,  1.151409150e+02f,  1.148840100e+02f,  1.153122300e+02f,  1.150549200e+02f,
                1.154835450e+02f,  1.152258300e+02f,  7.652962500e+01f,  7.635768000e+01f,  3.803484000e+01f,  3.794880000e+01f,
                5.073787000e+01f,  5.062396000e+01f,  1.008678800e+02f,  1.006398800e+02f,  1.503889500e+02f,  1.500466800e+02f,
                1.506125100e+02f,  1.502697000e+02f,  1.508360700e+02f,  1.504927200e+02f,  1.510596300e+02f,  1.507157400e+02f,
                1.512831900e+02f,  1.509387600e+02f,  1.515067500e+02f,  1.511617800e+02f,  1.517303100e+02f,  1.513848000e+02f,
                1.519538700e+02f,  1.516078200e+02f,  1.521774300e+02f,  1.518308400e+02f,  1.008315200e+02f,  1.006002800e+02f,
                5.010535000e+01f,  4.998964000e+01f,  6.268287500e+01f,  6.253925000e+01f,  1.245964750e+02f,  1.243090000e+02f,
                1.857394500e+02f,  1.853079000e+02f,  1.860128250e+02f,  1.855806000e+02f,  1.862862000e+02f,  1.858533000e+02f,
                1.865595750e+02f,  1.861260000e+02f,  1.868329500e+02f,  1.863987000e+02f,  1.871063250e+02f,  1.866714000e+02f,
                1.873797000e+02f,  1.869441000e+02f,  1.876530750e+02f,  1.872168000e+02f,  1.879264500e+02f,  1.874895000e+02f,
                1.244997250e+02f,  1.242082000e+02f,  6.185712500e+01f,  6.171125000e+01f,  6.370010000e+01f,  6.355400000e+01f,
                1.266160750e+02f,  1.263236500e+02f,  1.887465750e+02f,  1.883076000e+02f,  1.890199500e+02f,  1.885803000e+02f,
                1.892933250e+02f,  1.888530000e+02f,  1.895667000e+02f,  1.891257000e+02f,  1.898400750e+02f,  1.893984000e+02f,
                1.901134500e+02f,  1.896711000e+02f,  1.903868250e+02f,  1.899438000e+02f,  1.906602000e+02f,  1.902165000e+02f,
                1.909335750e+02f,  1.904892000e+02f,  1.264896250e+02f,  1.261931500e+02f,  6.284465000e+01f,  6.269630000e+01f,
                6.471732500e+01f,  6.456875000e+01f,  1.286356750e+02f,  1.283383000e+02f,  1.917537000e+02f,  1.913073000e+02f,
                1.920270750e+02f,  1.915800000e+02f,  1.923004500e+02f,  1.918527000e+02f,  1.925738250e+02f,  1.921254000e+02f,
                1.928472000e+02f,  1.923981000e+02f,  1.931205750e+02f,  1.926708000e+02f,  1.933939500e+02f,  1.929435000e+02f,
                1.936673250e+02f,  1.932162000e+02f,  1.939407000e+02f,  1.934889000e+02f,  1.284795250e+02f,  1.281781000e+02f,
                6.383217500e+01f,  6.368135000e+01f,  6.573455000e+01f,  6.558350000e+01f,  1.306552750e+02f,  1.303529500e+02f,
                1.947608250e+02f,  1.943070000e+02f,  1.950342000e+02f,  1.945797000e+02f,  1.953075750e+02f,  1.948524000e+02f,
                1.955809500e+02f,  1.951251000e+02f,  1.958543250e+02f,  1.953978000e+02f,  1.961277000e+02f,  1.956705000e+02f,
                1.964010750e+02f,  1.959432000e+02f,  1.966744500e+02f,  1.962159000e+02f,  1.969478250e+02f,  1.964886000e+02f,
                1.304694250e+02f,  1.301630500e+02f,  6.481970000e+01f,  6.466640000e+01f,  5.187133000e+01f,  5.174950000e+01f,
                1.030840400e+02f,  1.028402000e+02f,  1.536370500e+02f,  1.532710200e+02f,  1.538508900e+02f,  1.534843200e+02f,
                1.540647300e+02f,  1.536976200e+02f,  1.542785700e+02f,  1.539109200e+02f,  1.544924100e+02f,  1.541242200e+02f,
                1.547062500e+02f,  1.543375200e+02f,  1.549200900e+02f,  1.545508200e+02f,  1.551339300e+02f,  1.547641200e+02f,
                1.553477700e+02f,  1.549774200e+02f,  1.028943200e+02f,  1.026472400e+02f,  5.111137000e+01f,  5.098774000e+01f,
                3.835735500e+01f,  3.826524000e+01f,  7.621507500e+01f,  7.603071000e+01f,  1.135723500e+02f,  1.132956000e+02f,
                1.137290850e+02f,  1.134519300e+02f,  1.138858200e+02f,  1.136082600e+02f,  1.140425550e+02f,  1.137645900e+02f,
                1.141992900e+02f,  1.139209200e+02f,  1.143560250e+02f,  1.140772500e+02f,  1.145127600e+02f,  1.142335800e+02f,
                1.146694950e+02f,  1.143899100e+02f,  1.148262300e+02f,  1.145462400e+02f,  7.604200500e+01f,  7.585521000e+01f,
                3.776632500e+01f,  3.767286000e+01f,  2.520153500e+01f,  2.513963000e+01f,  5.006620000e+01f,  4.994230000e+01f,
                7.459345500e+01f,  7.440747000e+01f,  7.469551500e+01f,  7.450926000e+01f,  7.479757500e+01f,  7.461105000e+01f,
                7.489963500e+01f,  7.471284000e+01f,  7.500169500e+01f,  7.481463000e+01f,  7.510375500e+01f,  7.491642000e+01f,
                7.520581500e+01f,  7.501821000e+01f,  7.530787500e+01f,  7.512000000e+01f,  7.540993500e+01f,  7.522179000e+01f,
                4.993030000e+01f,  4.980478000e+01f,  2.479347500e+01f,  2.473067000e+01f,  1.241278000e+01f,  1.238158000e+01f,
                2.465523500e+01f,  2.459279000e+01f,  3.672709500e+01f,  3.663336000e+01f,  3.677691000e+01f,  3.668304000e+01f,
                3.682672500e+01f,  3.673272000e+01f,  3.687654000e+01f,  3.678240000e+01f,  3.692635500e+01f,  3.683208000e+01f,
                3.697617000e+01f,  3.688176000e+01f,  3.702598500e+01f,  3.693144000e+01f,  3.707580000e+01f,  3.698112000e+01f,
                3.712561500e+01f,  3.703080000e+01f,  2.457702500e+01f,  2.451377000e+01f,  1.220173000e+01f,  1.217008000e+01f,
                1.061417500e+01f,  1.058644000e+01f,  2.107800500e+01f,  2.102249000e+01f,  3.139122000e+01f,  3.130788000e+01f,
                3.143860500e+01f,  3.135513000e+01f,  3.148599000e+01f,  3.140238000e+01f,  3.153337500e+01f,  3.144963000e+01f,
                3.158076000e+01f,  3.149688000e+01f,  3.162814500e+01f,  3.154413000e+01f,  3.167553000e+01f,  3.159138000e+01f,
                3.172291500e+01f,  3.163863000e+01f,  3.177030000e+01f,  3.168588000e+01f,  2.102679500e+01f,  2.097047000e+01f,
                1.043660500e+01f,  1.040842000e+01f,  2.090583500e+01f,  2.084987000e+01f,  4.150720000e+01f,  4.139518000e+01f,
                6.180355500e+01f,  6.163539000e+01f,  6.189589500e+01f,  6.172746000e+01f,  6.198823500e+01f,  6.181953000e+01f,
                6.208057500e+01f,  6.191160000e+01f,  6.217291500e+01f,  6.200367000e+01f,  6.226525500e+01f,  6.209574000e+01f,
                6.235759500e+01f,  6.218781000e+01f,  6.244993500e+01f,  6.227988000e+01f,  6.254227500e+01f,  6.237195000e+01f,
                4.138426000e+01f,  4.127062000e+01f,  2.053665500e+01f,  2.047979000e+01f,  3.086607000e+01f,  3.078138000e+01f,
                6.126976500e+01f,  6.110025000e+01f,  9.121027500e+01f,  9.095580000e+01f,  9.134514000e+01f,  9.109026000e+01f,
                9.148000500e+01f,  9.122472000e+01f,  9.161487000e+01f,  9.135918000e+01f,  9.174973500e+01f,  9.149364000e+01f,
                9.188460000e+01f,  9.162810000e+01f,  9.201946500e+01f,  9.176256000e+01f,  9.215433000e+01f,  9.189702000e+01f,
                9.228919500e+01f,  9.203148000e+01f,  6.105457500e+01f,  6.088263000e+01f,  3.029124000e+01f,  3.020520000e+01f,
                4.048597000e+01f,  4.037206000e+01f,  8.034788000e+01f,  8.011988000e+01f,  1.195846500e+02f,  1.192423800e+02f,
                1.197596100e+02f,  1.194168000e+02f,  1.199345700e+02f,  1.195912200e+02f,  1.201095300e+02f,  1.197656400e+02f,
                1.202844900e+02f,  1.199400600e+02f,  1.204594500e+02f,  1.201144800e+02f,  1.206344100e+02f,  1.202889000e+02f,
                1.208093700e+02f,  1.204633200e+02f,  1.209843300e+02f,  1.206377400e+02f,  8.001992000e+01f,  7.978868000e+01f,
                3.969145000e+01f,  3.957574000e+01f,  4.975662500e+01f,  4.961300000e+01f,  9.872372500e+01f,  9.843625000e+01f,
                1.468999500e+02f,  1.464684000e+02f,  1.471125750e+02f,  1.466803500e+02f,  1.473252000e+02f,  1.468923000e+02f,
                1.475378250e+02f,  1.471042500e+02f,  1.477504500e+02f,  1.473162000e+02f,  1.479630750e+02f,  1.475281500e+02f,
                1.481757000e+02f,  1.477401000e+02f,  1.483883250e+02f,  1.479520500e+02f,  1.486009500e+02f,  1.481640000e+02f,
                9.826247500e+01f,  9.797095000e+01f,  4.872837500e+01f,  4.858250000e+01f,  5.055110000e+01f,  5.040500000e+01f,
                1.002978250e+02f,  1.000054000e+02f,  1.492388250e+02f,  1.487998500e+02f,  1.494514500e+02f,  1.490118000e+02f,
                1.496640750e+02f,  1.492237500e+02f,  1.498767000e+02f,  1.494357000e+02f,  1.500893250e+02f,  1.496476500e+02f,
                1.503019500e+02f,  1.498596000e+02f,  1.505145750e+02f,  1.500715500e+02f,  1.507272000e+02f,  1.502835000e+02f,
                1.509398250e+02f,  1.504954500e+02f,  9.980687500e+01f,  9.951040000e+01f,  4.949315000e+01f,  4.934480000e+01f,
                5.134557500e+01f,  5.119700000e+01f,  1.018719250e+02f,  1.015745500e+02f,  1.515777000e+02f,  1.511313000e+02f,
                1.517903250e+02f,  1.513432500e+02f,  1.520029500e+02f,  1.515552000e+02f,  1.522155750e+02f,  1.517671500e+02f,
                1.524282000e+02f,  1.519791000e+02f,  1.526408250e+02f,  1.521910500e+02f,  1.528534500e+02f,  1.524030000e+02f,
                1.530660750e+02f,  1.526149500e+02f,  1.532787000e+02f,  1.528269000e+02f,  1.013512750e+02f,  1.010498500e+02f,
                5.025792500e+01f,  5.010710000e+01f,  5.214005000e+01f,  5.198900000e+01f,  1.034460250e+02f,  1.031437000e+02f,
                1.539165750e+02f,  1.534627500e+02f,  1.541292000e+02f,  1.536747000e+02f,  1.543418250e+02f,  1.538866500e+02f,
                1.545544500e+02f,  1.540986000e+02f,  1.547670750e+02f,  1.543105500e+02f,  1.549797000e+02f,  1.545225000e+02f,
                1.551923250e+02f,  1.547344500e+02f,  1.554049500e+02f,  1.549464000e+02f,  1.556175750e+02f,  1.551583500e+02f,
                1.028956750e+02f,  1.025893000e+02f,  5.102270000e+01f,  5.086940000e+01f,  4.090663000e+01f,  4.078480000e+01f,
                8.113844000e+01f,  8.089460000e+01f,  1.206943500e+02f,  1.203283200e+02f,  1.208595900e+02f,  1.204930200e+02f,
                1.210248300e+02f,  1.206577200e+02f,  1.211900700e+02f,  1.208224200e+02f,  1.213553100e+02f,  1.209871200e+02f,
                1.215205500e+02f,  1.211518200e+02f,  1.216857900e+02f,  1.213165200e+02f,  1.218510300e+02f,  1.214812200e+02f,
                1.220162700e+02f,  1.216459200e+02f,  8.065712000e+01f,  8.041004000e+01f,  3.998467000e+01f,  3.986104000e+01f,
                3.006700500e+01f,  2.997489000e+01f,  5.962222500e+01f,  5.943786000e+01f,  8.866485000e+01f,  8.838810000e+01f,
                8.878513500e+01f,  8.850798000e+01f,  8.890542000e+01f,  8.862786000e+01f,  8.902570500e+01f,  8.874774000e+01f,
                8.914599000e+01f,  8.886762000e+01f,  8.926627500e+01f,  8.898750000e+01f,  8.938656000e+01f,  8.910738000e+01f,
                8.950684500e+01f,  8.922726000e+01f,  8.962713000e+01f,  8.934714000e+01f,  5.923045500e+01f,  5.904366000e+01f,
                2.935447500e+01f,  2.926101000e+01f,  1.963008500e+01f,  1.956818000e+01f,  3.891520000e+01f,  3.879130000e+01f,
                5.785480500e+01f,  5.766882000e+01f,  5.793256500e+01f,  5.774631000e+01f,  5.801032500e+01f,  5.782380000e+01f,
                5.808808500e+01f,  5.790129000e+01f,  5.816584500e+01f,  5.797878000e+01f,  5.824360500e+01f,  5.805627000e+01f,
                5.832136500e+01f,  5.813376000e+01f,  5.839912500e+01f,  5.821125000e+01f,  5.847688500e+01f,  5.828874000e+01f,
                3.863350000e+01f,  3.850798000e+01f,  1.914102500e+01f,  1.907822000e+01f,  9.604780000e+00f,  9.573580000e+00f,
                1.903518500e+01f,  1.897274000e+01f,  2.829094500e+01f,  2.819721000e+01f,  2.832861000e+01f,  2.823474000e+01f,
                2.836627500e+01f,  2.827227000e+01f,  2.840394000e+01f,  2.830980000e+01f,  2.844160500e+01f,  2.834733000e+01f,
                2.847927000e+01f,  2.838486000e+01f,  2.851693500e+01f,  2.842239000e+01f,  2.855460000e+01f,  2.845992000e+01f,
                2.859226500e+01f,  2.849745000e+01f,  1.888407500e+01f,  1.882082000e+01f,  9.353230000e+00f,  9.321580000e+00f,
                8.118025000e+00f,  8.090290000e+00f,  1.608165500e+01f,  1.602614000e+01f,  2.389062000e+01f,  2.380728000e+01f,
                2.392585500e+01f,  2.384238000e+01f,  2.396109000e+01f,  2.387748000e+01f,  2.399632500e+01f,  2.391258000e+01f,
                2.403156000e+01f,  2.394768000e+01f,  2.406679500e+01f,  2.398278000e+01f,  2.410203000e+01f,  2.401788000e+01f,
                2.413726500e+01f,  2.405298000e+01f,  2.417250000e+01f,  2.408808000e+01f,  1.595754500e+01f,  1.590122000e+01f,
                7.899955000e+00f,  7.871770000e+00f,  1.586898500e+01f,  1.581302000e+01f,  3.142540000e+01f,  3.131338000e+01f,
                4.666870500e+01f,  4.650054000e+01f,  4.673674500e+01f,  4.656831000e+01f,  4.680478500e+01f,  4.663608000e+01f,
                4.687282500e+01f,  4.670385000e+01f,  4.694086500e+01f,  4.677162000e+01f,  4.700890500e+01f,  4.683939000e+01f,
                4.707694500e+01f,  4.690716000e+01f,  4.714498500e+01f,  4.697493000e+01f,  4.721302500e+01f,  4.704270000e+01f,
                3.115666000e+01f,  3.104302000e+01f,  1.541880500e+01f,  1.536194000e+01f,  2.324397000e+01f,  2.315928000e+01f,
                4.601341500e+01f,  4.584390000e+01f,  6.830752500e+01f,  6.805305000e+01f,  6.840594000e+01f,  6.815106000e+01f,
                6.850435500e+01f,  6.824907000e+01f,  6.860277000e+01f,  6.834708000e+01f,  6.870118500e+01f,  6.844509000e+01f,
                6.879960000e+01f,  6.854310000e+01f,  6.889801500e+01f,  6.864111000e+01f,  6.899643000e+01f,  6.873912000e+01f,
                6.909484500e+01f,  6.883713000e+01f,  4.557952500e+01f,  4.540758000e+01f,  2.254764000e+01f,  2.246160000e+01f,
                3.023407000e+01f,  3.012016000e+01f,  5.982788000e+01f,  5.959988000e+01f,  8.878035000e+01f,  8.843808000e+01f,
                8.890671000e+01f,  8.856390000e+01f,  8.903307000e+01f,  8.868972000e+01f,  8.915943000e+01f,  8.881554000e+01f,
                8.928579000e+01f,  8.894136000e+01f,  8.941215000e+01f,  8.906718000e+01f,  8.953851000e+01f,  8.919300000e+01f,
                8.966487000e+01f,  8.931882000e+01f,  8.979123000e+01f,  8.944464000e+01f,  5.920832000e+01f,  5.897708000e+01f,
                2.927755000e+01f,  2.916184000e+01f,  3.683037500e+01f,  3.668675000e+01f,  7.285097500e+01f,  7.256350000e+01f,
                1.080604500e+02f,  1.076289000e+02f,  1.082123250e+02f,  1.077801000e+02f,  1.083642000e+02f,  1.079313000e+02f,
                1.085160750e+02f,  1.080825000e+02f,  1.086679500e+02f,  1.082337000e+02f,  1.088198250e+02f,  1.083849000e+02f,
                1.089717000e+02f,  1.085361000e+02f,  1.091235750e+02f,  1.086873000e+02f,  1.092754500e+02f,  1.088385000e+02f,
                7.202522500e+01f,  7.173370000e+01f,  3.559962500e+01f,  3.545375000e+01f,  3.740210000e+01f,  3.725600000e+01f,
                7.397957500e+01f,  7.368715000e+01f,  1.097310750e+02f,  1.092921000e+02f,  1.098829500e+02f,  1.094433000e+02f,
                1.100348250e+02f,  1.095945000e+02f,  1.101867000e+02f,  1.097457000e+02f,  1.103385750e+02f,  1.098969000e+02f,
                1.104904500e+02f,  1.100481000e+02f,  1.106423250e+02f,  1.101993000e+02f,  1.107942000e+02f,  1.103505000e+02f,
                1.109460750e+02f,  1.105017000e+02f,  7.312412500e+01f,  7.282765000e+01f,  3.614165000e+01f,  3.599330000e+01f,
                3.797382500e+01f,  3.782525000e+01f,  7.510817500e+01f,  7.481080000e+01f,  1.114017000e+02f,  1.109553000e+02f,
                1.115535750e+02f,  1.111065000e+02f,  1.117054500e+02f,  1.112577000e+02f,  1.118573250e+02f,  1.114089000e+02f,
                1.120092000e+02f,  1.115601000e+02f,  1.121610750e+02f,  1.117113000e+02f,  1.123129500e+02f,  1.118625000e+02f,
                1.124648250e+02f,  1.120137000e+02f,  1.126167000e+02f,  1.121649000e+02f,  7.422302500e+01f,  7.392160000e+01f,
                3.668367500e+01f,  3.653285000e+01f,  3.854555000e+01f,  3.839450000e+01f,  7.623677500e+01f,  7.593445000e+01f,
                1.130723250e+02f,  1.126185000e+02f,  1.132242000e+02f,  1.127697000e+02f,  1.133760750e+02f,  1.129209000e+02f,
                1.135279500e+02f,  1.130721000e+02f,  1.136798250e+02f,  1.132233000e+02f,  1.138317000e+02f,  1.133745000e+02f,
                1.139835750e+02f,  1.135257000e+02f,  1.141354500e+02f,  1.136769000e+02f,  1.142873250e+02f,  1.138281000e+02f,
                7.532192500e+01f,  7.501555000e+01f,  3.722570000e+01f,  3.707240000e+01f,  2.994193000e+01f,  2.982010000e+01f,
                5.919284000e+01f,  5.894900000e+01f,  8.775165000e+01f,  8.738562000e+01f,  8.786829000e+01f,  8.750172000e+01f,
                8.798493000e+01f,  8.761782000e+01f,  8.810157000e+01f,  8.773392000e+01f,  8.821821000e+01f,  8.785002000e+01f,
                8.833485000e+01f,  8.796612000e+01f,  8.845149000e+01f,  8.808222000e+01f,  8.856813000e+01f,  8.819832000e+01f,
                8.868477000e+01f,  8.831442000e+01f,  5.841992000e+01f,  5.817284000e+01f,  2.885797000e+01f,  2.873434000e+01f,
                2.177665500e+01f,  2.168454000e+01f,  4.302937500e+01f,  4.284501000e+01f,  6.375735000e+01f,  6.348060000e+01f,
                6.384118500e+01f,  6.356403000e+01f,  6.392502000e+01f,  6.364746000e+01f,  6.400885500e+01f,  6.373089000e+01f,
                6.409269000e+01f,  6.381432000e+01f,  6.417652500e+01f,  6.389775000e+01f,  6.426036000e+01f,  6.398118000e+01f,
                6.434419500e+01f,  6.406461000e+01f,  6.442803000e+01f,  6.414804000e+01f,  4.241890500e+01f,  4.223211000e+01f,
                2.094262500e+01f,  2.084916000e+01f,  1.405863500e+01f,  1.399673000e+01f,  2.776420000e+01f,  2.764030000e+01f,
                4.111615500e+01f,  4.093017000e+01f,  4.116961500e+01f,  4.098336000e+01f,  4.122307500e+01f,  4.103655000e+01f,
                4.127653500e+01f,  4.108974000e+01f,  4.132999500e+01f,  4.114293000e+01f,  4.138345500e+01f,  4.119612000e+01f,
                4.143691500e+01f,  4.124931000e+01f,  4.149037500e+01f,  4.130250000e+01f,  4.154383500e+01f,  4.135569000e+01f,
                2.733670000e+01f,  2.721118000e+01f,  1.348857500e+01f,  1.342577000e+01f,  6.796780000e+00f,  6.765580000e+00f,
                1.341513500e+01f,  1.335269000e+01f,  1.985479500e+01f,  1.976106000e+01f,  1.988031000e+01f,  1.978644000e+01f,
                1.990582500e+01f,  1.981182000e+01f,  1.993134000e+01f,  1.983720000e+01f,  1.995685500e+01f,  1.986258000e+01f,
                1.998237000e+01f,  1.988796000e+01f,  2.000788500e+01f,  1.991334000e+01f,  2.003340000e+01f,  1.993872000e+01f,
                2.005891500e+01f,  1.996410000e+01f,  1.319112500e+01f,  1.312787000e+01f,  6.504730000e+00f,  6.473080000e+00f,
                5.633768000e+00f,  5.609996000e+00f,  1.113289600e+01f,  1.108531600e+01f,  1.649716800e+01f,  1.642574400e+01f,
                1.652049600e+01f,  1.644896400e+01f,  1.654382400e+01f,  1.647218400e+01f,  1.656715200e+01f,  1.649540400e+01f,
                1.659048000e+01f,  1.651862400e+01f,  1.661380800e+01f,  1.654184400e+01f,  1.663713600e+01f,  1.656506400e+01f,
                1.666046400e+01f,  1.658828400e+01f,  1.668379200e+01f,  1.661150400e+01f,  1.098544000e+01f,  1.093721200e+01f,
                5.424104000e+00f,  5.399972000e+00f,  1.092755200e+01f,  1.087961200e+01f,  2.158280000e+01f,  2.148684800e+01f,
                3.196531200e+01f,  3.182127600e+01f,  3.201002400e+01f,  3.186577200e+01f,  3.205473600e+01f,  3.191026800e+01f,
                3.209944800e+01f,  3.195476400e+01f,  3.214416000e+01f,  3.199926000e+01f,  3.218887200e+01f,  3.204375600e+01f,
                3.223358400e+01f,  3.208825200e+01f,  3.227829600e+01f,  3.213274800e+01f,  3.232300800e+01f,  3.217724400e+01f,
                2.127147200e+01f,  2.117422400e+01f,  1.049699200e+01f,  1.044833200e+01f,  1.587422400e+01f,  1.580172000e+01f,
                3.133545600e+01f,  3.119034000e+01f,  4.638304800e+01f,  4.616521200e+01f,  4.644720000e+01f,  4.622904000e+01f,
                4.651135200e+01f,  4.629286800e+01f,  4.657550400e+01f,  4.635669600e+01f,  4.663965600e+01f,  4.642052400e+01f,
                4.670380800e+01f,  4.648435200e+01f,  4.676796000e+01f,  4.654818000e+01f,  4.683211200e+01f,  4.661200800e+01f,
                4.689626400e+01f,  4.667583600e+01f,  3.084384000e+01f,  3.069678000e+01f,  1.521153600e+01f,  1.513795200e+01f,
                2.046665600e+01f,  2.036919200e+01f,  4.037660800e+01f,  4.018153600e+01f,  5.972899200e+01f,  5.943616800e+01f,
                5.981064000e+01f,  5.951738400e+01f,  5.989228800e+01f,  5.959860000e+01f,  5.997393600e+01f,  5.967981600e+01f,
                6.005558400e+01f,  5.976103200e+01f,  6.013723200e+01f,  5.984224800e+01f,  6.021888000e+01f,  5.992346400e+01f,
                6.030052800e+01f,  6.000468000e+01f,  6.038217600e+01f,  6.008589600e+01f,  3.968828800e+01f,  3.949062400e+01f,
                1.956060800e+01f,  1.946170400e+01f,  2.469772000e+01f,  2.457490000e+01f,  4.869200000e+01f,  4.844618000e+01f,
                7.198176000e+01f,  7.161276000e+01f,  7.207896000e+01f,  7.170942000e+01f,  7.217616000e+01f,  7.180608000e+01f,
                7.227336000e+01f,  7.190274000e+01f,  7.237056000e+01f,  7.199940000e+01f,  7.246776000e+01f,  7.209606000e+01f,
                7.256496000e+01f,  7.219272000e+01f,  7.266216000e+01f,  7.228938000e+01f,  7.275936000e+01f,  7.238604000e+01f,
                4.779056000e+01f,  4.754150000e+01f,  2.353708000e+01f,  2.341246000e+01f,  2.506600000e+01f,  2.494120000e+01f,
                4.941668000e+01f,  4.916690000e+01f,  7.305096000e+01f,  7.267602000e+01f,  7.314816000e+01f,  7.277268000e+01f,
                7.324536000e+01f,  7.286934000e+01f,  7.334256000e+01f,  7.296600000e+01f,  7.343976000e+01f,  7.306266000e+01f,
                7.353696000e+01f,  7.315932000e+01f,  7.363416000e+01f,  7.325598000e+01f,  7.373136000e+01f,  7.335264000e+01f,
                7.382856000e+01f,  7.344930000e+01f,  4.849148000e+01f,  4.823846000e+01f,  2.388160000e+01f,  2.375500000e+01f,
                2.543428000e+01f,  2.530750000e+01f,  5.014136000e+01f,  4.988762000e+01f,  7.412016000e+01f,  7.373928000e+01f,
                7.421736000e+01f,  7.383594000e+01f,  7.431456000e+01f,  7.393260000e+01f,  7.441176000e+01f,  7.402926000e+01f,
                7.450896000e+01f,  7.412592000e+01f,  7.460616000e+01f,  7.422258000e+01f,  7.470336000e+01f,  7.431924000e+01f,
                7.480056000e+01f,  7.441590000e+01f,  7.489776000e+01f,  7.451256000e+01f,  4.919240000e+01f,  4.893542000e+01f,
                2.422612000e+01f,  2.409754000e+01f,  2.580256000e+01f,  2.567380000e+01f,  5.086604000e+01f,  5.060834000e+01f,
                7.518936000e+01f,  7.480254000e+01f,  7.528656000e+01f,  7.489920000e+01f,  7.538376000e+01f,  7.499586000e+01f,
                7.548096000e+01f,  7.509252000e+01f,  7.557816000e+01f,  7.518918000e+01f,  7.567536000e+01f,  7.528584000e+01f,
                7.577256000e+01f,  7.538250000e+01f,  7.586976000e+01f,  7.547916000e+01f,  7.596696000e+01f,  7.557582000e+01f,
                4.989332000e+01f,  4.963238000e+01f,  2.457064000e+01f,  2.444008000e+01f,  1.983377600e+01f,  1.972997600e+01f,
                3.907024000e+01f,  3.886249600e+01f,  5.770852800e+01f,  5.739669600e+01f,  5.778240000e+01f,  5.747013600e+01f,
                5.785627200e+01f,  5.754357600e+01f,  5.793014400e+01f,  5.761701600e+01f,  5.800401600e+01f,  5.769045600e+01f,
                5.807788800e+01f,  5.776389600e+01f,  5.815176000e+01f,  5.783733600e+01f,  5.822563200e+01f,  5.791077600e+01f,
                5.829950400e+01f,  5.798421600e+01f,  3.825923200e+01f,  3.804889600e+01f,  1.882577600e+01f,  1.872053600e+01f,
                1.426200000e+01f,  1.418355600e+01f,  2.807148000e+01f,  2.791448400e+01f,  4.142779200e+01f,  4.119213600e+01f,
                4.148028000e+01f,  4.124430000e+01f,  4.153276800e+01f,  4.129646400e+01f,  4.158525600e+01f,  4.134862800e+01f,
                4.163774400e+01f,  4.140079200e+01f,  4.169023200e+01f,  4.145295600e+01f,  4.174272000e+01f,  4.150512000e+01f,
                4.179520800e+01f,  4.155728400e+01f,  4.184769600e+01f,  4.160944800e+01f,  2.743860000e+01f,  2.727966000e+01f,
                1.348915200e+01f,  1.340962800e+01f,  9.094360000e+00f,  9.041668000e+00f,  1.788401600e+01f,  1.777856000e+01f,
                2.636853600e+01f,  2.621024400e+01f,  2.640158400e+01f,  2.624307600e+01f,  2.643463200e+01f,  2.627590800e+01f,
                2.646768000e+01f,  2.630874000e+01f,  2.650072800e+01f,  2.634157200e+01f,  2.653377600e+01f,  2.637440400e+01f,
                2.656682400e+01f,  2.640723600e+01f,  2.659987200e+01f,  2.644006800e+01f,  2.663292000e+01f,  2.647290000e+01f,
                1.744568000e+01f,  1.733892800e+01f,  8.567896000e+00f,  8.514484000e+00f,  4.337984000e+00f,  4.311440000e+00f,
                8.522104000e+00f,  8.468980000e+00f,  1.255214400e+01f,  1.247240400e+01f,  1.256769600e+01f,  1.248784800e+01f,
                1.258324800e+01f,  1.250329200e+01f,  1.259880000e+01f,  1.251873600e+01f,  1.261435200e+01f,  1.253418000e+01f,
                1.262990400e+01f,  1.254962400e+01f,  1.264545600e+01f,  1.256506800e+01f,  1.266100800e+01f,  1.258051200e+01f,
                1.267656000e+01f,  1.259595600e+01f,  8.294728000e+00f,  8.240956000e+00f,  4.069136000e+00f,  4.042232000e+00f,
                3.508557000e+00f,  3.489540000e+00f,  6.907791000e+00f,  6.869730000e+00f,  1.019754000e+01f,  1.014040800e+01f,
                1.021139100e+01f,  1.015417800e+01f,  1.022524200e+01f,  1.016794800e+01f,  1.023909300e+01f,  1.018171800e+01f,
                1.025294400e+01f,  1.019548800e+01f,  1.026679500e+01f,  1.020925800e+01f,  1.028064600e+01f,  1.022302800e+01f,
                1.029449700e+01f,  1.023679800e+01f,  1.030834800e+01f,  1.025056800e+01f,  6.761073000e+00f,  6.722526000e+00f,
                3.324903000e+00f,  3.305616000e+00f,  6.727377000e+00f,  6.689046000e+00f,  1.323384000e+01f,  1.315712400e+01f,
                1.951906500e+01f,  1.940391000e+01f,  1.954530900e+01f,  1.942999200e+01f,  1.957155300e+01f,  1.945607400e+01f,
                1.959779700e+01f,  1.948215600e+01f,  1.962404100e+01f,  1.950823800e+01f,  1.965028500e+01f,  1.953432000e+01f,
                1.967652900e+01f,  1.956040200e+01f,  1.970277300e+01f,  1.958648400e+01f,  1.972901700e+01f,  1.961256600e+01f,
                1.292809200e+01f,  1.285040400e+01f,  6.351645000e+00f,  6.312774000e+00f,  9.651114000e+00f,  9.593172000e+00f,
                1.896745500e+01f,  1.885149000e+01f,  2.794853700e+01f,  2.777446800e+01f,  2.798571600e+01f,  2.781140400e+01f,
                2.802289500e+01f,  2.784834000e+01f,  2.806007400e+01f,  2.788527600e+01f,  2.809725300e+01f,  2.792221200e+01f,
                2.813443200e+01f,  2.795914800e+01f,  2.817161100e+01f,  2.799608400e+01f,  2.820879000e+01f,  2.803302000e+01f,
                2.824596900e+01f,  2.806995600e+01f,  1.849036500e+01f,  1.837294200e+01f,  9.074880000e+00f,  9.016128000e+00f,
                1.227442200e+01f,  1.219657200e+01f,  2.409794400e+01f,  2.394213600e+01f,  3.546991800e+01f,  3.523604400e+01f,
                3.551657400e+01f,  3.528237600e+01f,  3.556323000e+01f,  3.532870800e+01f,  3.560988600e+01f,  3.537504000e+01f,
                3.565654200e+01f,  3.542137200e+01f,  3.570319800e+01f,  3.546770400e+01f,  3.574985400e+01f,  3.551403600e+01f,
                3.579651000e+01f,  3.556036800e+01f,  3.584316600e+01f,  3.560670000e+01f,  2.343720000e+01f,  2.327944800e+01f,
                1.148926200e+01f,  1.141033200e+01f,  1.459195500e+01f,  1.449390000e+01f,  2.861461500e+01f,  2.841837000e+01f,
                4.206717000e+01f,  4.177260000e+01f,  4.212184500e+01f,  4.182687000e+01f,  4.217652000e+01f,  4.188114000e+01f,
                4.223119500e+01f,  4.193541000e+01f,  4.228587000e+01f,  4.198968000e+01f,  4.234054500e+01f,  4.204395000e+01f,
                4.239522000e+01f,  4.209822000e+01f,  4.244989500e+01f,  4.215249000e+01f,  4.250457000e+01f,  4.220676000e+01f,
                2.775790500e+01f,  2.755923000e+01f,  1.358944500e+01f,  1.349004000e+01f,  1.480134000e+01f,  1.470180000e+01f,
                2.902447500e+01f,  2.882526000e+01f,  4.266859500e+01f,  4.236957000e+01f,  4.272327000e+01f,  4.242384000e+01f,
                4.277794500e+01f,  4.247811000e+01f,  4.283262000e+01f,  4.253238000e+01f,  4.288729500e+01f,  4.258665000e+01f,
                4.294197000e+01f,  4.264092000e+01f,  4.299664500e+01f,  4.269519000e+01f,  4.305132000e+01f,  4.274946000e+01f,
                4.310599500e+01f,  4.280373000e+01f,  2.814994500e+01f,  2.794830000e+01f,  1.378101000e+01f,  1.368012000e+01f,
                1.501072500e+01f,  1.490970000e+01f,  2.943433500e+01f,  2.923215000e+01f,  4.327002000e+01f,  4.296654000e+01f,
                4.332469500e+01f,  4.302081000e+01f,  4.337937000e+01f,  4.307508000e+01f,  4.343404500e+01f,  4.312935000e+01f,
                4.348872000e+01f,  4.318362000e+01f,  4.354339500e+01f,  4.323789000e+01f,  4.359807000e+01f,  4.329216000e+01f,
                4.365274500e+01f,  4.334643000e+01f,  4.370742000e+01f,  4.340070000e+01f,  2.854198500e+01f,  2.833737000e+01f,
                1.397257500e+01f,  1.387020000e+01f,  1.522011000e+01f,  1.511760000e+01f,  2.984419500e+01f,  2.963904000e+01f,
                4.387144500e+01f,  4.356351000e+01f,  4.392612000e+01f,  4.361778000e+01f,  4.398079500e+01f,  4.367205000e+01f,
                4.403547000e+01f,  4.372632000e+01f,  4.409014500e+01f,  4.378059000e+01f,  4.414482000e+01f,  4.383486000e+01f,
                4.419949500e+01f,  4.388913000e+01f,  4.425417000e+01f,  4.394340000e+01f,  4.430884500e+01f,  4.399767000e+01f,
                2.893402500e+01f,  2.872644000e+01f,  1.416414000e+01f,  1.406028000e+01f,  1.150038600e+01f,  1.141778400e+01f,
                2.251941600e+01f,  2.235410400e+01f,  3.305644200e+01f,  3.280831200e+01f,  3.309726600e+01f,  3.284881200e+01f,
                3.313809000e+01f,  3.288931200e+01f,  3.317891400e+01f,  3.292981200e+01f,  3.321973800e+01f,  3.297031200e+01f,
                3.326056200e+01f,  3.301081200e+01f,  3.330138600e+01f,  3.305131200e+01f,  3.334221000e+01f,  3.309181200e+01f,
                3.338303400e+01f,  3.313231200e+01f,  2.176665600e+01f,  2.159940000e+01f,  1.063876200e+01f,  1.055508000e+01f,
                8.113167000e+00f,  8.050770000e+00f,  1.586191500e+01f,  1.573704000e+01f,  2.324575800e+01f,  2.305832400e+01f,
                2.327418900e+01f,  2.308651200e+01f,  2.330262000e+01f,  2.311470000e+01f,  2.333105100e+01f,  2.314288800e+01f,
                2.335948200e+01f,  2.317107600e+01f,  2.338791300e+01f,  2.319926400e+01f,  2.341634400e+01f,  2.322745200e+01f,
                2.344477500e+01f,  2.325564000e+01f,  2.347320600e+01f,  2.328382800e+01f,  1.527887700e+01f,  1.515254400e+01f,
                7.454313000e+00f,  7.391106000e+00f,  5.063799000e+00f,  5.021904000e+00f,  9.882384000e+00f,  9.798540000e+00f,
                1.445543100e+01f,  1.432958400e+01f,  1.447292700e+01f,  1.434691800e+01f,  1.449042300e+01f,  1.436425200e+01f,
                1.450791900e+01f,  1.438158600e+01f,  1.452541500e+01f,  1.439892000e+01f,  1.454291100e+01f,  1.441625400e+01f,
                1.456040700e+01f,  1.443358800e+01f,  1.457790300e+01f,  1.445092200e+01f,  1.459539900e+01f,  1.446825600e+01f,
                9.481380000e+00f,  9.396564000e+00f,  4.616139000e+00f,  4.573704000e+00f,  2.357628000e+00f,  2.336532000e+00f,
                4.591515000e+00f,  4.549296000e+00f,  6.701499000e+00f,  6.638130000e+00f,  6.709518000e+00f,  6.646068000e+00f,
                6.717537000e+00f,  6.654006000e+00f,  6.725556000e+00f,  6.661944000e+00f,  6.733575000e+00f,  6.669882000e+00f,
                6.741594000e+00f,  6.677820000e+00f,  6.749613000e+00f,  6.685758000e+00f,  6.757632000e+00f,  6.693696000e+00f,
                6.765651000e+00f,  6.701634000e+00f,  4.384857000e+00f,  4.342152000e+00f,  2.129586000e+00f,  2.108220000e+00f,
                1.813672000e+00f,  1.800202000e+00f,  3.548900000e+00f,  3.521942000e+00f,  5.205576000e+00f,  5.165112000e+00f,
                5.212380000e+00f,  5.171862000e+00f,  5.219184000e+00f,  5.178612000e+00f,  5.225988000e+00f,  5.185362000e+00f,
                5.232792000e+00f,  5.192112000e+00f,  5.239596000e+00f,  5.198862000e+00f,  5.246400000e+00f,  5.205612000e+00f,
                5.253204000e+00f,  5.212362000e+00f,  5.260008000e+00f,  5.219112000e+00f,  3.427004000e+00f,  3.399722000e+00f,
                1.673632000e+00f,  1.659982000e+00f,  3.411020000e+00f,  3.383882000e+00f,  6.663640000e+00f,  6.609328000e+00f,
                9.757644000e+00f,  9.676122000e+00f,  9.770280000e+00f,  9.688650000e+00f,  9.782916000e+00f,  9.701178000e+00f,
                9.795552000e+00f,  9.713706000e+00f,  9.808188000e+00f,  9.726234000e+00f,  9.820824000e+00f,  9.738762000e+00f,
                9.833460000e+00f,  9.751290000e+00f,  9.846096000e+00f,  9.763818000e+00f,  9.858732000e+00f,  9.776346000e+00f,
                6.411640000e+00f,  6.356680000e+00f,  3.125324000e+00f,  3.097826000e+00f,  4.788480000e+00f,  4.747476000e+00f,
                9.337092000e+00f,  9.255030000e+00f,  1.364551200e+01f,  1.352233800e+01f,  1.366300800e+01f,  1.353967200e+01f,
                1.368050400e+01f,  1.355700600e+01f,  1.369800000e+01f,  1.357434000e+01f,  1.371549600e+01f,  1.359167400e+01f,
                1.373299200e+01f,  1.360900800e+01f,  1.375048800e+01f,  1.362634200e+01f,  1.376798400e+01f,  1.364367600e+01f,
                1.378548000e+01f,  1.366101000e+01f,  8.946780000e+00f,  8.863746000e+00f,  4.351512000e+00f,  4.309968000e+00f,
                5.942488000e+00f,  5.887420000e+00f,  1.156212800e+01f,  1.145192000e+01f,  1.685848800e+01f,  1.669306800e+01f,
                1.687987200e+01f,  1.671423600e+01f,  1.690125600e+01f,  1.673540400e+01f,  1.692264000e+01f,  1.675657200e+01f,
                1.694402400e+01f,  1.677774000e+01f,  1.696540800e+01f,  1.679890800e+01f,  1.698679200e+01f,  1.682007600e+01f,
                1.700817600e+01f,  1.684124400e+01f,  1.702956000e+01f,  1.686241200e+01f,  1.102529600e+01f,  1.091379200e+01f,
                5.348632000e+00f,  5.292844000e+00f,  6.869480000e+00f,  6.800150000e+00f,  1.333162000e+01f,  1.319287000e+01f,
                1.938588000e+01f,  1.917762000e+01f,  1.941018000e+01f,  1.920165000e+01f,  1.943448000e+01f,  1.922568000e+01f,
                1.945878000e+01f,  1.924971000e+01f,  1.948308000e+01f,  1.927374000e+01f,  1.950738000e+01f,  1.929777000e+01f,
                1.953168000e+01f,  1.932180000e+01f,  1.955598000e+01f,  1.934583000e+01f,  1.958028000e+01f,  1.936986000e+01f,
                1.264006000e+01f,  1.249969000e+01f,  6.113120000e+00f,  6.042890000e+00f,  6.964520000e+00f,  6.894200000e+00f,
                1.351576000e+01f,  1.337503000e+01f,  1.965318000e+01f,  1.944195000e+01f,  1.967748000e+01f,  1.946598000e+01f,
                1.970178000e+01f,  1.949001000e+01f,  1.972608000e+01f,  1.951404000e+01f,  1.975038000e+01f,  1.953807000e+01f,
                1.977468000e+01f,  1.956210000e+01f,  1.979898000e+01f,  1.958613000e+01f,  1.982328000e+01f,  1.961016000e+01f,
                1.984758000e+01f,  1.963419000e+01f,  1.281232000e+01f,  1.266997000e+01f,  6.196280000e+00f,  6.125060000e+00f,
                7.059560000e+00f,  6.988250000e+00f,  1.369990000e+01f,  1.355719000e+01f,  1.992048000e+01f,  1.970628000e+01f,
                1.994478000e+01f,  1.973031000e+01f,  1.996908000e+01f,  1.975434000e+01f,  1.999338000e+01f,  1.977837000e+01f,
                2.001768000e+01f,  1.980240000e+01f,  2.004198000e+01f,  1.982643000e+01f,  2.006628000e+01f,  1.985046000e+01f,
                2.009058000e+01f,  1.987449000e+01f,  2.011488000e+01f,  1.989852000e+01f,  1.298458000e+01f,  1.284025000e+01f,
                6.279440000e+00f,  6.207230000e+00f,  7.154600000e+00f,  7.082300000e+00f,  1.388404000e+01f,  1.373935000e+01f,
                2.018778000e+01f,  1.997061000e+01f,  2.021208000e+01f,  1.999464000e+01f,  2.023638000e+01f,  2.001867000e+01f,
                2.026068000e+01f,  2.004270000e+01f,  2.028498000e+01f,  2.006673000e+01f,  2.030928000e+01f,  2.009076000e+01f,
                2.033358000e+01f,  2.011479000e+01f,  2.035788000e+01f,  2.013882000e+01f,  2.038218000e+01f,  2.016285000e+01f,
                1.315684000e+01f,  1.301053000e+01f,  6.362600000e+00f,  6.289400000e+00f,  5.226880000e+00f,  5.168644000e+00f,
                1.011060800e+01f,  9.994064000e+00f,  1.465075200e+01f,  1.447582800e+01f,  1.466824800e+01f,  1.449310800e+01f,
                1.468574400e+01f,  1.451038800e+01f,  1.470324000e+01f,  1.452766800e+01f,  1.472073600e+01f,  1.454494800e+01f,
                1.473823200e+01f,  1.456222800e+01f,  1.475572800e+01f,  1.457950800e+01f,  1.477322400e+01f,  1.459678800e+01f,
                1.479072000e+01f,  1.461406800e+01f,  9.512432000e+00f,  9.394592000e+00f,  4.582048000e+00f,  4.523092000e+00f,
                3.543996000e+00f,  3.500022000e+00f,  6.828360000e+00f,  6.740358000e+00f,  9.852768000e+00f,  9.720684000e+00f,
                9.864432000e+00f,  9.732186000e+00f,  9.876096000e+00f,  9.743688000e+00f,  9.887760000e+00f,  9.755190000e+00f,
                9.899424000e+00f,  9.766692000e+00f,  9.911088000e+00f,  9.778194000e+00f,  9.922752000e+00f,  9.789696000e+00f,
                9.934416000e+00f,  9.801198000e+00f,  9.946080000e+00f,  9.812700000e+00f,  6.367416000e+00f,  6.278442000e+00f,
                3.051948000e+00f,  3.007434000e+00f,  2.109512000e+00f,  2.079998000e+00f,  4.044424000e+00f,  3.985360000e+00f,
                5.804520000e+00f,  5.715870000e+00f,  5.811324000e+00f,  5.722566000e+00f,  5.818128000e+00f,  5.729262000e+00f,
                5.824932000e+00f,  5.735958000e+00f,  5.831736000e+00f,  5.742654000e+00f,  5.838540000e+00f,  5.749350000e+00f,
                5.845344000e+00f,  5.756046000e+00f,  5.852148000e+00f,  5.762742000e+00f,  5.858952000e+00f,  5.769438000e+00f,
                3.728920000e+00f,  3.669208000e+00f,  1.775864000e+00f,  1.745990000e+00f,  9.269920000e-01f,  9.121360000e-01f,
                1.765928000e+00f,  1.736198000e+00f,  2.516700000e+00f,  2.472078000e+00f,  2.519616000e+00f,  2.474940000e+00f,
                2.522532000e+00f,  2.477802000e+00f,  2.525448000e+00f,  2.480664000e+00f,  2.528364000e+00f,  2.483526000e+00f,
                2.531280000e+00f,  2.486388000e+00f,  2.534196000e+00f,  2.489250000e+00f,  2.537112000e+00f,  2.492112000e+00f,
                2.540028000e+00f,  2.494974000e+00f,  1.604072000e+00f,  1.574018000e+00f,  7.573600000e-01f,  7.423240000e-01f,
                6.203930000e-01f,  6.132620000e-01f,  1.198783000e+00f,  1.184512000e+00f,  1.735116000e+00f,  1.713696000e+00f,
                1.737303000e+00f,  1.715856000e+00f,  1.739490000e+00f,  1.718016000e+00f,  1.741677000e+00f,  1.720176000e+00f,
                1.743864000e+00f,  1.722336000e+00f,  1.746051000e+00f,  1.724496000e+00f,  1.748238000e+00f,  1.726656000e+00f,
                1.750425000e+00f,  1.728816000e+00f,  1.752612000e+00f,  1.730976000e+00f,  1.125793000e+00f,  1.111360000e+00f,
                5.415710000e-01f,  5.343500000e-01f,  1.121041000e+00f,  1.106680000e+00f,  2.157320000e+00f,  2.128580000e+00f,
                3.108729000e+00f,  3.065592000e+00f,  3.112617000e+00f,  3.069426000e+00f,  3.116505000e+00f,  3.073260000e+00f,
                3.120393000e+00f,  3.077094000e+00f,  3.124281000e+00f,  3.080928000e+00f,  3.128169000e+00f,  3.084762000e+00f,
                3.132057000e+00f,  3.088596000e+00f,  3.135945000e+00f,  3.092430000e+00f,  3.139833000e+00f,  3.096264000e+00f,
                2.007236000e+00f,  1.978172000e+00f,  9.605890000e-01f,  9.460480000e-01f,  1.500162000e+00f,  1.478472000e+00f,
                2.872047000e+00f,  2.828640000e+00f,  4.115493000e+00f,  4.050342000e+00f,  4.120596000e+00f,  4.055364000e+00f,
                4.125699000e+00f,  4.060386000e+00f,  4.130802000e+00f,  4.065408000e+00f,  4.135905000e+00f,  4.070430000e+00f,
                4.141008000e+00f,  4.075452000e+00f,  4.146111000e+00f,  4.080474000e+00f,  4.151214000e+00f,  4.085496000e+00f,
                4.156317000e+00f,  4.090518000e+00f,  2.640765000e+00f,  2.596872000e+00f,  1.255272000e+00f,  1.233312000e+00f,
                1.755974000e+00f,  1.726856000e+00f,  3.339400000e+00f,  3.281128000e+00f,  4.750062000e+00f,  4.662600000e+00f,
                4.755894000e+00f,  4.668324000e+00f,  4.761726000e+00f,  4.674048000e+00f,  4.767558000e+00f,  4.679772000e+00f,
                4.773390000e+00f,  4.685496000e+00f,  4.779222000e+00f,  4.691220000e+00f,  4.785054000e+00f,  4.696944000e+00f,
                4.790886000e+00f,  4.702668000e+00f,  4.796718000e+00f,  4.708392000e+00f,  3.022816000e+00f,  2.963896000e+00f,
                1.423838000e+00f,  1.394360000e+00f,  1.886695000e+00f,  1.850050000e+00f,  3.555815000e+00f,  3.482480000e+00f,
                5.007090000e+00f,  4.897020000e+00f,  5.013165000e+00f,  4.902960000e+00f,  5.019240000e+00f,  4.908900000e+00f,
                5.025315000e+00f,  4.914840000e+00f,  5.031390000e+00f,  4.920780000e+00f,  5.037465000e+00f,  4.926720000e+00f,
                5.043540000e+00f,  4.932660000e+00f,  5.049615000e+00f,  4.938600000e+00f,  5.055690000e+00f,  4.944540000e+00f,
                3.149825000e+00f,  3.075680000e+00f,  1.464505000e+00f,  1.427410000e+00f,  1.911940000e+00f,  1.874800000e+00f,
                3.603335000e+00f,  3.529010000e+00f,  5.073915000e+00f,  4.962360000e+00f,  5.079990000e+00f,  4.968300000e+00f,
                5.086065000e+00f,  4.974240000e+00f,  5.092140000e+00f,  4.980180000e+00f,  5.098215000e+00f,  4.986120000e+00f,
                5.104290000e+00f,  4.992060000e+00f,  5.110365000e+00f,  4.998000000e+00f,  5.116440000e+00f,  5.003940000e+00f,
                5.122515000e+00f,  5.009880000e+00f,  3.191405000e+00f,  3.116270000e+00f,  1.483810000e+00f,  1.446220000e+00f,
                1.937185000e+00f,  1.899550000e+00f,  3.650855000e+00f,  3.575540000e+00f,  5.140740000e+00f,  5.027700000e+00f,
                5.146815000e+00f,  5.033640000e+00f,  5.152890000e+00f,  5.039580000e+00f,  5.158965000e+00f,  5.045520000e+00f,
                5.165040000e+00f,  5.051460000e+00f,  5.171115000e+00f,  5.057400000e+00f,  5.177190000e+00f,  5.063340000e+00f,
                5.183265000e+00f,  5.069280000e+00f,  5.189340000e+00f,  5.075220000e+00f,  3.232985000e+00f,  3.156860000e+00f,
                1.503115000e+00f,  1.465030000e+00f,  1.962430000e+00f,  1.924300000e+00f,  3.698375000e+00f,  3.622070000e+00f,
                5.207565000e+00f,  5.093040000e+00f,  5.213640000e+00f,  5.098980000e+00f,  5.219715000e+00f,  5.104920000e+00f,
                5.225790000e+00f,  5.110860000e+00f,  5.231865000e+00f,  5.116800000e+00f,  5.237940000e+00f,  5.122740000e+00f,
                5.244015000e+00f,  5.128680000e+00f,  5.250090000e+00f,  5.134620000e+00f,  5.256165000e+00f,  5.140560000e+00f,
                3.274565000e+00f,  3.197450000e+00f,  1.522420000e+00f,  1.483840000e+00f,  1.298378000e+00f,  1.267676000e+00f,
                2.414056000e+00f,  2.352616000e+00f,  3.346818000e+00f,  3.254604000e+00f,  3.350706000e+00f,  3.258384000e+00f,
                3.354594000e+00f,  3.262164000e+00f,  3.358482000e+00f,  3.265944000e+00f,  3.362370000e+00f,  3.269724000e+00f,
                3.366258000e+00f,  3.273504000e+00f,  3.370146000e+00f,  3.277284000e+00f,  3.374034000e+00f,  3.281064000e+00f,
                3.377922000e+00f,  3.284844000e+00f,  2.066800000e+00f,  2.004712000e+00f,  9.407540000e-01f,  9.096920000e-01f,
                7.683270000e-01f,  7.451520000e-01f,  1.398495000e+00f,  1.352118000e+00f,  1.890342000e+00f,  1.820736000e+00f,
                1.892529000e+00f,  1.822842000e+00f,  1.894716000e+00f,  1.824948000e+00f,  1.896903000e+00f,  1.827054000e+00f,
                1.899090000e+00f,  1.829160000e+00f,  1.901277000e+00f,  1.831266000e+00f,  1.903464000e+00f,  1.833372000e+00f,
                1.905651000e+00f,  1.835478000e+00f,  1.907838000e+00f,  1.837584000e+00f,  1.131897000e+00f,  1.085034000e+00f,
                4.958970000e-01f,  4.724520000e-01f,  3.740590000e-01f,  3.585100000e-01f,  6.552560000e-01f,  6.241400000e-01f,
                8.434830000e-01f,  7.967820000e-01f,  8.444550000e-01f,  7.977000000e-01f,  8.454270000e-01f,  7.986180000e-01f,
                8.463990000e-01f,  7.995360000e-01f,  8.473710000e-01f,  8.004540000e-01f,  8.483430000e-01f,  8.013720000e-01f,
                8.493150000e-01f,  8.022900000e-01f,  8.502870000e-01f,  8.032080000e-01f,  8.512590000e-01f,  8.041260000e-01f,
                4.734200000e-01f,  4.419800000e-01f,  1.896310000e-01f,  1.739020000e-01f,  1.173560000e-01f,  1.095320000e-01f,
                1.879030000e-01f,  1.722460000e-01f,  2.115870000e-01f,  1.880880000e-01f,  2.118300000e-01f,  1.883040000e-01f,
                2.120730000e-01f,  1.885200000e-01f,  2.123160000e-01f,  1.887360000e-01f,  2.125590000e-01f,  1.889520000e-01f,
                2.128020000e-01f,  1.891680000e-01f,  2.130450000e-01f,  1.893840000e-01f,  2.132880000e-01f,  1.896000000e-01f,
                2.135310000e-01f,  1.898160000e-01f,  9.493300000e-02f,  7.911400000e-02f,  2.373800000e-02f,  1.582400000e-02f,
            };

            float[] x_actual = x.ToArray();

            AssertError.Tolerance(x_expect, x_actual, 1e-7f, 1e-5f, $"mismatch value {inchannels},{outchannels},{kwidth},{kheight},{kdepth},{inwidth},{inheight},{indepth},{batch}");
        }
    }
}
