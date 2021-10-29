using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;
using TensorShader;
using TensorShader.Operators.QuaternionConvolution;
using TensorShaderCudaBackend.API;

namespace TensorShaderTest.Operators.Quaternion {
    [TestClass]
    public class QuaternionConvolution2DTest {
        [TestMethod]
        public void ExecuteFPTest() {
            TensorShaderCudaBackend.Environment.Precision = TensorShaderCudaBackend.Environment.PrecisionMode.Float;
            TensorShaderCudaBackend.Environment.CudnnEnabled = false;

            Random random = new(1234);

            float max_err = 0;

            foreach (int batch in new int[] { 1, 2 }) {
                foreach (int inchannels in new int[] { 4, 8, 20, 32, 36 }) {
                    foreach (int outchannels in new int[] { 4, 8, 20, 32, 36 }) {
                        foreach (int kheight in new int[] { 1, 3, 5 }) {
                            foreach (int kwidth in new int[] { 1, 3, 5 }) {
                                foreach (int inwidth in new int[] { kwidth, kwidth * 2, 8, 9, 13, 17, 25 }) {
                                    foreach (int inheight in new int[] { kheight, kheight * 2, 8, 9, 13, 17, 25 }) {
                                        int outwidth = inwidth - kwidth + 1, outheight = inheight - kheight + 1;

                                        float[] xval = (new float[inwidth * inheight * inchannels * batch]).Select((_, idx) => (float)random.NextDouble() * 1e-2f).ToArray();
                                        float[] wval = (new float[kwidth * kheight * inchannels * outchannels / 4]).Select((_, idx) => (float)random.NextDouble() * 1e-2f).ToArray();

                                        Quaternion[] xcval = (new Quaternion[xval.Length / 4])
                                            .Select((_, idx) => new Quaternion(xval[idx * 4], xval[idx * 4 + 1], xval[idx * 4 + 2], xval[idx * 4 + 3])).ToArray();

                                        Quaternion[] wcval = (new Quaternion[wval.Length / 4])
                                            .Select((_, idx) => new Quaternion(wval[idx * 4], wval[idx * 4 + 1], wval[idx * 4 + 2], wval[idx * 4 + 3])).ToArray();

                                        QuaternionMap2D x = new(inchannels / 4, inwidth, inheight, batch, xcval);
                                        QuaternionFilter2D w = new(inchannels / 4, outchannels / 4, kwidth, kheight, wcval);

                                        QuaternionMap2D y = Reference(x, w, kwidth, kheight);

                                        OverflowCheckedTensor x_tensor = new(Shape.Map2D(inchannels, inwidth, inheight, batch), xval);
                                        OverflowCheckedTensor w_tensor = new(Shape.Kernel2D(inchannels, outchannels / 4, kwidth, kheight), wval);

                                        OverflowCheckedTensor y_tensor = new(Shape.Map2D(outchannels, outwidth, outheight, batch));

                                        QuaternionConvolution2D ope = new(inwidth, inheight, inchannels, outchannels, kwidth, kheight, gradmode: false, batch);

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
                foreach (int inchannels in new int[] { 4, 8, 20, 32, 36 }) {
                    foreach (int outchannels in new int[] { 4, 8, 20, 32, 36 }) {
                        foreach (int kheight in new int[] { 1, 3, 5 }) {
                            foreach (int kwidth in new int[] { 1, 3, 5 }) {
                                foreach (int inwidth in new int[] { kwidth, kwidth * 2, 8, 9, 13, 17, 25 }) {
                                    foreach (int inheight in new int[] { kheight, kheight * 2, 8, 9, 13, 17, 25 }) {
                                        int outwidth = inwidth - kwidth + 1, outheight = inheight - kheight + 1;

                                        float[] xval = (new float[inwidth * inheight * inchannels * batch]).Select((_, idx) => (float)random.NextDouble() * 1e-2f).ToArray();
                                        float[] wval = (new float[kwidth * kheight * inchannels * outchannels / 4]).Select((_, idx) => (float)random.NextDouble() * 1e-2f).ToArray();

                                        Quaternion[] xcval = (new Quaternion[xval.Length / 4])
                                            .Select((_, idx) => new Quaternion(xval[idx * 4], xval[idx * 4 + 1], xval[idx * 4 + 2], xval[idx * 4 + 3])).ToArray();

                                        Quaternion[] wcval = (new Quaternion[wval.Length / 4])
                                            .Select((_, idx) => new Quaternion(wval[idx * 4], wval[idx * 4 + 1], wval[idx * 4 + 2], wval[idx * 4 + 3])).ToArray();

                                        QuaternionMap2D x = new(inchannels / 4, inwidth, inheight, batch, xcval);
                                        QuaternionFilter2D w = new(inchannels / 4, outchannels / 4, kwidth, kheight, wcval);

                                        QuaternionMap2D y = Reference(x, w, kwidth, kheight);

                                        OverflowCheckedTensor x_tensor = new(Shape.Map2D(inchannels, inwidth, inheight, batch), xval);
                                        OverflowCheckedTensor w_tensor = new(Shape.Kernel2D(inchannels, outchannels / 4, kwidth, kheight), wval);

                                        OverflowCheckedTensor y_tensor = new(Shape.Map2D(outchannels, outwidth, outheight, batch));

                                        QuaternionConvolution2D ope = new(inwidth, inheight, inchannels, outchannels, kwidth, kheight, gradmode: false, batch);

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
            int inchannels = 196, outchannels = 200;
            int kwidth = 5, kheight = 3;
            int inwidth = 250, inheight = 196;
            int outwidth = inwidth - kwidth + 1, outheight = inheight - kheight + 1;

            float[] xval = (new float[inwidth * inheight * inchannels * batch]).Select((_, idx) => (float)random.NextDouble() * 1e-2f).ToArray();
            float[] wval = (new float[kwidth * kheight * inchannels * outchannels / 4]).Select((_, idx) => (float)random.NextDouble() * 1e-2f).ToArray();

            Quaternion[] xcval = (new Quaternion[xval.Length / 4])
                .Select((_, idx) => new Quaternion(xval[idx * 4], xval[idx * 4 + 1], xval[idx * 4 + 2], xval[idx * 4 + 3])).ToArray();

            Quaternion[] wcval = (new Quaternion[wval.Length / 4])
                .Select((_, idx) => new Quaternion(wval[idx * 4], wval[idx * 4 + 1], wval[idx * 4 + 2], wval[idx * 4 + 3])).ToArray();

            QuaternionMap2D x = new(inchannels / 4, inwidth, inheight, batch, xcval);
            QuaternionFilter2D w = new(inchannels / 4, outchannels / 4, kwidth, kheight, wcval);

            QuaternionMap2D y = Reference(x, w, kwidth, kheight);

            OverflowCheckedTensor x_tensor = new(Shape.Map2D(inchannels, inwidth, inheight, batch), xval);
            OverflowCheckedTensor w_tensor = new(Shape.Kernel2D(inchannels, outchannels / 4, kwidth, kheight), wval);

            OverflowCheckedTensor y_tensor = new(Shape.Map2D(outchannels, outwidth, outheight, batch));

            QuaternionConvolution2D ope = new(inwidth, inheight, inchannels, outchannels, kwidth, kheight, gradmode: false, batch);

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
        public void SpeedTest() {
            int inwidth = 512, inheight = 512, inchannels = 32, outchannels = 32, ksize = 3;
            int outwidth = inwidth - ksize + 1, outheight = inheight - ksize + 1;

            OverflowCheckedTensor x_tensor = new(Shape.Map2D(inchannels, inwidth, inheight));
            OverflowCheckedTensor w_tensor = new(Shape.Kernel2D(inchannels, outchannels / 4, ksize, ksize));

            OverflowCheckedTensor y_tensor = new(Shape.Map2D(outchannels, outwidth, outheight));

            QuaternionConvolution2D ope = new(inwidth, inheight, inchannels, outchannels, ksize, ksize);

            Cuda.Profiler.Initialize("../../../../profiler.nvsetting", "../../nvprofiles/quaternion_convolution_2d.nvvp");
            Cuda.Profiler.Start();

            ope.Execute(x_tensor, w_tensor, y_tensor);

            Cuda.Profiler.Stop();
        }

        public static QuaternionMap2D Reference(QuaternionMap2D x, QuaternionFilter2D w, int kwidth, int kheight) {
            int inchannels = x.Channels, outchannels = w.OutChannels, batch = x.Batch;
            int inw = x.Width, inh = x.Height, outw = inw - kwidth + 1, outh = inh - kheight + 1;

            QuaternionMap2D y = new(outchannels, outw, outh, batch);

            for (int kx, ky = 0; ky < kheight; ky++) {
                for (kx = 0; kx < kwidth; kx++) {
                    for (int th = 0; th < batch; th++) {
                        for (int ox, oy = 0; oy < outh; oy++) {
                            for (ox = 0; ox < outw; ox++) {
                                for (int outch = 0; outch < outchannels; outch++) {
                                    Quaternion sum = y[outch, ox, oy, th];

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
            int inchannels = 8, outchannels = 12, kwidth = 3, kheight = 5, inwidth = 7, inheight = 8, batch = 3;
            int outwidth = inwidth - kwidth + 1, outheight = inheight - kheight + 1;

            float[] xval = (new float[batch * inwidth * inheight * inchannels]).Select((_, idx) => idx * 1e-3f).ToArray();
            float[] wval = (new float[kwidth * kheight * outchannels * inchannels / 4]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

            Quaternion[] xcval = (new Quaternion[xval.Length / 4])
                .Select((_, idx) => new Quaternion(xval[idx * 4], xval[idx * 4 + 1], xval[idx * 4 + 2], xval[idx * 4 + 3])).ToArray();

            Quaternion[] wcval = (new Quaternion[wval.Length / 4])
                .Select((_, idx) => new Quaternion(wval[idx * 4], wval[idx * 4 + 1], wval[idx * 4 + 2], wval[idx * 4 + 3])).ToArray();

            QuaternionMap2D x = new(inchannels / 4, inwidth, inheight, batch, xcval);
            QuaternionFilter2D w = new(inchannels / 4, outchannels / 4, kwidth, kheight, wcval);

            QuaternionMap2D y = Reference(x, w, kwidth, kheight);

            float[] y_expect = {
                -9.033600000e-01f,  8.844000000e-01f,  9.144000000e-01f,  8.884200000e-01f,  -8.433600000e-01f,  8.258400000e-01f,
                8.548800000e-01f,  8.293800000e-01f,  -7.833600000e-01f,  7.672800000e-01f,  7.953600000e-01f,  7.703400000e-01f,
                -9.926400000e-01f,  9.746400000e-01f,  1.005120000e+00f,  9.781800000e-01f,  -9.288000000e-01f,  9.122400000e-01f,
                9.417600000e-01f,  9.153000000e-01f,  -8.649600000e-01f,  8.498400000e-01f,  8.784000000e-01f,  8.524200000e-01f,
                -1.081920000e+00f,  1.064880000e+00f,  1.095840000e+00f,  1.067940000e+00f,  -1.014240000e+00f,  9.986400000e-01f,
                1.028640000e+00f,  1.001220000e+00f,  -9.465600000e-01f,  9.324000000e-01f,  9.614400000e-01f,  9.345000000e-01f,
                -1.171200000e+00f,  1.155120000e+00f,  1.186560000e+00f,  1.157700000e+00f,  -1.099680000e+00f,  1.085040000e+00f,
                1.115520000e+00f,  1.087140000e+00f,  -1.028160000e+00f,  1.014960000e+00f,  1.044480000e+00f,  1.016580000e+00f,
                -1.260480000e+00f,  1.245360000e+00f,  1.277280000e+00f,  1.247460000e+00f,  -1.185120000e+00f,  1.171440000e+00f,
                1.202400000e+00f,  1.173060000e+00f,  -1.109760000e+00f,  1.097520000e+00f,  1.127520000e+00f,  1.098660000e+00f,
                -1.528320000e+00f,  1.516080000e+00f,  1.549440000e+00f,  1.516740000e+00f,  -1.441440000e+00f,  1.430640000e+00f,
                1.463040000e+00f,  1.430820000e+00f,  -1.354560000e+00f,  1.345200000e+00f,  1.376640000e+00f,  1.344900000e+00f,
                -1.617600000e+00f,  1.606320000e+00f,  1.640160000e+00f,  1.606500000e+00f,  -1.526880000e+00f,  1.517040000e+00f,
                1.549920000e+00f,  1.516740000e+00f,  -1.436160000e+00f,  1.427760000e+00f,  1.459680000e+00f,  1.426980000e+00f,
                -1.706880000e+00f,  1.696560000e+00f,  1.730880000e+00f,  1.696260000e+00f,  -1.612320000e+00f,  1.603440000e+00f,
                1.636800000e+00f,  1.602660000e+00f,  -1.517760000e+00f,  1.510320000e+00f,  1.542720000e+00f,  1.509060000e+00f,
                -1.796160000e+00f,  1.786800000e+00f,  1.821600000e+00f,  1.786020000e+00f,  -1.697760000e+00f,  1.689840000e+00f,
                1.723680000e+00f,  1.688580000e+00f,  -1.599360000e+00f,  1.592880000e+00f,  1.625760000e+00f,  1.591140000e+00f,
                -1.885440000e+00f,  1.877040000e+00f,  1.912320000e+00f,  1.875780000e+00f,  -1.783200000e+00f,  1.776240000e+00f,
                1.810560000e+00f,  1.774500000e+00f,  -1.680960000e+00f,  1.675440000e+00f,  1.708800000e+00f,  1.673220000e+00f,
                -2.153280000e+00f,  2.147760000e+00f,  2.184480000e+00f,  2.145060000e+00f,  -2.039520000e+00f,  2.035440000e+00f,
                2.071200000e+00f,  2.032260000e+00f,  -1.925760000e+00f,  1.923120000e+00f,  1.957920000e+00f,  1.919460000e+00f,
                -2.242560000e+00f,  2.238000000e+00f,  2.275200000e+00f,  2.234820000e+00f,  -2.124960000e+00f,  2.121840000e+00f,
                2.158080000e+00f,  2.118180000e+00f,  -2.007360000e+00f,  2.005680000e+00f,  2.040960000e+00f,  2.001540000e+00f,
                -2.331840000e+00f,  2.328240000e+00f,  2.365920000e+00f,  2.324580000e+00f,  -2.210400000e+00f,  2.208240000e+00f,
                2.244960000e+00f,  2.204100000e+00f,  -2.088960000e+00f,  2.088240000e+00f,  2.124000000e+00f,  2.083620000e+00f,
                -2.421120000e+00f,  2.418480000e+00f,  2.456640000e+00f,  2.414340000e+00f,  -2.295840000e+00f,  2.294640000e+00f,
                2.331840000e+00f,  2.290020000e+00f,  -2.170560000e+00f,  2.170800000e+00f,  2.207040000e+00f,  2.165700000e+00f,
                -2.510400000e+00f,  2.508720000e+00f,  2.547360000e+00f,  2.504100000e+00f,  -2.381280000e+00f,  2.381040000e+00f,
                2.418720000e+00f,  2.375940000e+00f,  -2.252160000e+00f,  2.253360000e+00f,  2.290080000e+00f,  2.247780000e+00f,
                -2.778240000e+00f,  2.779440000e+00f,  2.819520000e+00f,  2.773380000e+00f,  -2.637600000e+00f,  2.640240000e+00f,
                2.679360000e+00f,  2.633700000e+00f,  -2.496960000e+00f,  2.501040000e+00f,  2.539200000e+00f,  2.494020000e+00f,
                -2.867520000e+00f,  2.869680000e+00f,  2.910240000e+00f,  2.863140000e+00f,  -2.723040000e+00f,  2.726640000e+00f,
                2.766240000e+00f,  2.719620000e+00f,  -2.578560000e+00f,  2.583600000e+00f,  2.622240000e+00f,  2.576100000e+00f,
                -2.956800000e+00f,  2.959920000e+00f,  3.000960000e+00f,  2.952900000e+00f,  -2.808480000e+00f,  2.813040000e+00f,
                2.853120000e+00f,  2.805540000e+00f,  -2.660160000e+00f,  2.666160000e+00f,  2.705280000e+00f,  2.658180000e+00f,
                -3.046080000e+00f,  3.050160000e+00f,  3.091680000e+00f,  3.042660000e+00f,  -2.893920000e+00f,  2.899440000e+00f,
                2.940000000e+00f,  2.891460000e+00f,  -2.741760000e+00f,  2.748720000e+00f,  2.788320000e+00f,  2.740260000e+00f,
                -3.135360000e+00f,  3.140400000e+00f,  3.182400000e+00f,  3.132420000e+00f,  -2.979360000e+00f,  2.985840000e+00f,
                3.026880000e+00f,  2.977380000e+00f,  -2.823360000e+00f,  2.831280000e+00f,  2.871360000e+00f,  2.822340000e+00f,
                -5.903040000e+00f,  5.937840000e+00f,  5.994720000e+00f,  5.914980000e+00f,  -5.628000000e+00f,  5.664240000e+00f,
                5.720160000e+00f,  5.640900000e+00f,  -5.352960000e+00f,  5.390640000e+00f,  5.445600000e+00f,  5.366820000e+00f,
                -5.992320000e+00f,  6.028080000e+00f,  6.085440000e+00f,  6.004740000e+00f,  -5.713440000e+00f,  5.750640000e+00f,
                5.807040000e+00f,  5.726820000e+00f,  -5.434560000e+00f,  5.473200000e+00f,  5.528640000e+00f,  5.448900000e+00f,
                -6.081600000e+00f,  6.118320000e+00f,  6.176160000e+00f,  6.094500000e+00f,  -5.798880000e+00f,  5.837040000e+00f,
                5.893920000e+00f,  5.812740000e+00f,  -5.516160000e+00f,  5.555760000e+00f,  5.611680000e+00f,  5.530980000e+00f,
                -6.170880000e+00f,  6.208560000e+00f,  6.266880000e+00f,  6.184260000e+00f,  -5.884320000e+00f,  5.923440000e+00f,
                5.980800000e+00f,  5.898660000e+00f,  -5.597760000e+00f,  5.638320000e+00f,  5.694720000e+00f,  5.613060000e+00f,
                -6.260160000e+00f,  6.298800000e+00f,  6.357600000e+00f,  6.274020000e+00f,  -5.969760000e+00f,  6.009840000e+00f,
                6.067680000e+00f,  5.984580000e+00f,  -5.679360000e+00f,  5.720880000e+00f,  5.777760000e+00f,  5.695140000e+00f,
                -6.528000000e+00f,  6.569520000e+00f,  6.629760000e+00f,  6.543300000e+00f,  -6.226080000e+00f,  6.269040000e+00f,
                6.328320000e+00f,  6.242340000e+00f,  -5.924160000e+00f,  5.968560000e+00f,  6.026880000e+00f,  5.941380000e+00f,
                -6.617280000e+00f,  6.659760000e+00f,  6.720480000e+00f,  6.633060000e+00f,  -6.311520000e+00f,  6.355440000e+00f,
                6.415200000e+00f,  6.328260000e+00f,  -6.005760000e+00f,  6.051120000e+00f,  6.109920000e+00f,  6.023460000e+00f,
                -6.706560000e+00f,  6.750000000e+00f,  6.811200000e+00f,  6.722820000e+00f,  -6.396960000e+00f,  6.441840000e+00f,
                6.502080000e+00f,  6.414180000e+00f,  -6.087360000e+00f,  6.133680000e+00f,  6.192960000e+00f,  6.105540000e+00f,
                -6.795840000e+00f,  6.840240000e+00f,  6.901920000e+00f,  6.812580000e+00f,  -6.482400000e+00f,  6.528240000e+00f,
                6.588960000e+00f,  6.500100000e+00f,  -6.168960000e+00f,  6.216240000e+00f,  6.276000000e+00f,  6.187620000e+00f,
                -6.885120000e+00f,  6.930480000e+00f,  6.992640000e+00f,  6.902340000e+00f,  -6.567840000e+00f,  6.614640000e+00f,
                6.675840000e+00f,  6.586020000e+00f,  -6.250560000e+00f,  6.298800000e+00f,  6.359040000e+00f,  6.269700000e+00f,
                -7.152960000e+00f,  7.201200000e+00f,  7.264800000e+00f,  7.171620000e+00f,  -6.824160000e+00f,  6.873840000e+00f,
                6.936480000e+00f,  6.843780000e+00f,  -6.495360000e+00f,  6.546480000e+00f,  6.608160000e+00f,  6.515940000e+00f,
                -7.242240000e+00f,  7.291440000e+00f,  7.355520000e+00f,  7.261380000e+00f,  -6.909600000e+00f,  6.960240000e+00f,
                7.023360000e+00f,  6.929700000e+00f,  -6.576960000e+00f,  6.629040000e+00f,  6.691200000e+00f,  6.598020000e+00f,
                -7.331520000e+00f,  7.381680000e+00f,  7.446240000e+00f,  7.351140000e+00f,  -6.995040000e+00f,  7.046640000e+00f,
                7.110240000e+00f,  7.015620000e+00f,  -6.658560000e+00f,  6.711600000e+00f,  6.774240000e+00f,  6.680100000e+00f,
                -7.420800000e+00f,  7.471920000e+00f,  7.536960000e+00f,  7.440900000e+00f,  -7.080480000e+00f,  7.133040000e+00f,
                7.197120000e+00f,  7.101540000e+00f,  -6.740160000e+00f,  6.794160000e+00f,  6.857280000e+00f,  6.762180000e+00f,
                -7.510080000e+00f,  7.562160000e+00f,  7.627680000e+00f,  7.530660000e+00f,  -7.165920000e+00f,  7.219440000e+00f,
                7.284000000e+00f,  7.187460000e+00f,  -6.821760000e+00f,  6.876720000e+00f,  6.940320000e+00f,  6.844260000e+00f,
                -7.777920000e+00f,  7.832880000e+00f,  7.899840000e+00f,  7.799940000e+00f,  -7.422240000e+00f,  7.478640000e+00f,
                7.544640000e+00f,  7.445220000e+00f,  -7.066560000e+00f,  7.124400000e+00f,  7.189440000e+00f,  7.090500000e+00f,
                -7.867200000e+00f,  7.923120000e+00f,  7.990560000e+00f,  7.889700000e+00f,  -7.507680000e+00f,  7.565040000e+00f,
                7.631520000e+00f,  7.531140000e+00f,  -7.148160000e+00f,  7.206960000e+00f,  7.272480000e+00f,  7.172580000e+00f,
                -7.956480000e+00f,  8.013360000e+00f,  8.081280000e+00f,  7.979460000e+00f,  -7.593120000e+00f,  7.651440000e+00f,
                7.718400000e+00f,  7.617060000e+00f,  -7.229760000e+00f,  7.289520000e+00f,  7.355520000e+00f,  7.254660000e+00f,
                -8.045760000e+00f,  8.103600000e+00f,  8.172000000e+00f,  8.069220000e+00f,  -7.678560000e+00f,  7.737840000e+00f,
                7.805280000e+00f,  7.702980000e+00f,  -7.311360000e+00f,  7.372080000e+00f,  7.438560000e+00f,  7.336740000e+00f,
                -8.135040000e+00f,  8.193840000e+00f,  8.262720000e+00f,  8.158980000e+00f,  -7.764000000e+00f,  7.824240000e+00f,
                7.892160000e+00f,  7.788900000e+00f,  -7.392960000e+00f,  7.454640000e+00f,  7.521600000e+00f,  7.418820000e+00f,
                -1.090272000e+01f,  1.099128000e+01f,  1.107504000e+01f,  1.094154000e+01f,  -1.041264000e+01f,  1.050264000e+01f,
                1.058544000e+01f,  1.045242000e+01f,  -9.922560000e+00f,  1.001400000e+01f,  1.009584000e+01f,  9.963300000e+00f,
                -1.099200000e+01f,  1.108152000e+01f,  1.116576000e+01f,  1.103130000e+01f,  -1.049808000e+01f,  1.058904000e+01f,
                1.067232000e+01f,  1.053834000e+01f,  -1.000416000e+01f,  1.009656000e+01f,  1.017888000e+01f,  1.004538000e+01f,
                -1.108128000e+01f,  1.117176000e+01f,  1.125648000e+01f,  1.112106000e+01f,  -1.058352000e+01f,  1.067544000e+01f,
                1.075920000e+01f,  1.062426000e+01f,  -1.008576000e+01f,  1.017912000e+01f,  1.026192000e+01f,  1.012746000e+01f,
                -1.117056000e+01f,  1.126200000e+01f,  1.134720000e+01f,  1.121082000e+01f,  -1.066896000e+01f,  1.076184000e+01f,
                1.084608000e+01f,  1.071018000e+01f,  -1.016736000e+01f,  1.026168000e+01f,  1.034496000e+01f,  1.020954000e+01f,
                -1.125984000e+01f,  1.135224000e+01f,  1.143792000e+01f,  1.130058000e+01f,  -1.075440000e+01f,  1.084824000e+01f,
                1.093296000e+01f,  1.079610000e+01f,  -1.024896000e+01f,  1.034424000e+01f,  1.042800000e+01f,  1.029162000e+01f,
                -1.152768000e+01f,  1.162296000e+01f,  1.171008000e+01f,  1.156986000e+01f,  -1.101072000e+01f,  1.110744000e+01f,
                1.119360000e+01f,  1.105386000e+01f,  -1.049376000e+01f,  1.059192000e+01f,  1.067712000e+01f,  1.053786000e+01f,
                -1.161696000e+01f,  1.171320000e+01f,  1.180080000e+01f,  1.165962000e+01f,  -1.109616000e+01f,  1.119384000e+01f,
                1.128048000e+01f,  1.113978000e+01f,  -1.057536000e+01f,  1.067448000e+01f,  1.076016000e+01f,  1.061994000e+01f,
                -1.170624000e+01f,  1.180344000e+01f,  1.189152000e+01f,  1.174938000e+01f,  -1.118160000e+01f,  1.128024000e+01f,
                1.136736000e+01f,  1.122570000e+01f,  -1.065696000e+01f,  1.075704000e+01f,  1.084320000e+01f,  1.070202000e+01f,
                -1.179552000e+01f,  1.189368000e+01f,  1.198224000e+01f,  1.183914000e+01f,  -1.126704000e+01f,  1.136664000e+01f,
                1.145424000e+01f,  1.131162000e+01f,  -1.073856000e+01f,  1.083960000e+01f,  1.092624000e+01f,  1.078410000e+01f,
                -1.188480000e+01f,  1.198392000e+01f,  1.207296000e+01f,  1.192890000e+01f,  -1.135248000e+01f,  1.145304000e+01f,
                1.154112000e+01f,  1.139754000e+01f,  -1.082016000e+01f,  1.092216000e+01f,  1.100928000e+01f,  1.086618000e+01f,
                -1.215264000e+01f,  1.225464000e+01f,  1.234512000e+01f,  1.219818000e+01f,  -1.160880000e+01f,  1.171224000e+01f,
                1.180176000e+01f,  1.165530000e+01f,  -1.106496000e+01f,  1.116984000e+01f,  1.125840000e+01f,  1.111242000e+01f,
                -1.224192000e+01f,  1.234488000e+01f,  1.243584000e+01f,  1.228794000e+01f,  -1.169424000e+01f,  1.179864000e+01f,
                1.188864000e+01f,  1.174122000e+01f,  -1.114656000e+01f,  1.125240000e+01f,  1.134144000e+01f,  1.119450000e+01f,
                -1.233120000e+01f,  1.243512000e+01f,  1.252656000e+01f,  1.237770000e+01f,  -1.177968000e+01f,  1.188504000e+01f,
                1.197552000e+01f,  1.182714000e+01f,  -1.122816000e+01f,  1.133496000e+01f,  1.142448000e+01f,  1.127658000e+01f,
                -1.242048000e+01f,  1.252536000e+01f,  1.261728000e+01f,  1.246746000e+01f,  -1.186512000e+01f,  1.197144000e+01f,
                1.206240000e+01f,  1.191306000e+01f,  -1.130976000e+01f,  1.141752000e+01f,  1.150752000e+01f,  1.135866000e+01f,
                -1.250976000e+01f,  1.261560000e+01f,  1.270800000e+01f,  1.255722000e+01f,  -1.195056000e+01f,  1.205784000e+01f,
                1.214928000e+01f,  1.199898000e+01f,  -1.139136000e+01f,  1.150008000e+01f,  1.159056000e+01f,  1.144074000e+01f,
                -1.277760000e+01f,  1.288632000e+01f,  1.298016000e+01f,  1.282650000e+01f,  -1.220688000e+01f,  1.231704000e+01f,
                1.240992000e+01f,  1.225674000e+01f,  -1.163616000e+01f,  1.174776000e+01f,  1.183968000e+01f,  1.168698000e+01f,
                -1.286688000e+01f,  1.297656000e+01f,  1.307088000e+01f,  1.291626000e+01f,  -1.229232000e+01f,  1.240344000e+01f,
                1.249680000e+01f,  1.234266000e+01f,  -1.171776000e+01f,  1.183032000e+01f,  1.192272000e+01f,  1.176906000e+01f,
                -1.295616000e+01f,  1.306680000e+01f,  1.316160000e+01f,  1.300602000e+01f,  -1.237776000e+01f,  1.248984000e+01f,
                1.258368000e+01f,  1.242858000e+01f,  -1.179936000e+01f,  1.191288000e+01f,  1.200576000e+01f,  1.185114000e+01f,
                -1.304544000e+01f,  1.315704000e+01f,  1.325232000e+01f,  1.309578000e+01f,  -1.246320000e+01f,  1.257624000e+01f,
                1.267056000e+01f,  1.251450000e+01f,  -1.188096000e+01f,  1.199544000e+01f,  1.208880000e+01f,  1.193322000e+01f,
                -1.313472000e+01f,  1.324728000e+01f,  1.334304000e+01f,  1.318554000e+01f,  -1.254864000e+01f,  1.266264000e+01f,
                1.275744000e+01f,  1.260042000e+01f,  -1.196256000e+01f,  1.207800000e+01f,  1.217184000e+01f,  1.201530000e+01f,
            };

            float[] y_actual = y.ToArray();

            AssertError.Tolerance(y_expect, y_actual, 1e-7f, 1e-5f, $"mismatch value {inchannels},{outchannels},{kwidth},{kheight},{inwidth},{inheight},{batch}");
        }
    }
}
