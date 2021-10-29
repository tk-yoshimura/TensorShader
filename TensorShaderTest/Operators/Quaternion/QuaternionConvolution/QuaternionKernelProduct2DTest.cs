using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;
using TensorShader;
using TensorShader.Operators.QuaternionConvolution;
using TensorShaderCudaBackend.API;

namespace TensorShaderTest.Operators.Quaternion {
    [TestClass]
    public class QuaternionKernelProduct2DTest {
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
                                        float[] yval = (new float[outwidth * outheight * outchannels * batch]).Select((_, idx) => (float)random.NextDouble() * 1e-2f).ToArray();

                                        Quaternion[] xcval = (new Quaternion[xval.Length / 4])
                                            .Select((_, idx) => new Quaternion(xval[idx * 4], xval[idx * 4 + 1], xval[idx * 4 + 2], xval[idx * 4 + 3])).ToArray();

                                        Quaternion[] ycval = (new Quaternion[yval.Length / 4])
                                            .Select((_, idx) => new Quaternion(yval[idx * 4], yval[idx * 4 + 1], yval[idx * 4 + 2], yval[idx * 4 + 3])).ToArray();

                                        QuaternionMap2D x = new(inchannels / 4, inwidth, inheight, batch, xcval);
                                        QuaternionMap2D y = new(outchannels / 4, outwidth, outheight, batch, ycval);

                                        QuaternionFilter2D gw = Reference(x, y, kwidth, kheight);

                                        OverflowCheckedTensor x_tensor = new(Shape.Map2D(inchannels, inwidth, inheight, batch), xval);
                                        OverflowCheckedTensor y_tensor = new(Shape.Map2D(outchannels, outwidth, outheight, batch), yval);

                                        OverflowCheckedTensor gw_tensor = new(Shape.Kernel2D(inchannels, outchannels / 4, kwidth, kheight));

                                        QuaternionKernelProduct2D ope = new(inwidth, inheight, inchannels, outchannels, kwidth, kheight, transpose: false, batch);

                                        ope.Execute(x_tensor, y_tensor, gw_tensor);

                                        float[] gw_expect = gw.ToArray();
                                        float[] gw_actual = gw_tensor.State.Value;

                                        CollectionAssert.AreEqual(xval, x_tensor.State.Value);
                                        CollectionAssert.AreEqual(yval, y_tensor.State.Value);

                                        AssertError.Tolerance(gw_expect, gw_actual, 1e-6f, 1e-4f, ref max_err, $"mismatch value {inchannels},{outchannels},{kwidth},{kheight},{inwidth},{inheight},{batch}");

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
                                        float[] yval = (new float[outwidth * outheight * outchannels * batch]).Select((_, idx) => (float)random.NextDouble() * 1e-2f).ToArray();

                                        Quaternion[] xcval = (new Quaternion[xval.Length / 4])
                                            .Select((_, idx) => new Quaternion(xval[idx * 4], xval[idx * 4 + 1], xval[idx * 4 + 2], xval[idx * 4 + 3])).ToArray();

                                        Quaternion[] ycval = (new Quaternion[yval.Length / 4])
                                            .Select((_, idx) => new Quaternion(yval[idx * 4], yval[idx * 4 + 1], yval[idx * 4 + 2], yval[idx * 4 + 3])).ToArray();

                                        QuaternionMap2D x = new(inchannels / 4, inwidth, inheight, batch, xcval);
                                        QuaternionMap2D y = new(outchannels / 4, outwidth, outheight, batch, ycval);

                                        QuaternionFilter2D gw = Reference(x, y, kwidth, kheight);

                                        OverflowCheckedTensor x_tensor = new(Shape.Map2D(inchannels, inwidth, inheight, batch), xval);
                                        OverflowCheckedTensor y_tensor = new(Shape.Map2D(outchannels, outwidth, outheight, batch), yval);

                                        OverflowCheckedTensor gw_tensor = new(Shape.Kernel2D(inchannels, outchannels / 4, kwidth, kheight));

                                        QuaternionKernelProduct2D ope = new(inwidth, inheight, inchannels, outchannels, kwidth, kheight, transpose: false, batch);

                                        ope.Execute(x_tensor, y_tensor, gw_tensor);

                                        float[] gw_expect = gw.ToArray();
                                        float[] gw_actual = gw_tensor.State.Value;

                                        CollectionAssert.AreEqual(xval, x_tensor.State.Value);
                                        CollectionAssert.AreEqual(yval, y_tensor.State.Value);

                                        AssertError.Tolerance(gw_expect, gw_actual, 1e-7f, 1e-5f, ref max_err, $"mismatch value {inchannels},{outchannels},{kwidth},{kheight},{inwidth},{inheight},{batch}");

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
            float[] yval = (new float[outwidth * outheight * outchannels * batch]).Select((_, idx) => (float)random.NextDouble() * 1e-2f).ToArray();

            Quaternion[] xcval = (new Quaternion[xval.Length / 4])
                .Select((_, idx) => new Quaternion(xval[idx * 4], xval[idx * 4 + 1], xval[idx * 4 + 2], xval[idx * 4 + 3])).ToArray();

            Quaternion[] ycval = (new Quaternion[yval.Length / 4])
                .Select((_, idx) => new Quaternion(yval[idx * 4], yval[idx * 4 + 1], yval[idx * 4 + 2], yval[idx * 4 + 3])).ToArray();

            QuaternionMap2D x = new(inchannels / 4, inwidth, inheight, batch, xcval);
            QuaternionMap2D y = new(outchannels / 4, outwidth, outheight, batch, ycval);

            QuaternionFilter2D gw = Reference(x, y, kwidth, kheight);

            OverflowCheckedTensor x_tensor = new(Shape.Map2D(inchannels, inwidth, inheight, batch), xval);
            OverflowCheckedTensor y_tensor = new(Shape.Map2D(outchannels, outwidth, outheight, batch), yval);

            OverflowCheckedTensor gw_tensor = new(Shape.Kernel2D(inchannels, outchannels / 4, kwidth, kheight));

            QuaternionKernelProduct2D ope = new(inwidth, inheight, inchannels, outchannels, kwidth, kheight, transpose: false, batch);

            ope.Execute(x_tensor, y_tensor, gw_tensor);

            float[] gw_expect = gw.ToArray();
            float[] gw_actual = gw_tensor.State.Value;

            CollectionAssert.AreEqual(xval, x_tensor.State.Value);
            CollectionAssert.AreEqual(yval, y_tensor.State.Value);

            AssertError.Tolerance(gw_expect, gw_actual, 1e-7f, 1e-5f, ref max_err, $"mismatch value {inchannels},{outchannels},{kwidth},{kheight},{inwidth},{inheight},{batch}");

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
            OverflowCheckedTensor y_tensor = new(Shape.Map2D(outchannels, outwidth, outheight));

            OverflowCheckedTensor gw_tensor = new(Shape.Kernel2D(inchannels, outchannels / 4, ksize, ksize));

            QuaternionKernelProduct2D ope = new(inwidth, inheight, inchannels, outchannels, ksize, ksize);

            Cuda.Profiler.Initialize("../../../../profiler.nvsetting", "../../nvprofiles/quaternion_kernelproduct_2d_fp.nvvp");
            Cuda.Profiler.Start();

            ope.Execute(x_tensor, y_tensor, gw_tensor);

            Cuda.Profiler.Stop();
        }

        [TestMethod]
        public void SpeedFFPTest() {
            TensorShaderCudaBackend.Environment.CudnnEnabled = false;
            TensorShaderCudaBackend.Environment.Precision = TensorShaderCudaBackend.Environment.PrecisionMode.FloatFloat;

            int inwidth = 512, inheight = 512, inchannels = 32, outchannels = 32, ksize = 3;
            int outwidth = inwidth - ksize + 1, outheight = inheight - ksize + 1;

            OverflowCheckedTensor x_tensor = new(Shape.Map2D(inchannels, inwidth, inheight));
            OverflowCheckedTensor y_tensor = new(Shape.Map2D(outchannels, outwidth, outheight));

            OverflowCheckedTensor gw_tensor = new(Shape.Kernel2D(inchannels, outchannels / 4, ksize, ksize));

            QuaternionKernelProduct2D ope = new(inwidth, inheight, inchannels, outchannels, ksize, ksize);

            Cuda.Profiler.Initialize("../../../../profiler.nvsetting", "../../nvprofiles/quaternion_kernelproduct_2d_ffp.nvvp");
            Cuda.Profiler.Start();

            ope.Execute(x_tensor, y_tensor, gw_tensor);

            Cuda.Profiler.Stop();
        }

        public static QuaternionFilter2D Reference(QuaternionMap2D x, QuaternionMap2D gy, int kwidth, int kheight) {
            int inchannels = x.Channels, outchannels = gy.Channels, batch = x.Batch;
            int inw = x.Width, inh = x.Height, outw = gy.Width, outh = gy.Height;

            if (outw != inw - kwidth + 1 || outh != inh - kheight + 1) {
                throw new ArgumentException("mismatch shape");
            }

            QuaternionFilter2D w = new(inchannels, outchannels, kwidth, kheight);

            for (int kx, ky = 0; ky < kheight; ky++) {
                for (kx = 0; kx < kwidth; kx++) {
                    for (int th = 0; th < batch; th++) {
                        for (int ix, iy = ky, ox, oy = 0; oy < outh; iy++, oy++) {
                            for (ix = kx, ox = 0; ox < outw; ix++, ox++) {
                                for (int inch, outch = 0; outch < outchannels; outch++) {
                                    for (inch = 0; inch < inchannels; inch++) {
                                        w[inch, outch, kx, ky] += Quaternion.MulGrad(gy[outch, ox, oy, th], x[inch, ix, iy, th]);
                                    }
                                }
                            }
                        }
                    }
                }
            }

            return w;
        }

        [TestMethod]
        public void ReferenceTest() {
            int inchannels = 8, outchannels = 12, kwidth = 3, kheight = 5, inwidth = 13, inheight = 17;
            int outwidth = inwidth - kwidth + 1, outheight = inheight - kheight + 1, batch = 1;

            float[] xval = (new float[inwidth * inheight * inchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
            float[] yval = (new float[outwidth * outheight * outchannels * batch]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

            Quaternion[] xcval = (new Quaternion[xval.Length / 4])
                .Select((_, idx) => new Quaternion(xval[idx * 4], xval[idx * 4 + 1], xval[idx * 4 + 2], xval[idx * 4 + 3])).ToArray();

            Quaternion[] ycval = (new Quaternion[yval.Length / 4])
                .Select((_, idx) => new Quaternion(yval[idx * 4], yval[idx * 4 + 1], yval[idx * 4 + 2], yval[idx * 4 + 3])).ToArray();

            QuaternionMap2D x = new(inchannels / 4, inwidth, inheight, batch, xcval);
            QuaternionMap2D y = new(outchannels / 4, outwidth, outheight, batch, ycval);

            QuaternionFilter2D gw = Reference(x, y, kwidth, kheight);

            float[] gw_expect = {
                2.174601000e+02f,  -0.000000000e+00f,  -8.734440000e-01f,  -4.367220000e-01f,  2.194312120e+02f,  1.421085472e-14f,
                -8.757320000e-01f,  -4.378660000e-01f,  2.159374360e+02f,  -1.421085472e-14f,  -8.711560000e-01f,  -4.355780000e-01f,
                2.178993960e+02f,  2.131628207e-14f,  -8.734440000e-01f,  -4.367220000e-01f,  2.144147720e+02f,  -7.105427358e-15f,
                -8.688680000e-01f,  -4.344340000e-01f,  2.163675800e+02f,  7.105427358e-15f,  -8.711560000e-01f,  -4.355780000e-01f,
                2.214023240e+02f,  2.131628207e-14f,  -8.780200000e-01f,  -4.390100000e-01f,  2.233734360e+02f,  -1.421085472e-14f,
                -8.803080000e-01f,  -4.401540000e-01f,  2.198613560e+02f,  -2.842170943e-14f,  -8.757320000e-01f,  -4.378660000e-01f,
                2.218233160e+02f,  -2.131628207e-14f,  -8.780200000e-01f,  -4.390100000e-01f,  2.183203880e+02f,  7.105427358e-15f,
                -8.734440000e-01f,  -4.367220000e-01f,  2.202731960e+02f,  -7.105427358e-15f,  -8.757320000e-01f,  -4.378660000e-01f,
                2.253445480e+02f,  1.421085472e-14f,  -8.825960000e-01f,  -4.412980000e-01f,  2.273156600e+02f,  7.105427358e-15f,
                -8.848840000e-01f,  -4.424420000e-01f,  2.237852760e+02f,  2.131628207e-14f,  -8.803080000e-01f,  -4.401540000e-01f,
                2.257472360e+02f,  -1.421085472e-14f,  -8.825960000e-01f,  -4.412980000e-01f,  2.222260040e+02f,  -0.000000000e+00f,
                -8.780200000e-01f,  -4.390100000e-01f,  2.241788120e+02f,  -7.105427358e-15f,  -8.803080000e-01f,  -4.401540000e-01f,
                2.687090120e+02f,  -0.000000000e+00f,  -9.329320000e-01f,  -4.664660000e-01f,  2.706801240e+02f,  -0.000000000e+00f,
                -9.352200000e-01f,  -4.676100000e-01f,  2.669483960e+02f,  -1.421085472e-14f,  -9.306440000e-01f,  -4.653220000e-01f,
                2.689103560e+02f,  -0.000000000e+00f,  -9.329320000e-01f,  -4.664660000e-01f,  2.651877800e+02f,  -0.000000000e+00f,
                -9.283560000e-01f,  -4.641780000e-01f,  2.671405880e+02f,  2.842170943e-14f,  -9.306440000e-01f,  -4.653220000e-01f,
                2.726512360e+02f,  1.421085472e-14f,  -9.375080000e-01f,  -4.687540000e-01f,  2.746223480e+02f,  2.842170943e-14f,
                -9.397960000e-01f,  -4.698980000e-01f,  2.708723160e+02f,  -0.000000000e+00f,  -9.352200000e-01f,  -4.676100000e-01f,
                2.728342760e+02f,  -1.421085472e-14f,  -9.375080000e-01f,  -4.687540000e-01f,  2.690933960e+02f,  -0.000000000e+00f,
                -9.329320000e-01f,  -4.664660000e-01f,  2.710462040e+02f,  -0.000000000e+00f,  -9.352200000e-01f,  -4.676100000e-01f,
                2.765934600e+02f,  1.421085472e-14f,  -9.420840000e-01f,  -4.710420000e-01f,  2.785645720e+02f,  1.421085472e-14f,
                -9.443720000e-01f,  -4.721860000e-01f,  2.747962360e+02f,  4.263256415e-14f,  -9.397960000e-01f,  -4.698980000e-01f,
                2.767581960e+02f,  -0.000000000e+00f,  -9.420840000e-01f,  -4.710420000e-01f,  2.729990120e+02f,  -2.842170943e-14f,
                -9.375080000e-01f,  -4.687540000e-01f,  2.749518200e+02f,  -1.421085472e-14f,  -9.397960000e-01f,  -4.698980000e-01f,
                3.199579240e+02f,  -4.263256415e-14f,  -9.924200000e-01f,  -4.962100000e-01f,  3.219290360e+02f,  -4.263256415e-14f,
                -9.947080000e-01f,  -4.973540000e-01f,  3.179593560e+02f,  1.421085472e-14f,  -9.901320000e-01f,  -4.950660000e-01f,
                3.199213160e+02f,  -0.000000000e+00f,  -9.924200000e-01f,  -4.962100000e-01f,  3.159607880e+02f,  -2.842170943e-14f,
                -9.878440000e-01f,  -4.939220000e-01f,  3.179135960e+02f,  -1.421085472e-14f,  -9.901320000e-01f,  -4.950660000e-01f,
                3.239001480e+02f,  2.842170943e-14f,  -9.969960000e-01f,  -4.984980000e-01f,  3.258712600e+02f,  -2.842170943e-14f,
                -9.992840000e-01f,  -4.996420000e-01f,  3.218832760e+02f,  -2.842170943e-14f,  -9.947080000e-01f,  -4.973540000e-01f,
                3.238452360e+02f,  -0.000000000e+00f,  -9.969960000e-01f,  -4.984980000e-01f,  3.198664040e+02f,  1.421085472e-14f,
                -9.924200000e-01f,  -4.962100000e-01f,  3.218192120e+02f,  -1.421085472e-14f,  -9.947080000e-01f,  -4.973540000e-01f,
                3.278423720e+02f,  -1.421085472e-14f,  -1.001572000e+00f,  -5.007860000e-01f,  3.298134840e+02f,  -5.684341886e-14f,
                -1.003860000e+00f,  -5.019300000e-01f,  3.258071960e+02f,  -0.000000000e+00f,  -9.992840000e-01f,  -4.996420000e-01f,
                3.277691560e+02f,  -0.000000000e+00f,  -1.001572000e+00f,  -5.007860000e-01f,  3.237720200e+02f,  -0.000000000e+00f,
                -9.969960000e-01f,  -4.984980000e-01f,  3.257248280e+02f,  -0.000000000e+00f,  -9.992840000e-01f,  -4.996420000e-01f,
                3.712068360e+02f,  1.421085472e-14f,  -1.051908000e+00f,  -5.259540000e-01f,  3.731779480e+02f,  1.421085472e-14f,
                -1.054196000e+00f,  -5.270980000e-01f,  3.689703160e+02f,  -0.000000000e+00f,  -1.049620000e+00f,  -5.248100000e-01f,
                3.709322760e+02f,  2.842170943e-14f,  -1.051908000e+00f,  -5.259540000e-01f,  3.667337960e+02f,  -0.000000000e+00f,
                -1.047332000e+00f,  -5.236660000e-01f,  3.686866040e+02f,  -1.421085472e-14f,  -1.049620000e+00f,  -5.248100000e-01f,
                3.751490600e+02f,  -4.263256415e-14f,  -1.056484000e+00f,  -5.282420000e-01f,  3.771201720e+02f,  -1.421085472e-14f,
                -1.058772000e+00f,  -5.293860000e-01f,  3.728942360e+02f,  -0.000000000e+00f,  -1.054196000e+00f,  -5.270980000e-01f,
                3.748561960e+02f,  -4.263256415e-14f,  -1.056484000e+00f,  -5.282420000e-01f,  3.706394120e+02f,  -1.421085472e-14f,
                -1.051908000e+00f,  -5.259540000e-01f,  3.725922200e+02f,  1.421085472e-14f,  -1.054196000e+00f,  -5.270980000e-01f,
                3.790912840e+02f,  5.684341886e-14f,  -1.061060000e+00f,  -5.305300000e-01f,  3.810623960e+02f,  2.842170943e-14f,
                -1.063348000e+00f,  -5.316740000e-01f,  3.768181560e+02f,  5.684341886e-14f,  -1.058772000e+00f,  -5.293860000e-01f,
                3.787801160e+02f,  2.842170943e-14f,  -1.061060000e+00f,  -5.305300000e-01f,  3.745450280e+02f,  -0.000000000e+00f,
                -1.056484000e+00f,  -5.282420000e-01f,  3.764978360e+02f,  1.421085472e-14f,  -1.058772000e+00f,  -5.293860000e-01f,
                4.224557480e+02f,  5.684341886e-14f,  -1.111396000e+00f,  -5.556980000e-01f,  4.244268600e+02f,  7.105427358e-14f,
                -1.113684000e+00f,  -5.568420000e-01f,  4.199812760e+02f,  -8.526512829e-14f,  -1.109108000e+00f,  -5.545540000e-01f,
                4.219432360e+02f,  -0.000000000e+00f,  -1.111396000e+00f,  -5.556980000e-01f,  4.175068040e+02f,  -1.421085472e-14f,
                -1.106820000e+00f,  -5.534100000e-01f,  4.194596120e+02f,  -0.000000000e+00f,  -1.109108000e+00f,  -5.545540000e-01f,
                4.263979720e+02f,  -1.421085472e-14f,  -1.115972000e+00f,  -5.579860000e-01f,  4.283690840e+02f,  2.842170943e-14f,
                -1.118260000e+00f,  -5.591300000e-01f,  4.239051960e+02f,  -0.000000000e+00f,  -1.113684000e+00f,  -5.568420000e-01f,
                4.258671560e+02f,  -0.000000000e+00f,  -1.115972000e+00f,  -5.579860000e-01f,  4.214124200e+02f,  -0.000000000e+00f,
                -1.111396000e+00f,  -5.556980000e-01f,  4.233652280e+02f,  -5.684341886e-14f,  -1.113684000e+00f,  -5.568420000e-01f,
                4.303401960e+02f,  -2.842170943e-14f,  -1.120548000e+00f,  -5.602740000e-01f,  4.323113080e+02f,  -2.842170943e-14f,
                -1.122836000e+00f,  -5.614180000e-01f,  4.278291160e+02f,  8.526512829e-14f,  -1.118260000e+00f,  -5.591300000e-01f,
                4.297910760e+02f,  2.842170943e-14f,  -1.120548000e+00f,  -5.602740000e-01f,  4.253180360e+02f,  -1.421085472e-14f,
                -1.115972000e+00f,  -5.579860000e-01f,  4.272708440e+02f,  -5.684341886e-14f,  -1.118260000e+00f,  -5.591300000e-01f,
            };

            float[] gw_actual = gw.ToArray();

            AssertError.Tolerance(gw_expect, gw_actual, 1e-6f, 1e-4f, $"mismatch value {inchannels},{outchannels},{kwidth},{kheight},{inwidth},{inheight},{batch}"); /*many fma tolerance*/
        }
    }
}
