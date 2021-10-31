using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;
using TensorShader;
using TensorShader.Operators.Connection1D;
using TensorShaderCudaBackend.API;

namespace TensorShaderTest.Operators.Connection1D {
    [TestClass]
    public class KernelProductTest {
        [TestMethod]
        public void ExecuteFPTest() {
            TensorShaderCudaBackend.Environment.Precision = TensorShaderCudaBackend.Environment.PrecisionMode.Float;
            TensorShaderCudaBackend.Environment.CudnnEnabled = false;

            float max_err = 0;

            foreach (int batch in new int[] { 1, 2 }) {
                foreach (int inchannels in new int[] { 1, 2, 3, 4, 5, 10, 15, 20, 32, 33 }) {
                    foreach (int outchannels in new int[] { 1, 2, 3, 4, 5, 10, 15, 20, 32, 33 }) {
                        foreach (int kwidth in new int[] { 1, 3, 5 }) {
                            foreach (int inwidth in new int[] { kwidth, kwidth * 2, 8, 9, 13, 17, 25 }) {
                                int outwidth = inwidth - kwidth + 1;

                                float[] xval = (new float[inwidth * inchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
                                float[] gyval = (new float[outwidth * outchannels * batch]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

                                Map1D x = new(inchannels, inwidth, batch, xval);
                                Map1D gy = new(outchannels, outwidth, batch, gyval);

                                Filter1D gw = Reference(x, gy, kwidth);

                                OverflowCheckedTensor x_tensor = new(Shape.Map1D(inchannels, inwidth, batch), xval);
                                OverflowCheckedTensor gy_tensor = new(Shape.Map1D(outchannels, outwidth, batch), gyval);

                                OverflowCheckedTensor gw_tensor = new(Shape.Kernel1D(inchannels, outchannels, kwidth));

                                KernelProduct ope = new(inwidth, inchannels, outchannels, kwidth, batch);

                                ope.Execute(x_tensor, gy_tensor, gw_tensor);

                                float[] gw_expect = gw.ToArray();
                                float[] gw_actual = gw_tensor.State.Value;

                                CollectionAssert.AreEqual(xval, x_tensor.State.Value);
                                CollectionAssert.AreEqual(gyval, gy_tensor.State.Value);

                                AssertError.Tolerance(gw_expect, gw_actual, 1e-6f, 1e-4f, ref max_err, $"mismatch value {inchannels},{outchannels},{kwidth},{inwidth},{batch}");

                                Console.WriteLine($"pass: {inchannels},{outchannels},{kwidth},{inwidth},{batch}");

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
                foreach (int inchannels in new int[] { 1, 2, 3, 4, 5, 10, 15, 20, 32, 33 }) {
                    foreach (int outchannels in new int[] { 1, 2, 3, 4, 5, 10, 15, 20, 32, 33 }) {
                        foreach (int kwidth in new int[] { 1, 3, 5 }) {
                            foreach (int inwidth in new int[] { kwidth, kwidth * 2, 8, 9, 13, 17, 25 }) {
                                int outwidth = inwidth - kwidth + 1;

                                float[] xval = (new float[inwidth * inchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
                                float[] gyval = (new float[outwidth * outchannels * batch]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

                                Map1D x = new(inchannels, inwidth, batch, xval);
                                Map1D gy = new(outchannels, outwidth, batch, gyval);

                                Filter1D gw = Reference(x, gy, kwidth);

                                OverflowCheckedTensor x_tensor = new(Shape.Map1D(inchannels, inwidth, batch), xval);
                                OverflowCheckedTensor gy_tensor = new(Shape.Map1D(outchannels, outwidth, batch), gyval);

                                OverflowCheckedTensor gw_tensor = new(Shape.Kernel1D(inchannels, outchannels, kwidth));

                                KernelProduct ope = new(inwidth, inchannels, outchannels, kwidth, batch);

                                ope.Execute(x_tensor, gy_tensor, gw_tensor);

                                float[] gw_expect = gw.ToArray();
                                float[] gw_actual = gw_tensor.State.Value;

                                CollectionAssert.AreEqual(xval, x_tensor.State.Value);
                                CollectionAssert.AreEqual(gyval, gy_tensor.State.Value);

                                AssertError.Tolerance(gw_expect, gw_actual, 1e-7f, 1e-5f, ref max_err, $"mismatch value {inchannels},{outchannels},{kwidth},{inwidth},{batch}");

                                Console.WriteLine($"pass: {inchannels},{outchannels},{kwidth},{inwidth},{batch}");

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
                foreach (int inchannels in new int[] { 1, 2, 3, 4, 5, 10, 15, 20, 32, 33 }) {
                    foreach (int outchannels in new int[] { 1, 2, 3, 4, 5, 10, 15, 20, 32, 33 }) {
                        foreach (int kwidth in new int[] { 1, 3, 5 }) {
                            foreach (int inwidth in new int[] { kwidth, kwidth * 2, 8, 9, 13, 17, 25 }) {
                                int outwidth = inwidth - kwidth + 1;

                                float[] xval = (new float[inwidth * inchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
                                float[] gyval = (new float[outwidth * outchannels * batch]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

                                Map1D x = new(inchannels, inwidth, batch, xval);
                                Map1D gy = new(outchannels, outwidth, batch, gyval);

                                Filter1D gw = Reference(x, gy, kwidth);

                                OverflowCheckedTensor x_tensor = new(Shape.Map1D(inchannels, inwidth, batch), xval);
                                OverflowCheckedTensor gy_tensor = new(Shape.Map1D(outchannels, outwidth, batch), gyval);

                                OverflowCheckedTensor gw_tensor = new(Shape.Kernel1D(inchannels, outchannels, kwidth));

                                KernelProduct ope = new(inwidth, inchannels, outchannels, kwidth, batch);

                                ope.Execute(x_tensor, gy_tensor, gw_tensor);

                                float[] gw_expect = gw.ToArray();
                                float[] gw_actual = gw_tensor.State.Value;

                                CollectionAssert.AreEqual(xval, x_tensor.State.Value);
                                CollectionAssert.AreEqual(gyval, gy_tensor.State.Value);

                                AssertError.Tolerance(gw_expect, gw_actual, 1e-6f, 1e-4f, ref max_err, $"mismatch value {inchannels},{outchannels},{kwidth},{inwidth},{batch}");

                                Console.WriteLine($"pass: {inchannels},{outchannels},{kwidth},{inwidth},{batch}");

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
            int kwidth = 5;
            int inwidth = 125;
            int outwidth = inwidth - kwidth + 1;

            float[] xval = (new float[inwidth * inchannels * batch]).Select((_, idx) => (float)random.NextDouble() * 1e-2f).ToArray();
            float[] gyval = (new float[outwidth * outchannels * batch]).Select((_, idx) => (float)random.NextDouble() * 1e-2f).ToArray();

            Map1D x = new(inchannels, inwidth, batch, xval);
            Map1D gy = new(outchannels, outwidth, batch, gyval);

            Filter1D gw = Reference(x, gy, kwidth);

            OverflowCheckedTensor x_tensor = new(Shape.Map1D(inchannels, inwidth, batch), xval);
            OverflowCheckedTensor gy_tensor = new(Shape.Map1D(outchannels, outwidth, batch), gyval);

            OverflowCheckedTensor gw_tensor = new(Shape.Kernel1D(inchannels, outchannels, kwidth));

            KernelProduct ope = new(inwidth, inchannels, outchannels, kwidth, batch);

            ope.Execute(x_tensor, gy_tensor, gw_tensor);

            float[] gw_expect = gw.ToArray();
            float[] gw_actual = gw_tensor.State.Value;

            CollectionAssert.AreEqual(xval, x_tensor.State.Value);
            CollectionAssert.AreEqual(gyval, gy_tensor.State.Value);

            AssertError.Tolerance(gw_expect, gw_actual, 1e-7f, 1e-5f, ref max_err, $"mismatch value {inchannels},{outchannels},{kwidth},{inwidth},{batch}");

            Console.WriteLine($"pass: {inchannels},{outchannels},{kwidth},{inwidth},{batch}");

            Console.WriteLine($"maxerr:{max_err}");
        }

        [TestMethod]
        public void SpeedFPTest() {
            TensorShaderCudaBackend.Environment.Precision = TensorShaderCudaBackend.Environment.PrecisionMode.Float;
            TensorShaderCudaBackend.Environment.CudnnEnabled = false;

            int inwidth = 512, inchannels = 32, outchannels = 32, ksize = 3;
            int outwidth = inwidth - ksize + 1;

            OverflowCheckedTensor x_tensor = new(Shape.Map1D(inchannels, inwidth));
            OverflowCheckedTensor gy_tensor = new(Shape.Map1D(outchannels, outwidth));

            OverflowCheckedTensor gw_tensor = new(Shape.Kernel1D(inchannels, outchannels, ksize));

            KernelProduct ope = new(inwidth, inchannels, outchannels, ksize);

            Cuda.Profiler.Initialize("../../../../profiler.nvsetting", "../../nvprofiles/kernelproduct_1d_fp.nvvp");
            Cuda.Profiler.Start();

            ope.Execute(x_tensor, gy_tensor, gw_tensor);

            Cuda.Profiler.Stop();
        }

        [TestMethod]
        public void SpeedFFPTest() {
            TensorShaderCudaBackend.Environment.CudnnEnabled = false;
            TensorShaderCudaBackend.Environment.Precision = TensorShaderCudaBackend.Environment.PrecisionMode.FloatFloat;

            int inwidth = 512, inchannels = 32, outchannels = 32, ksize = 3;
            int outwidth = inwidth - ksize + 1;

            OverflowCheckedTensor x_tensor = new(Shape.Map1D(inchannels, inwidth));
            OverflowCheckedTensor gy_tensor = new(Shape.Map1D(outchannels, outwidth));

            OverflowCheckedTensor gw_tensor = new(Shape.Kernel1D(inchannels, outchannels, ksize));

            KernelProduct ope = new(inwidth, inchannels, outchannels, ksize);

            Cuda.Profiler.Initialize("../../../../profiler.nvsetting", "../../nvprofiles/kernelproduct_1d_ffp.nvvp");
            Cuda.Profiler.Start();

            ope.Execute(x_tensor, gy_tensor, gw_tensor);

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

            int inwidth = 512, inchannels = 32, outchannels = 32, ksize = 3;
            int outwidth = inwidth - ksize + 1;

            OverflowCheckedTensor x_tensor = new(Shape.Map1D(inchannels, inwidth));
            OverflowCheckedTensor gy_tensor = new(Shape.Map1D(outchannels, outwidth));

            OverflowCheckedTensor gw_tensor = new(Shape.Kernel1D(inchannels, outchannels, ksize));

            KernelProduct ope = new(inwidth, inchannels, outchannels, ksize);

            Cuda.Profiler.Initialize("../../../../profiler.nvsetting", "../../nvprofiles/kernelproduct_1d_cudnn.nvvp");
            Cuda.Profiler.Start();

            ope.Execute(x_tensor, gy_tensor, gw_tensor);

            Cuda.Profiler.Stop();
        }

        public static Filter1D Reference(Map1D x, Map1D gy, int kwidth) {
            int inchannels = x.Channels, outchannels = gy.Channels, batch = x.Batch;
            int inw = x.Width, outw = gy.Width;

            if (outw != inw - kwidth + 1) {
                throw new ArgumentException("mismatch shape");
            }

            Filter1D w = new(inchannels, outchannels, kwidth);

            for (int kx = 0; kx < kwidth; kx++) {
                for (int th = 0; th < batch; th++) {
                    for (int ix = kx, ox = 0; ox < outw; ix++, ox++) {
                        for (int inch, outch = 0; outch < outchannels; outch++) {
                            for (inch = 0; inch < inchannels; inch++) {
                                w[inch, outch, kx] += x[inch, ix, th] * gy[outch, ox, th];
                            }
                        }
                    }
                }
            }

            return w;
        }

        [TestMethod]
        public void ReferenceTest() {
            int inchannels = 7, outchannels = 11, kwidth = 3, inwidth = 13;
            int outwidth = inwidth - kwidth + 1;

            float[] xval = (new float[inwidth * inchannels]).Select((_, idx) => idx * 1e-3f).ToArray();
            float[] gyval = (new float[outwidth * outchannels]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

            Map1D x = new(inchannels, inwidth, 1, xval);
            Map1D gy = new(outchannels, outwidth, 1, gyval);

            Filter1D gw = Reference(x, gy, kwidth);

            float[] gw_expect = {
                1.655500196e-02f, 1.727000065e-02f, 1.798500121e-02f, 1.870000176e-02f, 1.941500232e-02f, 2.013000101e-02f, 2.084500156e-02f,
                1.617000252e-02f, 1.687400229e-02f, 1.757800020e-02f, 1.828200184e-02f, 1.898600161e-02f, 1.969000138e-02f, 2.039400116e-02f,
                1.578499936e-02f, 1.647800207e-02f, 1.717100292e-02f, 1.786400191e-02f, 1.855700277e-02f, 1.925000176e-02f, 1.994300261e-02f,
                1.540000085e-02f, 1.608200185e-02f, 1.676400378e-02f, 1.744600199e-02f, 1.812800020e-02f, 1.881000213e-02f, 1.949200220e-02f,
                1.501500141e-02f, 1.568600163e-02f, 1.635700092e-02f, 1.702800021e-02f, 1.769900322e-02f, 1.837000065e-02f, 1.904100180e-02f,
                1.463000197e-02f, 1.529000141e-02f, 1.595000364e-02f, 1.661000215e-02f, 1.727000065e-02f, 1.793000288e-02f, 1.859000139e-02f,
                1.424500067e-02f, 1.489400119e-02f, 1.554300170e-02f, 1.619200036e-02f, 1.684100181e-02f, 1.748999953e-02f, 1.813900098e-02f,
                1.386000216e-02f, 1.449800096e-02f, 1.513600163e-02f, 1.577400044e-02f, 1.641200110e-02f, 1.704999991e-02f, 1.768800244e-02f,
                1.347500179e-02f, 1.410200167e-02f, 1.472900156e-02f, 1.535600144e-02f, 1.598300040e-02f, 1.661000215e-02f, 1.723700017e-02f,
                1.309000142e-02f, 1.370600238e-02f, 1.432200149e-02f, 1.493800152e-02f, 1.555400062e-02f, 1.617000066e-02f, 1.678599976e-02f,
                1.270500105e-02f, 1.331000030e-02f, 1.391500328e-02f, 1.452000160e-02f, 1.512500178e-02f, 1.573000103e-02f, 1.633500308e-02f,
                2.156000212e-02f, 2.227500267e-02f, 2.299000323e-02f, 2.370500192e-02f, 2.442000061e-02f, 2.513500303e-02f, 2.585000359e-02f,
                2.109800279e-02f, 2.180200256e-02f, 2.250600234e-02f, 2.321000211e-02f, 2.391400188e-02f, 2.461800165e-02f, 2.532200143e-02f,
                2.063600160e-02f, 2.132900059e-02f, 2.202200145e-02f, 2.271500044e-02f, 2.340800315e-02f, 2.410100028e-02f, 2.479400113e-02f,
                2.017400041e-02f, 2.085600235e-02f, 2.153800428e-02f, 2.222000249e-02f, 2.290200070e-02f, 2.358400449e-02f, 2.426600456e-02f,
                1.971199922e-02f, 2.038300410e-02f, 2.105400153e-02f, 2.172500081e-02f, 2.239600196e-02f, 2.306699939e-02f, 2.373800240e-02f,
                1.924999803e-02f, 1.991000213e-02f, 2.057000250e-02f, 2.123000100e-02f, 2.189000323e-02f, 2.255000174e-02f, 2.321000211e-02f,
                1.878800057e-02f, 1.943700202e-02f, 2.008600160e-02f, 2.073500119e-02f, 2.138400078e-02f, 2.203300223e-02f, 2.268200181e-02f,
                1.832600310e-02f, 1.896400377e-02f, 1.960200071e-02f, 2.024000138e-02f, 2.087800391e-02f, 2.151600085e-02f, 2.215400152e-02f,
                1.786400191e-02f, 1.849100180e-02f, 1.911800355e-02f, 1.974500343e-02f, 2.037200145e-02f, 2.099899948e-02f, 2.162600309e-02f,
                1.740200259e-02f, 1.801800169e-02f, 1.863400266e-02f, 1.925000176e-02f, 1.986600272e-02f, 2.048199996e-02f, 2.109800093e-02f,
                1.694000326e-02f, 1.754500158e-02f, 1.815000363e-02f, 1.875500195e-02f, 1.936000027e-02f, 1.996500231e-02f, 2.057000250e-02f,
                2.656500228e-02f, 2.728000470e-02f, 2.799500339e-02f, 2.871000394e-02f, 2.942500077e-02f, 3.014000319e-02f, 3.085500188e-02f,
                2.602600120e-02f, 2.673000284e-02f, 2.743400075e-02f, 2.813800424e-02f, 2.884200402e-02f, 2.954600379e-02f, 3.025000170e-02f,
                2.548700199e-02f, 2.618000284e-02f, 2.687300183e-02f, 2.756600082e-02f, 2.825899981e-02f, 2.895200253e-02f, 2.964500338e-02f,
                2.494800277e-02f, 2.563000284e-02f, 2.631200291e-02f, 2.699400112e-02f, 2.767600305e-02f, 2.835800499e-02f, 2.904000506e-02f,
                2.440900356e-02f, 2.508000098e-02f, 2.575100400e-02f, 2.642200328e-02f, 2.709300257e-02f, 2.776400186e-02f, 2.843500301e-02f,
                2.387000062e-02f, 2.453000285e-02f, 2.519000322e-02f, 2.585000172e-02f, 2.651000209e-02f, 2.717000432e-02f, 2.783000283e-02f,
                2.333100326e-02f, 2.398000099e-02f, 2.462900244e-02f, 2.527800389e-02f, 2.592700347e-02f, 2.657599933e-02f, 2.722500451e-02f,
                2.279200032e-02f, 2.343000099e-02f, 2.406800352e-02f, 2.470600046e-02f, 2.534400299e-02f, 2.598200366e-02f, 2.662000433e-02f,
                2.225300111e-02f, 2.288000099e-02f, 2.350700088e-02f, 2.413400449e-02f, 2.476100065e-02f, 2.538800240e-02f, 2.601500414e-02f,
                2.171400189e-02f, 2.233000100e-02f, 2.294600196e-02f, 2.356200106e-02f, 2.417800389e-02f, 2.479400299e-02f, 2.541000023e-02f,
                2.117500454e-02f, 2.177999914e-02f, 2.238500305e-02f, 2.299000137e-02f, 2.359500155e-02f, 2.420000173e-02f, 2.480500005e-02f,
            };

            float[] gw_actual = gw.ToArray();

            AssertError.Tolerance(gw_expect, gw_actual, 1e-7f, 1e-5f, $"mismatch value {inchannels},{outchannels},{kwidth},{inwidth}");
        }
    }
}
