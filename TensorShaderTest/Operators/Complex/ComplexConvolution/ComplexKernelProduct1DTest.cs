using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;
using TensorShader;
using TensorShader.Operators.ComplexConvolution;
using TensorShaderCudaBackend.API;

namespace TensorShaderTest.Operators.Complex {
    [TestClass]
    public class ComplexKernelProduct1DTest {
        [TestMethod]
        public void ExecuteFPTest() {
            TensorShaderCudaBackend.Environment.Precision = TensorShaderCudaBackend.Environment.PrecisionMode.Float;
            TensorShaderCudaBackend.Environment.CudnnEnabled = false;

            Random random = new(1234);

            float max_err = 0;

            foreach (int batch in new int[] { 1, 2 }) {
                foreach (int inchannels in new int[] { 2, 4, 10, 20, 32, 34 }) {
                    foreach (int outchannels in new int[] { 2, 4, 10, 20, 32, 34 }) {
                        foreach (int kwidth in new int[] { 1, 3, 5 }) {
                            foreach (int inwidth in new int[] { kwidth, kwidth * 2, 8, 9, 13, 17, 25 }) {
                                int outwidth = inwidth - kwidth + 1;

                                float[] xval = (new float[inwidth * inchannels * batch]).Select((_, idx) => (float)random.NextDouble() * 1e-2f).ToArray();
                                float[] yval = (new float[outwidth * outchannels * batch]).Select((_, idx) => (float)random.NextDouble() * 1e-2f).ToArray();

                                System.Numerics.Complex[] xcval = (new System.Numerics.Complex[xval.Length / 2])
                                    .Select((_, idx) => new System.Numerics.Complex(xval[idx * 2], xval[idx * 2 + 1])).ToArray();

                                System.Numerics.Complex[] ycval = (new System.Numerics.Complex[yval.Length / 2])
                                    .Select((_, idx) => new System.Numerics.Complex(yval[idx * 2], yval[idx * 2 + 1])).ToArray();

                                ComplexMap1D x = new(inchannels / 2, inwidth, batch, xcval);
                                ComplexMap1D y = new(outchannels / 2, outwidth, batch, ycval);

                                ComplexFilter1D gw = Reference(x, y, kwidth);

                                OverflowCheckedTensor x_tensor = new(Shape.Map1D(inchannels, inwidth, batch), xval);
                                OverflowCheckedTensor y_tensor = new(Shape.Map1D(outchannels, outwidth, batch), yval);

                                OverflowCheckedTensor gw_tensor = new(Shape.Kernel1D(inchannels, outchannels / 2, kwidth));

                                ComplexKernelProduct1D ope = new(inwidth, inchannels, outchannels, kwidth, transpose: false, batch);

                                ope.Execute(x_tensor, y_tensor, gw_tensor);

                                float[] gw_expect = gw.ToArray();
                                float[] gw_actual = gw_tensor.State.Value;

                                CollectionAssert.AreEqual(xval, x_tensor.State.Value);
                                CollectionAssert.AreEqual(yval, y_tensor.State.Value);

                                AssertError.Tolerance(gw_expect, gw_actual, 1e-6f, 1e-4f, ref max_err, $"mismatch value {inchannels},{outchannels},{kwidth},{inwidth},{batch}");

                                Console.WriteLine($"pass: {inchannels},{outchannels},{kwidth},{inwidth},{batch}");

                            }
                        }
                    }
                }
            }

            Console.WriteLine($"maxerr:{max_err}");
        }

        public void ExecuteFFPTest() {
            TensorShaderCudaBackend.Environment.CudnnEnabled = false;
            TensorShaderCudaBackend.Environment.Precision = TensorShaderCudaBackend.Environment.PrecisionMode.FloatFloat;

            Random random = new(1234);

            float max_err = 0;

            foreach (int batch in new int[] { 1, 2 }) {
                foreach (int inchannels in new int[] { 2, 4, 10, 20, 32, 34 }) {
                    foreach (int outchannels in new int[] { 2, 4, 10, 20, 32, 34 }) {
                        foreach (int kwidth in new int[] { 1, 3, 5 }) {
                            foreach (int inwidth in new int[] { kwidth, kwidth * 2, 8, 9, 13, 17, 25 }) {
                                int outwidth = inwidth - kwidth + 1;

                                float[] xval = (new float[inwidth * inchannels * batch]).Select((_, idx) => (float)random.NextDouble() * 1e-2f).ToArray();
                                float[] yval = (new float[outwidth * outchannels * batch]).Select((_, idx) => (float)random.NextDouble() * 1e-2f).ToArray();

                                System.Numerics.Complex[] xcval = (new System.Numerics.Complex[xval.Length / 2])
                                    .Select((_, idx) => new System.Numerics.Complex(xval[idx * 2], xval[idx * 2 + 1])).ToArray();

                                System.Numerics.Complex[] ycval = (new System.Numerics.Complex[yval.Length / 2])
                                    .Select((_, idx) => new System.Numerics.Complex(yval[idx * 2], yval[idx * 2 + 1])).ToArray();

                                ComplexMap1D x = new(inchannels / 2, inwidth, batch, xcval);
                                ComplexMap1D y = new(outchannels / 2, outwidth, batch, ycval);

                                ComplexFilter1D gw = Reference(x, y, kwidth);

                                OverflowCheckedTensor x_tensor = new(Shape.Map1D(inchannels, inwidth, batch), xval);
                                OverflowCheckedTensor y_tensor = new(Shape.Map1D(outchannels, outwidth, batch), yval);

                                OverflowCheckedTensor gw_tensor = new(Shape.Kernel1D(inchannels, outchannels / 2, kwidth));

                                ComplexKernelProduct1D ope = new(inwidth, inchannels, outchannels, kwidth, transpose: false, batch);

                                ope.Execute(x_tensor, y_tensor, gw_tensor);

                                float[] gw_expect = gw.ToArray();
                                float[] gw_actual = gw_tensor.State.Value;

                                CollectionAssert.AreEqual(xval, x_tensor.State.Value);
                                CollectionAssert.AreEqual(yval, y_tensor.State.Value);

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
        public void LargeMapTest() {
            TensorShaderCudaBackend.Environment.CudnnEnabled = false;
            TensorShaderCudaBackend.Environment.Precision = TensorShaderCudaBackend.Environment.PrecisionMode.FloatFloat;

            Random random = new(1234);

            float max_err = 0;

            int batch = 3;
            int inchannels = 96;
            int outchannels = 98;
            int kwidth = 5;
            int inwidth = 125;
            int outwidth = inwidth - kwidth + 1;

            float[] xval = (new float[inwidth * inchannels * batch]).Select((_, idx) => (float)random.NextDouble() * 1e-2f).ToArray();
            float[] yval = (new float[outwidth * outchannels * batch]).Select((_, idx) => (float)random.NextDouble() * 1e-2f).ToArray();

            System.Numerics.Complex[] xcval = (new System.Numerics.Complex[xval.Length / 2])
                .Select((_, idx) => new System.Numerics.Complex(xval[idx * 2], xval[idx * 2 + 1])).ToArray();

            System.Numerics.Complex[] ycval = (new System.Numerics.Complex[yval.Length / 2])
                .Select((_, idx) => new System.Numerics.Complex(yval[idx * 2], yval[idx * 2 + 1])).ToArray();

            ComplexMap1D x = new(inchannels / 2, inwidth, batch, xcval);
            ComplexMap1D y = new(outchannels / 2, outwidth, batch, ycval);

            ComplexFilter1D gw = Reference(x, y, kwidth);

            OverflowCheckedTensor x_tensor = new(Shape.Map1D(inchannels, inwidth, batch), xval);
            OverflowCheckedTensor y_tensor = new(Shape.Map1D(outchannels, outwidth, batch), yval);

            OverflowCheckedTensor gw_tensor = new(Shape.Kernel1D(inchannels, outchannels / 2, kwidth));

            ComplexKernelProduct1D ope = new(inwidth, inchannels, outchannels, kwidth, transpose: false, batch);

            ope.Execute(x_tensor, y_tensor, gw_tensor);

            float[] gw_expect = gw.ToArray();
            float[] gw_actual = gw_tensor.State.Value;

            CollectionAssert.AreEqual(xval, x_tensor.State.Value);
            CollectionAssert.AreEqual(yval, y_tensor.State.Value);

            AssertError.Tolerance(gw_expect, gw_actual, 1e-6f, 1e-4f, ref max_err, $"mismatch value {inchannels},{outchannels},{kwidth},{inwidth},{batch}");

            Console.WriteLine($"pass: {inchannels},{outchannels},{kwidth},{inwidth},{batch}");

            Console.WriteLine($"maxerr:{max_err}");
        }

        [TestMethod]
        public void SpeedTest() {
            int inwidth = 512, inchannels = 32, outchannels = 32, ksize = 3;
            int outwidth = inwidth - ksize + 1;

            OverflowCheckedTensor x_tensor = new(Shape.Map1D(inchannels, inwidth));
            OverflowCheckedTensor y_tensor = new(Shape.Map1D(outchannels, outwidth));

            OverflowCheckedTensor gw_tensor = new(Shape.Kernel1D(inchannels, outchannels / 2, ksize));

            ComplexKernelProduct1D ope = new(inwidth, inchannels, outchannels, ksize);

            Cuda.Profiler.Initialize("../../../../profiler.nvsetting", "../../nvprofiles/complex_kernelproduct_1d.nvvp");
            Cuda.Profiler.Start();

            ope.Execute(x_tensor, y_tensor, gw_tensor);

            Cuda.Profiler.Stop();
        }

        public static ComplexFilter1D Reference(ComplexMap1D x, ComplexMap1D gy, int kwidth) {
            int inchannels = x.Channels, outchannels = gy.Channels, batch = x.Batch;
            int inw = x.Width, outw = gy.Width;

            if (outw != inw - kwidth + 1) {
                throw new ArgumentException("mismatch shape");
            }

            ComplexFilter1D w = new(inchannels, outchannels, kwidth);

            Func<System.Numerics.Complex, System.Numerics.Complex, System.Numerics.Complex> mul_grad = (z1, z2) => {
                return new System.Numerics.Complex(z1.Real * z2.Real + z1.Imaginary * z2.Imaginary, z1.Imaginary * z2.Real - z1.Real * z2.Imaginary);
            };

            for (int kx = 0; kx < kwidth; kx++) {
                for (int th = 0; th < batch; th++) {
                    for (int ix = kx, ox = 0; ox < outw; ix++, ox++) {
                        for (int inch, outch = 0; outch < outchannels; outch++) {
                            for (inch = 0; inch < inchannels; inch++) {
                                w[inch, outch, kx] += mul_grad(gy[outch, ox, th], x[inch, ix, th]);
                            }
                        }
                    }
                }
            }

            return w;
        }

        [TestMethod]
        public void ReferenceTest() {
            int inchannels = 6, outchannels = 8, kwidth = 3, inwidth = 13;
            int outwidth = inwidth - kwidth + 1, batch = 1;

            float[] xval = (new float[inwidth * inchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
            float[] yval = (new float[outwidth * outchannels * batch]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

            System.Numerics.Complex[] xcval = (new System.Numerics.Complex[xval.Length / 2])
                .Select((_, idx) => new System.Numerics.Complex(xval[idx * 2], xval[idx * 2 + 1])).ToArray();

            System.Numerics.Complex[] ycval = (new System.Numerics.Complex[yval.Length / 2])
                .Select((_, idx) => new System.Numerics.Complex(yval[idx * 2], yval[idx * 2 + 1])).ToArray();

            ComplexMap1D x = new(inchannels / 2, inwidth, batch, xcval);
            ComplexMap1D y = new(outchannels / 2, outwidth, batch, ycval);

            ComplexFilter1D gw = Reference(x, y, kwidth);

            float[] gw_expect = {
                2.063600000e-02f,  -8.470000000e-04f,  2.268200000e-02f,  -8.690000000e-04f,  2.472800000e-02f,  -8.910000000e-04f,
                1.929400000e-02f,  -8.250000000e-04f,  2.125200000e-02f,  -8.470000000e-04f,  2.321000000e-02f,  -8.690000000e-04f,
                1.795200000e-02f,  -8.030000000e-04f,  1.982200000e-02f,  -8.250000000e-04f,  2.169200000e-02f,  -8.470000000e-04f,
                1.661000000e-02f,  -7.810000000e-04f,  1.839200000e-02f,  -8.030000000e-04f,  2.017400000e-02f,  -8.250000000e-04f,
                2.677400000e-02f,  -9.130000000e-04f,  2.882000000e-02f,  -9.350000000e-04f,  3.086600000e-02f,  -9.570000000e-04f,
                2.516800000e-02f,  -8.910000000e-04f,  2.712600000e-02f,  -9.130000000e-04f,  2.908400000e-02f,  -9.350000000e-04f,
                2.356200000e-02f,  -8.690000000e-04f,  2.543200000e-02f,  -8.910000000e-04f,  2.730200000e-02f,  -9.130000000e-04f,
                2.195600000e-02f,  -8.470000000e-04f,  2.373800000e-02f,  -8.690000000e-04f,  2.552000000e-02f,  -8.910000000e-04f,
                3.291200000e-02f,  -9.790000000e-04f,  3.495800000e-02f,  -1.001000000e-03f,  3.700400000e-02f,  -1.023000000e-03f,
                3.104200000e-02f,  -9.570000000e-04f,  3.300000000e-02f,  -9.790000000e-04f,  3.495800000e-02f,  -1.001000000e-03f,
                2.917200000e-02f,  -9.350000000e-04f,  3.104200000e-02f,  -9.570000000e-04f,  3.291200000e-02f,  -9.790000000e-04f,
                2.730200000e-02f,  -9.130000000e-04f,  2.908400000e-02f,  -9.350000000e-04f,  3.086600000e-02f,  -9.570000000e-04f,
            };

            float[] gw_actual = gw.ToArray();

            AssertError.Tolerance(gw_expect, gw_actual, 1e-7f, 1e-5f, $"mismatch value {inchannels},{outchannels},{kwidth},{inwidth},{batch}");
        }
    }
}
