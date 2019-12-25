using System;
using System.Diagnostics;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using TensorShader.Operators.ComplexConvolution;
using TensorShaderCudaBackend.API;

namespace TensorShaderTest.Operators.Complex {
    [TestClass]
    public class ComplexKernelProductDenseTest {
        [TestMethod]
        public void ExecuteTest() {
            Random random = new Random(1234);

            float max_err = 0;

            foreach (int batch in new int[] { 1, 2 }) {
                foreach (int inchannels in new int[] { 2, 4, 10, 20, 32, 34 }) {
                    foreach (int outchannels in new int[] { 2, 4, 10, 20, 32, 34 }) {

                        float[] xval = (new float[inchannels * batch]).Select((_, idx) => (float)random.NextDouble() * 1e-2f).ToArray();
                        float[] yval = (new float[outchannels * batch]).Select((_, idx) => (float)random.NextDouble() * 1e-2f).ToArray();

                        System.Numerics.Complex[] xcval = (new System.Numerics.Complex[xval.Length / 2])
                            .Select((_, idx) => new System.Numerics.Complex(xval[idx * 2], xval[idx * 2 + 1])).ToArray();

                        System.Numerics.Complex[] ycval = (new System.Numerics.Complex[yval.Length / 2])
                            .Select((_, idx) => new System.Numerics.Complex(yval[idx * 2], yval[idx * 2 + 1])).ToArray();

                        ComplexMap0D x = new ComplexMap0D(inchannels / 2, batch, xcval);
                        ComplexMap0D y = new ComplexMap0D(outchannels / 2, batch, ycval);

                        ComplexFilter0D gw = Reference(x, y);

                        OverflowCheckedTensor x_tensor = new OverflowCheckedTensor(Shape.Map0D(inchannels, batch), xval);
                        OverflowCheckedTensor y_tensor = new OverflowCheckedTensor(Shape.Map0D(outchannels, batch), yval);

                        OverflowCheckedTensor gw_tensor = new OverflowCheckedTensor(Shape.Kernel0D(inchannels, outchannels / 2));

                        ComplexKernelProductDense ope = new ComplexKernelProductDense(inchannels, outchannels, transpose: false, batch);

                        ope.Execute(x_tensor, y_tensor, gw_tensor);

                        float[] gw_expect = gw.ToArray();
                        float[] gw_actual = gw_tensor.State;

                        CollectionAssert.AreEqual(xval, x_tensor.State);
                        CollectionAssert.AreEqual(yval, y_tensor.State);

                        AssertError.Tolerance(gw_expect, gw_actual, 1e-7f, 1e-5f, ref max_err, $"mismatch value {inchannels},{outchannels},{batch}");

                        Console.WriteLine($"pass: {inchannels},{outchannels},{batch}");

                    }
                }
            }

            Console.WriteLine($"maxerr:{max_err}");
        }

        [TestMethod]
        public void SpeedTest() {
            int inchannels = 32, outchannels = 32;

            OverflowCheckedTensor x_tensor = new OverflowCheckedTensor(Shape.Map0D(inchannels));
            OverflowCheckedTensor y_tensor = new OverflowCheckedTensor(Shape.Map0D(outchannels));

            OverflowCheckedTensor gw_tensor = new OverflowCheckedTensor(Shape.Kernel0D(inchannels, outchannels / 2));

            ComplexKernelProductDense ope = new ComplexKernelProductDense(inchannels, outchannels);

            Cuda.Profiler.Initialize("../../../profiler.nvsetting", "../../nvprofiles/complex_kernelproduct_dense.nvvp");
            Cuda.Profiler.Start();

            ope.Execute(x_tensor, y_tensor, gw_tensor);
            
            Cuda.Profiler.Stop();
        }

        public static ComplexFilter0D Reference(ComplexMap0D x, ComplexMap0D gy) {
            int inchannels = x.Channels, outchannels = gy.Channels, batch = x.Batch;

            ComplexFilter0D w = new ComplexFilter0D(inchannels, outchannels);

            Func<System.Numerics.Complex, System.Numerics.Complex, System.Numerics.Complex> mul_grad = (z1, z2) => {
                return new System.Numerics.Complex(z1.Real * z2.Real + z1.Imaginary * z2.Imaginary, z1.Imaginary * z2.Real - z1.Real * z2.Imaginary);
            };

            for (int inch, outch = 0; outch < outchannels; outch++) {
                for (inch = 0; inch < inchannels; inch++) {
                    System.Numerics.Complex sum = 0;
                    for (int th = 0; th < batch; th++) {
                        sum += mul_grad(gy[outch, th], x[inch, th]);
                    }

                    w[inch, outch] = sum;
                }

            }

            return w;
        }

        [TestMethod]
        public void ReferenceTest() {
            int inchannels = 6, outchannels = 8, batch = 3;

            float[] xval = (new float[inchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
            float[] yval = (new float[outchannels * batch]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

            System.Numerics.Complex[] xcval = (new System.Numerics.Complex[xval.Length / 2])
                .Select((_, idx) => new System.Numerics.Complex(xval[idx * 2], xval[idx * 2 + 1])).ToArray();

            System.Numerics.Complex[] ycval = (new System.Numerics.Complex[yval.Length / 2])
                .Select((_, idx) => new System.Numerics.Complex(yval[idx * 2], yval[idx * 2 + 1])).ToArray();

            ComplexMap0D x = new ComplexMap0D(inchannels / 2, batch, xcval);
            ComplexMap0D y = new ComplexMap0D(outchannels / 2, batch, ycval);

            ComplexFilter0D gw = Reference(x, y);

            float[] gw_expect = {
                3.720000000e-04f,  -6.300000000e-05f,  5.460000000e-04f,  -6.900000000e-05f,  7.200000000e-04f,  -7.500000000e-05f,
                2.940000000e-04f,  -5.700000000e-05f,  4.440000000e-04f,  -6.300000000e-05f,  5.940000000e-04f,  -6.900000000e-05f,
                2.160000000e-04f,  -5.100000000e-05f,  3.420000000e-04f,  -5.700000000e-05f,  4.680000000e-04f,  -6.300000000e-05f,
                1.380000000e-04f,  -4.500000000e-05f,  2.400000000e-04f,  -5.100000000e-05f,  3.420000000e-04f,  -5.700000000e-05f,
            };

            float[] gw_actual = gw.ToArray();

            AssertError.Tolerance(gw_expect, gw_actual, 1e-9f, 1e-5f, $"mismatch value {inchannels},{outchannels},{batch}");
        }
    }
}
