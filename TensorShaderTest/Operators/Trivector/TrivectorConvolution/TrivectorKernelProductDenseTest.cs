using System;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using TensorShader.Operators.TrivectorConvolution;
using TensorShaderCudaBackend.API;

namespace TensorShaderTest.Operators.Trivector {
    [TestClass]
    public class TrivectorKernelProductDenseTest {
        [TestMethod]
        public void ExecuteTest() {
            Random random = new Random(1234);

            float max_err = 0;

            foreach (int batch in new int[] { 1, 2 }) {
                foreach (int inchannels in new int[] { 3, 6, 9, 15, 21, 33 }) {
                    foreach (int outchannels in new int[] { 3, 6, 9, 15, 21, 33 }) {

                        float[] xval = (new float[inchannels * batch]).Select((_, idx) => (float)random.NextDouble() * 1e-2f).ToArray();
                        float[] yval = (new float[outchannels * batch]).Select((_, idx) => (float)random.NextDouble() * 1e-2f).ToArray();
                        float[] wval = (new float[inchannels * outchannels / 9 * 4]).Select((_, idx) => (float)random.NextDouble() * 1e-2f).ToArray();

                        Trivector[] xcval = (new Trivector[xval.Length / 3])
                            .Select((_, idx) => new Trivector(xval[idx * 3], xval[idx * 3 + 1], xval[idx * 3 + 2])).ToArray();

                        Trivector[] ycval = (new Trivector[yval.Length / 3])
                            .Select((_, idx) => new Trivector(yval[idx * 3], yval[idx * 3 + 1], yval[idx * 3 + 2])).ToArray();

                        Quaternion.Quaternion[] wcval = (new Quaternion.Quaternion[wval.Length / 4])
                            .Select((_, idx) => new Quaternion.Quaternion(wval[idx * 4], wval[idx * 4 + 1], wval[idx * 4 + 2], wval[idx * 4 + 3])).ToArray();

                        TrivectorMap0D x = new TrivectorMap0D(inchannels / 3, batch, xcval);
                        TrivectorMap0D y = new TrivectorMap0D(outchannels / 3, batch, ycval);
                        Quaternion.QuaternionFilter0D w = new Quaternion.QuaternionFilter0D(inchannels / 3, outchannels / 3, wcval);

                        Quaternion.QuaternionFilter0D gw = Reference(x, y, w);

                        OverflowCheckedTensor x_tensor = new OverflowCheckedTensor(Shape.Map0D(inchannels, batch), xval);
                        OverflowCheckedTensor y_tensor = new OverflowCheckedTensor(Shape.Map0D(outchannels, batch), yval);
                        OverflowCheckedTensor w_tensor = new OverflowCheckedTensor(Shape.Kernel0D(inchannels / 3 * 4, outchannels / 3), wval);

                        OverflowCheckedTensor gw_tensor = new OverflowCheckedTensor(Shape.Kernel0D(inchannels / 3 * 4, outchannels / 3));

                        TrivectorKernelProductDense ope = new TrivectorKernelProductDense(inchannels, outchannels, transpose: false, batch);

                        ope.Execute(x_tensor, y_tensor, w_tensor, gw_tensor);

                        float[] gw_expect = gw.ToArray();
                        float[] gw_actual = gw_tensor.State;

                        CollectionAssert.AreEqual(xval, x_tensor.State);
                        CollectionAssert.AreEqual(yval, y_tensor.State);
                        CollectionAssert.AreEqual(wval, w_tensor.State);

                        AssertError.Tolerance(gw_expect, gw_actual, 1e-7f, 1e-5f, ref max_err, $"mismatch value {inchannels},{outchannels},{batch}");

                        Console.WriteLine($"pass: {inchannels},{outchannels},{batch}");

                    }
                }
            }

            Console.WriteLine($"maxerr:{max_err}");
        }

        [TestMethod]
        public void LargeMapTest() {
            Random random = new Random(1234);

            float max_err = 0;

            int batch = 3;
            int inchannels = 147, outchannels = 150;

            float[] xval = (new float[inchannels * batch]).Select((_, idx) => (float)random.NextDouble() * 1e-2f).ToArray();
            float[] yval = (new float[outchannels * batch]).Select((_, idx) => (float)random.NextDouble() * 1e-2f).ToArray();
            float[] wval = (new float[inchannels * outchannels / 9 * 4]).Select((_, idx) => (float)random.NextDouble() * 1e-2f).ToArray();

            Trivector[] xcval = (new Trivector[xval.Length / 3])
                .Select((_, idx) => new Trivector(xval[idx * 3], xval[idx * 3 + 1], xval[idx * 3 + 2])).ToArray();

            Trivector[] ycval = (new Trivector[yval.Length / 3])
                .Select((_, idx) => new Trivector(yval[idx * 3], yval[idx * 3 + 1], yval[idx * 3 + 2])).ToArray();

            Quaternion.Quaternion[] wcval = (new Quaternion.Quaternion[wval.Length / 4])
                .Select((_, idx) => new Quaternion.Quaternion(wval[idx * 4], wval[idx * 4 + 1], wval[idx * 4 + 2], wval[idx * 4 + 3])).ToArray();

            TrivectorMap0D x = new TrivectorMap0D(inchannels / 3, batch, xcval);
            TrivectorMap0D y = new TrivectorMap0D(outchannels / 3, batch, ycval);
            Quaternion.QuaternionFilter0D w = new Quaternion.QuaternionFilter0D(inchannels / 3, outchannels / 3, wcval);

            Quaternion.QuaternionFilter0D gw = Reference(x, y, w);

            OverflowCheckedTensor x_tensor = new OverflowCheckedTensor(Shape.Map0D(inchannels, batch), xval);
            OverflowCheckedTensor y_tensor = new OverflowCheckedTensor(Shape.Map0D(outchannels, batch), yval);
            OverflowCheckedTensor w_tensor = new OverflowCheckedTensor(Shape.Kernel0D(inchannels / 3 * 4, outchannels / 3), wval);

            OverflowCheckedTensor gw_tensor = new OverflowCheckedTensor(Shape.Kernel0D(inchannels / 3 * 4, outchannels / 3));

            TrivectorKernelProductDense ope = new TrivectorKernelProductDense(inchannels, outchannels, transpose: false, batch);

            ope.Execute(x_tensor, y_tensor, w_tensor, gw_tensor);

            float[] gw_expect = gw.ToArray();
            float[] gw_actual = gw_tensor.State;

            CollectionAssert.AreEqual(xval, x_tensor.State);
            CollectionAssert.AreEqual(yval, y_tensor.State);
            CollectionAssert.AreEqual(wval, w_tensor.State);

            AssertError.Tolerance(gw_expect, gw_actual, 1e-7f, 1e-5f, ref max_err, $"mismatch value {inchannels},{outchannels},{batch}");

            Console.WriteLine($"pass: {inchannels},{outchannels},{batch}");

            Console.WriteLine($"maxerr:{max_err}");
        }

        [TestMethod]
        public void SpeedTest() {
            int inchannels = 33, outchannels = 33;

            OverflowCheckedTensor x_tensor = new OverflowCheckedTensor(Shape.Map0D(inchannels));
            OverflowCheckedTensor y_tensor = new OverflowCheckedTensor(Shape.Map0D(outchannels));
            OverflowCheckedTensor w_tensor = new OverflowCheckedTensor(Shape.Kernel0D(inchannels / 3 * 4, outchannels / 3));

            OverflowCheckedTensor gw_tensor = new OverflowCheckedTensor(Shape.Kernel0D(inchannels / 3 * 4, outchannels / 3));

            TrivectorKernelProductDense ope = new TrivectorKernelProductDense(inchannels, outchannels);

            Cuda.Profiler.Initialize("../../../../profiler.nvsetting", "../../nvprofiles/trivector_kernelproduct_dense.nvvp");
            Cuda.Profiler.Start();

            ope.Execute(x_tensor, y_tensor, w_tensor, gw_tensor);

            Cuda.Profiler.Stop();
        }

        public static Quaternion.QuaternionFilter0D Reference(TrivectorMap0D x, TrivectorMap0D gy, Quaternion.QuaternionFilter0D w) {
            int inchannels = x.Channels, outchannels = gy.Channels, batch = x.Batch;

            Quaternion.QuaternionFilter0D gw = new Quaternion.QuaternionFilter0D(inchannels, outchannels);

            for (int th = 0; th < batch; th++) {
                for (int inch, outch = 0; outch < outchannels; outch++) {
                    for (inch = 0; inch < inchannels; inch++) {
                        gw[inch, outch] += Trivector.MulQGrad(x[inch, th], gy[outch, th], w[inch, outch]);
                    }
                }
            }

            return gw;
        }
    }
}
