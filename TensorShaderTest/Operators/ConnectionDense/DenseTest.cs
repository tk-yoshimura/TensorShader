using System;
using System.Diagnostics;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using TensorShader.Operators.ConnectionDense;
using TensorShaderCudaBackend.API;

namespace TensorShaderTest.Operators.ConnectionDense {
    [TestClass]
    public class DenseTest {
        [TestMethod]
        public void ExecuteTest() {
            float max_err = 0;

            foreach (int batch in new int[] { 1, 2 }) {
                foreach (int inchannels in new int[] { 1, 2, 3, 4, 5, 10, 15, 20, 1024 }) {
                    foreach (int outchannels in new int[] { 1, 2, 3, 4, 5, 10, 15, 20, 1024 }) {
                        float[] xval = (new float[inchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
                        float[] wval = (new float[inchannels * outchannels]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

                        Map0D x = new Map0D(inchannels, batch, xval);
                        Filter0D w = new Filter0D(inchannels, outchannels, 1, wval);

                        Map0D y = Reference(x, w);

                        OverflowCheckedTensor x_tensor = new OverflowCheckedTensor(Shape.Map0D(inchannels, batch), xval);
                        OverflowCheckedTensor w_tensor = new OverflowCheckedTensor(Shape.Kernel0D(inchannels, outchannels), wval);

                        OverflowCheckedTensor y_tensor = new OverflowCheckedTensor(Shape.Map0D(outchannels, batch));

                        Dense ope = new Dense(inchannels, outchannels, batch);

                        ope.Execute(x_tensor, w_tensor, y_tensor);

                        float[] y_expect = y.ToArray();
                        float[] y_actual = y_tensor.State;

                        CollectionAssert.AreEqual(xval, x_tensor.State);
                        CollectionAssert.AreEqual(wval, w_tensor.State);

                        AssertError.Tolerance(y_expect, y_actual, 1e-7f, 1e-5f, ref max_err, $"mismatch value {inchannels},{outchannels},{batch}");

                        Console.WriteLine($"pass: {inchannels},{outchannels},{batch}");
                    }
                }
            }

            Console.WriteLine($"maxerr:{max_err}");
        }

        [TestMethod]
        public void SpeedTest() {
            int inchannels = 1024, outchannels = 1024;

            OverflowCheckedTensor x_tensor = new OverflowCheckedTensor(Shape.Map0D(inchannels));
            OverflowCheckedTensor w_tensor = new OverflowCheckedTensor(Shape.Kernel0D(inchannels, outchannels));

            OverflowCheckedTensor y_tensor = new OverflowCheckedTensor(Shape.Map0D(outchannels));

            Dense ope = new Dense(inchannels, outchannels);

            Cuda.Profiler.Initialize("../../../profiler.nvsetting", "../../nvprofiles/dense_trans.nvvp");
            Cuda.Profiler.Start();

            ope.Execute(x_tensor, w_tensor, y_tensor);

            Cuda.Profiler.Stop();
        }

        public static Map0D Reference(Map0D x, Filter0D w) {
            int inchannels = x.Channels, outchannels = w.OutChannels, batch = x.Batch;

            Map0D y = new Map0D(outchannels, batch);

            for (int th = 0; th < batch; th++) {
                for (int outch = 0; outch < outchannels; outch++) {
                    double sum = 0;

                    for (int inch = 0; inch < inchannels; inch++) {
                        sum += x[inch, th] * w[inch, outch, 0];
                    }

                    y[outch, th] = sum;
                }
            }

            return y;
        }

        public static Map0D OptimizedReference(Map0D x, Filter0D w) {
            int inchannels = x.Channels, outchannels = w.OutChannels, batch = x.Batch;

            Map0D y = new Map0D(outchannels, batch);

            for (int th = 0; th < batch; th++) {
                int inmap_org = th * inchannels;
                int outmap_idx = th * outchannels;
                int kernel_idx = 0;

                for (int outch = 0; outch < outchannels; outch++) {
                    double sum = 0;

                    int inmap_idx = inmap_org;

                    for (int inch = 0; inch < inchannels; inch++) {
                        sum += x[inmap_idx] * w[kernel_idx];

                        inmap_idx++;
                        kernel_idx++;
                    }

                    y[outmap_idx] = sum;

                    outmap_idx++;
                }
            }

            return y;
        }

        [TestMethod]
        public void ReferenceTest() {
            int inchannels = 7, outchannels = 11, batch = 2;

            float[] xval = (new float[inchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
            float[] wval = (new float[outchannels * inchannels]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

            Map0D x = new Map0D(inchannels, batch, xval);
            Filter0D w = new Filter0D(inchannels, outchannels, 1, wval);

            Map0D y = Reference(x, w);

            float[] y_expect = {
                1.5050e-03f, 1.3580e-03f, 1.2110e-03f, 1.0640e-03f, 9.1700e-04f,
                7.7000e-04f, 6.2300e-04f, 4.7600e-04f, 3.2900e-04f, 1.8200e-04f,
                3.5000e-05f, 5.0820e-03f, 4.5920e-03f, 4.1020e-03f, 3.6120e-03f,
                3.1220e-03f, 2.6320e-03f, 2.1420e-03f, 1.6520e-03f, 1.1620e-03f,
                6.7200e-04f, 1.8200e-04f
            };

            float[] y_actual = y.ToArray();

            AssertError.Tolerance(y_expect, y_actual, 1e-7f, 1e-5f, $"mismatch value {inchannels},{outchannels},{batch}");
        }

        [TestMethod]
        public void OptimizeTest() {
            float max_err = 0;

            foreach (int batch in new int[] { 1, 2 }) {
                foreach (int inchannels in new int[] { 1, 2, 3, 4, 5, 10, 15, 20 }) {
                    foreach (int outchannels in new int[] { 7, 13 }) {
                        float[] xval = (new float[inchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
                        float[] wval = (new float[inchannels * outchannels]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

                        Map0D x = new Map0D(inchannels, batch, xval);
                        Filter0D w = new Filter0D(inchannels, outchannels, 1, wval);

                        Map0D y = Reference(x, w);
                        Map0D y_optimized = OptimizedReference(x, w);

                        float[] y_expect = y.ToArray();
                        float[] y_actual = y_optimized.ToArray();

                        AssertError.Tolerance(y_expect, y_actual, 1e-7f, 1e-5f, ref max_err, $"mismatch value {inchannels},{outchannels},{batch}");

                        Console.WriteLine($"pass: {inchannels},{outchannels},{batch}");
                    }
                }
            }

            Console.WriteLine($"maxerr:{max_err}");
        }
    }
}
