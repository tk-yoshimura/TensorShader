using System;
using System.Diagnostics;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using TensorShader.Operators.TrivectorConvolution;

namespace TensorShaderTest.Operators.Trivector {
    [TestClass]
    public class TrivectorDenseTest {
        [TestMethod]
        public void ExecuteTest() {
            float max_err = 0;

            foreach (int batch in new int[] { 1, 2, 3 }) {
                foreach (int inchannels in new int[] { 3, 6, 9, 12 }) {
                    foreach (int outchannels in new int[] { 3, 6, 9, 12 }) {
                        float[] xval = (new float[inchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
                        float[] wval = (new float[inchannels * outchannels / 9 * 4]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

                        Trivector[] xcval = (new Trivector[xval.Length / 3])
                            .Select((_, idx) => new Trivector(xval[idx * 3], xval[idx * 3 + 1], xval[idx * 3 + 2])).ToArray();

                        Quaternion.Quaternion[] wcval = (new Quaternion.Quaternion[wval.Length / 4])
                            .Select((_, idx) => new Quaternion.Quaternion(wval[idx * 4], wval[idx * 4 + 1], wval[idx * 4 + 2], wval[idx * 4 + 3])).ToArray();

                        TrivectorMap0D x = new TrivectorMap0D(inchannels / 3, batch, xcval);
                        Quaternion.QuaternionFilter0D w = new Quaternion.QuaternionFilter0D(inchannels / 3, outchannels / 3, wcval);

                        TrivectorMap0D y = Reference(x, w);

                        OverflowCheckedTensor x_tensor = new OverflowCheckedTensor(Shape.Map0D(inchannels, batch), xval);
                        OverflowCheckedTensor w_tensor = new OverflowCheckedTensor(Shape.Kernel0D(inchannels / 3 * 4, outchannels / 3), wval);

                        OverflowCheckedTensor y_tensor = new OverflowCheckedTensor(Shape.Map0D(outchannels, batch));

                        TrivectorDense ope = new TrivectorDense(inchannels, outchannels, gradmode: false, batch);

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
        public void OverflowTest() {
            foreach (bool gradmode in new bool[] { false, true }) {
                foreach (int batch in new int[] { 1, 2, 3 }) {
                    foreach (int inchannels in new int[] { 3, 6, 9, 12 }) {
                        foreach (int outchannels in new int[] { 3, 6, 9, 12 }) {
                            float[] xval = (new float[inchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
                            float[] wval = (new float[inchannels * outchannels / 9 * 4]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

                            OverflowCheckedTensor x_tensor = new OverflowCheckedTensor(Shape.Map0D(inchannels, batch), xval);
                            OverflowCheckedTensor w_tensor = new OverflowCheckedTensor(Shape.Kernel0D(inchannels / 3 * 4, outchannels / 3), wval);

                            OverflowCheckedTensor y_tensor = new OverflowCheckedTensor(Shape.Map0D(outchannels, batch));

                            TrivectorDense ope = new TrivectorDense(inchannels, outchannels, gradmode, batch);

                            ope.Execute(x_tensor, w_tensor, y_tensor);

                            CollectionAssert.AreEqual(xval, x_tensor.State);
                            CollectionAssert.AreEqual(wval, w_tensor.State);

                            y_tensor.CheckOverflow();

                            Console.WriteLine($"pass: {inchannels},{outchannels},{batch},{gradmode}");
                        }
                    }
                }
            }
        }

        [TestMethod]
        public void SpeedTest() {
            int inchannels = 33, outchannels = 33;

            OverflowCheckedTensor x_tensor = new OverflowCheckedTensor(Shape.Map0D(inchannels));
            OverflowCheckedTensor w_tensor = new OverflowCheckedTensor(Shape.Kernel0D(inchannels / 3 * 4, outchannels / 3));

            OverflowCheckedTensor y_tensor = new OverflowCheckedTensor(Shape.Map0D(outchannels));

            TrivectorDense ope = new TrivectorDense(inchannels, outchannels);

            Stopwatch sw = new Stopwatch();
            sw.Start();

            ope.Execute(x_tensor, w_tensor, y_tensor);
            ope.Execute(x_tensor, w_tensor, y_tensor);
            ope.Execute(x_tensor, w_tensor, y_tensor);
            ope.Execute(x_tensor, w_tensor, y_tensor);

            sw.Stop();

            Console.WriteLine($"{sw.ElapsedMilliseconds / 4} msec");
        }

        public static TrivectorMap0D Reference(TrivectorMap0D x, Quaternion.QuaternionFilter0D w) {
            int inchannels = x.Channels, outchannels = w.OutChannels, batch = x.Batch;

            TrivectorMap0D y = new TrivectorMap0D(outchannels, batch);

            for (int th = 0; th < batch; th++) {
                for (int outch = 0; outch < outchannels; outch++) {
                    Trivector sum = y[outch, th];

                    for (int inch = 0; inch < inchannels; inch++) {
                        sum += x[inch, th] * w[inch, outch];
                    }

                    y[outch, th] = sum;
                }
            }

            return y;
        }

        [TestMethod]
        public void ReferenceTest() {
            int inchannels = 9, outchannels = 12, batch = 3;

            float[] xval = (new float[batch * inchannels]).Select((_, idx) => idx * 1e-2f).ToArray();
            float[] wval = (new float[outchannels * inchannels / 9 * 4]).Select((_, idx) => idx * 1e-2f).Reverse().ToArray();

            Trivector[] xcval = (new Trivector[xval.Length / 3])
                .Select((_, idx) => new Trivector(xval[idx * 3], xval[idx * 3 + 1], xval[idx * 3 + 2])).ToArray();

            Quaternion.Quaternion[] wcval = (new Quaternion.Quaternion[wval.Length / 4])
                .Select((_, idx) => new Quaternion.Quaternion(wval[idx * 4], wval[idx * 4 + 1], wval[idx * 4 + 2], wval[idx * 4 + 3])).ToArray();

            TrivectorMap0D x = new TrivectorMap0D(inchannels / 3, batch, xcval);
            Quaternion.QuaternionFilter0D w = new Quaternion.QuaternionFilter0D(inchannels / 3, outchannels / 3, wcval);

            TrivectorMap0D y = Reference(x, w);

            float[] y_expect = {
                9.880600000e-02f,  5.148000000e-02f,  7.391000000e-02f,  4.912600000e-02f,  2.397600000e-02f,  3.575000000e-02f,
                1.672600000e-02f,  6.840000000e-03f,  1.141400000e-02f,  1.606000000e-03f,  7.200000000e-05f,  9.020000000e-04f,
                2.949520000e-01f,  2.340180000e-01f,  2.567720000e-01f,  1.506640000e-01f,  1.157940000e-01f,  1.278920000e-01f,
                5.476000000e-02f,  3.904200000e-02f,  4.394000000e-02f,  7.240000000e-03f,  3.762000000e-03f,  4.916000000e-03f,
                4.910980000e-01f,  4.165560000e-01f,  4.396340000e-01f,  2.522020000e-01f,  2.076120000e-01f,  2.200340000e-01f,
                9.279400000e-02f,  7.124400000e-02f,  7.646600000e-02f,  1.287400000e-02f,  7.452000000e-03f,  8.930000000e-03f,
            };

            float[] y_actual = y.ToArray();

            AssertError.Tolerance(y_expect, y_actual, 1e-7f, 1e-5f, $"mismatch value {inchannels},{outchannels},{batch}");
        }
    }
}
