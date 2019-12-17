using System;
using System.Diagnostics;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using TensorShader.Operators.TrivectorConvolution;

namespace TensorShaderTest.Operators.Trivector {
    [TestClass]
    public class TrivectorDeconvolution1DTest {
        [TestMethod]
        public void ExecuteTest() {
            float max_err = 0;

            foreach (int batch in new int[] { 1, 2, 3 }) {
                foreach (int inchannels in new int[] { 3, 6, 9, 12 }) {
                    foreach (int outchannels in new int[] { 3, 6, 9, 12 }) {
                        foreach (int kwidth in new int[] { 1, 3, 5 }) {
                            foreach (int inwidth in new int[] { 8, 9, 13, 17 }) {
                                int outwidth = inwidth - kwidth + 1;

                                float[] yval = (new float[outwidth * outchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
                                float[] wval = (new float[kwidth * inchannels * outchannels / 9 * 4]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

                                Trivector[] ycval = (new Trivector[yval.Length / 3])
                                    .Select((_, idx) => new Trivector(yval[idx * 3], yval[idx * 3 + 1], yval[idx * 3 + 2])).ToArray();

                                Quaternion.Quaternion[] wcval = (new Quaternion.Quaternion[wval.Length / 4])
                                    .Select((_, idx) => new Quaternion.Quaternion(wval[idx * 4], wval[idx * 4 + 1], wval[idx * 4 + 2], wval[idx * 4 + 3])).ToArray();

                                TrivectorMap1D y = new TrivectorMap1D(outchannels / 3, outwidth, batch, ycval);
                                Quaternion.QuaternionFilter1D w = new Quaternion.QuaternionFilter1D(inchannels / 3, outchannels / 3, kwidth, wcval);

                                TrivectorMap1D x = Reference(y, w, inwidth, kwidth);

                                OverflowCheckedTensor y_tensor = new OverflowCheckedTensor(Shape.Map1D(outchannels, outwidth, batch), yval);
                                OverflowCheckedTensor w_tensor = new OverflowCheckedTensor(Shape.Kernel1D(inchannels / 3 * 4, outchannels / 3, kwidth), wval);

                                OverflowCheckedTensor x_tensor = new OverflowCheckedTensor(Shape.Map1D(inchannels, inwidth, batch));

                                TrivectorDeconvolution1D ope = new TrivectorDeconvolution1D(inwidth, outchannels, inchannels, kwidth, gradmode: false, batch);

                                ope.Execute(y_tensor, w_tensor, x_tensor);

                                float[] x_expect = x.ToArray();
                                float[] x_actual = x_tensor.State;

                                CollectionAssert.AreEqual(yval, y_tensor.State);
                                CollectionAssert.AreEqual(wval, w_tensor.State);

                                AssertError.Tolerance(x_expect, x_actual, 1e-7f, 1e-5f, ref max_err, $"mismatch value {inchannels},{outchannels},{kwidth},{inwidth},{batch}");

                                Console.WriteLine($"pass: {inchannels},{outchannels},{kwidth},{inwidth},{batch}");
                            }
                        }
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
                            foreach (int kwidth in new int[] { 1, 3, 5 }) {
                                foreach (int inwidth in new int[] { 8, 9, 13, 17 }) {
                                    int outwidth = inwidth - kwidth + 1;

                                    float[] yval = (new float[outwidth * outchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
                                    float[] wval = (new float[kwidth * inchannels * outchannels / 9 * 4]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

                                    OverflowCheckedTensor y_tensor = new OverflowCheckedTensor(Shape.Map1D(outchannels, outwidth, batch), yval);
                                    OverflowCheckedTensor w_tensor = new OverflowCheckedTensor(Shape.Kernel1D(inchannels / 3 * 4, outchannels / 3, kwidth), wval);

                                    OverflowCheckedTensor x_tensor = new OverflowCheckedTensor(Shape.Map1D(inchannels, inwidth, batch));

                                    TrivectorDeconvolution1D ope = new TrivectorDeconvolution1D(inwidth, outchannels, inchannels, kwidth, gradmode, batch);

                                    ope.Execute(y_tensor, w_tensor, x_tensor);

                                    CollectionAssert.AreEqual(yval, y_tensor.State);
                                    CollectionAssert.AreEqual(wval, w_tensor.State);

                                    x_tensor.CheckOverflow();

                                    Console.WriteLine($"pass: {inchannels},{outchannels},{kwidth},{inwidth},{batch},{gradmode}");
                                }
                            }
                        }
                    }
                }
            }
        }

        [TestMethod]
        public void SpeedTest() {
            int inwidth = 32, inchannels = 33, outchannels = 33, ksize = 3;
            int outwidth = inwidth - ksize + 1;

            OverflowCheckedTensor y_tensor = new OverflowCheckedTensor(Shape.Map1D(outchannels, outwidth));
            OverflowCheckedTensor w_tensor = new OverflowCheckedTensor(Shape.Kernel1D(inchannels / 3 * 4, outchannels / 3, ksize));

            OverflowCheckedTensor x_tensor = new OverflowCheckedTensor(Shape.Map1D(inchannels, inwidth));

            TrivectorDeconvolution1D ope = new TrivectorDeconvolution1D(inwidth, outchannels, inchannels, ksize);

            ope.Execute(y_tensor, w_tensor, x_tensor);

            Stopwatch sw = new Stopwatch();
            sw.Start();

            ope.Execute(y_tensor, w_tensor, x_tensor);
            ope.Execute(y_tensor, w_tensor, x_tensor);
            ope.Execute(y_tensor, w_tensor, x_tensor);
            ope.Execute(y_tensor, w_tensor, x_tensor);

            sw.Stop();

            Console.WriteLine($"{sw.ElapsedMilliseconds / 4} msec");
        }

        public static TrivectorMap1D Reference(TrivectorMap1D y, Quaternion.QuaternionFilter1D w, int inw, int kwidth) {
            int inchannels = w.InChannels, outchannels = w.OutChannels, batch = y.Batch;
            int outw = inw - kwidth + 1;

            if (y.Width != outw) {
                throw new ArgumentException("mismatch shape");
            }

            TrivectorMap1D x = new TrivectorMap1D(inchannels, inw, batch);

            for (int kx = 0; kx < kwidth; kx++) {
                for (int th = 0; th < batch; th++) {
                    for (int ox = 0; ox < outw; ox++) {
                        for (int outch = 0; outch < outchannels; outch++) {
                            Trivector v = y[outch, ox, th];

                            for (int inch = 0; inch < inchannels; inch++) {
                                x[inch, kx + ox, th] += v * w[inch, outch, kx];
                            }
                        }
                    }
                }
            }

            return x;
        }

        [TestMethod]
        public void ReferenceTest() {
            int inchannels = 9, outchannels = 12, kwidth = 3, inwidth = 7, batch = 3;
            int outwidth = inwidth - kwidth + 1;

            float[] yval = (new float[outwidth * outchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
            float[] wval = (new float[kwidth * outchannels * inchannels / 9 * 4]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

            Trivector[] ycval = (new Trivector[yval.Length / 3])
                .Select((_, idx) => new Trivector(yval[idx * 3], yval[idx * 3 + 1], yval[idx * 3 + 2])).ToArray();

            Quaternion.Quaternion[] wcval = (new Quaternion.Quaternion[wval.Length / 4])
                .Select((_, idx) => new Quaternion.Quaternion(wval[idx * 4], wval[idx * 4 + 1], wval[idx * 4 + 2], wval[idx * 4 + 3])).ToArray();

            TrivectorMap1D y = new TrivectorMap1D(outchannels / 3, outwidth, batch, ycval);
            Quaternion.QuaternionFilter1D w = new Quaternion.QuaternionFilter1D(inchannels / 3, outchannels / 3, kwidth, wcval);

            TrivectorMap1D x = Reference(y, w, inwidth, kwidth);

            float[] x_expect = {
                1.443468000e-03f,  9.190920000e-04f,  1.172196000e-03f,  1.347564000e-03f,  8.553480000e-04f,  1.092708000e-03f,
                1.254988000e-03f,  7.939080000e-04f,  1.016036000e-03f,  5.122680000e-04f,  3.062280000e-04f,  4.041960000e-04f,
                4.563000000e-04f,  2.701320000e-04f,  3.585000000e-04f,  4.036600000e-04f,  2.363400000e-04f,  3.156200000e-04f,
                4.514232000e-03f,  3.883272000e-03f,  4.153512000e-03f,  4.214136000e-03f,  3.625224000e-03f,  3.876264000e-03f,
                3.926840000e-03f,  3.377928000e-03f,  3.610792000e-03f,  1.670316000e-03f,  1.420500000e-03f,  1.519044000e-03f,
                1.499916000e-03f,  1.272276000e-03f,  1.361220000e-03f,  1.338988000e-03f,  1.132500000e-03f,  1.212356000e-03f,
                7.715064000e-03f,  6.996552000e-03f,  7.267944000e-03f,  7.186104000e-03f,  6.514248000e-03f,  6.766440000e-03f,
                6.682232000e-03f,  6.054984000e-03f,  6.289000000e-03f,  2.828364000e-03f,  2.534772000e-03f,  2.633892000e-03f,
                2.543532000e-03f,  2.274420000e-03f,  2.363940000e-03f,  2.274316000e-03f,  2.028660000e-03f,  2.109092000e-03f,
                4.410360000e-04f,  3.736200000e-04f,  3.913320000e-04f,  3.435960000e-04f,  2.883720000e-04f,  3.026280000e-04f,
                2.617720000e-04f,  2.177160000e-04f,  2.290280000e-04f,  1.047486000e-02f,  9.736212000e-03f,  9.991044000e-03f,
                9.814476000e-03f,  9.114900000e-03f,  9.353988000e-03f,  9.175852000e-03f,  8.514324000e-03f,  8.738180000e-03f,
                3.986412000e-03f,  3.649044000e-03f,  3.748740000e-03f,  3.587148000e-03f,  3.276564000e-03f,  3.366660000e-03f,
                3.209644000e-03f,  2.924820000e-03f,  3.005828000e-03f,  1.411672800e-02f,  1.322311200e-02f,  1.349680800e-02f,
                1.313004000e-02f,  1.229229600e-02f,  1.254679200e-02f,  1.219301600e-02f,  1.140909600e-02f,  1.164541600e-02f,
                5.144460000e-03f,  4.763316000e-03f,  4.863588000e-03f,  4.630764000e-03f,  4.278708000e-03f,  4.369380000e-03f,
                4.144972000e-03f,  3.820980000e-03f,  3.902564000e-03f,  1.731756000e-02f,  1.633639200e-02f,  1.661124000e-02f,
                1.610200800e-02f,  1.518132000e-02f,  1.543696800e-02f,  1.494840800e-02f,  1.408615200e-02f,  1.432362400e-02f,
                6.302508000e-03f,  5.877588000e-03f,  5.978436000e-03f,  5.674380000e-03f,  5.280852000e-03f,  5.372100000e-03f,
                5.080300000e-03f,  4.717140000e-03f,  4.799300000e-03f,  1.012140000e-03f,  8.963400000e-04f,  9.157800000e-04f,
                7.925880000e-04f,  6.958920000e-04f,  7.118760000e-04f,  6.070840000e-04f,  5.284680000e-04f,  5.415080000e-04f,
                1.950625200e-02f,  1.855333200e-02f,  1.880989200e-02f,  1.828138800e-02f,  1.737445200e-02f,  1.761526800e-02f,
                1.709671600e-02f,  1.623474000e-02f,  1.646032400e-02f,  7.460556000e-03f,  6.991860000e-03f,  7.093284000e-03f,
                6.717996000e-03f,  6.282996000e-03f,  6.374820000e-03f,  6.015628000e-03f,  5.613300000e-03f,  5.696036000e-03f,
                2.371922400e-02f,  2.256295200e-02f,  2.284010400e-02f,  2.204594400e-02f,  2.095936800e-02f,  2.121732000e-02f,
                2.045919200e-02f,  1.944026400e-02f,  1.968004000e-02f,  8.618604000e-03f,  8.106132000e-03f,  8.208132000e-03f,
                7.761612000e-03f,  7.285140000e-03f,  7.377540000e-03f,  6.950956000e-03f,  6.509460000e-03f,  6.592772000e-03f,
                2.692005600e-02f,  2.567623200e-02f,  2.595453600e-02f,  2.501791200e-02f,  2.384839200e-02f,  2.410749600e-02f,
                2.321458400e-02f,  2.211732000e-02f,  2.235824800e-02f,  9.776652000e-03f,  9.220404000e-03f,  9.322980000e-03f,
                8.805228000e-03f,  8.287284000e-03f,  8.380260000e-03f,  7.886284000e-03f,  7.405620000e-03f,  7.489508000e-03f,
                1.583244000e-03f,  1.419060000e-03f,  1.440228000e-03f,  1.241580000e-03f,  1.103412000e-03f,  1.121124000e-03f,
                9.523960000e-04f,  8.392200000e-04f,  8.539880000e-04f
            };

            float[] x_actual = x.ToArray();

            AssertError.Tolerance(x_expect, x_actual, 1e-7f, 1e-5f, $"mismatch value {inchannels},{outchannels},{kwidth},{inwidth},{batch}");
        }
    }
}
