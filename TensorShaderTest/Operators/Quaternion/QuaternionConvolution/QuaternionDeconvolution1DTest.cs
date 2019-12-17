using System;
using System.Diagnostics;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using TensorShader.Operators.QuaternionConvolution;

namespace TensorShaderTest.Operators.Quaternion {
    [TestClass]
    public class QuaternionDeconvolution1DTest {
        [TestMethod]
        public void ExecuteTest() {
            float max_err = 0;

            foreach (int batch in new int[] { 1, 2, 3 }) {
                foreach (int inchannels in new int[] { 4, 8, 12 }) {
                    foreach (int outchannels in new int[] { 4, 8, 12 }) {
                        foreach (int kwidth in new int[] { 1, 3, 5 }) {
                            foreach (int inwidth in new int[] { 8, 9, 13, 17 }) {
                                int outwidth = inwidth - kwidth + 1;

                                float[] yval = (new float[outwidth * outchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
                                float[] wval = (new float[kwidth * inchannels * outchannels / 4]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

                                Quaternion[] ycval = (new Quaternion[yval.Length / 4])
                                    .Select((_, idx) => new Quaternion(yval[idx * 4], yval[idx * 4 + 1], yval[idx * 4 + 2], yval[idx * 4 + 3])).ToArray();

                                Quaternion[] wcval = (new Quaternion[wval.Length / 4])
                                    .Select((_, idx) => new Quaternion(wval[idx * 4], wval[idx * 4 + 1], wval[idx * 4 + 2], wval[idx * 4 + 3])).ToArray();

                                QuaternionMap1D y = new QuaternionMap1D(outchannels / 4, outwidth, batch, ycval);
                                QuaternionFilter1D w = new QuaternionFilter1D(inchannels / 4, outchannels / 4, kwidth, wcval);

                                QuaternionMap1D x = Reference(y, w, inwidth, kwidth);

                                OverflowCheckedTensor y_tensor = new OverflowCheckedTensor(Shape.Map1D(outchannels, outwidth, batch), yval);
                                OverflowCheckedTensor w_tensor = new OverflowCheckedTensor(Shape.Kernel1D(inchannels, outchannels / 4, kwidth), wval);

                                OverflowCheckedTensor x_tensor = new OverflowCheckedTensor(Shape.Map1D(inchannels, inwidth, batch));

                                QuaternionDeconvolution1D ope = new QuaternionDeconvolution1D(inwidth, outchannels, inchannels, kwidth, gradmode: false, batch);

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
                    foreach (int inchannels in new int[] { 4, 8, 12 }) {
                        foreach (int outchannels in new int[] { 4, 8, 12 }) {
                            foreach (int kwidth in new int[] { 1, 3, 5 }) {
                                foreach (int inwidth in new int[] { 8, 9, 13, 17 }) {
                                    int outwidth = inwidth - kwidth + 1;

                                    float[] yval = (new float[outwidth * outchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
                                    float[] wval = (new float[kwidth * inchannels * outchannels / 4]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

                                    OverflowCheckedTensor y_tensor = new OverflowCheckedTensor(Shape.Map1D(outchannels, outwidth, batch), yval);
                                    OverflowCheckedTensor w_tensor = new OverflowCheckedTensor(Shape.Kernel1D(inchannels, outchannels / 4, kwidth), wval);

                                    OverflowCheckedTensor x_tensor = new OverflowCheckedTensor(Shape.Map1D(inchannels, inwidth, batch));

                                    QuaternionDeconvolution1D ope = new QuaternionDeconvolution1D(inwidth, outchannels, inchannels, kwidth, gradmode, batch);

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
            int inwidth = 512, inchannels = 32, outchannels = 32, ksize = 3;
            int outwidth = inwidth - ksize + 1;

            OverflowCheckedTensor y_tensor = new OverflowCheckedTensor(Shape.Map1D(outchannels, outwidth));
            OverflowCheckedTensor w_tensor = new OverflowCheckedTensor(Shape.Kernel1D(inchannels, outchannels / 4, ksize));

            OverflowCheckedTensor x_tensor = new OverflowCheckedTensor(Shape.Map1D(inchannels, inwidth));

            QuaternionDeconvolution1D ope = new QuaternionDeconvolution1D(inwidth, outchannels, inchannels, ksize);

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

        public static QuaternionMap1D Reference(QuaternionMap1D y, QuaternionFilter1D w, int inw, int kwidth) {
            int inchannels = w.InChannels, outchannels = w.OutChannels, batch = y.Batch;
            int outw = inw - kwidth + 1;

            if (y.Width != outw) {
                throw new ArgumentException("mismatch shape");
            }

            QuaternionMap1D x = new QuaternionMap1D(inchannels, inw, batch);

            for (int kx = 0; kx < kwidth; kx++) {
                for (int th = 0; th < batch; th++) {
                    for (int ox = 0; ox < outw; ox++) {
                        for (int outch = 0; outch < outchannels; outch++) {
                            Quaternion v = y[outch, ox, th];

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
            int inchannels = 8, outchannels = 12, kwidth = 3, inwidth = 13, batch = 3;
            int outwidth = inwidth - kwidth + 1;

            float[] yval = (new float[outwidth * outchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
            float[] wval = (new float[kwidth * outchannels * inchannels / 4]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

            Quaternion[] ycval = (new Quaternion[yval.Length / 4])
                .Select((_, idx) => new Quaternion(yval[idx * 4], yval[idx * 4 + 1], yval[idx * 4 + 2], yval[idx * 4 + 3])).ToArray();

            Quaternion[] wcval = (new Quaternion[wval.Length / 4])
                .Select((_, idx) => new Quaternion(wval[idx * 4], wval[idx * 4 + 1], wval[idx * 4 + 2], wval[idx * 4 + 3])).ToArray();

            QuaternionMap1D y = new QuaternionMap1D(outchannels / 4, outwidth, batch, ycval);
            QuaternionFilter1D w = new QuaternionFilter1D(inchannels / 4, outchannels / 4, kwidth, wcval);

            QuaternionMap1D x = Reference(y, w, inwidth, kwidth);

            float[] x_expect = {
                -2.404000000e-03f,  1.360000000e-03f,  2.140000000e-03f,  1.714000000e-03f,  -2.236000000e-03f,  1.264000000e-03f,
                1.996000000e-03f,  1.594000000e-03f,  -1.396000000e-03f,  7.840000000e-04f,  1.276000000e-03f,  9.940000000e-04f,
                -1.228000000e-03f,  6.880000000e-04f,  1.132000000e-03f,  8.740000000e-04f,  -7.112000000e-03f,  6.032000000e-03f,
                7.088000000e-03f,  6.380000000e-03f,  -6.488000000e-03f,  5.552000000e-03f,  6.512000000e-03f,  5.852000000e-03f,
                -3.988000000e-03f,  3.520000000e-03f,  4.084000000e-03f,  3.658000000e-03f,  -3.532000000e-03f,  3.136000000e-03f,
                3.652000000e-03f,  3.250000000e-03f,  -1.229600000e-02f,  1.150400000e-02f,  1.270400000e-02f,  1.170800000e-02f,
                -1.109600000e-02f,  1.044800000e-02f,  1.155200000e-02f,  1.060400000e-02f,  -6.580000000e-03f,  6.256000000e-03f,
                6.892000000e-03f,  6.322000000e-03f,  -5.836000000e-03f,  5.584000000e-03f,  6.172000000e-03f,  5.626000000e-03f,
                -1.748000000e-02f,  1.697600000e-02f,  1.832000000e-02f,  1.703600000e-02f,  -1.570400000e-02f,  1.534400000e-02f,
                1.659200000e-02f,  1.535600000e-02f,  -9.172000000e-03f,  8.992000000e-03f,  9.700000000e-03f,  8.986000000e-03f,
                -8.140000000e-03f,  8.032000000e-03f,  8.692000000e-03f,  8.002000000e-03f,  -2.266400000e-02f,  2.244800000e-02f,
                2.393600000e-02f,  2.236400000e-02f,  -2.031200000e-02f,  2.024000000e-02f,  2.163200000e-02f,  2.010800000e-02f,
                -1.176400000e-02f,  1.172800000e-02f,  1.250800000e-02f,  1.165000000e-02f,  -1.044400000e-02f,  1.048000000e-02f,
                1.121200000e-02f,  1.037800000e-02f,  -2.784800000e-02f,  2.792000000e-02f,  2.955200000e-02f,  2.769200000e-02f,
                -2.492000000e-02f,  2.513600000e-02f,  2.667200000e-02f,  2.486000000e-02f,  -1.435600000e-02f,  1.446400000e-02f,
                1.531600000e-02f,  1.431400000e-02f,  -1.274800000e-02f,  1.292800000e-02f,  1.373200000e-02f,  1.275400000e-02f,
                -4.708000000e-03f,  5.248000000e-03f,  5.812000000e-03f,  4.954000000e-03f,  -3.100000000e-03f,  3.712000000e-03f,
                4.228000000e-03f,  3.394000000e-03f,  -2.832400000e-02f,  2.814400000e-02f,  2.935600000e-02f,  2.806600000e-02f,
                -2.642800000e-02f,  2.632000000e-02f,  2.748400000e-02f,  2.621800000e-02f,  -1.694800000e-02f,  1.720000000e-02f,
                1.812400000e-02f,  1.697800000e-02f,  -1.505200000e-02f,  1.537600000e-02f,  1.625200000e-02f,  1.513000000e-02f,
                -3.821600000e-02f,  3.886400000e-02f,  4.078400000e-02f,  3.834800000e-02f,  -3.413600000e-02f,  3.492800000e-02f,
                3.675200000e-02f,  3.436400000e-02f,  -1.954000000e-02f,  1.993600000e-02f,  2.093200000e-02f,  1.964200000e-02f,
                -1.735600000e-02f,  1.782400000e-02f,  1.877200000e-02f,  1.750600000e-02f,  -4.340000000e-02f,  4.433600000e-02f,
                4.640000000e-02f,  4.367600000e-02f,  -3.874400000e-02f,  3.982400000e-02f,  4.179200000e-02f,  3.911600000e-02f,
                -2.213200000e-02f,  2.267200000e-02f,  2.374000000e-02f,  2.230600000e-02f,  -1.966000000e-02f,  2.027200000e-02f,
                2.129200000e-02f,  1.988200000e-02f,  -4.858400000e-02f,  4.980800000e-02f,  5.201600000e-02f,  4.900400000e-02f,
                -4.335200000e-02f,  4.472000000e-02f,  4.683200000e-02f,  4.386800000e-02f,  -2.472400000e-02f,  2.540800000e-02f,
                2.654800000e-02f,  2.497000000e-02f,  -2.196400000e-02f,  2.272000000e-02f,  2.381200000e-02f,  2.225800000e-02f,
                -5.376800000e-02f,  5.528000000e-02f,  5.763200000e-02f,  5.433200000e-02f,  -4.796000000e-02f,  4.961600000e-02f,
                5.187200000e-02f,  4.862000000e-02f,  -2.731600000e-02f,  2.814400000e-02f,  2.935600000e-02f,  2.763400000e-02f,
                -2.426800000e-02f,  2.516800000e-02f,  2.633200000e-02f,  2.463400000e-02f,  -5.895200000e-02f,  6.075200000e-02f,
                6.324800000e-02f,  5.966000000e-02f,  -5.256800000e-02f,  5.451200000e-02f,  5.691200000e-02f,  5.337200000e-02f,
                -2.990800000e-02f,  3.088000000e-02f,  3.216400000e-02f,  3.029800000e-02f,  -2.657200000e-02f,  2.761600000e-02f,
                2.885200000e-02f,  2.701000000e-02f,  -9.892000000e-03f,  1.129600000e-02f,  1.229200000e-02f,  1.057000000e-02f,
                -6.556000000e-03f,  8.032000000e-03f,  8.980000000e-03f,  7.282000000e-03f,  -5.424400000e-02f,  5.492800000e-02f,
                5.657200000e-02f,  5.441800000e-02f,  -5.062000000e-02f,  5.137600000e-02f,  5.297200000e-02f,  5.084200000e-02f,
                -3.250000000e-02f,  3.361600000e-02f,  3.497200000e-02f,  3.296200000e-02f,  -2.887600000e-02f,  3.006400000e-02f,
                3.137200000e-02f,  2.938600000e-02f,  -6.932000000e-02f,  7.169600000e-02f,  7.448000000e-02f,  7.031600000e-02f,
                -6.178400000e-02f,  6.430400000e-02f,  6.699200000e-02f,  6.287600000e-02f,  -3.509200000e-02f,  3.635200000e-02f,
                3.778000000e-02f,  3.562600000e-02f,  -3.118000000e-02f,  3.251200000e-02f,  3.389200000e-02f,  3.176200000e-02f,
                -7.450400000e-02f,  7.716800000e-02f,  8.009600000e-02f,  7.564400000e-02f,  -6.639200000e-02f,  6.920000000e-02f,
                7.203200000e-02f,  6.762800000e-02f,  -3.768400000e-02f,  3.908800000e-02f,  4.058800000e-02f,  3.829000000e-02f,
                -3.348400000e-02f,  3.496000000e-02f,  3.641200000e-02f,  3.413800000e-02f,  -7.968800000e-02f,  8.264000000e-02f,
                8.571200000e-02f,  8.097200000e-02f,  -7.100000000e-02f,  7.409600000e-02f,  7.707200000e-02f,  7.238000000e-02f,
                -4.027600000e-02f,  4.182400000e-02f,  4.339600000e-02f,  4.095400000e-02f,  -3.578800000e-02f,  3.740800000e-02f,
                3.893200000e-02f,  3.651400000e-02f,  -8.487200000e-02f,  8.811200000e-02f,  9.132800000e-02f,  8.630000000e-02f,
                -7.560800000e-02f,  7.899200000e-02f,  8.211200000e-02f,  7.713200000e-02f,  -4.286800000e-02f,  4.456000000e-02f,
                4.620400000e-02f,  4.361800000e-02f,  -3.809200000e-02f,  3.985600000e-02f,  4.145200000e-02f,  3.889000000e-02f,
                -9.005600000e-02f,  9.358400000e-02f,  9.694400000e-02f,  9.162800000e-02f,  -8.021600000e-02f,  8.388800000e-02f,
                8.715200000e-02f,  8.188400000e-02f,  -4.546000000e-02f,  4.729600000e-02f,  4.901200000e-02f,  4.628200000e-02f,
                -4.039600000e-02f,  4.230400000e-02f,  4.397200000e-02f,  4.126600000e-02f,  -1.507600000e-02f,  1.734400000e-02f,
                1.877200000e-02f,  1.618600000e-02f,  -1.001200000e-02f,  1.235200000e-02f,  1.373200000e-02f,  1.117000000e-02f,
            };

            float[] x_actual = x.ToArray();

            AssertError.Tolerance(x_expect, x_actual, 1e-7f, 1e-5f, $"mismatch value {inchannels},{outchannels},{kwidth},{inwidth},{batch}");
        }
    }
}
