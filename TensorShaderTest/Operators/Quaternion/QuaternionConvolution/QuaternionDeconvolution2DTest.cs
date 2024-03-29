using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;
using TensorShader;
using TensorShader.Operators.QuaternionConvolution;
using TensorShaderCudaBackend.API;

namespace TensorShaderTest.Operators.Quaternion {
    [TestClass]
    public class QuaternionDeconvolution2DTest {
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

                                        float[] yval = (new float[outwidth * outheight * outchannels * batch]).Select((_, idx) => (float)random.NextDouble() * 1e-2f).ToArray();
                                        float[] wval = (new float[kwidth * kheight * inchannels * outchannels / 4]).Select((_, idx) => (float)random.NextDouble() * 1e-2f).ToArray();

                                        Quaternion[] ycval = (new Quaternion[yval.Length / 4])
                                            .Select((_, idx) => new Quaternion(yval[idx * 4], yval[idx * 4 + 1], yval[idx * 4 + 2], yval[idx * 4 + 3])).ToArray();

                                        Quaternion[] wcval = (new Quaternion[wval.Length / 4])
                                            .Select((_, idx) => new Quaternion(wval[idx * 4], wval[idx * 4 + 1], wval[idx * 4 + 2], wval[idx * 4 + 3])).ToArray();

                                        QuaternionMap2D y = new(outchannels / 4, outwidth, outheight, batch, ycval);
                                        QuaternionFilter2D w = new(inchannels / 4, outchannels / 4, kwidth, kheight, wcval);

                                        QuaternionMap2D x = Reference(y, w, inwidth, inheight, kwidth, kheight);

                                        OverflowCheckedTensor y_tensor = new(Shape.Map2D(outchannels, outwidth, outheight, batch), yval);
                                        OverflowCheckedTensor w_tensor = new(Shape.Kernel2D(inchannels, outchannels / 4, kwidth, kheight), wval);

                                        OverflowCheckedTensor x_tensor = new(Shape.Map2D(inchannels, inwidth, inheight, batch));

                                        QuaternionDeconvolution2D ope = new(outwidth, outheight, outchannels, inchannels, kwidth, kheight, gradmode: false, batch);

                                        ope.Execute(y_tensor, w_tensor, x_tensor);

                                        float[] x_expect = x.ToArray();
                                        float[] x_actual = x_tensor.State.Value;

                                        CollectionAssert.AreEqual(yval, y_tensor.State.Value);
                                        CollectionAssert.AreEqual(wval, w_tensor.State.Value);

                                        AssertError.Tolerance(x_expect, x_actual, 1e-6f, 1e-4f, ref max_err, $"mismatch value {inchannels},{outchannels},{kwidth},{kheight},{inwidth},{inheight},{batch}");

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

                                        float[] yval = (new float[outwidth * outheight * outchannels * batch]).Select((_, idx) => (float)random.NextDouble() * 1e-2f).ToArray();
                                        float[] wval = (new float[kwidth * kheight * inchannels * outchannels / 4]).Select((_, idx) => (float)random.NextDouble() * 1e-2f).ToArray();

                                        Quaternion[] ycval = (new Quaternion[yval.Length / 4])
                                            .Select((_, idx) => new Quaternion(yval[idx * 4], yval[idx * 4 + 1], yval[idx * 4 + 2], yval[idx * 4 + 3])).ToArray();

                                        Quaternion[] wcval = (new Quaternion[wval.Length / 4])
                                            .Select((_, idx) => new Quaternion(wval[idx * 4], wval[idx * 4 + 1], wval[idx * 4 + 2], wval[idx * 4 + 3])).ToArray();

                                        QuaternionMap2D y = new(outchannels / 4, outwidth, outheight, batch, ycval);
                                        QuaternionFilter2D w = new(inchannels / 4, outchannels / 4, kwidth, kheight, wcval);

                                        QuaternionMap2D x = Reference(y, w, inwidth, inheight, kwidth, kheight);

                                        OverflowCheckedTensor y_tensor = new(Shape.Map2D(outchannels, outwidth, outheight, batch), yval);
                                        OverflowCheckedTensor w_tensor = new(Shape.Kernel2D(inchannels, outchannels / 4, kwidth, kheight), wval);

                                        OverflowCheckedTensor x_tensor = new(Shape.Map2D(inchannels, inwidth, inheight, batch));

                                        QuaternionDeconvolution2D ope = new(outwidth, outheight, outchannels, inchannels, kwidth, kheight, gradmode: false, batch);

                                        ope.Execute(y_tensor, w_tensor, x_tensor);

                                        float[] x_expect = x.ToArray();
                                        float[] x_actual = x_tensor.State.Value;

                                        CollectionAssert.AreEqual(yval, y_tensor.State.Value);
                                        CollectionAssert.AreEqual(wval, w_tensor.State.Value);

                                        AssertError.Tolerance(x_expect, x_actual, 1e-7f, 1e-5f, ref max_err, $"mismatch value {inchannels},{outchannels},{kwidth},{kheight},{inwidth},{inheight},{batch}");

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

            float[] yval = (new float[outwidth * outheight * outchannels * batch]).Select((_, idx) => (float)random.NextDouble() * 1e-2f).ToArray();
            float[] wval = (new float[kwidth * kheight * inchannels * outchannels / 4]).Select((_, idx) => (float)random.NextDouble() * 1e-2f).ToArray();

            Quaternion[] ycval = (new Quaternion[yval.Length / 4])
                .Select((_, idx) => new Quaternion(yval[idx * 4], yval[idx * 4 + 1], yval[idx * 4 + 2], yval[idx * 4 + 3])).ToArray();

            Quaternion[] wcval = (new Quaternion[wval.Length / 4])
                .Select((_, idx) => new Quaternion(wval[idx * 4], wval[idx * 4 + 1], wval[idx * 4 + 2], wval[idx * 4 + 3])).ToArray();

            QuaternionMap2D y = new(outchannels / 4, outwidth, outheight, batch, ycval);
            QuaternionFilter2D w = new(inchannels / 4, outchannels / 4, kwidth, kheight, wcval);

            QuaternionMap2D x = Reference(y, w, inwidth, inheight, kwidth, kheight);

            OverflowCheckedTensor y_tensor = new(Shape.Map2D(outchannels, outwidth, outheight, batch), yval);
            OverflowCheckedTensor w_tensor = new(Shape.Kernel2D(inchannels, outchannels / 4, kwidth, kheight), wval);

            OverflowCheckedTensor x_tensor = new(Shape.Map2D(inchannels, inwidth, inheight, batch));

            QuaternionDeconvolution2D ope = new(outwidth, outheight, outchannels, inchannels, kwidth, kheight, gradmode: false, batch);

            ope.Execute(y_tensor, w_tensor, x_tensor);

            float[] x_expect = x.ToArray();
            float[] x_actual = x_tensor.State.Value;

            CollectionAssert.AreEqual(yval, y_tensor.State.Value);
            CollectionAssert.AreEqual(wval, w_tensor.State.Value);

            AssertError.Tolerance(x_expect, x_actual, 1e-7f, 1e-5f, ref max_err, $"mismatch value {inchannels},{outchannels},{kwidth},{kheight},{inwidth},{inheight},{batch}");

            Console.WriteLine($"pass: {inchannels},{outchannels},{kwidth},{kheight},{inwidth},{inheight},{batch}");

            Console.WriteLine($"maxerr:{max_err}");
        }

        [TestMethod]
        public void SpeedFPTest() {
            TensorShaderCudaBackend.Environment.Precision = TensorShaderCudaBackend.Environment.PrecisionMode.Float;
            TensorShaderCudaBackend.Environment.CudnnEnabled = false;

            int inwidth = 512, inheight = 512, inchannels = 32, outchannels = 32, ksize = 3;
            int outwidth = inwidth - ksize + 1, outheight = inheight - ksize + 1;

            OverflowCheckedTensor y_tensor = new(Shape.Map2D(outchannels, outwidth, outheight));
            OverflowCheckedTensor w_tensor = new(Shape.Kernel2D(inchannels, outchannels / 4, ksize, ksize));

            OverflowCheckedTensor x_tensor = new(Shape.Map2D(inchannels, inwidth, inheight));

            QuaternionDeconvolution2D ope = new(outwidth, outheight, outchannels, inchannels, ksize, ksize);

            ope.Execute(y_tensor, w_tensor, x_tensor);

            Cuda.Profiler.Initialize("../../../../profiler.nvsetting", "../../nvprofiles/quaternion_deconvolution_2d_fp.nvvp");
            Cuda.Profiler.Start();

            ope.Execute(y_tensor, w_tensor, x_tensor);

            Cuda.Profiler.Stop();
        }

        [TestMethod]
        public void SpeedFFPTest() {
            TensorShaderCudaBackend.Environment.CudnnEnabled = false;
            TensorShaderCudaBackend.Environment.Precision = TensorShaderCudaBackend.Environment.PrecisionMode.FloatFloat;

            int inwidth = 512, inheight = 512, inchannels = 32, outchannels = 32, ksize = 3;
            int outwidth = inwidth - ksize + 1, outheight = inheight - ksize + 1;

            OverflowCheckedTensor y_tensor = new(Shape.Map2D(outchannels, outwidth, outheight));
            OverflowCheckedTensor w_tensor = new(Shape.Kernel2D(inchannels, outchannels / 4, ksize, ksize));

            OverflowCheckedTensor x_tensor = new(Shape.Map2D(inchannels, inwidth, inheight));

            QuaternionDeconvolution2D ope = new(outwidth, outheight, outchannels, inchannels, ksize, ksize);

            ope.Execute(y_tensor, w_tensor, x_tensor);

            Cuda.Profiler.Initialize("../../../../profiler.nvsetting", "../../nvprofiles/quaternion_deconvolution_2d_ffp.nvvp");
            Cuda.Profiler.Start();

            ope.Execute(y_tensor, w_tensor, x_tensor);

            Cuda.Profiler.Stop();
        }

        public static QuaternionMap2D Reference(QuaternionMap2D y, QuaternionFilter2D w, int inw, int inh, int kwidth, int kheight) {
            int inchannels = w.InChannels, outchannels = w.OutChannels, batch = y.Batch;
            int outw = inw - kwidth + 1, outh = inh - kheight + 1;

            if (y.Width != outw || y.Height != outh) {
                throw new ArgumentException("mismatch shape");
            }

            QuaternionMap2D x = new(inchannels, inw, inh, batch);

            for (int kx, ky = 0; ky < kheight; ky++) {
                for (kx = 0; kx < kwidth; kx++) {
                    for (int th = 0; th < batch; th++) {
                        for (int ox, oy = 0; oy < outh; oy++) {
                            for (ox = 0; ox < outw; ox++) {
                                for (int outch = 0; outch < outchannels; outch++) {
                                    Quaternion v = y[outch, ox, oy, th];

                                    for (int inch = 0; inch < inchannels; inch++) {
                                        x[inch, kx + ox, ky + oy, th] += v * w[inch, outch, kx, ky];
                                    }
                                }
                            }
                        }
                    }
                }
            }

            return x;
        }

        [TestMethod]
        public void ReferenceTest() {
            int inchannels = 8, outchannels = 12, kwidth = 3, kheight = 5, inwidth = 7, inheight = 8, batch = 3;
            int outwidth = inwidth - kwidth + 1, outheight = inheight - kheight + 1;

            float[] yval = (new float[outwidth * outheight * outchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
            float[] wval = (new float[kwidth * kheight * outchannels * inchannels / 4]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

            Quaternion[] ycval = (new Quaternion[yval.Length / 4])
                .Select((_, idx) => new Quaternion(yval[idx * 4], yval[idx * 4 + 1], yval[idx * 4 + 2], yval[idx * 4 + 3])).ToArray();

            Quaternion[] wcval = (new Quaternion[wval.Length / 4])
                .Select((_, idx) => new Quaternion(wval[idx * 4], wval[idx * 4 + 1], wval[idx * 4 + 2], wval[idx * 4 + 3])).ToArray();

            QuaternionMap2D y = new(outchannels / 4, outwidth, outheight, batch, ycval);
            QuaternionFilter2D w = new(inchannels / 4, outchannels / 4, kwidth, kheight, wcval);

            QuaternionMap2D x = Reference(y, w, inwidth, inheight, kwidth, kheight);

            float[] x_expect = {
                -1.450000000e-02f,  8.272000000e-03f,  1.250800000e-02f,  1.035400000e-02f,  -1.433200000e-02f,  8.176000000e-03f,
                1.236400000e-02f,  1.023400000e-02f,  -5.304800000e-02f,  4.116800000e-02f,  4.942400000e-02f,  4.511600000e-02f,
                -5.242400000e-02f,  4.068800000e-02f,  4.884800000e-02f,  4.458800000e-02f,  -1.139160000e-01f,  9.696000000e-02f,
                1.090200000e-01f,  1.025580000e-01f,  -1.125480000e-01f,  9.580800000e-02f,  1.077240000e-01f,  1.013340000e-01f,
                -1.839000000e-01f,  1.673760000e-01f,  1.796520000e-01f,  1.727580000e-01f,  -1.816680000e-01f,  1.653600000e-01f,
                1.774920000e-01f,  1.706700000e-01f,  -2.538840000e-01f,  2.377920000e-01f,  2.502840000e-01f,  2.429580000e-01f,
                -2.507880000e-01f,  2.349120000e-01f,  2.472600000e-01f,  2.400060000e-01f,  -1.840880000e-01f,  1.739360000e-01f,
                1.820480000e-01f,  1.771640000e-01f,  -1.817360000e-01f,  1.717280000e-01f,  1.797440000e-01f,  1.749080000e-01f,
                -9.888400000e-02f,  9.409600000e-02f,  9.804400000e-02f,  9.560200000e-02f,  -9.756400000e-02f,  9.284800000e-02f,
                9.674800000e-02f,  9.433000000e-02f,  -1.512560000e-01f,  1.408160000e-01f,  1.487840000e-01f,  1.441880000e-01f,
                -1.494800000e-01f,  1.391840000e-01f,  1.470560000e-01f,  1.425080000e-01f,  -3.367840000e-01f,  3.170560000e-01f,
                3.325600000e-01f,  3.233680000e-01f,  -3.326560000e-01f,  3.132160000e-01f,  3.285280000e-01f,  3.194320000e-01f,
                -5.531280000e-01f,  5.252640000e-01f,  5.478720000e-01f,  5.340840000e-01f,  -5.460720000e-01f,  5.186400000e-01f,
                5.409600000e-01f,  5.273160000e-01f,  -6.775440000e-01f,  6.505440000e-01f,  6.735840000e-01f,  6.589320000e-01f,
                -6.687600000e-01f,  6.421920000e-01f,  6.649440000e-01f,  6.504360000e-01f,  -8.019600000e-01f,  7.758240000e-01f,
                7.992960000e-01f,  7.837800000e-01f,  -7.914480000e-01f,  7.657440000e-01f,  7.889280000e-01f,  7.735560000e-01f,
                -5.504800000e-01f,  5.342080000e-01f,  5.494240000e-01f,  5.390800000e-01f,  -5.428960000e-01f,  5.269120000e-01f,
                5.419360000e-01f,  5.316880000e-01f,  -2.820080000e-01f,  2.744480000e-01f,  2.818400000e-01f,  2.766680000e-01f,
                -2.779280000e-01f,  2.705120000e-01f,  2.778080000e-01f,  2.726840000e-01f,  -3.843480000e-01f,  3.717120000e-01f,
                3.829080000e-01f,  3.755820000e-01f,  -3.795240000e-01f,  3.671040000e-01f,  3.781560000e-01f,  3.709020000e-01f,
                -7.993680000e-01f,  7.758240000e-01f,  7.975680000e-01f,  7.829160000e-01f,  -7.888560000e-01f,  7.657440000e-01f,
                7.872000000e-01f,  7.726920000e-01f,  -1.239876000e+00f,  1.207152000e+00f,  1.238796000e+00f,  1.216818000e+00f,
                -1.222812000e+00f,  1.190736000e+00f,  1.221948000e+00f,  1.200186000e+00f,  -1.403172000e+00f,  1.371744000e+00f,
                1.404036000e+00f,  1.380762000e+00f,  -1.383516000e+00f,  1.352736000e+00f,  1.384596000e+00f,  1.361538000e+00f,
                -1.566468000e+00f,  1.536336000e+00f,  1.569276000e+00f,  1.544706000e+00f,  -1.544220000e+00f,  1.514736000e+00f,
                1.547244000e+00f,  1.522890000e+00f,  -1.047336000e+00f,  1.028976000e+00f,  1.050288000e+00f,  1.033908000e+00f,
                -1.031640000e+00f,  1.013712000e+00f,  1.034736000e+00f,  1.018500000e+00f,  -5.234520000e-01f,  5.151360000e-01f,
                5.254680000e-01f,  5.172780000e-01f,  -5.151720000e-01f,  5.070720000e-01f,  5.172600000e-01f,  5.091420000e-01f,
                -6.878560000e-01f,  6.750400000e-01f,  6.889600000e-01f,  6.786160000e-01f,  -6.785440000e-01f,  6.660160000e-01f,
                6.797440000e-01f,  6.694960000e-01f,  -1.388960000e+00f,  1.365632000e+00f,  1.392608000e+00f,  1.371920000e+00f,
                -1.369184000e+00f,  1.346432000e+00f,  1.373024000e+00f,  1.352528000e+00f,  -2.096400000e+00f,  2.064864000e+00f,
                2.104032000e+00f,  2.073000000e+00f,  -2.065008000e+00f,  2.034336000e+00f,  2.072928000e+00f,  2.042184000e+00f,
                -2.283024000e+00f,  2.253216000e+00f,  2.293248000e+00f,  2.260488000e+00f,  -2.248176000e+00f,  2.219232000e+00f,
                2.258688000e+00f,  2.226216000e+00f,  -2.469648000e+00f,  2.441568000e+00f,  2.482464000e+00f,  2.447976000e+00f,
                -2.431344000e+00f,  2.404128000e+00f,  2.444448000e+00f,  2.410248000e+00f,  -1.622816000e+00f,  1.606400000e+00f,
                1.632800000e+00f,  1.609808000e+00f,  -1.596128000e+00f,  1.580288000e+00f,  1.606304000e+00f,  1.583504000e+00f,
                -7.972960000e-01f,  7.902400000e-01f,  8.030080000e-01f,  7.915120000e-01f,  -7.833760000e-01f,  7.766080000e-01f,
                7.891840000e-01f,  7.777840000e-01f,  -5.202400000e-01f,  5.126080000e-01f,  5.230720000e-01f,  5.144560000e-01f,
                -5.109280000e-01f,  5.035840000e-01f,  5.138560000e-01f,  5.053360000e-01f,  -1.032992000e+00f,  1.020032000e+00f,
                1.040096000e+00f,  1.022864000e+00f,  -1.013216000e+00f,  1.000832000e+00f,  1.020512000e+00f,  1.003472000e+00f,
                -1.531344000e+00f,  1.515360000e+00f,  1.544160000e+00f,  1.518312000e+00f,  -1.499952000e+00f,  1.484832000e+00f,
                1.513056000e+00f,  1.487496000e+00f,  -1.655760000e+00f,  1.641504000e+00f,  1.671168000e+00f,  1.643592000e+00f,
                -1.620912000e+00f,  1.607520000e+00f,  1.636608000e+00f,  1.609320000e+00f,  -1.780176000e+00f,  1.767648000e+00f,
                1.798176000e+00f,  1.768872000e+00f,  -1.741872000e+00f,  1.730208000e+00f,  1.760160000e+00f,  1.731144000e+00f,
                -1.142432000e+00f,  1.136384000e+00f,  1.155872000e+00f,  1.136336000e+00f,  -1.115744000e+00f,  1.110272000e+00f,
                1.129376000e+00f,  1.110032000e+00f,  -5.467360000e-01f,  5.448640000e-01f,  5.541760000e-01f,  5.444080000e-01f,
                -5.328160000e-01f,  5.312320000e-01f,  5.403520000e-01f,  5.306800000e-01f,  -3.532440000e-01f,  3.505440000e-01f,
                3.576360000e-01f,  3.507420000e-01f,  -3.441000000e-01f,  3.416160000e-01f,  3.485640000e-01f,  3.417420000e-01f,
                -6.801360000e-01f,  6.764640000e-01f,  6.900000000e-01f,  6.762120000e-01f,  -6.609840000e-01f,  6.577440000e-01f,
                6.709920000e-01f,  6.573480000e-01f,  -9.754920000e-01f,  9.725760000e-01f,  9.919080000e-01f,  9.712260000e-01f,
                -9.454680000e-01f,  9.432000000e-01f,  9.621000000e-01f,  9.416340000e-01f,  -1.045476000e+00f,  1.043856000e+00f,
                1.063836000e+00f,  1.041858000e+00f,  -1.012860000e+00f,  1.011888000e+00f,  1.031436000e+00f,  1.009674000e+00f,
                -1.115460000e+00f,  1.115136000e+00f,  1.135764000e+00f,  1.112490000e+00f,  -1.080252000e+00f,  1.080576000e+00f,
                1.100772000e+00f,  1.077714000e+00f,  -6.896400000e-01f,  6.911520000e-01f,  7.042560000e-01f,  6.887400000e-01f,
                -6.653040000e-01f,  6.672480000e-01f,  6.800640000e-01f,  6.646920000e-01f,  -3.160920000e-01f,  3.177120000e-01f,
                3.239400000e-01f,  3.161820000e-01f,  -3.034920000e-01f,  3.053280000e-01f,  3.114120000e-01f,  3.037260000e-01f,
                -1.935920000e-01f,  1.938080000e-01f,  1.980320000e-01f,  1.931480000e-01f,  -1.860560000e-01f,  1.864160000e-01f,
                1.905440000e-01f,  1.857080000e-01f,  -3.557920000e-01f,  3.573760000e-01f,  3.653920000e-01f,  3.556240000e-01f,
                -3.401440000e-01f,  3.420160000e-01f,  3.498400000e-01f,  3.401680000e-01f,  -4.831440000e-01f,  4.872480000e-01f,
                4.986240000e-01f,  4.839720000e-01f,  -4.588080000e-01f,  4.633440000e-01f,  4.744320000e-01f,  4.599240000e-01f,
                -5.142480000e-01f,  5.192160000e-01f,  5.310240000e-01f,  5.155080000e-01f,  -4.881840000e-01f,  4.935840000e-01f,
                5.051040000e-01f,  4.897320000e-01f,  -5.453520000e-01f,  5.511840000e-01f,  5.634240000e-01f,  5.470440000e-01f,
                -5.175600000e-01f,  5.238240000e-01f,  5.357760000e-01f,  5.195400000e-01f,  -3.137440000e-01f,  3.187840000e-01f,
                3.265120000e-01f,  3.155920000e-01f,  -2.946400000e-01f,  2.999680000e-01f,  3.075040000e-01f,  2.966800000e-01f,
                -1.308080000e-01f,  1.339040000e-01f,  1.375520000e-01f,  1.320920000e-01f,  -1.209680000e-01f,  1.242080000e-01f,
                1.277600000e-01f,  1.223480000e-01f,  -6.720400000e-02f,  6.832000000e-02f,  7.018000000e-02f,  6.759400000e-02f,
                -6.271600000e-02f,  6.390400000e-02f,  6.571600000e-02f,  6.315400000e-02f,  -1.118000000e-01f,  1.146080000e-01f,
                1.181120000e-01f,  1.129400000e-01f,  -1.025360000e-01f,  1.054880000e-01f,  1.088960000e-01f,  1.037720000e-01f,
                -1.320600000e-01f,  1.371360000e-01f,  1.420680000e-01f,  1.343100000e-01f,  -1.177320000e-01f,  1.230240000e-01f,
                1.278120000e-01f,  1.201260000e-01f,  -1.398360000e-01f,  1.453440000e-01f,  1.504920000e-01f,  1.423020000e-01f,
                -1.246440000e-01f,  1.303680000e-01f,  1.353720000e-01f,  1.272540000e-01f,  -1.476120000e-01f,  1.535520000e-01f,
                1.589160000e-01f,  1.502940000e-01f,  -1.315560000e-01f,  1.377120000e-01f,  1.429320000e-01f,  1.343820000e-01f,
                -6.658400000e-02f,  7.112000000e-02f,  7.448000000e-02f,  6.873200000e-02f,  -5.559200000e-02f,  6.027200000e-02f,
                6.353600000e-02f,  5.783600000e-02f,  -1.680400000e-02f,  1.936000000e-02f,  2.093200000e-02f,  1.805800000e-02f,
                -1.116400000e-02f,  1.379200000e-02f,  1.531600000e-02f,  1.246600000e-02f,  -5.156200000e-01f,  5.122720000e-01f,
                5.179480000e-01f,  5.129140000e-01f,  -5.096920000e-01f,  5.064160000e-01f,  5.120440000e-01f,  5.070340000e-01f,
                -1.020728000e+00f,  1.014608000e+00f,  1.025744000e+00f,  1.015676000e+00f,  -1.008584000e+00f,  1.002608000e+00f,
                1.013648000e+00f,  1.003628000e+00f,  -1.513596000e+00f,  1.505280000e+00f,  1.521660000e+00f,  1.506558000e+00f,
                -1.494948000e+00f,  1.486848000e+00f,  1.503084000e+00f,  1.488054000e+00f,  -1.583580000e+00f,  1.575696000e+00f,
                1.592292000e+00f,  1.576758000e+00f,  -1.564068000e+00f,  1.556400000e+00f,  1.572852000e+00f,  1.557390000e+00f,
                -1.653564000e+00f,  1.646112000e+00f,  1.662924000e+00f,  1.646958000e+00f,  -1.633188000e+00f,  1.625952000e+00f,
                1.642620000e+00f,  1.626726000e+00f,  -1.082648000e+00f,  1.078256000e+00f,  1.089248000e+00f,  1.078604000e+00f,
                -1.068776000e+00f,  1.064528000e+00f,  1.075424000e+00f,  1.064828000e+00f,  -5.308840000e-01f,  5.289760000e-01f,
                5.343640000e-01f,  5.290420000e-01f,  -5.238040000e-01f,  5.219680000e-01f,  5.273080000e-01f,  5.220100000e-01f,
                -1.049816000e+00f,  1.045136000e+00f,  1.055984000e+00f,  1.045628000e+00f,  -1.036520000e+00f,  1.031984000e+00f,
                1.042736000e+00f,  1.032428000e+00f,  -2.064784000e+00f,  2.056576000e+00f,  2.077840000e+00f,  2.057128000e+00f,
                -2.037616000e+00f,  2.029696000e+00f,  2.050768000e+00f,  2.030152000e+00f,  -3.041448000e+00f,  3.030864000e+00f,
                3.062112000e+00f,  3.031044000e+00f,  -2.999832000e+00f,  2.989680000e+00f,  3.020640000e+00f,  2.989716000e+00f,
                -3.165864000e+00f,  3.156144000e+00f,  3.187824000e+00f,  3.155892000e+00f,  -3.122520000e+00f,  3.113232000e+00f,
                3.144624000e+00f,  3.112836000e+00f,  -3.290280000e+00f,  3.281424000e+00f,  3.313536000e+00f,  3.280740000e+00f,
                -3.245208000e+00f,  3.236784000e+00f,  3.268608000e+00f,  3.235956000e+00f,  -2.140240000e+00f,  2.135488000e+00f,
                2.156464000e+00f,  2.134600000e+00f,  -2.109616000e+00f,  2.105152000e+00f,  2.125936000e+00f,  2.104168000e+00f,
                -1.042328000e+00f,  1.040528000e+00f,  1.050800000e+00f,  1.039868000e+00f,  -1.026728000e+00f,  1.025072000e+00f,
                1.035248000e+00f,  1.024364000e+00f,  -1.576668000e+00f,  1.572672000e+00f,  1.588188000e+00f,  1.572222000e+00f,
                -1.554564000e+00f,  1.550784000e+00f,  1.566156000e+00f,  1.550262000e+00f,  -3.080328000e+00f,  3.074064000e+00f,
                3.104448000e+00f,  3.072516000e+00f,  -3.035256000e+00f,  3.029424000e+00f,  3.059520000e+00f,  3.027732000e+00f,
                -4.505796000e+00f,  4.498992000e+00f,  4.543596000e+00f,  4.495698000e+00f,  -4.436892000e+00f,  4.430736000e+00f,
                4.474908000e+00f,  4.427226000e+00f,  -4.669092000e+00f,  4.663584000e+00f,  4.708836000e+00f,  4.659642000e+00f,
                -4.597596000e+00f,  4.592736000e+00f,  4.637556000e+00f,  4.588578000e+00f,  -4.832388000e+00f,  4.828176000e+00f,
                4.874076000e+00f,  4.823586000e+00f,  -4.758300000e+00f,  4.754736000e+00f,  4.800204000e+00f,  4.749930000e+00f,
                -3.120936000e+00f,  3.119856000e+00f,  3.149808000e+00f,  3.116148000e+00f,  -3.070680000e+00f,  3.070032000e+00f,
                3.099696000e+00f,  3.066180000e+00f,  -1.508412000e+00f,  1.508736000e+00f,  1.523388000e+00f,  1.506558000e+00f,
                -1.482852000e+00f,  1.483392000e+00f,  1.497900000e+00f,  1.481142000e+00f,  -2.070256000e+00f,  2.068960000e+00f,
                2.088640000e+00f,  2.066776000e+00f,  -2.037904000e+00f,  2.036896000e+00f,  2.056384000e+00f,  2.034616000e+00f,
                -4.015520000e+00f,  4.015232000e+00f,  4.053728000e+00f,  4.010000000e+00f,  -3.949664000e+00f,  3.949952000e+00f,
                3.988064000e+00f,  3.944528000e+00f,  -5.828880000e+00f,  5.831904000e+00f,  5.888352000e+00f,  5.822760000e+00f,
                -5.728368000e+00f,  5.732256000e+00f,  5.788128000e+00f,  5.722824000e+00f,  -6.015504000e+00f,  6.020256000e+00f,
                6.077568000e+00f,  6.010248000e+00f,  -5.911536000e+00f,  5.917152000e+00f,  5.973888000e+00f,  5.906856000e+00f,
                -6.202128000e+00f,  6.208608000e+00f,  6.266784000e+00f,  6.197736000e+00f,  -6.094704000e+00f,  6.102048000e+00f,
                6.159648000e+00f,  6.090888000e+00f,  -3.972896000e+00f,  3.979520000e+00f,  4.017440000e+00f,  3.971408000e+00f,
                -3.900128000e+00f,  3.907328000e+00f,  3.944864000e+00f,  3.899024000e+00f,  -1.903216000e+00f,  1.907680000e+00f,
                1.926208000e+00f,  1.903192000e+00f,  -1.866256000e+00f,  1.871008000e+00f,  1.889344000e+00f,  1.866424000e+00f,
                -1.487920000e+00f,  1.491808000e+00f,  1.508032000e+00f,  1.487896000e+00f,  -1.455568000e+00f,  1.459744000e+00f,
                1.475776000e+00f,  1.455736000e+00f,  -2.830112000e+00f,  2.840192000e+00f,  2.871776000e+00f,  2.831504000e+00f,
                -2.764256000e+00f,  2.774912000e+00f,  2.806112000e+00f,  2.766032000e+00f,  -4.019664000e+00f,  4.038240000e+00f,
                4.084320000e+00f,  4.023912000e+00f,  -3.919152000e+00f,  3.938592000e+00f,  3.984096000e+00f,  3.923976000e+00f,
                -4.144080000e+00f,  4.164384000e+00f,  4.211328000e+00f,  4.149192000e+00f,  -4.040112000e+00f,  4.061280000e+00f,
                4.107648000e+00f,  4.045800000e+00f,  -4.268496000e+00f,  4.290528000e+00f,  4.338336000e+00f,  4.274472000e+00f,
                -4.161072000e+00f,  4.183968000e+00f,  4.231200000e+00f,  4.167624000e+00f,  -2.663072000e+00f,  2.680064000e+00f,
                2.711072000e+00f,  2.668496000e+00f,  -2.590304000e+00f,  2.607872000e+00f,  2.638496000e+00f,  2.596112000e+00f,
                -1.237936000e+00f,  1.247584000e+00f,  1.262656000e+00f,  1.241368000e+00f,  -1.200976000e+00f,  1.210912000e+00f,
                1.225792000e+00f,  1.204600000e+00f,  -9.234840000e-01f,  9.294240000e-01f,  9.408360000e-01f,  9.253020000e-01f,
                -8.970600000e-01f,  9.032160000e-01f,  9.144840000e-01f,  8.990220000e-01f,  -1.716936000e+00f,  1.730544000e+00f,
                1.752720000e+00f,  1.721652000e+00f,  -1.663224000e+00f,  1.677264000e+00f,  1.699152000e+00f,  1.668228000e+00f,
                -2.375172000e+00f,  2.398176000e+00f,  2.430468000e+00f,  2.383866000e+00f,  -2.293308000e+00f,  2.316960000e+00f,
                2.348820000e+00f,  2.302434000e+00f,  -2.445156000e+00f,  2.469456000e+00f,  2.502396000e+00f,  2.454498000e+00f,
                -2.360700000e+00f,  2.385648000e+00f,  2.418156000e+00f,  2.370474000e+00f,  -2.515140000e+00f,  2.540736000e+00f,
                2.574324000e+00f,  2.525130000e+00f,  -2.428092000e+00f,  2.454336000e+00f,  2.487492000e+00f,  2.438514000e+00f,
                -1.519080000e+00f,  1.537872000e+00f,  1.559616000e+00f,  1.526820000e+00f,  -1.460184000e+00f,  1.479408000e+00f,
                1.500864000e+00f,  1.468212000e+00f,  -6.789720000e-01f,  6.892320000e-01f,  6.997800000e-01f,  6.833820000e-01f,
                -6.490920000e-01f,  6.595680000e-01f,  6.699720000e-01f,  6.536460000e-01f,  -4.700720000e-01f,  4.760480000e-01f,
                4.831520000e-01f,  4.725080000e-01f,  -4.510160000e-01f,  4.571360000e-01f,  4.641440000e-01f,  4.535480000e-01f,
                -8.396320000e-01f,  8.527360000e-01f,  8.665120000e-01f,  8.452240000e-01f,  -8.009440000e-01f,  8.143360000e-01f,
                8.279200000e-01f,  8.067280000e-01f,  -1.105224000e+00f,  1.126608000e+00f,  1.146624000e+00f,  1.114692000e+00f,
                -1.046328000e+00f,  1.068144000e+00f,  1.087872000e+00f,  1.056084000e+00f,  -1.136328000e+00f,  1.158576000e+00f,
                1.179024000e+00f,  1.146228000e+00f,  -1.075704000e+00f,  1.098384000e+00f,  1.118544000e+00f,  1.085892000e+00f,
                -1.167432000e+00f,  1.190544000e+00f,  1.211424000e+00f,  1.177764000e+00f,  -1.105080000e+00f,  1.128624000e+00f,
                1.149216000e+00f,  1.115700000e+00f,  -6.593440000e-01f,  6.759040000e-01f,  6.893920000e-01f,  6.669520000e-01f,
                -6.172000000e-01f,  6.340480000e-01f,  6.473440000e-01f,  6.250000000e-01f,  -2.690480000e-01f,  2.779040000e-01f,
                2.844320000e-01f,  2.732120000e-01f,  -2.476880000e-01f,  2.566880000e-01f,  2.631200000e-01f,  2.519480000e-01f,
                -1.536040000e-01f,  1.576000000e-01f,  1.609000000e-01f,  1.554340000e-01f,  -1.433560000e-01f,  1.474240000e-01f,
                1.506760000e-01f,  1.452340000e-01f,  -2.500400000e-01f,  2.586080000e-01f,  2.649920000e-01f,  2.540600000e-01f,
                -2.292560000e-01f,  2.379680000e-01f,  2.442560000e-01f,  2.333720000e-01f,  -2.875800000e-01f,  3.012960000e-01f,
                3.105480000e-01f,  2.941500000e-01f,  -2.559720000e-01f,  2.699040000e-01f,  2.790120000e-01f,  2.626860000e-01f,
                -2.953560000e-01f,  3.095040000e-01f,  3.189720000e-01f,  3.021420000e-01f,  -2.628840000e-01f,  2.772480000e-01f,
                2.865720000e-01f,  2.698140000e-01f,  -3.031320000e-01f,  3.177120000e-01f,  3.273960000e-01f,  3.101340000e-01f,
                -2.697960000e-01f,  2.845920000e-01f,  2.941320000e-01f,  2.769420000e-01f,  -1.357040000e-01f,  1.460000000e-01f,
                1.522400000e-01f,  1.407320000e-01f,  -1.131920000e-01f,  1.236320000e-01f,  1.297760000e-01f,  1.183160000e-01f,
                -3.408400000e-02f,  3.952000000e-02f,  4.253200000e-02f,  3.677800000e-02f,  -2.268400000e-02f,  2.819200000e-02f,
                3.115600000e-02f,  2.542600000e-02f,  -1.016740000e+00f,  1.016272000e+00f,  1.023388000e+00f,  1.015474000e+00f,
                -1.005052000e+00f,  1.004656000e+00f,  1.011724000e+00f,  1.003834000e+00f,  -1.988408000e+00f,  1.988048000e+00f,
                2.002064000e+00f,  1.986236000e+00f,  -1.964744000e+00f,  1.964528000e+00f,  1.978448000e+00f,  1.962668000e+00f,
                -2.913276000e+00f,  2.913600000e+00f,  2.934300000e+00f,  2.910558000e+00f,  -2.877348000e+00f,  2.877888000e+00f,
                2.898444000e+00f,  2.874774000e+00f,  -2.983260000e+00f,  2.984016000e+00f,  3.004932000e+00f,  2.980758000e+00f,
                -2.946468000e+00f,  2.947440000e+00f,  2.968212000e+00f,  2.944110000e+00f,  -3.053244000e+00f,  3.054432000e+00f,
                3.075564000e+00f,  3.050958000e+00f,  -3.015588000e+00f,  3.016992000e+00f,  3.037980000e+00f,  3.013446000e+00f,
                -1.981208000e+00f,  1.982576000e+00f,  1.996448000e+00f,  1.980044000e+00f,  -1.955816000e+00f,  1.957328000e+00f,
                1.971104000e+00f,  1.954748000e+00f,  -9.628840000e-01f,  9.638560000e-01f,  9.706840000e-01f,  9.624820000e-01f,
                -9.500440000e-01f,  9.510880000e-01f,  9.578680000e-01f,  9.496900000e-01f,  -1.948376000e+00f,  1.949456000e+00f,
                1.963184000e+00f,  1.947068000e+00f,  -1.923560000e+00f,  1.924784000e+00f,  1.938416000e+00f,  1.922348000e+00f,
                -3.792784000e+00f,  3.796096000e+00f,  3.823120000e+00f,  3.790888000e+00f,  -3.742576000e+00f,  3.746176000e+00f,
                3.773008000e+00f,  3.740872000e+00f,  -5.529768000e+00f,  5.536464000e+00f,  5.576352000e+00f,  5.528004000e+00f,
                -5.453592000e+00f,  5.460720000e+00f,  5.500320000e+00f,  5.452116000e+00f,  -5.654184000e+00f,  5.661744000e+00f,
                5.702064000e+00f,  5.652852000e+00f,  -5.576280000e+00f,  5.584272000e+00f,  5.624304000e+00f,  5.575236000e+00f,
                -5.778600000e+00f,  5.787024000e+00f,  5.827776000e+00f,  5.777700000e+00f,  -5.698968000e+00f,  5.707824000e+00f,
                5.748288000e+00f,  5.698356000e+00f,  -3.730000000e+00f,  3.736768000e+00f,  3.763504000e+00f,  3.730120000e+00f,
                -3.676336000e+00f,  3.683392000e+00f,  3.709936000e+00f,  3.676648000e+00f,  -1.802648000e+00f,  1.806608000e+00f,
                1.819760000e+00f,  1.803068000e+00f,  -1.775528000e+00f,  1.779632000e+00f,  1.792688000e+00f,  1.776044000e+00f,
                -2.768988000e+00f,  2.773632000e+00f,  2.793468000e+00f,  2.768862000e+00f,  -2.729604000e+00f,  2.734464000e+00f,
                2.754156000e+00f,  2.729622000e+00f,  -5.361288000e+00f,  5.372304000e+00f,  5.411328000e+00f,  5.362116000e+00f,
                -5.281656000e+00f,  5.293104000e+00f,  5.331840000e+00f,  5.282772000e+00f,  -7.771716000e+00f,  7.790832000e+00f,
                7.848396000e+00f,  7.774578000e+00f,  -7.650972000e+00f,  7.670736000e+00f,  7.727868000e+00f,  7.654266000e+00f,
                -7.935012000e+00f,  7.955424000e+00f,  8.013636000e+00f,  7.938522000e+00f,  -7.811676000e+00f,  7.832736000e+00f,
                7.890516000e+00f,  7.815618000e+00f,  -8.098308000e+00f,  8.120016000e+00f,  8.178876000e+00f,  8.102466000e+00f,
                -7.972380000e+00f,  7.994736000e+00f,  8.053164000e+00f,  7.976970000e+00f,  -5.194536000e+00f,  5.210736000e+00f,
                5.249328000e+00f,  5.198388000e+00f,  -5.109720000e+00f,  5.126352000e+00f,  5.164656000e+00f,  5.113860000e+00f,
                -2.493372000e+00f,  2.502336000e+00f,  2.521308000e+00f,  2.495838000e+00f,  -2.450532000e+00f,  2.459712000e+00f,
                2.478540000e+00f,  2.453142000e+00f,  -3.452656000e+00f,  3.462880000e+00f,  3.488320000e+00f,  3.454936000e+00f,
                -3.397264000e+00f,  3.407776000e+00f,  3.433024000e+00f,  3.399736000e+00f,  -6.642080000e+00f,  6.664832000e+00f,
                6.714848000e+00f,  6.648080000e+00f,  -6.530144000e+00f,  6.553472000e+00f,  6.603104000e+00f,  6.536528000e+00f,
                -9.561360000e+00f,  9.598944000e+00f,  9.672672000e+00f,  9.572520000e+00f,  -9.391728000e+00f,  9.430176000e+00f,
                9.503328000e+00f,  9.403464000e+00f,  -9.747984000e+00f,  9.787296000e+00f,  9.861888000e+00f,  9.760008000e+00f,
                -9.574896000e+00f,  9.615072000e+00f,  9.689088000e+00f,  9.587496000e+00f,  -9.934608000e+00f,  9.975648000e+00f,
                1.005110400e+01f,  9.947496000e+00f,  -9.758064000e+00f,  9.799968000e+00f,  9.874848000e+00f,  9.771528000e+00f,
                -6.322976000e+00f,  6.352640000e+00f,  6.402080000e+00f,  6.333008000e+00f,  -6.204128000e+00f,  6.234368000e+00f,
                6.283424000e+00f,  6.214544000e+00f,  -3.009136000e+00f,  3.025120000e+00f,  3.049408000e+00f,  3.014872000e+00f,
                -2.949136000e+00f,  2.965408000e+00f,  2.989504000e+00f,  2.955064000e+00f,  -2.455600000e+00f,  2.471008000e+00f,
                2.492992000e+00f,  2.461336000e+00f,  -2.400208000e+00f,  2.415904000e+00f,  2.437696000e+00f,  2.406136000e+00f,
                -4.627232000e+00f,  4.660352000e+00f,  4.703456000e+00f,  4.640144000e+00f,  -4.515296000e+00f,  4.548992000e+00f,
                4.591712000e+00f,  4.528592000e+00f,  -6.507984000e+00f,  6.561120000e+00f,  6.624480000e+00f,  6.529512000e+00f,
                -6.338352000e+00f,  6.392352000e+00f,  6.455136000e+00f,  6.360456000e+00f,  -6.632400000e+00f,  6.687264000e+00f,
                6.751488000e+00f,  6.654792000e+00f,  -6.459312000e+00f,  6.515040000e+00f,  6.578688000e+00f,  6.482280000e+00f,
                -6.756816000e+00f,  6.813408000e+00f,  6.878496000e+00f,  6.780072000e+00f,  -6.580272000e+00f,  6.637728000e+00f,
                6.702240000e+00f,  6.604104000e+00f,  -4.183712000e+00f,  4.223744000e+00f,  4.266272000e+00f,  4.200656000e+00f,
                -4.064864000e+00f,  4.105472000e+00f,  4.147616000e+00f,  4.082192000e+00f,  -1.929136000e+00f,  1.950304000e+00f,
                1.971136000e+00f,  1.938328000e+00f,  -1.869136000e+00f,  1.890592000e+00f,  1.911232000e+00f,  1.878520000e+00f,
                -1.493724000e+00f,  1.508304000e+00f,  1.524036000e+00f,  1.499862000e+00f,  -1.450020000e+00f,  1.464816000e+00f,
                1.480404000e+00f,  1.456302000e+00f,  -2.753736000e+00f,  2.784624000e+00f,  2.815440000e+00f,  2.767092000e+00f,
                -2.665464000e+00f,  2.696784000e+00f,  2.727312000e+00f,  2.679108000e+00f,  -3.774852000e+00f,  3.823776000e+00f,
                3.869028000e+00f,  3.796506000e+00f,  -3.641148000e+00f,  3.690720000e+00f,  3.735540000e+00f,  3.663234000e+00f,
                -3.844836000e+00f,  3.895056000e+00f,  3.940956000e+00f,  3.867138000e+00f,  -3.708540000e+00f,  3.759408000e+00f,
                3.804876000e+00f,  3.731274000e+00f,  -3.914820000e+00f,  3.966336000e+00f,  4.012884000e+00f,  3.937770000e+00f,
                -3.775932000e+00f,  3.828096000e+00f,  3.874212000e+00f,  3.799314000e+00f,  -2.348520000e+00f,  2.384592000e+00f,
                2.414976000e+00f,  2.364900000e+00f,  -2.255064000e+00f,  2.291568000e+00f,  2.321664000e+00f,  2.271732000e+00f,
                -1.041852000e+00f,  1.060752000e+00f,  1.075620000e+00f,  1.050582000e+00f,  -9.946920000e-01f,  1.013808000e+00f,
                1.028532000e+00f,  1.003566000e+00f,  -7.465520000e-01f,  7.582880000e-01f,  7.682720000e-01f,  7.518680000e-01f,
                -7.159760000e-01f,  7.278560000e-01f,  7.377440000e-01f,  7.213880000e-01f,  -1.323472000e+00f,  1.348096000e+00f,
                1.367632000e+00f,  1.334824000e+00f,  -1.261744000e+00f,  1.286656000e+00f,  1.306000000e+00f,  1.273288000e+00f,
                -1.727304000e+00f,  1.765968000e+00f,  1.794624000e+00f,  1.745412000e+00f,  -1.633848000e+00f,  1.672944000e+00f,
                1.701312000e+00f,  1.652244000e+00f,  -1.758408000e+00f,  1.797936000e+00f,  1.827024000e+00f,  1.776948000e+00f,
                -1.663224000e+00f,  1.703184000e+00f,  1.731984000e+00f,  1.682052000e+00f,  -1.789512000e+00f,  1.829904000e+00f,
                1.859424000e+00f,  1.808484000e+00f,  -1.692600000e+00f,  1.733424000e+00f,  1.762656000e+00f,  1.711860000e+00f,
                -1.004944000e+00f,  1.033024000e+00f,  1.052272000e+00f,  1.018312000e+00f,  -9.397600000e-01f,  9.681280000e-01f,
                9.871840000e-01f,  9.533200000e-01f,  -4.072880000e-01f,  4.219040000e-01f,  4.313120000e-01f,  4.143320000e-01f,
                -3.744080000e-01f,  3.891680000e-01f,  3.984800000e-01f,  3.815480000e-01f,  -2.400040000e-01f,  2.468800000e-01f,
                2.516200000e-01f,  2.432740000e-01f,  -2.239960000e-01f,  2.309440000e-01f,  2.356360000e-01f,  2.273140000e-01f,
                -3.882800000e-01f,  4.026080000e-01f,  4.118720000e-01f,  3.951800000e-01f,  -3.559760000e-01f,  3.704480000e-01f,
                3.796160000e-01f,  3.629720000e-01f,  -4.431000000e-01f,  4.654560000e-01f,  4.790280000e-01f,  4.539900000e-01f,
                -3.942120000e-01f,  4.167840000e-01f,  4.302120000e-01f,  4.052460000e-01f,  -4.508760000e-01f,  4.736640000e-01f,
                4.874520000e-01f,  4.619820000e-01f,  -4.011240000e-01f,  4.241280000e-01f,  4.377720000e-01f,  4.123740000e-01f,
                -4.586520000e-01f,  4.818720000e-01f,  4.958760000e-01f,  4.699740000e-01f,  -4.080360000e-01f,  4.314720000e-01f,
                4.453320000e-01f,  4.195020000e-01f,  -2.048240000e-01f,  2.208800000e-01f,  2.300000000e-01f,  2.127320000e-01f,
                -1.707920000e-01f,  1.869920000e-01f,  1.960160000e-01f,  1.787960000e-01f,  -5.136400000e-02f,  5.968000000e-02f,
                6.413200000e-02f,  5.549800000e-02f,  -3.420400000e-02f,  4.259200000e-02f,  4.699600000e-02f,  3.838600000e-02f,
            };

            float[] x_actual = x.ToArray();

            AssertError.Tolerance(x_expect, x_actual, 1e-7f, 1e-5f, $"mismatch value {inchannels},{outchannels},{kwidth},{kheight},{inwidth},{inheight},{batch}");
        }
    }
}
