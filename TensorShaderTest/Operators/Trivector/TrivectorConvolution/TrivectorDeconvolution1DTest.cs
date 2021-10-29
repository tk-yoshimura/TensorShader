using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;
using TensorShader;
using TensorShader.Operators.TrivectorConvolution;
using TensorShaderCudaBackend.API;

namespace TensorShaderTest.Operators.Trivector {
    [TestClass]
    public class TrivectorDeconvolution1DTest {
        [TestMethod]
        public void ExecuteFPTest() {
            TensorShaderCudaBackend.Environment.Precision = TensorShaderCudaBackend.Environment.PrecisionMode.Float;
            TensorShaderCudaBackend.Environment.CudnnEnabled = false;

            Random random = new(1234);

            float max_err = 0;

            foreach (int batch in new int[] { 1, 2 }) {
                foreach (int inchannels in new int[] { 3, 6, 9, 15, 21, 33 }) {
                    foreach (int outchannels in new int[] { 3, 6, 9, 15, 21, 33 }) {
                        foreach (int kwidth in new int[] { 1, 3, 5 }) {
                            foreach (int inwidth in new int[] { kwidth, kwidth * 2, 8, 9, 13, 17, 25 }) {
                                int outwidth = inwidth - kwidth + 1;

                                float[] yval = (new float[outwidth * outchannels * batch]).Select((_, idx) => (float)random.NextDouble() * 1e-2f).ToArray();
                                float[] wval = (new float[kwidth * inchannels * outchannels / 9 * 4]).Select((_, idx) => (float)random.NextDouble() * 1e-2f).ToArray();

                                Trivector[] ycval = (new Trivector[yval.Length / 3])
                                    .Select((_, idx) => new Trivector(yval[idx * 3], yval[idx * 3 + 1], yval[idx * 3 + 2])).ToArray();

                                Quaternion.Quaternion[] wcval = (new Quaternion.Quaternion[wval.Length / 4])
                                    .Select((_, idx) => new Quaternion.Quaternion(wval[idx * 4], wval[idx * 4 + 1], wval[idx * 4 + 2], wval[idx * 4 + 3])).ToArray();

                                TrivectorMap1D y = new(outchannels / 3, outwidth, batch, ycval);
                                Quaternion.QuaternionFilter1D w = new(inchannels / 3, outchannels / 3, kwidth, wcval);

                                TrivectorMap1D x = Reference(y, w, inwidth, kwidth);

                                OverflowCheckedTensor y_tensor = new(Shape.Map1D(outchannels, outwidth, batch), yval);
                                OverflowCheckedTensor w_tensor = new(Shape.Kernel1D(inchannels / 3 * 4, outchannels / 3, kwidth), wval);

                                OverflowCheckedTensor x_tensor = new(Shape.Map1D(inchannels, inwidth, batch));

                                TrivectorDeconvolution1D ope = new(outwidth, outchannels, inchannels, kwidth, gradmode: false, batch);

                                ope.Execute(y_tensor, w_tensor, x_tensor);

                                float[] x_expect = x.ToArray();
                                float[] x_actual = x_tensor.State.Value;

                                CollectionAssert.AreEqual(yval, y_tensor.State.Value);
                                CollectionAssert.AreEqual(wval, w_tensor.State.Value);

                                AssertError.Tolerance(x_expect, x_actual, 1e-6f, 1e-4f, ref max_err, $"mismatch value {inchannels},{outchannels},{kwidth},{inwidth},{batch}");

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

            Random random = new(1234);

            float max_err = 0;

            foreach (int batch in new int[] { 1, 2 }) {
                foreach (int inchannels in new int[] { 3, 6, 9, 15, 21, 33 }) {
                    foreach (int outchannels in new int[] { 3, 6, 9, 15, 21, 33 }) {
                        foreach (int kwidth in new int[] { 1, 3, 5 }) {
                            foreach (int inwidth in new int[] { kwidth, kwidth * 2, 8, 9, 13, 17, 25 }) {
                                int outwidth = inwidth - kwidth + 1;

                                float[] yval = (new float[outwidth * outchannels * batch]).Select((_, idx) => (float)random.NextDouble() * 1e-2f).ToArray();
                                float[] wval = (new float[kwidth * inchannels * outchannels / 9 * 4]).Select((_, idx) => (float)random.NextDouble() * 1e-2f).ToArray();

                                Trivector[] ycval = (new Trivector[yval.Length / 3])
                                    .Select((_, idx) => new Trivector(yval[idx * 3], yval[idx * 3 + 1], yval[idx * 3 + 2])).ToArray();

                                Quaternion.Quaternion[] wcval = (new Quaternion.Quaternion[wval.Length / 4])
                                    .Select((_, idx) => new Quaternion.Quaternion(wval[idx * 4], wval[idx * 4 + 1], wval[idx * 4 + 2], wval[idx * 4 + 3])).ToArray();

                                TrivectorMap1D y = new(outchannels / 3, outwidth, batch, ycval);
                                Quaternion.QuaternionFilter1D w = new(inchannels / 3, outchannels / 3, kwidth, wcval);

                                TrivectorMap1D x = Reference(y, w, inwidth, kwidth);

                                OverflowCheckedTensor y_tensor = new(Shape.Map1D(outchannels, outwidth, batch), yval);
                                OverflowCheckedTensor w_tensor = new(Shape.Kernel1D(inchannels / 3 * 4, outchannels / 3, kwidth), wval);

                                OverflowCheckedTensor x_tensor = new(Shape.Map1D(inchannels, inwidth, batch));

                                TrivectorDeconvolution1D ope = new(outwidth, outchannels, inchannels, kwidth, gradmode: false, batch);

                                ope.Execute(y_tensor, w_tensor, x_tensor);

                                float[] x_expect = x.ToArray();
                                float[] x_actual = x_tensor.State.Value;

                                CollectionAssert.AreEqual(yval, y_tensor.State.Value);
                                CollectionAssert.AreEqual(wval, w_tensor.State.Value);

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
        public void LargeMapTest() {
            TensorShaderCudaBackend.Environment.CudnnEnabled = false;
            TensorShaderCudaBackend.Environment.Precision = TensorShaderCudaBackend.Environment.PrecisionMode.FloatFloat;

            Random random = new(1234);

            float max_err = 0;

            int batch = 3;
            int inchannels = 147, outchannels = 150;
            int kwidth = 5;
            int inwidth = 125;
            int outwidth = inwidth - kwidth + 1;

            float[] yval = (new float[outwidth * outchannels * batch]).Select((_, idx) => (float)random.NextDouble() * 1e-2f).ToArray();
            float[] wval = (new float[kwidth * inchannels * outchannels / 9 * 4]).Select((_, idx) => (float)random.NextDouble() * 1e-2f).ToArray();

            Trivector[] ycval = (new Trivector[yval.Length / 3])
                .Select((_, idx) => new Trivector(yval[idx * 3], yval[idx * 3 + 1], yval[idx * 3 + 2])).ToArray();

            Quaternion.Quaternion[] wcval = (new Quaternion.Quaternion[wval.Length / 4])
                .Select((_, idx) => new Quaternion.Quaternion(wval[idx * 4], wval[idx * 4 + 1], wval[idx * 4 + 2], wval[idx * 4 + 3])).ToArray();

            TrivectorMap1D y = new(outchannels / 3, outwidth, batch, ycval);
            Quaternion.QuaternionFilter1D w = new(inchannels / 3, outchannels / 3, kwidth, wcval);

            TrivectorMap1D x = Reference(y, w, inwidth, kwidth);

            OverflowCheckedTensor y_tensor = new(Shape.Map1D(outchannels, outwidth, batch), yval);
            OverflowCheckedTensor w_tensor = new(Shape.Kernel1D(inchannels / 3 * 4, outchannels / 3, kwidth), wval);

            OverflowCheckedTensor x_tensor = new(Shape.Map1D(inchannels, inwidth, batch));

            TrivectorDeconvolution1D ope = new(outwidth, outchannels, inchannels, kwidth, gradmode: false, batch);

            ope.Execute(y_tensor, w_tensor, x_tensor);

            float[] x_expect = x.ToArray();
            float[] x_actual = x_tensor.State.Value;

            CollectionAssert.AreEqual(yval, y_tensor.State.Value);
            CollectionAssert.AreEqual(wval, w_tensor.State.Value);

            AssertError.Tolerance(x_expect, x_actual, 1e-7f, 1e-5f, ref max_err, $"mismatch value {inchannels},{outchannels},{kwidth},{inwidth},{batch}");

            Console.WriteLine($"pass: {inchannels},{outchannels},{kwidth},{inwidth},{batch}");

            Console.WriteLine($"maxerr:{max_err}");
        }

        [TestMethod]
        public void SpeedFPTest() {
            TensorShaderCudaBackend.Environment.Precision = TensorShaderCudaBackend.Environment.PrecisionMode.Float;
            TensorShaderCudaBackend.Environment.CudnnEnabled = false;

            int inwidth = 32, inchannels = 33, outchannels = 33, ksize = 3;
            int outwidth = inwidth - ksize + 1;

            OverflowCheckedTensor y_tensor = new(Shape.Map1D(outchannels, outwidth));
            OverflowCheckedTensor w_tensor = new(Shape.Kernel1D(inchannels / 3 * 4, outchannels / 3, ksize));

            OverflowCheckedTensor x_tensor = new(Shape.Map1D(inchannels, inwidth));

            TrivectorDeconvolution1D ope = new(outwidth, outchannels, inchannels, ksize);

            ope.Execute(y_tensor, w_tensor, x_tensor);

            Cuda.Profiler.Initialize("../../../../profiler.nvsetting", "../../nvprofiles/trivector_deconvolution_1d_fp.nvvp");
            Cuda.Profiler.Start();

            ope.Execute(y_tensor, w_tensor, x_tensor);

            Cuda.Profiler.Stop();
        }

        [TestMethod]
        public void SpeedFFPTest() {
            TensorShaderCudaBackend.Environment.CudnnEnabled = false;
            TensorShaderCudaBackend.Environment.Precision = TensorShaderCudaBackend.Environment.PrecisionMode.FloatFloat;

            int inwidth = 32, inchannels = 33, outchannels = 33, ksize = 3;
            int outwidth = inwidth - ksize + 1;

            OverflowCheckedTensor y_tensor = new(Shape.Map1D(outchannels, outwidth));
            OverflowCheckedTensor w_tensor = new(Shape.Kernel1D(inchannels / 3 * 4, outchannels / 3, ksize));

            OverflowCheckedTensor x_tensor = new(Shape.Map1D(inchannels, inwidth));

            TrivectorDeconvolution1D ope = new(outwidth, outchannels, inchannels, ksize);

            ope.Execute(y_tensor, w_tensor, x_tensor);

            Cuda.Profiler.Initialize("../../../../profiler.nvsetting", "../../nvprofiles/trivector_deconvolution_1d_ffp.nvvp");
            Cuda.Profiler.Start();

            ope.Execute(y_tensor, w_tensor, x_tensor);

            Cuda.Profiler.Stop();
        }

        public static TrivectorMap1D Reference(TrivectorMap1D y, Quaternion.QuaternionFilter1D w, int inw, int kwidth) {
            int inchannels = w.InChannels, outchannels = w.OutChannels, batch = y.Batch;
            int outw = inw - kwidth + 1;

            if (y.Width != outw) {
                throw new ArgumentException("mismatch shape");
            }

            TrivectorMap1D x = new(inchannels, inw, batch);

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

            TrivectorMap1D y = new(outchannels / 3, outwidth, batch, ycval);
            Quaternion.QuaternionFilter1D w = new(inchannels / 3, outchannels / 3, kwidth, wcval);

            TrivectorMap1D x = Reference(y, w, inwidth, kwidth);

            float[] x_expect = {
                1.443468000e-03f,  9.190920000e-04f,  1.172196000e-03f,  1.347564000e-03f,  8.553480000e-04f,  1.092708000e-03f,
                1.254988000e-03f,  7.939080000e-04f,  1.016036000e-03f,  4.966200000e-03f,  4.164360000e-03f,  4.516008000e-03f,
                4.626168000e-03f,  3.878664000e-03f,  4.204968000e-03f,  4.298936000e-03f,  3.603720000e-03f,  3.905704000e-03f,
                9.195012000e-03f,  8.242812000e-03f,  8.612172000e-03f,  8.536356000e-03f,  7.650684000e-03f,  7.991244000e-03f,
                7.906116000e-03f,  7.083900000e-03f,  7.397196000e-03f,  1.355389200e-02f,  1.247036400e-02f,  1.284145200e-02f,
                1.255194000e-02f,  1.154185200e-02f,  1.188414000e-02f,  1.159683600e-02f,  1.065711600e-02f,  1.097214000e-02f,
                1.791277200e-02f,  1.669791600e-02f,  1.707073200e-02f,  1.656752400e-02f,  1.543302000e-02f,  1.577703600e-02f,
                1.528755600e-02f,  1.423033200e-02f,  1.454708400e-02f,  5.775864000e-03f,  5.311176000e-03f,  5.429736000e-03f,
                5.124024000e-03f,  4.702920000e-03f,  4.808424000e-03f,  4.521848000e-03f,  4.142280000e-03f,  4.235752000e-03f,
                8.217720000e-04f,  7.221000000e-04f,  7.409640000e-04f,  6.429240000e-04f,  5.600520000e-04f,  5.754600000e-04f,
                4.919800000e-04f,  4.248840000e-04f,  4.373480000e-04f,  1.649578800e-02f,  1.561429200e-02f,  1.587027600e-02f,
                1.545908400e-02f,  1.462126800e-02f,  1.486150800e-02f,  1.445642800e-02f,  1.366126800e-02f,  1.388627600e-02f,
                2.580876000e-02f,  2.443092000e-02f,  2.478832800e-02f,  2.395576800e-02f,  2.265530400e-02f,  2.298736800e-02f,
                2.217701600e-02f,  2.095188000e-02f,  2.125962400e-02f,  3.098941200e-02f,  2.938057200e-02f,  2.975857200e-02f,
                2.861427600e-02f,  2.710652400e-02f,  2.745572400e-02f,  2.635971600e-02f,  2.494998000e-02f,  2.527191600e-02f,
                3.534829200e-02f,  3.360812400e-02f,  3.398785200e-02f,  3.262986000e-02f,  3.099769200e-02f,  3.134862000e-02f,
                3.005043600e-02f,  2.852319600e-02f,  2.884686000e-02f,  3.970717200e-02f,  3.783567600e-02f,  3.821713200e-02f,
                3.664544400e-02f,  3.488886000e-02f,  3.524151600e-02f,  3.374115600e-02f,  3.209641200e-02f,  3.242180400e-02f,
                1.251794400e-02f,  1.175373600e-02f,  1.187805600e-02f,  1.109042400e-02f,  1.039284000e-02f,  1.050410400e-02f,
                9.774008000e-03f,  9.141000000e-03f,  9.240232000e-03f,  1.773612000e-03f,  1.593300000e-03f,  1.615044000e-03f,
                1.391244000e-03f,  1.239252000e-03f,  1.257540000e-03f,  1.067500000e-03f,  9.428040000e-04f,  9.581480000e-04f,
                3.154810800e-02f,  3.030949200e-02f,  3.056835600e-02f,  2.957060400e-02f,  2.838718800e-02f,  2.863030800e-02f,
                2.765786800e-02f,  2.652862800e-02f,  2.675651600e-02f,  4.665132000e-02f,  4.469748000e-02f,  4.506064800e-02f,
                4.328536800e-02f,  4.143194400e-02f,  4.176976800e-02f,  4.005509600e-02f,  3.830004000e-02f,  3.861354400e-02f,
                5.278381200e-02f,  5.051833200e-02f,  5.090497200e-02f,  4.869219600e-02f,  4.656236400e-02f,  4.692020400e-02f,
                4.481331600e-02f,  4.281606000e-02f,  4.314663600e-02f,  5.714269200e-02f,  5.474588400e-02f,  5.513425200e-02f,
                5.270778000e-02f,  5.045353200e-02f,  5.081310000e-02f,  4.850403600e-02f,  4.638927600e-02f,  4.672158000e-02f,
                6.150157200e-02f,  5.897343600e-02f,  5.936353200e-02f,  5.672336400e-02f,  5.434470000e-02f,  5.470599600e-02f,
                5.219475600e-02f,  4.996249200e-02f,  5.029652400e-02f,  1.926002400e-02f,  1.819629600e-02f,  1.832637600e-02f,
                1.705682400e-02f,  1.608276000e-02f,  1.619978400e-02f,  1.502616800e-02f,  1.413972000e-02f,  1.424471200e-02f,
                2.725452000e-03f,  2.464500000e-03f,  2.489124000e-03f,  2.139564000e-03f,  1.918452000e-03f,  1.939620000e-03f,
                1.643020000e-03f,  1.460724000e-03f,  1.478948000e-03f
            };

            float[] x_actual = x.ToArray();

            AssertError.Tolerance(x_expect, x_actual, 1e-7f, 1e-5f, $"mismatch value {inchannels},{outchannels},{kwidth},{inwidth},{batch}");
        }
    }
}
