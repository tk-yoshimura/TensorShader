using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;
using TensorShader;
using TensorShader.Operators.TrivectorConvolution;
using TensorShaderCudaBackend.API;

namespace TensorShaderTest.Operators.Trivector {
    [TestClass]
    public class TrivectorDeconvolution2DTest {
        [TestMethod]
        public void ExecuteFPTest() {
            TensorShaderCudaBackend.Environment.Precision = TensorShaderCudaBackend.Environment.PrecisionMode.Float;
            TensorShaderCudaBackend.Environment.CudnnEnabled = false;

            Random random = new(1234);

            float max_err = 0;

            foreach (int batch in new int[] { 1, 2 }) {
                foreach (int inchannels in new int[] { 3, 6, 9, 15, 21, 33 }) {
                    foreach (int outchannels in new int[] { 3, 6, 9, 15, 21, 33 }) {
                        foreach (int kheight in new int[] { 1, 3, 5 }) {
                            foreach (int kwidth in new int[] { 1, 3, 5 }) {
                                foreach (int inwidth in new int[] { kwidth, kwidth * 2, 8, 9, 13, 17, 25 }) {
                                    foreach (int inheight in new int[] { kheight, kheight * 2, 8, 9, 13, 17, 25 }) {
                                        int outwidth = inwidth - kwidth + 1, outheight = inheight - kheight + 1;

                                        float[] yval = (new float[outwidth * outheight * outchannels * batch]).Select((_, idx) => (float)random.NextDouble() * 1e-2f).ToArray();
                                        float[] wval = (new float[kwidth * kheight * inchannels * outchannels / 9 * 4]).Select((_, idx) => (float)random.NextDouble() * 1e-2f).ToArray();

                                        Trivector[] ycval = (new Trivector[yval.Length / 3])
                                            .Select((_, idx) => new Trivector(yval[idx * 3], yval[idx * 3 + 1], yval[idx * 3 + 2])).ToArray();

                                        Quaternion.Quaternion[] wcval = (new Quaternion.Quaternion[wval.Length / 4])
                                            .Select((_, idx) => new Quaternion.Quaternion(wval[idx * 4], wval[idx * 4 + 1], wval[idx * 4 + 2], wval[idx * 4 + 3])).ToArray();

                                        TrivectorMap2D y = new(outchannels / 3, outwidth, outheight, batch, ycval);
                                        Quaternion.QuaternionFilter2D w = new(inchannels / 3, outchannels / 3, kwidth, kheight, wcval);

                                        TrivectorMap2D x = Reference(y, w, inwidth, inheight, kwidth, kheight);

                                        OverflowCheckedTensor y_tensor = new(Shape.Map2D(outchannels, outwidth, outheight, batch), yval);
                                        OverflowCheckedTensor w_tensor = new(Shape.Kernel2D(inchannels / 3 * 4, outchannels / 3, kwidth, kheight), wval);

                                        OverflowCheckedTensor x_tensor = new(Shape.Map2D(inchannels, inwidth, inheight, batch));

                                        TrivectorDeconvolution2D ope = new(outwidth, outheight, outchannels, inchannels, kwidth, kheight, gradmode: false, batch);

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
                foreach (int inchannels in new int[] { 3, 6, 9, 15, 21, 33 }) {
                    foreach (int outchannels in new int[] { 3, 6, 9, 15, 21, 33 }) {
                        foreach (int kheight in new int[] { 1, 3, 5 }) {
                            foreach (int kwidth in new int[] { 1, 3, 5 }) {
                                foreach (int inwidth in new int[] { kwidth, kwidth * 2, 8, 9, 13, 17, 25 }) {
                                    foreach (int inheight in new int[] { kheight, kheight * 2, 8, 9, 13, 17, 25 }) {
                                        int outwidth = inwidth - kwidth + 1, outheight = inheight - kheight + 1;

                                        float[] yval = (new float[outwidth * outheight * outchannels * batch]).Select((_, idx) => (float)random.NextDouble() * 1e-2f).ToArray();
                                        float[] wval = (new float[kwidth * kheight * inchannels * outchannels / 9 * 4]).Select((_, idx) => (float)random.NextDouble() * 1e-2f).ToArray();

                                        Trivector[] ycval = (new Trivector[yval.Length / 3])
                                            .Select((_, idx) => new Trivector(yval[idx * 3], yval[idx * 3 + 1], yval[idx * 3 + 2])).ToArray();

                                        Quaternion.Quaternion[] wcval = (new Quaternion.Quaternion[wval.Length / 4])
                                            .Select((_, idx) => new Quaternion.Quaternion(wval[idx * 4], wval[idx * 4 + 1], wval[idx * 4 + 2], wval[idx * 4 + 3])).ToArray();

                                        TrivectorMap2D y = new(outchannels / 3, outwidth, outheight, batch, ycval);
                                        Quaternion.QuaternionFilter2D w = new(inchannels / 3, outchannels / 3, kwidth, kheight, wcval);

                                        TrivectorMap2D x = Reference(y, w, inwidth, inheight, kwidth, kheight);

                                        OverflowCheckedTensor y_tensor = new(Shape.Map2D(outchannels, outwidth, outheight, batch), yval);
                                        OverflowCheckedTensor w_tensor = new(Shape.Kernel2D(inchannels / 3 * 4, outchannels / 3, kwidth, kheight), wval);

                                        OverflowCheckedTensor x_tensor = new(Shape.Map2D(inchannels, inwidth, inheight, batch));

                                        TrivectorDeconvolution2D ope = new(outwidth, outheight, outchannels, inchannels, kwidth, kheight, gradmode: false, batch);

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
            int inchannels = 147, outchannels = 150;
            int kwidth = 5, kheight = 3;
            int inwidth = 250, inheight = 196;
            int outwidth = inwidth - kwidth + 1, outheight = inheight - kheight + 1;

            float[] yval = (new float[outwidth * outheight * outchannels * batch]).Select((_, idx) => (float)random.NextDouble() * 1e-2f).ToArray();
            float[] wval = (new float[kwidth * kheight * inchannels * outchannels / 9 * 4]).Select((_, idx) => (float)random.NextDouble() * 1e-2f).ToArray();

            Trivector[] ycval = (new Trivector[yval.Length / 3])
                .Select((_, idx) => new Trivector(yval[idx * 3], yval[idx * 3 + 1], yval[idx * 3 + 2])).ToArray();

            Quaternion.Quaternion[] wcval = (new Quaternion.Quaternion[wval.Length / 4])
                .Select((_, idx) => new Quaternion.Quaternion(wval[idx * 4], wval[idx * 4 + 1], wval[idx * 4 + 2], wval[idx * 4 + 3])).ToArray();

            TrivectorMap2D y = new(outchannels / 3, outwidth, outheight, batch, ycval);
            Quaternion.QuaternionFilter2D w = new(inchannels / 3, outchannels / 3, kwidth, kheight, wcval);

            TrivectorMap2D x = Reference(y, w, inwidth, inheight, kwidth, kheight);

            OverflowCheckedTensor y_tensor = new(Shape.Map2D(outchannels, outwidth, outheight, batch), yval);
            OverflowCheckedTensor w_tensor = new(Shape.Kernel2D(inchannels / 3 * 4, outchannels / 3, kwidth, kheight), wval);

            OverflowCheckedTensor x_tensor = new(Shape.Map2D(inchannels, inwidth, inheight, batch));

            TrivectorDeconvolution2D ope = new(outwidth, outheight, outchannels, inchannels, kwidth, kheight, gradmode: false, batch);

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

            int inwidth = 32, inheight = 32, inchannels = 33, outchannels = 33, ksize = 3;
            int outwidth = inwidth - ksize + 1, outheight = inheight - ksize + 1;

            OverflowCheckedTensor y_tensor = new(Shape.Map2D(outchannels, outwidth, outheight));
            OverflowCheckedTensor w_tensor = new(Shape.Kernel2D(inchannels / 3 * 4, outchannels / 3, ksize, ksize));

            OverflowCheckedTensor x_tensor = new(Shape.Map2D(inchannels, inwidth, inheight));

            TrivectorDeconvolution2D ope = new(outwidth, outheight, outchannels, inchannels, ksize, ksize);

            ope.Execute(y_tensor, w_tensor, x_tensor);

            Cuda.Profiler.Initialize("../../../../profiler.nvsetting", "../../nvprofiles/trivector_deconvolution_2d_fp.nvvp");
            Cuda.Profiler.Start();

            ope.Execute(y_tensor, w_tensor, x_tensor);

            Cuda.Profiler.Stop();
        }

        [TestMethod]
        public void SpeedFFPTest() {
            TensorShaderCudaBackend.Environment.CudnnEnabled = false;
            TensorShaderCudaBackend.Environment.Precision = TensorShaderCudaBackend.Environment.PrecisionMode.FloatFloat;

            int inwidth = 32, inheight = 32, inchannels = 33, outchannels = 33, ksize = 3;
            int outwidth = inwidth - ksize + 1, outheight = inheight - ksize + 1;

            OverflowCheckedTensor y_tensor = new(Shape.Map2D(outchannels, outwidth, outheight));
            OverflowCheckedTensor w_tensor = new(Shape.Kernel2D(inchannels / 3 * 4, outchannels / 3, ksize, ksize));

            OverflowCheckedTensor x_tensor = new(Shape.Map2D(inchannels, inwidth, inheight));

            TrivectorDeconvolution2D ope = new(outwidth, outheight, outchannels, inchannels, ksize, ksize);

            ope.Execute(y_tensor, w_tensor, x_tensor);

            Cuda.Profiler.Initialize("../../../../profiler.nvsetting", "../../nvprofiles/trivector_deconvolution_2d_ffp.nvvp");
            Cuda.Profiler.Start();

            ope.Execute(y_tensor, w_tensor, x_tensor);

            Cuda.Profiler.Stop();
        }

        public static TrivectorMap2D Reference(TrivectorMap2D y, Quaternion.QuaternionFilter2D w, int inw, int inh, int kwidth, int kheight) {
            int inchannels = w.InChannels, outchannels = w.OutChannels, batch = y.Batch;
            int outw = inw - kwidth + 1, outh = inh - kheight + 1;

            if (y.Width != outw || y.Height != outh) {
                throw new ArgumentException("mismatch shape");
            }

            TrivectorMap2D x = new(inchannels, inw, inh, batch);

            for (int kx, ky = 0; ky < kheight; ky++) {
                for (kx = 0; kx < kwidth; kx++) {
                    for (int th = 0; th < batch; th++) {
                        for (int ox, oy = 0; oy < outh; oy++) {
                            for (ox = 0; ox < outw; ox++) {
                                for (int outch = 0; outch < outchannels; outch++) {
                                    Trivector v = y[outch, ox, oy, th];

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
            int inchannels = 9, outchannels = 12, kwidth = 3, kheight = 5, inwidth = 7, inheight = 8, batch = 3;
            int outwidth = inwidth - kwidth + 1, outheight = inheight - kheight + 1;

            float[] yval = (new float[outwidth * outheight * outchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
            float[] wval = (new float[kwidth * kheight * outchannels * inchannels / 9 * 4]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

            Trivector[] ycval = (new Trivector[yval.Length / 3])
                .Select((_, idx) => new Trivector(yval[idx * 3], yval[idx * 3 + 1], yval[idx * 3 + 2])).ToArray();

            Quaternion.Quaternion[] wcval = (new Quaternion.Quaternion[wval.Length / 4])
                .Select((_, idx) => new Quaternion.Quaternion(wval[idx * 4], wval[idx * 4 + 1], wval[idx * 4 + 2], wval[idx * 4 + 3])).ToArray();

            TrivectorMap2D y = new(outchannels / 3, outwidth, outheight, batch, ycval);
            Quaternion.QuaternionFilter2D w = new(inchannels / 3, outchannels / 3, kwidth, kheight, wcval);

            TrivectorMap2D x = Reference(y, w, inwidth, inheight, kwidth, kheight);

            float[] x_expect = {
                4.999796400e-02f,  3.415198800e-02f,  4.201750800e-02f,  4.942282800e-02f,  3.375646800e-02f,  4.153251600e-02f,
                4.885102000e-02f,  3.336325200e-02f,  4.105034000e-02f,  1.875628080e-01f,  1.575554640e-01f,  1.722472080e-01f,
                1.853795760e-01f,  1.557214800e-01f,  1.702404240e-01f,  1.832091440e-01f,  1.538982480e-01f,  1.682454160e-01f,
                4.007045160e-01f,  3.581006040e-01f,  3.786530040e-01f,  3.959539560e-01f,  3.538589400e-01f,  3.741613560e-01f,
                3.912318120e-01f,  3.496426200e-01f,  3.696965880e-01f,  6.469281000e-01f,  6.031975320e-01f,  6.237516600e-01f,
                6.391800360e-01f,  5.959652760e-01f,  6.162694200e-01f,  6.314788200e-01f,  5.887767960e-01f,  6.088324920e-01f,
                8.931516840e-01f,  8.482944600e-01f,  8.688503160e-01f,  8.824061160e-01f,  8.380716120e-01f,  8.583774840e-01f,
                8.717258280e-01f,  8.279109720e-01f,  8.479683960e-01f,  6.181329840e-01f,  5.900110800e-01f,  6.027002640e-01f,
                6.103295280e-01f,  5.825461200e-01f,  5.950747920e-01f,  6.025757360e-01f,  5.751287760e-01f,  5.874979600e-01f,
                3.178936440e-01f,  3.046842120e-01f,  3.105460200e-01f,  3.136966200e-01f,  3.006514440e-01f,  3.064360680e-01f,
                3.095275000e-01f,  2.966455560e-01f,  3.023535080e-01f,  5.526319920e-01f,  5.247450960e-01f,  5.375805840e-01f,
                5.462370480e-01f,  5.186840400e-01f,  5.313590160e-01f,  5.398794800e-01f,  5.126583120e-01f,  5.251738000e-01f,
                1.184735664e+00f,  1.132109904e+00f,  1.155923472e+00f,  1.170469680e+00f,  1.118491728e+00f,  1.141996560e+00f,
                1.156290736e+00f,  1.104956496e+00f,  1.128154640e+00f,  1.879408872e+00f,  1.804952664e+00f,  1.838034360e+00f,
                1.855943208e+00f,  1.782428760e+00f,  1.815065784e+00f,  1.832626536e+00f,  1.760047704e+00f,  1.792243128e+00f,
                2.275558056e+00f,  2.199097368e+00f,  2.232182520e+00f,  2.246760936e+00f,  2.171255832e+00f,  2.203896312e+00f,
                2.218149672e+00f,  2.143594008e+00f,  2.175792888e+00f,  2.671707240e+00f,  2.593242072e+00f,  2.626330680e+00f,
                2.637578664e+00f,  2.560082904e+00f,  2.592726840e+00f,  2.603672808e+00f,  2.527140312e+00f,  2.559342648e+00f,
                1.746503472e+00f,  1.697538000e+00f,  1.717788816e+00f,  1.723061424e+00f,  1.674722256e+00f,  1.694688912e+00f,
                1.699780144e+00f,  1.652063184e+00f,  1.671747728e+00f,  8.532494640e-01f,  8.303591760e-01f,  8.396296080e-01f,
                8.412327600e-01f,  8.186456400e-01f,  8.277801360e-01f,  8.293025840e-01f,  8.070165840e-01f,  8.160161680e-01f,
                1.338447492e+00f,  1.301412348e+00f,  1.316985804e+00f,  1.322093988e+00f,  1.285521660e+00f,  1.300881996e+00f,
                1.305842628e+00f,  1.269730044e+00f,  1.284878796e+00f,  2.673843048e+00f,  2.604163032e+00f,  2.632855608e+00f,
                2.639806632e+00f,  2.571022296e+00f,  2.599307064e+00f,  2.605992936e+00f,  2.538098136e+00f,  2.565978168e+00f,
                3.991450284e+00f,  3.893156244e+00f,  3.932734788e+00f,  3.938622732e+00f,  3.841627284e+00f,  3.880621764e+00f,
                3.886156908e+00f,  3.790450836e+00f,  3.828865860e+00f,  4.465114956e+00f,  4.364187444e+00f,  4.403771172e+00f,
                4.405285548e+00f,  4.305677364e+00f,  4.344677028e+00f,  4.345873164e+00f,  4.247575092e+00f,  4.285995300e+00f,
                4.938779628e+00f,  4.835218644e+00f,  4.874807556e+00f,  4.871948364e+00f,  4.769727444e+00f,  4.808732292e+00f,
                4.805589420e+00f,  4.704699348e+00f,  4.743124740e+00f,  3.141753768e+00f,  3.077398296e+00f,  3.101410296e+00f,
                3.097049832e+00f,  3.033557784e+00f,  3.057198840e+00f,  3.052679208e+00f,  2.990044440e+00f,  3.013317624e+00f,
                1.495005444e+00f,  1.465050492e+00f,  1.475939916e+00f,  1.472652324e+00f,  1.443114108e+00f,  1.453827276e+00f,
                1.450475076e+00f,  1.421350524e+00f,  1.431888972e+00f,  2.277802992e+00f,  2.233599888e+00f,  2.250342864e+00f,
                2.248458096e+00f,  2.204821392e+00f,  2.221317072e+00f,  2.219310832e+00f,  2.176236432e+00f,  2.192486864e+00f,
                4.416835680e+00f,  4.333840800e+00f,  4.364496672e+00f,  4.357552992e+00f,  4.275650976e+00f,  4.305836832e+00f,
                4.298690144e+00f,  4.217872800e+00f,  4.247592736e+00f,  6.411605328e+00f,  6.294750768e+00f,  6.336784368e+00f,
                6.322086864e+00f,  6.206811696e+00f,  6.248177136e+00f,  6.233235024e+00f,  6.119526960e+00f,  6.160230384e+00f,
                6.914263248e+00f,  6.794395056e+00f,  6.836435568e+00f,  6.816736080e+00f,  6.698474928e+00f,  6.739847280e+00f,
                6.719949264e+00f,  6.603282864e+00f,  6.643993200e+00f,  7.416921168e+00f,  7.294039344e+00f,  7.336086768e+00f,
                7.311385296e+00f,  7.190138160e+00f,  7.231517424e+00f,  7.206663504e+00f,  7.087038768e+00f,  7.127756016e+00f,
                4.640152416e+00f,  4.564035744e+00f,  4.589335584e+00f,  4.570775136e+00f,  4.495708320e+00f,  4.520587296e+00f,
                4.501965152e+00f,  4.427940000e+00f,  4.452402208e+00f,  2.171912688e+00f,  2.136596880e+00f,  2.147979216e+00f,
                2.137812336e+00f,  2.103001488e+00f,  2.114185680e+00f,  2.104007920e+00f,  2.069697936e+00f,  2.080686032e+00f,
                1.345894896e+00f,  1.319501712e+00f,  1.328632272e+00f,  1.323664752e+00f,  1.297690512e+00f,  1.306647504e+00f,
                1.301632240e+00f,  1.276072848e+00f,  1.284858320e+00f,  2.547158112e+00f,  2.498345376e+00f,  2.514661152e+00f,
                2.502989664e+00f,  2.454974880e+00f,  2.470968096e+00f,  2.459241056e+00f,  2.412016032e+00f,  2.427690784e+00f,
                3.608913744e+00f,  3.541175856e+00f,  3.563026416e+00f,  3.543393744e+00f,  3.476792880e+00f,  3.498196464e+00f,
                3.478540368e+00f,  3.413064240e+00f,  3.434026992e+00f,  3.869706960e+00f,  3.799950768e+00f,  3.821808240e+00f,
                3.798832464e+00f,  3.730240944e+00f,  3.751651440e+00f,  3.728698320e+00f,  3.661259184e+00f,  3.682228848e+00f,
                4.130500176e+00f,  4.058725680e+00f,  4.080590064e+00f,  4.054271184e+00f,  3.983689008e+00f,  4.005106416e+00f,
                3.978856272e+00f,  3.909454128e+00f,  3.930430704e+00f,  2.499966816e+00f,  2.456484000e+00f,  2.469213216e+00f,
                2.451012192e+00f,  2.408284320e+00f,  2.420740128e+00f,  2.402624864e+00f,  2.360643744e+00f,  2.372830240e+00f,
                1.130739696e+00f,  1.111021968e+00f,  1.116561360e+00f,  1.107293040e+00f,  1.087932816e+00f,  1.093347792e+00f,
                1.084142320e+00f,  1.065135504e+00f,  1.070428112e+00f,  6.700303080e-01f,  6.561503640e-01f,  6.603138360e-01f,
                6.548910120e-01f,  6.412872600e-01f,  6.453482040e-01f,  6.399460200e-01f,  6.266153880e-01f,  6.305753400e-01f,
                1.220804136e+00f,  1.195692696e+00f,  1.202892408e+00f,  1.191629160e+00f,  1.167040152e+00f,  1.174053240e+00f,
                1.162861224e+00f,  1.138788504e+00f,  1.145618040e+00f,  1.666781388e+00f,  1.632727476e+00f,  1.642057380e+00f,
                1.624895532e+00f,  1.591580340e+00f,  1.600657956e+00f,  1.583647884e+00f,  1.551062196e+00f,  1.559892132e+00f,
                1.777649004e+00f,  1.742454612e+00f,  1.751789700e+00f,  1.732742604e+00f,  1.698307668e+00f,  1.707390468e+00f,
                1.688529708e+00f,  1.654845012e+00f,  1.663680132e+00f,  1.888516620e+00f,  1.852181748e+00f,  1.861522020e+00f,
                1.840589676e+00f,  1.805034996e+00f,  1.814122980e+00f,  1.793411532e+00f,  1.758627828e+00f,  1.767468132e+00f,
                1.081398888e+00f,  1.060119000e+00f,  1.065292344e+00f,  1.051730856e+00f,  1.030941144e+00f,  1.035964728e+00f,
                1.022580456e+00f,  1.002274776e+00f,  1.007151672e+00f,  4.611369960e-01f,  4.518489240e-01f,  4.539825720e-01f,
                4.475183400e-01f,  4.384603800e-01f,  4.405283640e-01f,  4.341677160e-01f,  4.253367960e-01f,  4.273406520e-01f,
                2.468024880e-01f,  2.408842320e-01f,  2.423126160e-01f,  2.384691120e-01f,  2.327096400e-01f,  2.340881040e-01f,
                2.302959920e-01f,  2.246932560e-01f,  2.260228240e-01f,  4.210679520e-01f,  4.107758880e-01f,  4.131023520e-01f,
                4.057267680e-01f,  3.957323040e-01f,  3.979712160e-01f,  3.907183840e-01f,  3.810174240e-01f,  3.831708320e-01f,
                5.395142160e-01f,  5.261531760e-01f,  5.289948720e-01f,  5.186382480e-01f,  5.056936560e-01f,  5.084224560e-01f,
                4.982799120e-01f,  4.857456240e-01f,  4.883646000e-01f,  5.728663440e-01f,  5.589938160e-01f,  5.618389680e-01f,
                5.506402320e-01f,  5.371979760e-01f,  5.399302320e-01f,  5.289686160e-01f,  5.159504880e-01f,  5.185729200e-01f,
                6.062184720e-01f,  5.918344560e-01f,  5.946830640e-01f,  5.826422160e-01f,  5.687022960e-01f,  5.714380080e-01f,
                5.596573200e-01f,  5.461553520e-01f,  5.487812400e-01f,  3.083351520e-01f,  3.004868640e-01f,  3.019047840e-01f,
                2.947296480e-01f,  2.871574560e-01f,  2.885124000e-01f,  2.815306720e-01f,  2.742304800e-01f,  2.755244960e-01f,
                1.147840560e-01f,  1.116505680e-01f,  1.121681040e-01f,  1.090864560e-01f,  1.060810320e-01f,  1.065732240e-01f,
                1.035982640e-01f,  1.007188560e-01f,  1.011867280e-01f,  4.660042800e-02f,  4.500469200e-02f,  4.526643600e-02f,
                4.368212400e-02f,  4.215310800e-02f,  4.239910800e-02f,  4.085930800e-02f,  3.939598800e-02f,  3.962675600e-02f,
                6.749388000e-02f,  6.496404000e-02f,  6.533296800e-02f,  6.261496800e-02f,  6.020858400e-02f,  6.055216800e-02f,
                5.793317600e-02f,  5.564820000e-02f,  5.596746400e-02f,  7.457821200e-02f,  7.165609200e-02f,  7.205137200e-02f,
                6.877011600e-02f,  6.601820400e-02f,  6.638468400e-02f,  6.326691600e-02f,  6.068214000e-02f,  6.102135600e-02f,
                7.893709200e-02f,  7.588364400e-02f,  7.628065200e-02f,  7.278570000e-02f,  6.990937200e-02f,  7.027758000e-02f,
                6.695763600e-02f,  6.425535600e-02f,  6.459630000e-02f,  8.329597200e-02f,  8.011119600e-02f,  8.050993200e-02f,
                7.680128400e-02f,  7.380054000e-02f,  7.417047600e-02f,  7.064835600e-02f,  6.782857200e-02f,  6.817124400e-02f,
                2.600210400e-02f,  2.463885600e-02f,  2.477469600e-02f,  2.302322400e-02f,  2.177268000e-02f,  2.189546400e-02f,
                2.027832800e-02f,  1.913844000e-02f,  1.924919200e-02f,  3.677292000e-03f,  3.335700000e-03f,  3.363204000e-03f,
                2.887884000e-03f,  2.597652000e-03f,  2.621700000e-03f,  2.218540000e-03f,  1.978644000e-03f,  1.999748000e-03f,
                1.934975244e+00f,  1.911065268e+00f,  1.918942308e+00f,  1.912942188e+00f,  1.889257908e+00f,  1.897045476e+00f,
                1.891035340e+00f,  1.867575732e+00f,  1.875274340e+00f,  3.708132408e+00f,  3.662550024e+00f,  3.677264808e+00f,
                3.664507896e+00f,  3.619366920e+00f,  3.633908904e+00f,  3.621141944e+00f,  3.576440328e+00f,  3.590810536e+00f,
                5.325176196e+00f,  5.260039164e+00f,  5.280626124e+00f,  5.260475556e+00f,  5.195985660e+00f,  5.216322636e+00f,
                5.196171972e+00f,  5.132326140e+00f,  5.152414668e+00f,  5.571399780e+00f,  5.505136092e+00f,  5.525724780e+00f,
                5.503701636e+00f,  5.438091996e+00f,  5.458430700e+00f,  5.436418980e+00f,  5.371460316e+00f,  5.391550572e+00f,
                5.817623364e+00f,  5.750233020e+00f,  5.770823436e+00f,  5.746927716e+00f,  5.680198332e+00f,  5.700538764e+00f,
                5.676665988e+00f,  5.610594492e+00f,  5.630686476e+00f,  3.657627384e+00f,  3.615036360e+00f,  3.627748584e+00f,
                3.611331768e+00f,  3.569171400e+00f,  3.581723112e+00f,  3.565331576e+00f,  3.523599816e+00f,  3.535992040e+00f,
                1.721795724e+00f,  1.701628212e+00f,  1.707501540e+00f,  1.699089900e+00f,  1.679132724e+00f,  1.684928868e+00f,
                1.676534860e+00f,  1.656786996e+00f,  1.662506468e+00f,  3.627515832e+00f,  3.585159816e+00f,  3.598018344e+00f,
                3.582628728e+00f,  3.540698760e+00f,  3.553396776e+00f,  3.538024760e+00f,  3.496518792e+00f,  3.509057320e+00f,
                6.888817584e+00f,  6.808359504e+00f,  6.832219152e+00f,  6.800516400e+00f,  6.720890448e+00f,  6.744441360e+00f,
                6.712793776e+00f,  6.633995856e+00f,  6.657240080e+00f,  9.802392552e+00f,  9.687846744e+00f,  9.720997560e+00f,
                9.672297768e+00f,  9.558970200e+00f,  9.591676344e+00f,  9.543089256e+00f,  9.430973784e+00f,  9.463238328e+00f,
                1.019854174e+01f,  1.008199145e+01f,  1.011514572e+01f,  1.006311550e+01f,  9.947797272e+00f,  9.980506872e+00f,
                9.928612392e+00f,  9.814520088e+00f,  9.846788088e+00f,  1.059469092e+01f,  1.047613615e+01f,  1.050929388e+01f,
                1.045393322e+01f,  1.033662434e+01f,  1.036933740e+01f,  1.031413553e+01f,  1.019806639e+01f,  1.023033785e+01f,
                6.594603312e+00f,  6.520017360e+00f,  6.540314256e+00f,  6.503024304e+00f,  6.429248976e+00f,  6.449261712e+00f,
                6.412097584e+00f,  6.339128784e+00f,  6.358859408e+00f,  3.072151224e+00f,  3.037003656e+00f,  3.046297128e+00f,
                3.027540600e+00f,  2.992788360e+00f,  3.001945896e+00f,  2.983262264e+00f,  2.948903304e+00f,  2.957925928e+00f,
                5.067419652e+00f,  5.011169148e+00f,  5.026777164e+00f,  4.999963428e+00f,  4.944313980e+00f,  4.959708876e+00f,
                4.932977988e+00f,  4.877926524e+00f,  4.893109836e+00f,  9.542884968e+00f,  9.436433112e+00f,  9.465194808e+00f,
                9.411066792e+00f,  9.305787096e+00f,  9.334140984e+00f,  9.280208616e+00f,  9.176094936e+00f,  9.204044088e+00f,
                1.346474372e+01f,  1.331378024e+01f,  1.335346247e+01f,  1.327187905e+01f,  1.312262888e+01f,  1.316172704e+01f,
                1.308048203e+01f,  1.293293596e+01f,  1.297145466e+01f,  1.393840840e+01f,  1.378481144e+01f,  1.382449885e+01f,
                1.373854187e+01f,  1.358667896e+01f,  1.362578231e+01f,  1.354019828e+01f,  1.339006021e+01f,  1.342858410e+01f,
                1.441207307e+01f,  1.425584264e+01f,  1.429553524e+01f,  1.420520468e+01f,  1.405072904e+01f,  1.408983757e+01f,
                1.399991454e+01f,  1.384718447e+01f,  1.388571354e+01f,  8.886075048e+00f,  8.788265496e+00f,  8.812346616e+00f,
                8.752436712e+00f,  8.655767064e+00f,  8.679477240e+00f,  8.619868968e+00f,  8.524333080e+00f,  8.547675384e+00f,
                4.099256964e+00f,  4.053404412e+00f,  4.064328396e+00f,  4.034648484e+00f,  3.989350908e+00f,  4.000098636e+00f,
                3.970584516e+00f,  3.925838844e+00f,  3.936411852e+00f,  6.284297712e+00f,  6.217791888e+00f,  6.234580944e+00f,
                6.195663216e+00f,  6.129908112e+00f,  6.146449872e+00f,  6.107717872e+00f,  6.042709392e+00f,  6.059005904e+00f,
                1.175079024e+01f,  1.162540176e+01f,  1.165614979e+01f,  1.157882659e+01f,  1.145489962e+01f,  1.148517763e+01f,
                1.140826582e+01f,  1.128579216e+01f,  1.131560426e+01f,  1.646476373e+01f,  1.628763653e+01f,  1.632980837e+01f,
                1.621507118e+01f,  1.604007634e+01f,  1.608158002e+01f,  1.596751982e+01f,  1.579464504e+01f,  1.583548670e+01f,
                1.696742165e+01f,  1.678728082e+01f,  1.682945957e+01f,  1.670972040e+01f,  1.653173957e+01f,  1.657325016e+01f,
                1.645423406e+01f,  1.627840094e+01f,  1.631924952e+01f,  1.747007957e+01f,  1.728692510e+01f,  1.732911077e+01f,
                1.720436962e+01f,  1.702340280e+01f,  1.706492030e+01f,  1.694094830e+01f,  1.676215685e+01f,  1.680301234e+01f,
                1.068681610e+01f,  1.057272950e+01f,  1.059812150e+01f,  1.051655434e+01f,  1.040388624e+01f,  1.042885738e+01f,
                1.034784291e+01f,  1.023658512e+01f,  1.026113949e+01f,  4.891116528e+00f,  4.837921680e+00f,  4.849350096e+00f,
                4.809523056e+00f,  4.757017488e+00f,  4.768247760e+00f,  4.728717040e+00f,  4.676896656e+00f,  4.687930832e+00f,
                3.527621616e+00f,  3.485561232e+00f,  3.494737872e+00f,  3.463796592e+00f,  3.422339472e+00f,  3.431342544e+00f,
                3.400660720e+00f,  3.359802768e+00f,  3.368634320e+00f,  6.443913312e+00f,  6.365978016e+00f,  6.382385952e+00f,
                6.322453344e+00f,  6.245684640e+00f,  6.261770016e+00f,  6.202396256e+00f,  6.126785952e+00f,  6.142552864e+00f,
                8.824778064e+00f,  8.716674096e+00f,  8.738662896e+00f,  8.652168144e+00f,  8.545754160e+00f,  8.567295984e+00f,
                8.481699408e+00f,  8.376963120e+00f,  8.398064112e+00f,  9.085571280e+00f,  8.975449008e+00f,  8.997444720e+00f,
                8.907606864e+00f,  8.799202224e+00f,  8.820750960e+00f,  8.731857360e+00f,  8.625158064e+00f,  8.646265968e+00f,
                9.346364496e+00f,  9.234223920e+00f,  9.256226544e+00f,  9.163045584e+00f,  9.052650288e+00f,  9.074205936e+00f,
                8.982015312e+00f,  8.873353008e+00f,  8.894467824e+00f,  5.534104416e+00f,  5.465922720e+00f,  5.478744096e+00f,
                5.419654752e+00f,  5.352596640e+00f,  5.365144608e+00f,  5.306755424e+00f,  5.240812704e+00f,  5.253091360e+00f,
                2.449848816e+00f,  2.418887568e+00f,  2.424473040e+00f,  2.396603760e+00f,  2.366184336e+00f,  2.371645392e+00f,
                2.344146160e+00f,  2.314264464e+00f,  2.319603152e+00f,  1.661850468e+00f,  1.638708444e+00f,  1.642906476e+00f,
                1.622150532e+00f,  1.599422940e+00f,  1.603518444e+00f,  1.583013540e+00f,  1.560697308e+00f,  1.564691820e+00f,
                2.934047016e+00f,  2.892070296e+00f,  2.899339128e+00f,  2.860174440e+00f,  2.818996632e+00f,  2.826078840e+00f,
                2.787446184e+00f,  2.747061144e+00f,  2.753959800e+00f,  3.884133708e+00f,  3.827270196e+00f,  3.836703780e+00f,
                3.781836972e+00f,  3.726126900e+00f,  3.735308196e+00f,  3.681284364e+00f,  3.626718516e+00f,  3.635652132e+00f,
                3.995001324e+00f,  3.936997332e+00f,  3.946436100e+00f,  3.889684044e+00f,  3.832854228e+00f,  3.842040708e+00f,
                3.786166188e+00f,  3.730501332e+00f,  3.739440132e+00f,  4.105868940e+00f,  4.046724468e+00f,  4.056168420e+00f,
                3.997531116e+00f,  3.939581556e+00f,  3.948773220e+00f,  3.891048012e+00f,  3.834284148e+00f,  3.843228132e+00f,
                2.306931048e+00f,  2.272103640e+00f,  2.277346104e+00f,  2.241412776e+00f,  2.207352024e+00f,  2.212444728e+00f,
                2.177149416e+00f,  2.143849176e+00f,  2.148795192e+00f,  9.652464360e-01f,  9.500140440e-01f,  9.521822520e-01f,
                9.359145000e-01f,  9.210504600e-01f,  9.231530040e-01f,  9.072192360e-01f,  8.927204760e-01f,  8.947588920e-01f,
                5.845343280e-01f,  5.741002320e-01f,  5.755516560e-01f,  5.642508720e-01f,  5.540677200e-01f,  5.554692240e-01f,
                5.443734320e-01f,  5.344391760e-01f,  5.357917840e-01f,  9.693508320e-01f,  9.511330080e-01f,  9.535055520e-01f,
                9.330586080e-01f,  9.153227040e-01f,  9.176076960e-01f,  8.975907040e-01f,  8.803326240e-01f,  8.825321120e-01f,
                1.206556776e+00f,  1.182965976e+00f,  1.185876792e+00f,  1.158677928e+00f,  1.135780056e+00f,  1.138577976e+00f,
                1.112053992e+00f,  1.089842904e+00f,  1.092531000e+00f,  1.239908904e+00f,  1.215806616e+00f,  1.218720888e+00f,
                1.190679912e+00f,  1.167284376e+00f,  1.170085752e+00f,  1.142742696e+00f,  1.120047768e+00f,  1.122739320e+00f,
                1.273261032e+00f,  1.248647256e+00f,  1.251564984e+00f,  1.222681896e+00f,  1.198788696e+00f,  1.201593528e+00f,
                1.173431400e+00f,  1.150252632e+00f,  1.152947640e+00f,  6.376458720e-01f,  6.240836640e-01f,  6.255476640e-01f,
                6.089875680e-01f,  5.958857760e-01f,  5.972868000e-01f,  5.812273120e-01f,  5.685818400e-01f,  5.699219360e-01f,
                2.335437360e-01f,  2.281062480e-01f,  2.286468240e-01f,  2.217942960e-01f,  2.165770320e-01f,  2.170922640e-01f,
                2.105000240e-01f,  2.055009360e-01f,  2.059918480e-01f,  1.068097080e-01f,  1.037854920e-01f,  1.040587560e-01f,
                1.001282040e-01f,  9.721678800e-02f,  9.747430800e-02f,  9.366506800e-02f,  9.086542800e-02f,  9.110771600e-02f,
                1.508641200e-01f,  1.460302800e-01f,  1.464222480e-01f,  1.399333680e-01f,  1.353151440e-01f,  1.356817680e-01f,
                1.294454960e-01f,  1.250408400e-01f,  1.253831440e-01f,  1.617558120e-01f,  1.562071320e-01f,  1.566369720e-01f,
                1.490817960e-01f,  1.438415640e-01f,  1.442426040e-01f,  1.370813160e-01f,  1.321464600e-01f,  1.325202360e-01f,
                1.661146920e-01f,  1.604346840e-01f,  1.608662520e-01f,  1.530973800e-01f,  1.477327320e-01f,  1.481355000e-01f,
                1.407720360e-01f,  1.357196760e-01f,  1.360951800e-01f,  1.704735720e-01f,  1.646622360e-01f,  1.650955320e-01f,
                1.571129640e-01f,  1.516239000e-01f,  1.520283960e-01f,  1.444627560e-01f,  1.392928920e-01f,  1.396701240e-01f,
                5.297042400e-02f,  5.040909600e-02f,  5.056797600e-02f,  4.688882400e-02f,  4.453236000e-02f,  4.467818400e-02f,
                4.128696800e-02f,  3.913332000e-02f,  3.926711200e-02f,  7.484652000e-03f,  6.820500000e-03f,  6.859524000e-03f,
                5.881164000e-03f,  5.314452000e-03f,  5.350020000e-03f,  4.520620000e-03f,  4.050324000e-03f,  4.082948000e-03f,
                3.819952524e+00f,  3.787978548e+00f,  3.795867108e+00f,  3.776461548e+00f,  3.744759348e+00f,  3.752558436e+00f,
                3.733219660e+00f,  3.701788212e+00f,  3.709498340e+00f,  7.228702008e+00f,  7.167544584e+00f,  7.182282408e+00f,
                7.143636216e+00f,  7.083012360e+00f,  7.097577384e+00f,  7.059074744e+00f,  6.998982408e+00f,  7.013375656e+00f,
                1.024964788e+01f,  1.016197772e+01f,  1.018259924e+01f,  1.012499716e+01f,  1.003811238e+01f,  1.005848392e+01f,
                1.000111213e+01f,  9.915009660e+00f,  9.935132748e+00f,  1.049587146e+01f,  1.040707465e+01f,  1.042769790e+01f,
                1.036822324e+01f,  1.028021872e+01f,  1.030059198e+01f,  1.024135914e+01f,  1.015414384e+01f,  1.017426865e+01f,
                1.074209504e+01f,  1.065217158e+01f,  1.067279656e+01f,  1.061144932e+01f,  1.052232505e+01f,  1.054270004e+01f,
                1.048160615e+01f,  1.039327801e+01f,  1.041340456e+01f,  6.697121784e+00f,  6.640061640e+00f,  6.652796904e+00f,
                6.612334008e+00f,  6.555796680e+00f,  6.568371432e+00f,  6.528087416e+00f,  6.472070856e+00f,  6.484486120e+00f,
                3.125697804e+00f,  3.098572212e+00f,  3.104457060e+00f,  3.084483180e+00f,  3.057614004e+00f,  3.063421668e+00f,
                3.043542220e+00f,  3.016928436e+00f,  3.022659428e+00f,  6.702399672e+00f,  6.645574536e+00f,  6.658456104e+00f,
                6.619020408e+00f,  6.562713480e+00f,  6.575434536e+00f,  6.536170040e+00f,  6.480379272e+00f,  6.492940840e+00f,
                1.259289950e+01f,  1.248460910e+01f,  1.250851483e+01f,  1.243056312e+01f,  1.232328917e+01f,  1.234688616e+01f,
                1.226929682e+01f,  1.216303522e+01f,  1.218632552e+01f,  1.772537623e+01f,  1.757074082e+01f,  1.760396076e+01f,
                1.748865233e+01f,  1.733551164e+01f,  1.736828690e+01f,  1.725355198e+01f,  1.710189986e+01f,  1.713423353e+01f,
                1.812152542e+01f,  1.796488553e+01f,  1.799810892e+01f,  1.787947006e+01f,  1.772433871e+01f,  1.775711743e+01f,
                1.763907511e+01f,  1.748544617e+01f,  1.751778329e+01f,  1.851767460e+01f,  1.835903023e+01f,  1.839225708e+01f,
                1.827028778e+01f,  1.811316578e+01f,  1.814594796e+01f,  1.802459825e+01f,  1.786899247e+01f,  1.790133305e+01f,
                1.144270315e+01f,  1.134249672e+01f,  1.136283970e+01f,  1.128298718e+01f,  1.118377570e+01f,  1.120383451e+01f,
                1.112441502e+01f,  1.102619438e+01f,  1.104597109e+01f,  5.291052984e+00f,  5.243648136e+00f,  5.252964648e+00f,
                5.213848440e+00f,  5.166931080e+00f,  5.176111656e+00f,  5.137221944e+00f,  5.090790024e+00f,  5.099835688e+00f,
                8.796391812e+00f,  8.720925948e+00f,  8.736568524e+00f,  8.677832868e+00f,  8.603106300e+00f,  8.618535756e+00f,
                8.560113348e+00f,  8.486123004e+00f,  8.501340876e+00f,  1.641192689e+01f,  1.626870319e+01f,  1.629753401e+01f,
                1.618232695e+01f,  1.604055190e+01f,  1.606897490e+01f,  1.595442430e+01f,  1.581409174e+01f,  1.584211001e+01f,
                2.293803716e+01f,  2.273440424e+01f,  2.277419015e+01f,  2.260513537e+01f,  2.240363048e+01f,  2.244283232e+01f,
                2.227480715e+01f,  2.207542108e+01f,  2.211404346e+01f,  2.341170184e+01f,  2.320543544e+01f,  2.324522653e+01f,
                2.307179819e+01f,  2.286768056e+01f,  2.290688759e+01f,  2.273452340e+01f,  2.253254533e+01f,  2.257117290e+01f,
                2.388536651e+01f,  2.367646664e+01f,  2.371626292e+01f,  2.353846100e+01f,  2.333173064e+01f,  2.337094285e+01f,
                2.319423966e+01f,  2.298966959e+01f,  2.302830234e+01f,  1.463039633e+01f,  1.449913270e+01f,  1.452328294e+01f,
                1.440782359e+01f,  1.427797634e+01f,  1.430175564e+01f,  1.418705873e+01f,  1.405862172e+01f,  1.408203314e+01f,
                6.703508484e+00f,  6.641758332e+00f,  6.652716876e+00f,  6.596644644e+00f,  6.535587708e+00f,  6.546369996e+00f,
                6.490693956e+00f,  6.430327164e+00f,  6.440934732e+00f,  1.029079243e+01f,  1.020198389e+01f,  1.021881902e+01f,
                1.014286834e+01f,  1.005499483e+01f,  1.007158267e+01f,  9.996124912e+00f,  9.909182352e+00f,  9.925524944e+00f,
                1.908474480e+01f,  1.891696272e+01f,  1.894780291e+01f,  1.880010019e+01f,  1.863414826e+01f,  1.866451843e+01f,
                1.851784150e+01f,  1.835371152e+01f,  1.838361578e+01f,  2.651792213e+01f,  2.628052229e+01f,  2.632283237e+01f,
                2.610805550e+01f,  2.587334098e+01f,  2.591498290e+01f,  2.570180462e+01f,  2.546976312e+01f,  2.551074302e+01f,
                2.702058005e+01f,  2.678016658e+01f,  2.682248357e+01f,  2.660270472e+01f,  2.636500421e+01f,  2.640665304e+01f,
                2.618851886e+01f,  2.595351902e+01f,  2.599450584e+01f,  2.752323797e+01f,  2.727981086e+01f,  2.732213477e+01f,
                2.709735394e+01f,  2.685666744e+01f,  2.689832318e+01f,  2.667523310e+01f,  2.643727493e+01f,  2.647826866e+01f,
                1.673347978e+01f,  1.658142326e+01f,  1.660690742e+01f,  1.646233354e+01f,  1.631206416e+01f,  1.633712746e+01f,
                1.619372067e+01f,  1.604523024e+01f,  1.606987677e+01f,  7.610320368e+00f,  7.539246480e+00f,  7.550720976e+00f,
                7.481233776e+00f,  7.411033488e+00f,  7.422309840e+00f,  7.353426160e+00f,  7.284095376e+00f,  7.295175632e+00f,
                5.709348336e+00f,  5.651620752e+00f,  5.660843472e+00f,  5.603928432e+00f,  5.546988432e+00f,  5.556037584e+00f,
                5.499689200e+00f,  5.443532688e+00f,  5.452410320e+00f,  1.034066851e+01f,  1.023361066e+01f,  1.025011075e+01f,
                1.014191702e+01f,  1.003639440e+01f,  1.005257194e+01f,  9.945551456e+00f,  9.841555872e+00f,  9.857414944e+00f,
                1.404064238e+01f,  1.389217234e+01f,  1.391429938e+01f,  1.376094254e+01f,  1.361471544e+01f,  1.363639550e+01f,
                1.348485845e+01f,  1.334086200e+01f,  1.336210123e+01f,  1.430143560e+01f,  1.415094725e+01f,  1.417308120e+01f,
                1.401638126e+01f,  1.386816350e+01f,  1.388985048e+01f,  1.373501640e+01f,  1.358905694e+01f,  1.361030309e+01f,
                1.456222882e+01f,  1.440972216e+01f,  1.443186302e+01f,  1.427181998e+01f,  1.412161157e+01f,  1.414330546e+01f,
                1.398517435e+01f,  1.383725189e+01f,  1.385850494e+01f,  8.568242016e+00f,  8.475361440e+00f,  8.488274976e+00f,
                8.388297312e+00f,  8.296908960e+00f,  8.309549088e+00f,  8.210885984e+00f,  8.120981664e+00f,  8.133352480e+00f,
                3.768957936e+00f,  3.726753168e+00f,  3.732384720e+00f,  3.685914480e+00f,  3.644435856e+00f,  3.649942992e+00f,
                3.604150000e+00f,  3.563393424e+00f,  3.568778192e+00f,  2.653670628e+00f,  2.621266524e+00f,  2.625499116e+00f,
                2.589410052e+00f,  2.557558620e+00f,  2.561688684e+00f,  2.526081060e+00f,  2.494779228e+00f,  2.498808300e+00f,
                4.647289896e+00f,  4.588447896e+00f,  4.595785848e+00f,  4.528719720e+00f,  4.470953112e+00f,  4.478104440e+00f,
                4.412031144e+00f,  4.355333784e+00f,  4.362301560e+00f,  6.101486028e+00f,  6.021812916e+00f,  6.031350180e+00f,
                5.938778412e+00f,  5.860673460e+00f,  5.869958436e+00f,  5.778920844e+00f,  5.702374836e+00f,  5.711412132e+00f,
                6.212353644e+00f,  6.131540052e+00f,  6.141082500e+00f,  6.046625484e+00f,  5.967400788e+00f,  5.976690948e+00f,
                5.883802668e+00f,  5.806157652e+00f,  5.815200132e+00f,  6.323221260e+00f,  6.241267188e+00f,  6.250814820e+00f,
                6.154472556e+00f,  6.074128116e+00f,  6.083423460e+00f,  5.988684492e+00f,  5.909940468e+00f,  5.918988132e+00f,
                3.532463208e+00f,  3.484088280e+00f,  3.489399864e+00f,  3.431094696e+00f,  3.383762904e+00f,  3.388924728e+00f,
                3.331718376e+00f,  3.285423576e+00f,  3.290438712e+00f,  1.469355876e+00f,  1.448179164e+00f,  1.450381932e+00f,
                1.424310660e+00f,  1.403640540e+00f,  1.405777644e+00f,  1.380270756e+00f,  1.360104156e+00f,  1.362177132e+00f,
                9.222661680e-01f,  9.073162320e-01f,  9.087906960e-01f,  8.900326320e-01f,  8.754258000e-01f,  8.768503440e-01f,
                8.584508720e-01f,  8.441850960e-01f,  8.455607440e-01f,  1.517633712e+00f,  1.491490128e+00f,  1.493908752e+00f,
                1.460390448e+00f,  1.434913104e+00f,  1.437244176e+00f,  1.404463024e+00f,  1.379647824e+00f,  1.381893392e+00f,
                1.873599336e+00f,  1.839778776e+00f,  1.842758712e+00f,  1.798717608e+00f,  1.765866456e+00f,  1.768733496e+00f,
                1.725828072e+00f,  1.693940184e+00f,  1.696697400e+00f,  1.906951464e+00f,  1.872619416e+00f,  1.875602808e+00f,
                1.830719592e+00f,  1.797370776e+00f,  1.800241272e+00f,  1.756516776e+00f,  1.724145048e+00f,  1.726905720e+00f,
                1.940303592e+00f,  1.905460056e+00f,  1.908446904e+00f,  1.862721576e+00f,  1.828875096e+00f,  1.831749048e+00f,
                1.787205480e+00f,  1.754349912e+00f,  1.757114040e+00f,  9.669565920e-01f,  9.476804640e-01f,  9.491905440e-01f,
                9.232454880e-01f,  9.046140960e-01f,  9.060612000e-01f,  8.809239520e-01f,  8.629332000e-01f,  8.643193760e-01f,
                3.523034160e-01f,  3.445619280e-01f,  3.451255440e-01f,  3.345021360e-01f,  3.270730320e-01f,  3.276113040e-01f,
                3.174017840e-01f,  3.102830160e-01f,  3.107969680e-01f,  1.670189880e-01f,  1.625662920e-01f,  1.628510760e-01f,
                1.565742840e-01f,  1.522804680e-01f,  1.525495080e-01f,  1.464708280e-01f,  1.423348680e-01f,  1.425886760e-01f,
                2.342343600e-01f,  2.270965200e-01f,  2.275115280e-01f,  2.172517680e-01f,  2.104217040e-01f,  2.108113680e-01f,
                2.009578160e-01f,  1.944334800e-01f,  1.947988240e-01f,  2.489334120e-01f,  2.407581720e-01f,  2.412225720e-01f,
                2.293934760e-01f,  2.216649240e-01f,  2.221005240e-01f,  2.108957160e-01f,  2.036107800e-01f,  2.040191160e-01f,
                2.532922920e-01f,  2.449857240e-01f,  2.454518520e-01f,  2.334090600e-01f,  2.255560920e-01f,  2.259934200e-01f,
                2.145864360e-01f,  2.071839960e-01f,  2.075940600e-01f,  2.576511720e-01f,  2.492132760e-01f,  2.496811320e-01f,
                2.374246440e-01f,  2.294472600e-01f,  2.298863160e-01f,  2.182771560e-01f,  2.107572120e-01f,  2.111690040e-01f,
                7.993874400e-02f,  7.617933600e-02f,  7.636125600e-02f,  7.075442400e-02f,  6.729204000e-02f,  6.746090400e-02f,
                6.229560800e-02f,  5.912820000e-02f,  5.928503200e-02f,  1.129201200e-02f,  1.030530000e-02f,  1.035584400e-02f,
                8.874444000e-03f,  8.031252000e-03f,  8.078340000e-03f,  6.822700000e-03f,  6.122004000e-03f,  6.166148000e-03f,
            };

            float[] x_actual = x.ToArray();

            AssertError.Tolerance(x_expect, x_actual, 1e-7f, 1e-5f, $"mismatch value {inchannels},{outchannels},{kwidth},{kheight},{inwidth},{inheight},{batch}");
        }
    }
}
