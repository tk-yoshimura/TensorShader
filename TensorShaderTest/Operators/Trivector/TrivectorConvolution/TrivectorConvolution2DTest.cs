using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;
using TensorShader;
using TensorShader.Operators.TrivectorConvolution;
using TensorShaderCudaBackend.API;

namespace TensorShaderTest.Operators.Trivector {
    [TestClass]
    public class TrivectorConvolution2DTest {
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

                                        float[] xval = (new float[inwidth * inheight * inchannels * batch]).Select((_, idx) => (float)random.NextDouble() * 1e-2f).ToArray();
                                        float[] wval = (new float[kwidth * kheight * inchannels * outchannels / 9 * 4]).Select((_, idx) => (float)random.NextDouble() * 1e-2f).ToArray();

                                        Trivector[] xcval = (new Trivector[xval.Length / 3])
                                            .Select((_, idx) => new Trivector(xval[idx * 3], xval[idx * 3 + 1], xval[idx * 3 + 2])).ToArray();

                                        Quaternion.Quaternion[] wcval = (new Quaternion.Quaternion[wval.Length / 4])
                                            .Select((_, idx) => new Quaternion.Quaternion(wval[idx * 4], wval[idx * 4 + 1], wval[idx * 4 + 2], wval[idx * 4 + 3])).ToArray();

                                        TrivectorMap2D x = new(inchannels / 3, inwidth, inheight, batch, xcval);
                                        Quaternion.QuaternionFilter2D w = new(inchannels / 3, outchannels / 3, kwidth, kheight, wcval);

                                        TrivectorMap2D y = Reference(x, w, kwidth, kheight);

                                        OverflowCheckedTensor x_tensor = new(Shape.Map2D(inchannels, inwidth, inheight, batch), xval);
                                        OverflowCheckedTensor w_tensor = new(Shape.Kernel2D(inchannels / 3 * 4, outchannels / 3, kwidth, kheight), wval);

                                        OverflowCheckedTensor y_tensor = new(Shape.Map2D(outchannels, outwidth, outheight, batch));

                                        TrivectorConvolution2D ope = new(inwidth, inheight, inchannels, outchannels, kwidth, kheight, gradmode: false, batch);

                                        ope.Execute(x_tensor, w_tensor, y_tensor);

                                        float[] y_expect = y.ToArray();
                                        float[] y_actual = y_tensor.State.Value;

                                        CollectionAssert.AreEqual(xval, x_tensor.State.Value);
                                        CollectionAssert.AreEqual(wval, w_tensor.State.Value);

                                        AssertError.Tolerance(y_expect, y_actual, 1e-6f, 1e-4f, ref max_err, $"mismatch value {inchannels},{outchannels},{kwidth},{kheight},{inwidth},{inheight},{batch}");

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

                                        float[] xval = (new float[inwidth * inheight * inchannels * batch]).Select((_, idx) => (float)random.NextDouble() * 1e-2f).ToArray();
                                        float[] wval = (new float[kwidth * kheight * inchannels * outchannels / 9 * 4]).Select((_, idx) => (float)random.NextDouble() * 1e-2f).ToArray();

                                        Trivector[] xcval = (new Trivector[xval.Length / 3])
                                            .Select((_, idx) => new Trivector(xval[idx * 3], xval[idx * 3 + 1], xval[idx * 3 + 2])).ToArray();

                                        Quaternion.Quaternion[] wcval = (new Quaternion.Quaternion[wval.Length / 4])
                                            .Select((_, idx) => new Quaternion.Quaternion(wval[idx * 4], wval[idx * 4 + 1], wval[idx * 4 + 2], wval[idx * 4 + 3])).ToArray();

                                        TrivectorMap2D x = new(inchannels / 3, inwidth, inheight, batch, xcval);
                                        Quaternion.QuaternionFilter2D w = new(inchannels / 3, outchannels / 3, kwidth, kheight, wcval);

                                        TrivectorMap2D y = Reference(x, w, kwidth, kheight);

                                        OverflowCheckedTensor x_tensor = new(Shape.Map2D(inchannels, inwidth, inheight, batch), xval);
                                        OverflowCheckedTensor w_tensor = new(Shape.Kernel2D(inchannels / 3 * 4, outchannels / 3, kwidth, kheight), wval);

                                        OverflowCheckedTensor y_tensor = new(Shape.Map2D(outchannels, outwidth, outheight, batch));

                                        TrivectorConvolution2D ope = new(inwidth, inheight, inchannels, outchannels, kwidth, kheight, gradmode: false, batch);

                                        ope.Execute(x_tensor, w_tensor, y_tensor);

                                        float[] y_expect = y.ToArray();
                                        float[] y_actual = y_tensor.State.Value;

                                        CollectionAssert.AreEqual(xval, x_tensor.State.Value);
                                        CollectionAssert.AreEqual(wval, w_tensor.State.Value);

                                        AssertError.Tolerance(y_expect, y_actual, 1e-7f, 1e-5f, ref max_err, $"mismatch value {inchannels},{outchannels},{kwidth},{kheight},{inwidth},{inheight},{batch}");

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

            float[] xval = (new float[inwidth * inheight * inchannels * batch]).Select((_, idx) => (float)random.NextDouble() * 1e-2f).ToArray();
            float[] wval = (new float[kwidth * kheight * inchannels * outchannels / 9 * 4]).Select((_, idx) => (float)random.NextDouble() * 1e-2f).ToArray();

            Trivector[] xcval = (new Trivector[xval.Length / 3])
                .Select((_, idx) => new Trivector(xval[idx * 3], xval[idx * 3 + 1], xval[idx * 3 + 2])).ToArray();

            Quaternion.Quaternion[] wcval = (new Quaternion.Quaternion[wval.Length / 4])
                .Select((_, idx) => new Quaternion.Quaternion(wval[idx * 4], wval[idx * 4 + 1], wval[idx * 4 + 2], wval[idx * 4 + 3])).ToArray();

            TrivectorMap2D x = new(inchannels / 3, inwidth, inheight, batch, xcval);
            Quaternion.QuaternionFilter2D w = new(inchannels / 3, outchannels / 3, kwidth, kheight, wcval);

            TrivectorMap2D y = Reference(x, w, kwidth, kheight);

            OverflowCheckedTensor x_tensor = new(Shape.Map2D(inchannels, inwidth, inheight, batch), xval);
            OverflowCheckedTensor w_tensor = new(Shape.Kernel2D(inchannels / 3 * 4, outchannels / 3, kwidth, kheight), wval);

            OverflowCheckedTensor y_tensor = new(Shape.Map2D(outchannels, outwidth, outheight, batch));

            TrivectorConvolution2D ope = new(inwidth, inheight, inchannels, outchannels, kwidth, kheight, gradmode: false, batch);

            ope.Execute(x_tensor, w_tensor, y_tensor);

            float[] y_expect = y.ToArray();
            float[] y_actual = y_tensor.State.Value;

            CollectionAssert.AreEqual(xval, x_tensor.State.Value);
            CollectionAssert.AreEqual(wval, w_tensor.State.Value);

            AssertError.Tolerance(y_expect, y_actual, 1e-7f, 1e-5f, ref max_err, $"mismatch value {inchannels},{outchannels},{kwidth},{kheight},{inwidth},{inheight},{batch}");

            Console.WriteLine($"pass: {inchannels},{outchannels},{kwidth},{kheight},{inwidth},{inheight},{batch}");

            Console.WriteLine($"maxerr:{max_err}");
        }

        [TestMethod]
        public void SpeedTest() {
            int inwidth = 32, inheight = 32, inchannels = 33, outchannels = 33, ksize = 3;
            int outwidth = inwidth - ksize + 1, outheight = inheight - ksize + 1;

            OverflowCheckedTensor x_tensor = new(Shape.Map2D(inchannels, inwidth, inheight));
            OverflowCheckedTensor w_tensor = new(Shape.Kernel2D(inchannels / 3 * 4, outchannels / 3, ksize, ksize));

            OverflowCheckedTensor y_tensor = new(Shape.Map2D(outchannels, outwidth, outheight));

            TrivectorConvolution2D ope = new(inwidth, inheight, inchannels, outchannels, ksize, ksize);

            Cuda.Profiler.Initialize("../../../../profiler.nvsetting", "../../nvprofiles/trivector_convolution_2d.nvvp");
            Cuda.Profiler.Start();

            ope.Execute(x_tensor, w_tensor, y_tensor);

            Cuda.Profiler.Stop();
        }

        public static TrivectorMap2D Reference(TrivectorMap2D x, Quaternion.QuaternionFilter2D w, int kwidth, int kheight) {
            int inchannels = x.Channels, outchannels = w.OutChannels, batch = x.Batch;
            int inw = x.Width, inh = x.Height;
            int outw = inw - kwidth + 1, outh = inh - kheight + 1;

            TrivectorMap2D y = new(outchannels, outw, outh, batch);

            for (int kx, ky = 0; ky < kheight; ky++) {
                for (kx = 0; kx < kwidth; kx++) {
                    for (int th = 0; th < batch; th++) {
                        for (int ox, oy = 0; oy < outh; oy++) {
                            for (ox = 0; ox < outw; ox++) {
                                for (int outch = 0; outch < outchannels; outch++) {
                                    Trivector sum = y[outch, ox, oy, th];

                                    for (int inch = 0; inch < inchannels; inch++) {
                                        sum += x[inch, kx + ox, ky + oy, th] * w[inch, outch, kx, ky];
                                    }

                                    y[outch, ox, oy, th] = sum;
                                }
                            }
                        }
                    }
                }
            }

            return y;
        }

        [TestMethod]
        public void ReferenceTest() {
            int inchannels = 9, outchannels = 12, kwidth = 3, kheight = 5, inwidth = 7, inheight = 8, batch = 3;
            int outwidth = inwidth - kwidth + 1, outheight = inheight - kheight + 1;

            float[] xval = (new float[batch * inwidth * inheight * inchannels]).Select((_, idx) => idx * 1e-3f).ToArray();
            float[] wval = (new float[kwidth * kheight * outchannels * inchannels / 9 * 4]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

            Trivector[] xcval = (new Trivector[xval.Length / 3])
                .Select((_, idx) => new Trivector(xval[idx * 3], xval[idx * 3 + 1], xval[idx * 3 + 2])).ToArray();

            Quaternion.Quaternion[] wcval = (new Quaternion.Quaternion[wval.Length / 4])
                .Select((_, idx) => new Quaternion.Quaternion(wval[idx * 4], wval[idx * 4 + 1], wval[idx * 4 + 2], wval[idx * 4 + 3])).ToArray();

            TrivectorMap2D x = new(inchannels / 3, inwidth, inheight, batch, xcval);
            Quaternion.QuaternionFilter2D w = new(inchannels / 3, outchannels / 3, kwidth, kheight, wcval);

            TrivectorMap2D y = Reference(x, w, kwidth, kheight);

            float[] y_expect = {
                2.181353820e+00f,  2.096150130e+00f,  2.129823480e+00f,  2.035735260e+00f,  1.954642050e+00f,  1.986704040e+00f,
                1.897374300e+00f,  1.820287890e+00f,  1.850790360e+00f,  1.766270940e+00f,  1.693087650e+00f,  1.722082440e+00f,
                2.483127690e+00f,  2.396086920e+00f,  2.429765130e+00f,  2.323026330e+00f,  2.240154360e+00f,  2.272221210e+00f,
                2.170649130e+00f,  2.091842280e+00f,  2.122349610e+00f,  2.025996090e+00f,  1.951150680e+00f,  1.980150330e+00f,
                2.784901560e+00f,  2.696023710e+00f,  2.729706780e+00f,  2.610317400e+00f,  2.525666670e+00f,  2.557738380e+00f,
                2.443923960e+00f,  2.363396670e+00f,  2.393908860e+00f,  2.285721240e+00f,  2.209213710e+00f,  2.238218220e+00f,
                3.086675430e+00f,  2.995960500e+00f,  3.029648430e+00f,  2.897608470e+00f,  2.811178980e+00f,  2.843255550e+00f,
                2.717198790e+00f,  2.634951060e+00f,  2.665468110e+00f,  2.545446390e+00f,  2.467276740e+00f,  2.496286110e+00f,
                3.388449300e+00f,  3.295897290e+00f,  3.329590080e+00f,  3.184899540e+00f,  3.096691290e+00f,  3.128772720e+00f,
                2.990473620e+00f,  2.906505450e+00f,  2.937027360e+00f,  2.805171540e+00f,  2.725339770e+00f,  2.754354000e+00f,
                4.293770910e+00f,  4.195707660e+00f,  4.229415030e+00f,  4.046772750e+00f,  3.953228220e+00f,  3.985324230e+00f,
                3.810298110e+00f,  3.721168620e+00f,  3.751705110e+00f,  3.584346990e+00f,  3.499528860e+00f,  3.528557670e+00f,
                4.595544780e+00f,  4.495644450e+00f,  4.529356680e+00f,  4.334063820e+00f,  4.238740530e+00f,  4.270841400e+00f,
                4.083572940e+00f,  3.992723010e+00f,  4.023264360e+00f,  3.844072140e+00f,  3.757591890e+00f,  3.786625560e+00f,
                4.897318650e+00f,  4.795581240e+00f,  4.829298330e+00f,  4.621354890e+00f,  4.524252840e+00f,  4.556358570e+00f,
                4.356847770e+00f,  4.264277400e+00f,  4.294823610e+00f,  4.103797290e+00f,  4.015654920e+00f,  4.044693450e+00f,
                5.199092520e+00f,  5.095518030e+00f,  5.129239980e+00f,  4.908645960e+00f,  4.809765150e+00f,  4.841875740e+00f,
                4.630122600e+00f,  4.535831790e+00f,  4.566382860e+00f,  4.363522440e+00f,  4.273717950e+00f,  4.302761340e+00f,
                5.500866390e+00f,  5.395454820e+00f,  5.429181630e+00f,  5.195937030e+00f,  5.095277460e+00f,  5.127392910e+00f,
                4.903397430e+00f,  4.807386180e+00f,  4.837942110e+00f,  4.623247590e+00f,  4.531780980e+00f,  4.560829230e+00f,
                6.406188000e+00f,  6.295265190e+00f,  6.329006580e+00f,  6.057810240e+00f,  5.951814390e+00f,  5.983944420e+00f,
                5.723221920e+00f,  5.622049350e+00f,  5.652619860e+00f,  5.402423040e+00f,  5.305970070e+00f,  5.335032900e+00f,
                6.707961870e+00f,  6.595201980e+00f,  6.628948230e+00f,  6.345101310e+00f,  6.237326700e+00f,  6.269461590e+00f,
                5.996496750e+00f,  5.893603740e+00f,  5.924179110e+00f,  5.662148190e+00f,  5.564033100e+00f,  5.593100790e+00f,
                7.009735740e+00f,  6.895138770e+00f,  6.928889880e+00f,  6.632392380e+00f,  6.522839010e+00f,  6.554978760e+00f,
                6.269771580e+00f,  6.165158130e+00f,  6.195738360e+00f,  5.921873340e+00f,  5.822096130e+00f,  5.851168680e+00f,
                7.311509610e+00f,  7.195075560e+00f,  7.228831530e+00f,  6.919683450e+00f,  6.808351320e+00f,  6.840495930e+00f,
                6.543046410e+00f,  6.436712520e+00f,  6.467297610e+00f,  6.181598490e+00f,  6.080159160e+00f,  6.109236570e+00f,
                7.613283480e+00f,  7.495012350e+00f,  7.528773180e+00f,  7.206974520e+00f,  7.093863630e+00f,  7.126013100e+00f,
                6.816321240e+00f,  6.708266910e+00f,  6.738856860e+00f,  6.441323640e+00f,  6.338222190e+00f,  6.367304460e+00f,
                8.518605090e+00f,  8.394822720e+00f,  8.428598130e+00f,  8.068847730e+00f,  7.950400560e+00f,  7.982564610e+00f,
                7.636145730e+00f,  7.522930080e+00f,  7.553534610e+00f,  7.220499090e+00f,  7.112411280e+00f,  7.141508130e+00f,
                8.820378960e+00f,  8.694759510e+00f,  8.728539780e+00f,  8.356138800e+00f,  8.235912870e+00f,  8.268081780e+00f,
                7.909420560e+00f,  7.794484470e+00f,  7.825093860e+00f,  7.480224240e+00f,  7.370474310e+00f,  7.399576020e+00f,
                9.122152830e+00f,  8.994696300e+00f,  9.028481430e+00f,  8.643429870e+00f,  8.521425180e+00f,  8.553598950e+00f,
                8.182695390e+00f,  8.066038860e+00f,  8.096653110e+00f,  7.739949390e+00f,  7.628537340e+00f,  7.657643910e+00f,
                9.423926700e+00f,  9.294633090e+00f,  9.328423080e+00f,  8.930720940e+00f,  8.806937490e+00f,  8.839116120e+00f,
                8.455970220e+00f,  8.337593250e+00f,  8.368212360e+00f,  7.999674540e+00f,  7.886600370e+00f,  7.915711800e+00f,
                9.725700570e+00f,  9.594569880e+00f,  9.628364730e+00f,  9.218012010e+00f,  9.092449800e+00f,  9.124633290e+00f,
                8.729245050e+00f,  8.609147640e+00f,  8.639771610e+00f,  8.259399690e+00f,  8.144663400e+00f,  8.173779690e+00f,
                1.908069054e+01f,  1.889261037e+01f,  1.892655588e+01f,  1.812403518e+01f,  1.794333141e+01f,  1.797566556e+01f,
                1.720076478e+01f,  1.702733373e+01f,  1.705810836e+01f,  1.631087934e+01f,  1.614461733e+01f,  1.617388428e+01f,
                1.938246441e+01f,  1.919254716e+01f,  1.922649753e+01f,  1.841132625e+01f,  1.822884372e+01f,  1.826118273e+01f,
                1.747403961e+01f,  1.729888812e+01f,  1.732966761e+01f,  1.657060449e+01f,  1.640268036e+01f,  1.643195217e+01f,
                1.968423828e+01f,  1.949248395e+01f,  1.952643918e+01f,  1.869861732e+01f,  1.851435603e+01f,  1.854669990e+01f,
                1.774731444e+01f,  1.757044251e+01f,  1.760122686e+01f,  1.683032964e+01f,  1.666074339e+01f,  1.669002006e+01f,
                1.998601215e+01f,  1.979242074e+01f,  1.982638083e+01f,  1.898590839e+01f,  1.879986834e+01f,  1.883221707e+01f,
                1.802058927e+01f,  1.784199690e+01f,  1.787278611e+01f,  1.709005479e+01f,  1.691880642e+01f,  1.694808795e+01f,
                2.028778602e+01f,  2.009235753e+01f,  2.012632248e+01f,  1.927319946e+01f,  1.908538065e+01f,  1.911773424e+01f,
                1.829386410e+01f,  1.811355129e+01f,  1.814434536e+01f,  1.734977994e+01f,  1.717686945e+01f,  1.720615584e+01f,
                2.119310763e+01f,  2.099216790e+01f,  2.102614743e+01f,  2.013507267e+01f,  1.994191758e+01f,  1.997428575e+01f,
                1.911368859e+01f,  1.892821446e+01f,  1.895902311e+01f,  1.812895539e+01f,  1.795105854e+01f,  1.798035951e+01f,
                2.149488150e+01f,  2.129210469e+01f,  2.132608908e+01f,  2.042236374e+01f,  2.022742989e+01f,  2.025980292e+01f,
                1.938696342e+01f,  1.919976885e+01f,  1.923058236e+01f,  1.838868054e+01f,  1.820912157e+01f,  1.823842740e+01f,
                2.179665537e+01f,  2.159204148e+01f,  2.162603073e+01f,  2.070965481e+01f,  2.051294220e+01f,  2.054532009e+01f,
                1.966023825e+01f,  1.947132324e+01f,  1.950214161e+01f,  1.864840569e+01f,  1.846718460e+01f,  1.849649529e+01f,
                2.209842924e+01f,  2.189197827e+01f,  2.192597238e+01f,  2.099694588e+01f,  2.079845451e+01f,  2.083083726e+01f,
                1.993351308e+01f,  1.974287763e+01f,  1.977370086e+01f,  1.890813084e+01f,  1.872524763e+01f,  1.875456318e+01f,
                2.240020311e+01f,  2.219191506e+01f,  2.222591403e+01f,  2.128423695e+01f,  2.108396682e+01f,  2.111635443e+01f,
                2.020678791e+01f,  2.001443202e+01f,  2.004526011e+01f,  1.916785599e+01f,  1.898331066e+01f,  1.901263107e+01f,
                2.330552472e+01f,  2.309172543e+01f,  2.312573898e+01f,  2.214611016e+01f,  2.194050375e+01f,  2.197290594e+01f,
                2.102661240e+01f,  2.082909519e+01f,  2.085993786e+01f,  1.994703144e+01f,  1.975749975e+01f,  1.978683474e+01f,
                2.360729859e+01f,  2.339166222e+01f,  2.342568063e+01f,  2.243340123e+01f,  2.222601606e+01f,  2.225842311e+01f,
                2.129988723e+01f,  2.110064958e+01f,  2.113149711e+01f,  2.020675659e+01f,  2.001556278e+01f,  2.004490263e+01f,
                2.390907246e+01f,  2.369159901e+01f,  2.372562228e+01f,  2.272069230e+01f,  2.251152837e+01f,  2.254394028e+01f,
                2.157316206e+01f,  2.137220397e+01f,  2.140305636e+01f,  2.046648174e+01f,  2.027362581e+01f,  2.030297052e+01f,
                2.421084633e+01f,  2.399153580e+01f,  2.402556393e+01f,  2.300798337e+01f,  2.279704068e+01f,  2.282945745e+01f,
                2.184643689e+01f,  2.164375836e+01f,  2.167461561e+01f,  2.072620689e+01f,  2.053168884e+01f,  2.056103841e+01f,
                2.451262020e+01f,  2.429147259e+01f,  2.432550558e+01f,  2.329527444e+01f,  2.308255299e+01f,  2.311497462e+01f,
                2.211971172e+01f,  2.191531275e+01f,  2.194617486e+01f,  2.098593204e+01f,  2.078975187e+01f,  2.081910630e+01f,
                2.541794181e+01f,  2.519128296e+01f,  2.522533053e+01f,  2.415714765e+01f,  2.393908992e+01f,  2.397152613e+01f,
                2.293953621e+01f,  2.272997592e+01f,  2.276085261e+01f,  2.176510749e+01f,  2.156394096e+01f,  2.159330997e+01f,
                2.571971568e+01f,  2.549121975e+01f,  2.552527218e+01f,  2.444443872e+01f,  2.422460223e+01f,  2.425704330e+01f,
                2.321281104e+01f,  2.300153031e+01f,  2.303241186e+01f,  2.202483264e+01f,  2.182200399e+01f,  2.185137786e+01f,
                2.602148955e+01f,  2.579115654e+01f,  2.582521383e+01f,  2.473172979e+01f,  2.451011454e+01f,  2.454256047e+01f,
                2.348608587e+01f,  2.327308470e+01f,  2.330397111e+01f,  2.228455779e+01f,  2.208006702e+01f,  2.210944575e+01f,
                2.632326342e+01f,  2.609109333e+01f,  2.612515548e+01f,  2.501902086e+01f,  2.479562685e+01f,  2.482807764e+01f,
                2.375936070e+01f,  2.354463909e+01f,  2.357553036e+01f,  2.254428294e+01f,  2.233813005e+01f,  2.236751364e+01f,
                2.662503729e+01f,  2.639103012e+01f,  2.642509713e+01f,  2.530631193e+01f,  2.508113916e+01f,  2.511359481e+01f,
                2.403263553e+01f,  2.381619348e+01f,  2.384708961e+01f,  2.280400809e+01f,  2.259619308e+01f,  2.262558153e+01f,
                3.598002726e+01f,  3.568907061e+01f,  3.572328828e+01f,  3.421233510e+01f,  3.393202077e+01f,  3.396462708e+01f,
                3.250415526e+01f,  3.223437957e+01f,  3.226542636e+01f,  3.085548774e+01f,  3.059614701e+01f,  3.062568612e+01f,
                3.628180113e+01f,  3.598900740e+01f,  3.602322993e+01f,  3.449962617e+01f,  3.421753308e+01f,  3.425014425e+01f,
                3.277743009e+01f,  3.250593396e+01f,  3.253698561e+01f,  3.111521289e+01f,  3.085421004e+01f,  3.088375401e+01f,
                3.658357500e+01f,  3.628894419e+01f,  3.632317158e+01f,  3.478691724e+01f,  3.450304539e+01f,  3.453566142e+01f,
                3.305070492e+01f,  3.277748835e+01f,  3.280854486e+01f,  3.137493804e+01f,  3.111227307e+01f,  3.114182190e+01f,
                3.688534887e+01f,  3.658888098e+01f,  3.662311323e+01f,  3.507420831e+01f,  3.478855770e+01f,  3.482117859e+01f,
                3.332397975e+01f,  3.304904274e+01f,  3.308010411e+01f,  3.163466319e+01f,  3.137033610e+01f,  3.139988979e+01f,
                3.718712274e+01f,  3.688881777e+01f,  3.692305488e+01f,  3.536149938e+01f,  3.507407001e+01f,  3.510669576e+01f,
                3.359725458e+01f,  3.332059713e+01f,  3.335166336e+01f,  3.189438834e+01f,  3.162839913e+01f,  3.165795768e+01f,
                3.809244435e+01f,  3.778862814e+01f,  3.782287983e+01f,  3.622337259e+01f,  3.593060694e+01f,  3.596324727e+01f,
                3.441707907e+01f,  3.413526030e+01f,  3.416634111e+01f,  3.267356379e+01f,  3.240258822e+01f,  3.243216135e+01f,
                3.839421822e+01f,  3.808856493e+01f,  3.812282148e+01f,  3.651066366e+01f,  3.621611925e+01f,  3.624876444e+01f,
                3.469035390e+01f,  3.440681469e+01f,  3.443790036e+01f,  3.293328894e+01f,  3.266065125e+01f,  3.269022924e+01f,
                3.869599209e+01f,  3.838850172e+01f,  3.842276313e+01f,  3.679795473e+01f,  3.650163156e+01f,  3.653428161e+01f,
                3.496362873e+01f,  3.467836908e+01f,  3.470945961e+01f,  3.319301409e+01f,  3.291871428e+01f,  3.294829713e+01f,
                3.899776596e+01f,  3.868843851e+01f,  3.872270478e+01f,  3.708524580e+01f,  3.678714387e+01f,  3.681979878e+01f,
                3.523690356e+01f,  3.494992347e+01f,  3.498101886e+01f,  3.345273924e+01f,  3.317677731e+01f,  3.320636502e+01f,
                3.929953983e+01f,  3.898837530e+01f,  3.902264643e+01f,  3.737253687e+01f,  3.707265618e+01f,  3.710531595e+01f,
                3.551017839e+01f,  3.522147786e+01f,  3.525257811e+01f,  3.371246439e+01f,  3.343484034e+01f,  3.346443291e+01f,
                4.020486144e+01f,  3.988818567e+01f,  3.992247138e+01f,  3.823441008e+01f,  3.792919311e+01f,  3.796186746e+01f,
                3.633000288e+01f,  3.603614103e+01f,  3.606725586e+01f,  3.449163984e+01f,  3.420902943e+01f,  3.423863658e+01f,
                4.050663531e+01f,  4.018812246e+01f,  4.022241303e+01f,  3.852170115e+01f,  3.821470542e+01f,  3.824738463e+01f,
                3.660327771e+01f,  3.630769542e+01f,  3.633881511e+01f,  3.475136499e+01f,  3.446709246e+01f,  3.449670447e+01f,
                4.080840918e+01f,  4.048805925e+01f,  4.052235468e+01f,  3.880899222e+01f,  3.850021773e+01f,  3.853290180e+01f,
                3.687655254e+01f,  3.657924981e+01f,  3.661037436e+01f,  3.501109014e+01f,  3.472515549e+01f,  3.475477236e+01f,
                4.111018305e+01f,  4.078799604e+01f,  4.082229633e+01f,  3.909628329e+01f,  3.878573004e+01f,  3.881841897e+01f,
                3.714982737e+01f,  3.685080420e+01f,  3.688193361e+01f,  3.527081529e+01f,  3.498321852e+01f,  3.501284025e+01f,
                4.141195692e+01f,  4.108793283e+01f,  4.112223798e+01f,  3.938357436e+01f,  3.907124235e+01f,  3.910393614e+01f,
                3.742310220e+01f,  3.712235859e+01f,  3.715349286e+01f,  3.553054044e+01f,  3.524128155e+01f,  3.527090814e+01f,
                4.231727853e+01f,  4.198774320e+01f,  4.202206293e+01f,  4.024544757e+01f,  3.992777928e+01f,  3.996048765e+01f,
                3.824292669e+01f,  3.793702176e+01f,  3.796817061e+01f,  3.630971589e+01f,  3.601547064e+01f,  3.604511181e+01f,
                4.261905240e+01f,  4.228767999e+01f,  4.232200458e+01f,  4.053273864e+01f,  4.021329159e+01f,  4.024600482e+01f,
                3.851620152e+01f,  3.820857615e+01f,  3.823972986e+01f,  3.656944104e+01f,  3.627353367e+01f,  3.630317970e+01f,
                4.292082627e+01f,  4.258761678e+01f,  4.262194623e+01f,  4.082002971e+01f,  4.049880390e+01f,  4.053152199e+01f,
                3.878947635e+01f,  3.848013054e+01f,  3.851128911e+01f,  3.682916619e+01f,  3.653159670e+01f,  3.656124759e+01f,
                4.322260014e+01f,  4.288755357e+01f,  4.292188788e+01f,  4.110732078e+01f,  4.078431621e+01f,  4.081703916e+01f,
                3.906275118e+01f,  3.875168493e+01f,  3.878284836e+01f,  3.708889134e+01f,  3.678965973e+01f,  3.681931548e+01f,
                4.352437401e+01f,  4.318749036e+01f,  4.322182953e+01f,  4.139461185e+01f,  4.106982852e+01f,  4.110255633e+01f,
                3.933602601e+01f,  3.902323932e+01f,  3.905440761e+01f,  3.734861649e+01f,  3.704772276e+01f,  3.707738337e+01f,
            };

            float[] y_actual = y.ToArray();

            AssertError.Tolerance(y_expect, y_actual, 1e-7f, 1e-5f, $"mismatch value {inchannels},{outchannels},{kwidth},{kheight},{inwidth},{inheight},{batch}");
        }
    }
}
