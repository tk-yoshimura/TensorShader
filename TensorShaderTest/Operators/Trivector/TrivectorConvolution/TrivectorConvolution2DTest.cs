using System;
using System.Diagnostics;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using TensorShader.Operators.TrivectorConvolution;

namespace TensorShaderTest.Operators.Trivector {
    [TestClass]
    public class TrivectorConvolution2DTest {
        [TestMethod]
        public void ExecuteTest() {
            float max_err = 0;

            foreach (int batch in new int[] { 1, 2, 3 }) {
                foreach (int inchannels in new int[] { 3, 6, 9, 12 }) {
                    foreach (int outchannels in new int[] { 3, 6, 9, 12 }) {
                        foreach (int kheight in new int[] { 1, 3, 5 }) {
                            foreach (int kwidth in new int[] { 1, 3, 5 }) {
                                foreach (int stride in new int[] { 1, 2, 3 }) {
                                    foreach (int inwidth in new int[] { 8, 9, 13, 17 }) {
                                        foreach (int inheight in new int[] { 8, 9, 19, 23 }) {
                                            int outwidth = inwidth - kwidth + 1, outheight = inheight - kheight + 1;

                                            float[] xval = (new float[inwidth * inheight * inchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
                                            float[] wval = (new float[kwidth * kheight * inchannels * outchannels / 9 * 4]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

                                            Trivector[] xcval = (new Trivector[xval.Length / 3])
                                                .Select((_, idx) => new Trivector(xval[idx * 3], xval[idx * 3 + 1], xval[idx * 3 + 2])).ToArray();

                                            Quaternion.Quaternion[] wcval = (new Quaternion.Quaternion[wval.Length / 4])
                                                .Select((_, idx) => new Quaternion.Quaternion(wval[idx * 4], wval[idx * 4 + 1], wval[idx * 4 + 2], wval[idx * 4 + 3])).ToArray();

                                            TrivectorMap2D x = new TrivectorMap2D(inchannels / 3, inwidth, inheight, batch, xcval);
                                            Quaternion.QuaternionFilter2D w = new Quaternion.QuaternionFilter2D(inchannels / 3, outchannels / 3, kwidth, kheight, wcval);

                                            TrivectorMap2D y = Reference(x, w, kwidth, kheight, stride);

                                            OverflowCheckedTensor x_tensor = new OverflowCheckedTensor(Shape.Map2D(inchannels, inwidth, inheight, batch), xval);
                                            OverflowCheckedTensor w_tensor = new OverflowCheckedTensor(Shape.Kernel2D(inchannels / 3 * 4, outchannels / 3, kwidth, kheight), wval);

                                            OverflowCheckedTensor y_tensor = new OverflowCheckedTensor(Shape.Map2D(outchannels, outwidth, outheight, batch));

                                            TrivectorConvolution2D ope = new TrivectorConvolution2D(inwidth, inheight, inchannels, outchannels, kwidth, kheight, stride, gradmode: false, batch);

                                            ope.Execute(x_tensor, w_tensor, y_tensor);

                                            float[] y_expect = y.ToArray();
                                            float[] y_actual = y_tensor.State;

                                            CollectionAssert.AreEqual(xval, x_tensor.State);
                                            CollectionAssert.AreEqual(wval, w_tensor.State);

                                            AssertError.Tolerance(y_expect, y_actual, 1e-7f, 1e-5f, ref max_err, $"mismatch value {inchannels},{outchannels},{kwidth},{kheight},{stride},{inwidth},{inheight},{batch}");

                                            Console.WriteLine($"pass: {inchannels},{outchannels},{kwidth},{kheight},{stride},{inwidth},{inheight},{batch}");
                                        }
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
        public void OverflowTest() {
            foreach (bool gradmode in new bool[] { false, true }) {
                foreach (int batch in new int[] { 1, 2, 3 }) {
                    foreach (int inchannels in new int[] { 3, 6, 9, 12 }) {
                        foreach (int outchannels in new int[] { 3, 6, 9, 12 }) {
                            foreach (int kheight in new int[] { 1, 3, 5 }) {
                                foreach (int kwidth in new int[] { 1, 3, 5 }) {
                                    foreach (int stride in new int[] { 1, 2, 3 }) {
                                        foreach (int inwidth in new int[] { 8, 9, 13, 17 }) {
                                            foreach (int inheight in new int[] { 8, 9, 19, 23 }) {
                                                int outwidth = inwidth - kwidth + 1, outheight = inheight - kheight + 1;

                                                float[] xval = (new float[inwidth * inheight * inchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
                                                float[] wval = (new float[kwidth * kheight * inchannels * outchannels / 9 * 4]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

                                                OverflowCheckedTensor x_tensor = new OverflowCheckedTensor(Shape.Map2D(inchannels, inwidth, inheight, batch), xval);
                                                OverflowCheckedTensor w_tensor = new OverflowCheckedTensor(Shape.Kernel2D(inchannels / 3 * 4, outchannels / 3, kwidth, kheight), wval);

                                                OverflowCheckedTensor y_tensor = new OverflowCheckedTensor(Shape.Map2D(outchannels, outwidth, outheight, batch));

                                                TrivectorConvolution2D ope = new TrivectorConvolution2D(inwidth, inheight, inchannels, outchannels, kwidth, kheight, stride, gradmode, batch);

                                                ope.Execute(x_tensor, w_tensor, y_tensor);

                                                CollectionAssert.AreEqual(xval, x_tensor.State);
                                                CollectionAssert.AreEqual(wval, w_tensor.State);

                                                y_tensor.CheckOverflow();

                                                Console.WriteLine($"pass: {inchannels},{outchannels},{kwidth},{kheight},{stride},{inwidth},{inheight},{batch},{gradmode}");
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        [TestMethod]
        public void SpeedTest() {
            int inwidth = 32, inheight = 32, inchannels = 33, outchannels = 33, ksize = 3, stride = 2;
            int outwidth = (inwidth - ksize) / stride + 1, outheight = (inheight - ksize) / stride + 1;

            OverflowCheckedTensor x_tensor = new OverflowCheckedTensor(Shape.Map2D(inchannels, inwidth, inheight));
            OverflowCheckedTensor w_tensor = new OverflowCheckedTensor(Shape.Kernel2D(inchannels / 3 * 4, outchannels / 3, ksize, ksize));

            OverflowCheckedTensor y_tensor = new OverflowCheckedTensor(Shape.Map2D(outchannels, outwidth, outheight));

            TrivectorConvolution2D ope = new TrivectorConvolution2D(inwidth, inheight, inchannels, outchannels, ksize, ksize, stride);

            Stopwatch sw = new Stopwatch();
            sw.Start();

            ope.Execute(x_tensor, w_tensor, y_tensor);
            ope.Execute(x_tensor, w_tensor, y_tensor);
            ope.Execute(x_tensor, w_tensor, y_tensor);
            ope.Execute(x_tensor, w_tensor, y_tensor);

            sw.Stop();

            Console.WriteLine($"{sw.ElapsedMilliseconds / 4} msec");
        }

        public static TrivectorMap2D Reference(TrivectorMap2D x, Quaternion.QuaternionFilter2D w, int kwidth, int kheight, int stride) {
            int inchannels = x.Channels, outchannels = w.OutChannels, batch = x.Batch;
            int inw = x.Width, inh = x.Height;
            int outw = inw - kwidth + 1, outh = inh - kheight + 1;

            TrivectorMap2D y = new TrivectorMap2D(outchannels, outw, outh, batch);

            for (int kx, ky = 0; ky < kheight; ky++) {
                for (kx = 0; kx < kwidth; kx++) {
                    for (int th = 0; th < batch; th++) {
                        for (int ox, oy = 0; oy < outh; oy++) {
                            for (ox = 0; ox < outw; ox++) {
                                for (int outch = 0; outch < outchannels; outch++) {
                                    Trivector sum = y[outch, ox, oy, th];

                                    for (int inch = 0; inch < inchannels; inch++) {
                                        sum += x[inch, kx + ox * stride, ky + oy * stride, th] * w[inch, outch, kx, ky];
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
            int inchannels = 9, outchannels = 12, kwidth = 3, kheight = 5, stride = 2, inwidth = 7, inheight = 8, batch = 3;
            int outwidth = inwidth - kwidth + 1, outheight = inheight - kheight + 1;

            float[] xval = (new float[batch * inwidth * inheight * inchannels]).Select((_, idx) => idx * 1e-3f).ToArray();
            float[] wval = (new float[kwidth * kheight * outchannels * inchannels / 9 * 4]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

            Trivector[] xcval = (new Trivector[xval.Length / 3])
                .Select((_, idx) => new Trivector(xval[idx * 3], xval[idx * 3 + 1], xval[idx * 3 + 2])).ToArray();

            Quaternion.Quaternion[] wcval = (new Quaternion.Quaternion[wval.Length / 4])
                .Select((_, idx) => new Quaternion.Quaternion(wval[idx * 4], wval[idx * 4 + 1], wval[idx * 4 + 2], wval[idx * 4 + 3])).ToArray();

            TrivectorMap2D x = new TrivectorMap2D(inchannels / 3, inwidth, inheight, batch, xcval);
            Quaternion.QuaternionFilter2D w = new Quaternion.QuaternionFilter2D(inchannels / 3, outchannels / 3, kwidth, kheight, wcval);

            TrivectorMap2D y = Reference(x, w, kwidth, kheight, stride);

            float[] y_expect = {
                2.181353820e+00f,  2.096150130e+00f,  2.129823480e+00f,  2.035735260e+00f,  1.954642050e+00f,  1.986704040e+00f,
                1.897374300e+00f,  1.820287890e+00f,  1.850790360e+00f,  1.766270940e+00f,  1.693087650e+00f,  1.722082440e+00f,
                2.784901560e+00f,  2.696023710e+00f,  2.729706780e+00f,  2.610317400e+00f,  2.525666670e+00f,  2.557738380e+00f,
                2.443923960e+00f,  2.363396670e+00f,  2.393908860e+00f,  2.285721240e+00f,  2.209213710e+00f,  2.238218220e+00f,
                3.388449300e+00f,  3.295897290e+00f,  3.329590080e+00f,  3.184899540e+00f,  3.096691290e+00f,  3.128772720e+00f,
                2.990473620e+00f,  2.906505450e+00f,  2.937027360e+00f,  2.805171540e+00f,  2.725339770e+00f,  2.754354000e+00f,
                6.406188000e+00f,  6.295265190e+00f,  6.329006580e+00f,  6.057810240e+00f,  5.951814390e+00f,  5.983944420e+00f,
                5.723221920e+00f,  5.622049350e+00f,  5.652619860e+00f,  5.402423040e+00f,  5.305970070e+00f,  5.335032900e+00f,
                7.009735740e+00f,  6.895138770e+00f,  6.928889880e+00f,  6.632392380e+00f,  6.522839010e+00f,  6.554978760e+00f,
                6.269771580e+00f,  6.165158130e+00f,  6.195738360e+00f,  5.921873340e+00f,  5.822096130e+00f,  5.851168680e+00f,
                7.613283480e+00f,  7.495012350e+00f,  7.528773180e+00f,  7.206974520e+00f,  7.093863630e+00f,  7.126013100e+00f,
                6.816321240e+00f,  6.708266910e+00f,  6.738856860e+00f,  6.441323640e+00f,  6.338222190e+00f,  6.367304460e+00f,
                1.908069054e+01f,  1.889261037e+01f,  1.892655588e+01f,  1.812403518e+01f,  1.794333141e+01f,  1.797566556e+01f,
                1.720076478e+01f,  1.702733373e+01f,  1.705810836e+01f,  1.631087934e+01f,  1.614461733e+01f,  1.617388428e+01f,
                1.968423828e+01f,  1.949248395e+01f,  1.952643918e+01f,  1.869861732e+01f,  1.851435603e+01f,  1.854669990e+01f,
                1.774731444e+01f,  1.757044251e+01f,  1.760122686e+01f,  1.683032964e+01f,  1.666074339e+01f,  1.669002006e+01f,
                2.028778602e+01f,  2.009235753e+01f,  2.012632248e+01f,  1.927319946e+01f,  1.908538065e+01f,  1.911773424e+01f,
                1.829386410e+01f,  1.811355129e+01f,  1.814434536e+01f,  1.734977994e+01f,  1.717686945e+01f,  1.720615584e+01f,
                2.330552472e+01f,  2.309172543e+01f,  2.312573898e+01f,  2.214611016e+01f,  2.194050375e+01f,  2.197290594e+01f,
                2.102661240e+01f,  2.082909519e+01f,  2.085993786e+01f,  1.994703144e+01f,  1.975749975e+01f,  1.978683474e+01f,
                2.390907246e+01f,  2.369159901e+01f,  2.372562228e+01f,  2.272069230e+01f,  2.251152837e+01f,  2.254394028e+01f,
                2.157316206e+01f,  2.137220397e+01f,  2.140305636e+01f,  2.046648174e+01f,  2.027362581e+01f,  2.030297052e+01f,
                2.451262020e+01f,  2.429147259e+01f,  2.432550558e+01f,  2.329527444e+01f,  2.308255299e+01f,  2.311497462e+01f,
                2.211971172e+01f,  2.191531275e+01f,  2.194617486e+01f,  2.098593204e+01f,  2.078975187e+01f,  2.081910630e+01f,
                3.598002726e+01f,  3.568907061e+01f,  3.572328828e+01f,  3.421233510e+01f,  3.393202077e+01f,  3.396462708e+01f,
                3.250415526e+01f,  3.223437957e+01f,  3.226542636e+01f,  3.085548774e+01f,  3.059614701e+01f,  3.062568612e+01f,
                3.658357500e+01f,  3.628894419e+01f,  3.632317158e+01f,  3.478691724e+01f,  3.450304539e+01f,  3.453566142e+01f,
                3.305070492e+01f,  3.277748835e+01f,  3.280854486e+01f,  3.137493804e+01f,  3.111227307e+01f,  3.114182190e+01f,
                3.718712274e+01f,  3.688881777e+01f,  3.692305488e+01f,  3.536149938e+01f,  3.507407001e+01f,  3.510669576e+01f,
                3.359725458e+01f,  3.332059713e+01f,  3.335166336e+01f,  3.189438834e+01f,  3.162839913e+01f,  3.165795768e+01f,
                4.020486144e+01f,  3.988818567e+01f,  3.992247138e+01f,  3.823441008e+01f,  3.792919311e+01f,  3.796186746e+01f,
                3.633000288e+01f,  3.603614103e+01f,  3.606725586e+01f,  3.449163984e+01f,  3.420902943e+01f,  3.423863658e+01f,
                4.080840918e+01f,  4.048805925e+01f,  4.052235468e+01f,  3.880899222e+01f,  3.850021773e+01f,  3.853290180e+01f,
                3.687655254e+01f,  3.657924981e+01f,  3.661037436e+01f,  3.501109014e+01f,  3.472515549e+01f,  3.475477236e+01f,
                4.141195692e+01f,  4.108793283e+01f,  4.112223798e+01f,  3.938357436e+01f,  3.907124235e+01f,  3.910393614e+01f,
                3.742310220e+01f,  3.712235859e+01f,  3.715349286e+01f,  3.553054044e+01f,  3.524128155e+01f,  3.527090814e+01f,
            };

            float[] y_actual = y.ToArray();

            AssertError.Tolerance(y_expect, y_actual, 1e-7f, 1e-5f, $"mismatch value {inchannels},{outchannels},{kwidth},{kheight},{stride},{inwidth},{inheight},{batch}");
        }
    }
}
