using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;
using TensorShader;
using TensorShader.Operators.TrivectorConvolution;
using TensorShaderCudaBackend.API;

namespace TensorShaderTest.Operators.Trivector {
    [TestClass]
    public class TrivectorConvolution3DTest {
        [TestMethod]
        public void ExecuteFPTest() {
            TensorShaderCudaBackend.Environment.Precision = TensorShaderCudaBackend.Environment.PrecisionMode.Float;
            TensorShaderCudaBackend.Environment.CudnnEnabled = false;

            Random random = new(1234);

            float max_err = 0;

            foreach (int batch in new int[] { 1, 2 }) {
                foreach ((int inchannels, int outchannels) in new (int, int)[] { (3, 3), (3, 6), (6, 3), (6, 15), (15, 24), (24, 6), (24, 27), (27, 27) }) {
                    foreach ((int kwidth, int kheight, int kdepth) in new (int, int, int)[] { (1, 1, 1), (3, 3, 3), (5, 5, 5), (1, 3, 5), (3, 5, 1), (5, 1, 3) }) {
                        foreach ((int inwidth, int inheight, int indepth) in new (int, int, int)[] { (kwidth, kheight, kdepth), (kwidth * 2, kheight * 2, kdepth * 2), (13, 13, 13), (17, 17, 17), (19, 19, 19), (17, 19, 13), (13, 17, 19), (19, 13, 17) }) {

                            int outwidth = inwidth - kwidth + 1, outheight = inheight - kheight + 1, outdepth = indepth - kdepth + 1;

                            float[] xval = (new float[inwidth * inheight * indepth * inchannels * batch]).Select((_, idx) => (float)random.NextDouble() * 1e-2f).ToArray();
                            float[] wval = (new float[kwidth * kheight * kdepth * inchannels * outchannels / 9 * 4]).Select((_, idx) => (float)random.NextDouble() * 1e-2f).ToArray();

                            Trivector[] xcval = (new Trivector[xval.Length / 3])
                                .Select((_, idx) => new Trivector(xval[idx * 3], xval[idx * 3 + 1], xval[idx * 3 + 2])).ToArray();

                            Quaternion.Quaternion[] wcval = (new Quaternion.Quaternion[wval.Length / 4])
                                .Select((_, idx) => new Quaternion.Quaternion(wval[idx * 4], wval[idx * 4 + 1], wval[idx * 4 + 2], wval[idx * 4 + 3])).ToArray();

                            TrivectorMap3D x = new(inchannels / 3, inwidth, inheight, indepth, batch, xcval);
                            Quaternion.QuaternionFilter3D w = new(inchannels / 3, outchannels / 3, kwidth, kheight, kdepth, wcval);

                            TrivectorMap3D y = Reference(x, w, kwidth, kheight, kdepth);

                            OverflowCheckedTensor x_tensor = new(Shape.Map3D(inchannels, inwidth, inheight, indepth, batch), xval);
                            OverflowCheckedTensor w_tensor = new(Shape.Kernel3D(inchannels / 3 * 4, outchannels / 3, kwidth, kheight, kdepth), wval);

                            OverflowCheckedTensor y_tensor = new(Shape.Map3D(outchannels, outwidth, outheight, outdepth, batch));

                            TrivectorConvolution3D ope = new(inwidth, inheight, indepth, inchannels, outchannels, kwidth, kheight, kdepth, gradmode: false, batch);

                            ope.Execute(x_tensor, w_tensor, y_tensor);

                            float[] y_expect = y.ToArray();
                            float[] y_actual = y_tensor.State.Value;

                            CollectionAssert.AreEqual(xval, x_tensor.State.Value);
                            CollectionAssert.AreEqual(wval, w_tensor.State.Value);

                            AssertError.Tolerance(y_expect, y_actual, 1e-6f, 1e-4f, ref max_err, $"mismatch value {inchannels},{outchannels},{kwidth},{kheight},{kdepth},{inwidth},{inheight},{indepth},{batch}");

                            Console.WriteLine($"pass: {inchannels},{outchannels},{kwidth},{kheight},{kdepth},{inwidth},{inheight},{indepth},{batch}");
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
                foreach ((int inchannels, int outchannels) in new (int, int)[] { (3, 3), (3, 6), (6, 3), (6, 15), (15, 24), (24, 6), (24, 27), (27, 27) }) {
                    foreach ((int kwidth, int kheight, int kdepth) in new (int, int, int)[] { (1, 1, 1), (3, 3, 3), (5, 5, 5), (1, 3, 5), (3, 5, 1), (5, 1, 3) }) {
                        foreach ((int inwidth, int inheight, int indepth) in new (int, int, int)[] { (kwidth, kheight, kdepth), (kwidth * 2, kheight * 2, kdepth * 2), (13, 13, 13), (17, 17, 17), (19, 19, 19), (17, 19, 13), (13, 17, 19), (19, 13, 17) }) {

                            int outwidth = inwidth - kwidth + 1, outheight = inheight - kheight + 1, outdepth = indepth - kdepth + 1;

                            float[] xval = (new float[inwidth * inheight * indepth * inchannels * batch]).Select((_, idx) => (float)random.NextDouble() * 1e-2f).ToArray();
                            float[] wval = (new float[kwidth * kheight * kdepth * inchannels * outchannels / 9 * 4]).Select((_, idx) => (float)random.NextDouble() * 1e-2f).ToArray();

                            Trivector[] xcval = (new Trivector[xval.Length / 3])
                                .Select((_, idx) => new Trivector(xval[idx * 3], xval[idx * 3 + 1], xval[idx * 3 + 2])).ToArray();

                            Quaternion.Quaternion[] wcval = (new Quaternion.Quaternion[wval.Length / 4])
                                .Select((_, idx) => new Quaternion.Quaternion(wval[idx * 4], wval[idx * 4 + 1], wval[idx * 4 + 2], wval[idx * 4 + 3])).ToArray();

                            TrivectorMap3D x = new(inchannels / 3, inwidth, inheight, indepth, batch, xcval);
                            Quaternion.QuaternionFilter3D w = new(inchannels / 3, outchannels / 3, kwidth, kheight, kdepth, wcval);

                            TrivectorMap3D y = Reference(x, w, kwidth, kheight, kdepth);

                            OverflowCheckedTensor x_tensor = new(Shape.Map3D(inchannels, inwidth, inheight, indepth, batch), xval);
                            OverflowCheckedTensor w_tensor = new(Shape.Kernel3D(inchannels / 3 * 4, outchannels / 3, kwidth, kheight, kdepth), wval);

                            OverflowCheckedTensor y_tensor = new(Shape.Map3D(outchannels, outwidth, outheight, outdepth, batch));

                            TrivectorConvolution3D ope = new(inwidth, inheight, indepth, inchannels, outchannels, kwidth, kheight, kdepth, gradmode: false, batch);

                            ope.Execute(x_tensor, w_tensor, y_tensor);

                            float[] y_expect = y.ToArray();
                            float[] y_actual = y_tensor.State.Value;

                            CollectionAssert.AreEqual(xval, x_tensor.State.Value);
                            CollectionAssert.AreEqual(wval, w_tensor.State.Value);

                            AssertError.Tolerance(y_expect, y_actual, 1e-7f, 1e-5f, ref max_err, $"mismatch value {inchannels},{outchannels},{kwidth},{kheight},{kdepth},{inwidth},{inheight},{indepth},{batch}");

                            Console.WriteLine($"pass: {inchannels},{outchannels},{kwidth},{kheight},{kdepth},{inwidth},{inheight},{indepth},{batch}");
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
            int kwidth = 7, kheight = 5, kdepth = 3;
            int inwidth = 125, inheight = 196, indepth = 4;
            int outwidth = inwidth - kwidth + 1, outheight = inheight - kheight + 1, outdepth = indepth - kdepth + 1;

            float[] xval = (new float[inwidth * inheight * indepth * inchannels * batch]).Select((_, idx) => (float)random.NextDouble() * 1e-2f).ToArray();
            float[] wval = (new float[kwidth * kheight * kdepth * inchannels * outchannels / 9 * 4]).Select((_, idx) => (float)random.NextDouble() * 1e-2f).ToArray();

            Trivector[] xcval = (new Trivector[xval.Length / 3])
                .Select((_, idx) => new Trivector(xval[idx * 3], xval[idx * 3 + 1], xval[idx * 3 + 2])).ToArray();

            Quaternion.Quaternion[] wcval = (new Quaternion.Quaternion[wval.Length / 4])
                .Select((_, idx) => new Quaternion.Quaternion(wval[idx * 4], wval[idx * 4 + 1], wval[idx * 4 + 2], wval[idx * 4 + 3])).ToArray();

            TrivectorMap3D x = new(inchannels / 3, inwidth, inheight, indepth, batch, xcval);
            Quaternion.QuaternionFilter3D w = new(inchannels / 3, outchannels / 3, kwidth, kheight, kdepth, wcval);

            TrivectorMap3D y = Reference(x, w, kwidth, kheight, kdepth);

            OverflowCheckedTensor x_tensor = new(Shape.Map3D(inchannels, inwidth, inheight, indepth, batch), xval);
            OverflowCheckedTensor w_tensor = new(Shape.Kernel3D(inchannels / 3 * 4, outchannels / 3, kwidth, kheight, kdepth), wval);

            OverflowCheckedTensor y_tensor = new(Shape.Map3D(outchannels, outwidth, outheight, outdepth, batch));

            TrivectorConvolution3D ope = new(inwidth, inheight, indepth, inchannels, outchannels, kwidth, kheight, kdepth, gradmode: false, batch);

            ope.Execute(x_tensor, w_tensor, y_tensor);

            float[] y_expect = y.ToArray();
            float[] y_actual = y_tensor.State.Value;

            CollectionAssert.AreEqual(xval, x_tensor.State.Value);
            CollectionAssert.AreEqual(wval, w_tensor.State.Value);

            AssertError.Tolerance(y_expect, y_actual, 1e-7f, 1e-5f, ref max_err, $"mismatch value {inchannels},{outchannels},{kwidth},{kheight},{kdepth},{inwidth},{inheight},{indepth},{batch}");

            Console.WriteLine($"pass: {inchannels},{outchannels},{kwidth},{kheight},{kdepth},{inwidth},{inheight},{indepth},{batch}");

            Console.WriteLine($"maxerr:{max_err}");
        }

        [TestMethod]
        public void SpeedFPTest() {
            TensorShaderCudaBackend.Environment.Precision = TensorShaderCudaBackend.Environment.PrecisionMode.Float;
            TensorShaderCudaBackend.Environment.CudnnEnabled = false;

            int inwidth = 32, inheight = 32, indepth = 32, inchannels = 33, outchannels = 33, ksize = 3;
            int outwidth = inwidth - ksize + 1, outheight = inheight - ksize + 1, outdepth = indepth - ksize + 1;

            OverflowCheckedTensor x_tensor = new(Shape.Map3D(inchannels, inwidth, inheight, indepth));
            OverflowCheckedTensor w_tensor = new(Shape.Kernel3D(inchannels / 3 * 4, outchannels / 3, ksize, ksize, ksize));

            OverflowCheckedTensor y_tensor = new(Shape.Map3D(outchannels, outwidth, outheight, outdepth));

            TrivectorConvolution3D ope = new(inwidth, inheight, indepth, inchannels, outchannels, ksize, ksize, ksize);

            Cuda.Profiler.Initialize("../../../../profiler.nvsetting", "../../nvprofiles/trivector_convolution_3d_fp.nvvp");
            Cuda.Profiler.Start();

            ope.Execute(x_tensor, w_tensor, y_tensor);

            Cuda.Profiler.Stop();
        }

        [TestMethod]
        public void SpeedFFPTest() {
            TensorShaderCudaBackend.Environment.CudnnEnabled = false;
            TensorShaderCudaBackend.Environment.Precision = TensorShaderCudaBackend.Environment.PrecisionMode.FloatFloat;

            int inwidth = 32, inheight = 32, indepth = 32, inchannels = 33, outchannels = 33, ksize = 3;
            int outwidth = inwidth - ksize + 1, outheight = inheight - ksize + 1, outdepth = indepth - ksize + 1;

            OverflowCheckedTensor x_tensor = new(Shape.Map3D(inchannels, inwidth, inheight, indepth));
            OverflowCheckedTensor w_tensor = new(Shape.Kernel3D(inchannels / 3 * 4, outchannels / 3, ksize, ksize, ksize));

            OverflowCheckedTensor y_tensor = new(Shape.Map3D(outchannels, outwidth, outheight, outdepth));

            TrivectorConvolution3D ope = new(inwidth, inheight, indepth, inchannels, outchannels, ksize, ksize, ksize);

            Cuda.Profiler.Initialize("../../../../profiler.nvsetting", "../../nvprofiles/trivector_convolution_3d_ffp.nvvp");
            Cuda.Profiler.Start();

            ope.Execute(x_tensor, w_tensor, y_tensor);

            Cuda.Profiler.Stop();
        }

        public static TrivectorMap3D Reference(TrivectorMap3D x, Quaternion.QuaternionFilter3D w, int kwidth, int kheight, int kdepth) {
            int inchannels = x.Channels, outchannels = w.OutChannels, batch = x.Batch;
            int inw = x.Width, inh = x.Height, ind = x.Depth;
            int outw = inw - kwidth + 1, outh = inh - kheight + 1, outd = ind - kdepth + 1;

            TrivectorMap3D y = new(outchannels, outw, outh, outd, batch);

            for (int kx, ky, kz = 0; kz < kdepth; kz++) {
                for (ky = 0; ky < kheight; ky++) {
                    for (kx = 0; kx < kwidth; kx++) {
                        for (int th = 0; th < batch; th++) {
                            for (int ox, oy, oz = 0; oz < outd; oz++) {
                                for (oy = 0; oy < outh; oy++) {
                                    for (ox = 0; ox < outw; ox++) {
                                        for (int outch = 0; outch < outchannels; outch++) {
                                            Trivector sum = y[outch, ox, oy, oz, th];

                                            for (int inch = 0; inch < inchannels; inch++) {
                                                sum += x[inch, kx + ox, ky + oy, kz + oz, th] * w[inch, outch, kx, ky, kz];
                                            }

                                            y[outch, ox, oy, oz, th] = sum;
                                        }
                                    }
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
            int inchannels = 9, outchannels = 12, kwidth = 3, kheight = 5, kdepth = 7, inwidth = 7, inheight = 8, indepth = 9, batch = 3;
            int outwidth = inwidth - kwidth + 1, outheight = inheight - kheight + 1;

            float[] xval = (new float[batch * inwidth * inheight * indepth * inchannels]).Select((_, idx) => idx * 1e-3f).ToArray();
            float[] wval = (new float[kwidth * kheight * kdepth * outchannels * inchannels / 9 * 4]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

            Trivector[] xcval = (new Trivector[xval.Length / 3])
                .Select((_, idx) => new Trivector(xval[idx * 3], xval[idx * 3 + 1], xval[idx * 3 + 2])).ToArray();

            Quaternion.Quaternion[] wcval = (new Quaternion.Quaternion[wval.Length / 4])
                .Select((_, idx) => new Quaternion.Quaternion(wval[idx * 4], wval[idx * 4 + 1], wval[idx * 4 + 2], wval[idx * 4 + 3])).ToArray();

            TrivectorMap3D x = new(inchannels / 3, inwidth, inheight, indepth, batch, xcval);
            Quaternion.QuaternionFilter3D w = new(inchannels / 3, outchannels / 3, kwidth, kheight, kdepth, wcval);

            TrivectorMap3D y = Reference(x, w, kwidth, kheight, kdepth);

            float[] y_expect = {
                8.415684149e+03f,  8.383841578e+03f,  8.394637372e+03f,  8.333621015e+03f,  8.302006438e+03f,  8.312725634e+03f,
                8.252157358e+03f,  8.220770050e+03f,  8.231413011e+03f,  8.171293179e+03f,  8.140132414e+03f,  8.150699502e+03f,
                8.512761355e+03f,  8.480832441e+03f,  8.491628269e+03f,  8.430008975e+03f,  8.398308464e+03f,  8.409027694e+03f,
                8.347859339e+03f,  8.316386505e+03f,  8.327029500e+03f,  8.266312447e+03f,  8.235066564e+03f,  8.245633686e+03f,
                8.609838561e+03f,  8.577823305e+03f,  8.588619166e+03f,  8.526396936e+03f,  8.494610490e+03f,  8.505329754e+03f,
                8.443561321e+03f,  8.412002960e+03f,  8.422645989e+03f,  8.361331715e+03f,  8.330000714e+03f,  8.340567870e+03f,
                8.706915767e+03f,  8.674814168e+03f,  8.685610063e+03f,  8.622784897e+03f,  8.590912517e+03f,  8.601631814e+03f,
                8.539263302e+03f,  8.507619415e+03f,  8.518262478e+03f,  8.456350983e+03f,  8.424934864e+03f,  8.435502054e+03f,
                8.803992973e+03f,  8.771805031e+03f,  8.782600961e+03f,  8.719172857e+03f,  8.687214543e+03f,  8.697933875e+03f,
                8.634965283e+03f,  8.603235870e+03f,  8.613878967e+03f,  8.551370251e+03f,  8.519869014e+03f,  8.530436238e+03f,
                9.095224591e+03f,  9.062777620e+03f,  9.073573652e+03f,  9.008336740e+03f,  8.976120621e+03f,  8.986840055e+03f,
                8.922071228e+03f,  8.890085236e+03f,  8.900728435e+03f,  8.836428055e+03f,  8.804671464e+03f,  8.815238790e+03f,
                9.192301797e+03f,  9.159768483e+03f,  9.170564549e+03f,  9.104724700e+03f,  9.072422647e+03f,  9.083142115e+03f,
                9.017773209e+03f,  8.985701691e+03f,  8.996344924e+03f,  8.931447323e+03f,  8.899605614e+03f,  8.910172974e+03f,
                9.289379002e+03f,  9.256759347e+03f,  9.267555446e+03f,  9.201112661e+03f,  9.168724674e+03f,  9.179444176e+03f,
                9.113475190e+03f,  9.081318146e+03f,  9.091961413e+03f,  9.026466591e+03f,  8.994539764e+03f,  9.005107158e+03f,
                9.386456208e+03f,  9.353750210e+03f,  9.364546343e+03f,  9.297500622e+03f,  9.265026700e+03f,  9.275746236e+03f,
                9.209177172e+03f,  9.176934601e+03f,  9.187577902e+03f,  9.121485859e+03f,  9.089473914e+03f,  9.100041342e+03f,
                9.483533414e+03f,  9.450741073e+03f,  9.461537241e+03f,  9.393888582e+03f,  9.361328726e+03f,  9.372048296e+03f,
                9.304879153e+03f,  9.272551056e+03f,  9.283194391e+03f,  9.216505127e+03f,  9.184408064e+03f,  9.194975526e+03f,
                9.774765032e+03f,  9.741713662e+03f,  9.752509932e+03f,  9.683052464e+03f,  9.650234804e+03f,  9.660954476e+03f,
                9.591985098e+03f,  9.559400422e+03f,  9.570043859e+03f,  9.501562931e+03f,  9.469210514e+03f,  9.479778079e+03f,
                9.871842238e+03f,  9.838704525e+03f,  9.849500829e+03f,  9.779440425e+03f,  9.746536831e+03f,  9.757256537e+03f,
                9.687687079e+03f,  9.655016877e+03f,  9.665660348e+03f,  9.596582199e+03f,  9.564144664e+03f,  9.574712263e+03f,
                9.968919444e+03f,  9.935695388e+03f,  9.946491726e+03f,  9.875828386e+03f,  9.842838857e+03f,  9.853558597e+03f,
                9.783389060e+03f,  9.750633332e+03f,  9.761276837e+03f,  9.691601467e+03f,  9.659078814e+03f,  9.669646447e+03f,
                1.006599665e+04f,  1.003268625e+04f,  1.004348262e+04f,  9.972216346e+03f,  9.939140883e+03f,  9.949860657e+03f,
                9.879091042e+03f,  9.846249787e+03f,  9.856893326e+03f,  9.786620735e+03f,  9.754012964e+03f,  9.764580631e+03f,
                1.016307386e+04f,  1.012967711e+04f,  1.014047352e+04f,  1.006860431e+04f,  1.003544291e+04f,  1.004616272e+04f,
                9.974793023e+03f,  9.941866242e+03f,  9.952509815e+03f,  9.881640003e+03f,  9.848947114e+03f,  9.859514815e+03f,
                1.045430547e+04f,  1.042064970e+04f,  1.043144621e+04f,  1.035776819e+04f,  1.032434899e+04f,  1.033506890e+04f,
                1.026189897e+04f,  1.022871561e+04f,  1.023935928e+04f,  1.016669781e+04f,  1.013374956e+04f,  1.014431737e+04f,
                1.055138268e+04f,  1.051764057e+04f,  1.052843711e+04f,  1.045415615e+04f,  1.042065101e+04f,  1.043137096e+04f,
                1.035760095e+04f,  1.032433206e+04f,  1.033497577e+04f,  1.026171708e+04f,  1.022868371e+04f,  1.023925155e+04f,
                1.064845988e+04f,  1.061463143e+04f,  1.062542801e+04f,  1.055054411e+04f,  1.051695304e+04f,  1.052767302e+04f,
                1.045330293e+04f,  1.041994852e+04f,  1.043059226e+04f,  1.035673634e+04f,  1.032361786e+04f,  1.033418573e+04f,
                1.074553709e+04f,  1.071162229e+04f,  1.072241890e+04f,  1.064693207e+04f,  1.061325507e+04f,  1.062397508e+04f,
                1.054900491e+04f,  1.051556497e+04f,  1.052620875e+04f,  1.045175561e+04f,  1.041855201e+04f,  1.042911992e+04f,
                1.084261430e+04f,  1.080861316e+04f,  1.081940980e+04f,  1.074332003e+04f,  1.070955709e+04f,  1.072027714e+04f,
                1.064470689e+04f,  1.061118143e+04f,  1.062182524e+04f,  1.054677488e+04f,  1.051348616e+04f,  1.052405410e+04f,
                1.385200768e+04f,  1.381532991e+04f,  1.382612761e+04f,  1.373134681e+04f,  1.369491990e+04f,  1.370564100e+04f,
                1.361146832e+04f,  1.357529154e+04f,  1.358593640e+04f,  1.349237219e+04f,  1.345644481e+04f,  1.346701381e+04f,
                1.394908489e+04f,  1.391232078e+04f,  1.392311851e+04f,  1.382773477e+04f,  1.379122193e+04f,  1.380194306e+04f,
                1.370717030e+04f,  1.367090799e+04f,  1.368155289e+04f,  1.358739146e+04f,  1.355137896e+04f,  1.356194799e+04f,
                1.404616209e+04f,  1.400931164e+04f,  1.402010941e+04f,  1.392412273e+04f,  1.388752396e+04f,  1.389824512e+04f,
                1.380287228e+04f,  1.376652445e+04f,  1.377716938e+04f,  1.368241073e+04f,  1.364631311e+04f,  1.365688218e+04f,
                1.414323930e+04f,  1.410630250e+04f,  1.411710030e+04f,  1.402051070e+04f,  1.398382598e+04f,  1.399454719e+04f,
                1.389857426e+04f,  1.386214090e+04f,  1.387278587e+04f,  1.377742999e+04f,  1.374124726e+04f,  1.375181636e+04f,
                1.424031650e+04f,  1.420329337e+04f,  1.421409120e+04f,  1.411689866e+04f,  1.408012801e+04f,  1.409084925e+04f,
                1.399427624e+04f,  1.395775736e+04f,  1.396840236e+04f,  1.387244926e+04f,  1.383618141e+04f,  1.384675054e+04f,
                1.453154812e+04f,  1.449426596e+04f,  1.450506389e+04f,  1.440606254e+04f,  1.436903409e+04f,  1.437975543e+04f,
                1.428138219e+04f,  1.424460672e+04f,  1.425525183e+04f,  1.415750707e+04f,  1.412098386e+04f,  1.413155310e+04f,
                1.462862533e+04f,  1.459125682e+04f,  1.460205479e+04f,  1.450245050e+04f,  1.446533611e+04f,  1.447605749e+04f,
                1.437708417e+04f,  1.434022318e+04f,  1.435086832e+04f,  1.425252633e+04f,  1.421591801e+04f,  1.422648728e+04f,
                1.472570253e+04f,  1.468824768e+04f,  1.469904569e+04f,  1.459883846e+04f,  1.456163814e+04f,  1.457235955e+04f,
                1.447278615e+04f,  1.443583963e+04f,  1.444648481e+04f,  1.434754560e+04f,  1.431085216e+04f,  1.432142146e+04f,
                1.482277974e+04f,  1.478523854e+04f,  1.479603658e+04f,  1.469522642e+04f,  1.465794017e+04f,  1.466866161e+04f,
                1.456848813e+04f,  1.453145609e+04f,  1.454210129e+04f,  1.444256487e+04f,  1.440578631e+04f,  1.441635565e+04f,
                1.491985694e+04f,  1.488222941e+04f,  1.489302748e+04f,  1.479161438e+04f,  1.475424219e+04f,  1.476496367e+04f,
                1.466419011e+04f,  1.462707254e+04f,  1.463771778e+04f,  1.453758414e+04f,  1.450072046e+04f,  1.451128983e+04f,
                1.521108856e+04f,  1.517320200e+04f,  1.518400017e+04f,  1.508077826e+04f,  1.504314827e+04f,  1.505386985e+04f,
                1.495129606e+04f,  1.491392191e+04f,  1.492456725e+04f,  1.482264194e+04f,  1.478552291e+04f,  1.479609238e+04f,
                1.530816577e+04f,  1.527019286e+04f,  1.528099107e+04f,  1.517716622e+04f,  1.513945030e+04f,  1.515017191e+04f,
                1.504699804e+04f,  1.500953836e+04f,  1.502018374e+04f,  1.491766121e+04f,  1.488045706e+04f,  1.489102657e+04f,
                1.540524297e+04f,  1.536718372e+04f,  1.537798197e+04f,  1.527355418e+04f,  1.523575232e+04f,  1.524647397e+04f,
                1.514270002e+04f,  1.510515482e+04f,  1.511580023e+04f,  1.501268048e+04f,  1.497539121e+04f,  1.498596075e+04f,
                1.550232018e+04f,  1.546417459e+04f,  1.547497286e+04f,  1.536994215e+04f,  1.533205435e+04f,  1.534277603e+04f,
                1.523840200e+04f,  1.520077127e+04f,  1.521141672e+04f,  1.510769975e+04f,  1.507032536e+04f,  1.508089494e+04f,
                1.559939739e+04f,  1.556116545e+04f,  1.557196376e+04f,  1.546633011e+04f,  1.542835637e+04f,  1.543907809e+04f,
                1.533410398e+04f,  1.529638773e+04f,  1.530703321e+04f,  1.520271901e+04f,  1.516525951e+04f,  1.517582912e+04f,
                1.589062900e+04f,  1.585213804e+04f,  1.586293645e+04f,  1.575549399e+04f,  1.571726245e+04f,  1.572798427e+04f,
                1.562120993e+04f,  1.558323709e+04f,  1.559388268e+04f,  1.548777682e+04f,  1.545006196e+04f,  1.546063167e+04f,
                1.598770621e+04f,  1.594912890e+04f,  1.595992735e+04f,  1.585188195e+04f,  1.581356448e+04f,  1.582428633e+04f,
                1.571691191e+04f,  1.567885355e+04f,  1.568949916e+04f,  1.558279609e+04f,  1.554499611e+04f,  1.555556586e+04f,
                1.608478341e+04f,  1.604611977e+04f,  1.605691825e+04f,  1.594826991e+04f,  1.590986651e+04f,  1.592058839e+04f,
                1.581261389e+04f,  1.577447001e+04f,  1.578511565e+04f,  1.567781535e+04f,  1.563993026e+04f,  1.565050004e+04f,
                1.618186062e+04f,  1.614311063e+04f,  1.615390914e+04f,  1.604465787e+04f,  1.600616853e+04f,  1.601689045e+04f,
                1.590831587e+04f,  1.587008646e+04f,  1.588073214e+04f,  1.577283462e+04f,  1.573486441e+04f,  1.574543422e+04f,
                1.627893783e+04f,  1.624010149e+04f,  1.625090004e+04f,  1.614104583e+04f,  1.610247056e+04f,  1.611319251e+04f,
                1.600401785e+04f,  1.596570292e+04f,  1.597634863e+04f,  1.586785389e+04f,  1.582979856e+04f,  1.584036841e+04f,
                1.928833121e+04f,  1.924681825e+04f,  1.925761785e+04f,  1.912907261e+04f,  1.908783337e+04f,  1.909855638e+04f,
                1.897077928e+04f,  1.892981302e+04f,  1.894045980e+04f,  1.881345120e+04f,  1.877275721e+04f,  1.878332811e+04f,
                1.938540842e+04f,  1.934380911e+04f,  1.935460875e+04f,  1.922546057e+04f,  1.918413540e+04f,  1.919485844e+04f,
                1.906648126e+04f,  1.902542948e+04f,  1.903607628e+04f,  1.890847047e+04f,  1.886769136e+04f,  1.887826230e+04f,
                1.948248562e+04f,  1.944079998e+04f,  1.945159965e+04f,  1.932184853e+04f,  1.928043742e+04f,  1.929116050e+04f,
                1.916218324e+04f,  1.912104593e+04f,  1.913169277e+04f,  1.900348974e+04f,  1.896262551e+04f,  1.897319648e+04f,
                1.957956283e+04f,  1.953779084e+04f,  1.954859054e+04f,  1.941823649e+04f,  1.937673945e+04f,  1.938746256e+04f,
                1.925788522e+04f,  1.921666239e+04f,  1.922730926e+04f,  1.909850900e+04f,  1.905755966e+04f,  1.906813067e+04f,
                1.967664003e+04f,  1.963478170e+04f,  1.964558144e+04f,  1.951462445e+04f,  1.947304147e+04f,  1.948376462e+04f,
                1.935358720e+04f,  1.931227884e+04f,  1.932292575e+04f,  1.919352827e+04f,  1.915249381e+04f,  1.916306485e+04f,
                1.996787165e+04f,  1.992575429e+04f,  1.993655413e+04f,  1.980378834e+04f,  1.976194755e+04f,  1.977267080e+04f,
                1.964069315e+04f,  1.959912821e+04f,  1.960977522e+04f,  1.947858608e+04f,  1.943729626e+04f,  1.944786740e+04f,
                2.006494886e+04f,  2.002274515e+04f,  2.003354503e+04f,  1.990017630e+04f,  1.985824958e+04f,  1.986897286e+04f,
                1.973639513e+04f,  1.969474467e+04f,  1.970539171e+04f,  1.957360534e+04f,  1.953223041e+04f,  1.954280159e+04f,
                2.016202606e+04f,  2.011973602e+04f,  2.013053593e+04f,  1.999656426e+04f,  1.995455160e+04f,  1.996527492e+04f,
                1.983209711e+04f,  1.979036112e+04f,  1.980100820e+04f,  1.966862461e+04f,  1.962716456e+04f,  1.963773577e+04f,
                2.025910327e+04f,  2.021672688e+04f,  2.022752682e+04f,  2.009295222e+04f,  2.005085363e+04f,  2.006157698e+04f,
                1.992779909e+04f,  1.988597758e+04f,  1.989662469e+04f,  1.976364388e+04f,  1.972209871e+04f,  1.973266995e+04f,
                2.035618047e+04f,  2.031371774e+04f,  2.032451772e+04f,  2.018934018e+04f,  2.014715566e+04f,  2.015787904e+04f,
                2.002350107e+04f,  1.998159403e+04f,  1.999224118e+04f,  1.985866315e+04f,  1.981703286e+04f,  1.982760414e+04f,
                2.064741209e+04f,  2.060469033e+04f,  2.061549041e+04f,  2.047850406e+04f,  2.043606174e+04f,  2.044678522e+04f,
                2.031060702e+04f,  2.026844340e+04f,  2.027909064e+04f,  2.014372095e+04f,  2.010183531e+04f,  2.011240669e+04f,
                2.074448930e+04f,  2.070168120e+04f,  2.071248131e+04f,  2.057489202e+04f,  2.053236376e+04f,  2.054308728e+04f,
                2.040630900e+04f,  2.036405985e+04f,  2.037470713e+04f,  2.023874022e+04f,  2.019676946e+04f,  2.020734087e+04f,
                2.084156650e+04f,  2.079867206e+04f,  2.080947221e+04f,  2.067127998e+04f,  2.062866579e+04f,  2.063938934e+04f,
                2.050201098e+04f,  2.045967631e+04f,  2.047032362e+04f,  2.033375949e+04f,  2.029170361e+04f,  2.030227506e+04f,
                2.093864371e+04f,  2.089566292e+04f,  2.090646310e+04f,  2.076766794e+04f,  2.072496781e+04f,  2.073569140e+04f,
                2.059771296e+04f,  2.055529276e+04f,  2.056594011e+04f,  2.042877876e+04f,  2.038663776e+04f,  2.039720924e+04f,
                2.103572092e+04f,  2.099265379e+04f,  2.100345400e+04f,  2.086405590e+04f,  2.082126984e+04f,  2.083199346e+04f,
                2.069341494e+04f,  2.065090922e+04f,  2.066155660e+04f,  2.052379803e+04f,  2.048157191e+04f,  2.049214343e+04f,
                2.132695253e+04f,  2.128362637e+04f,  2.129442669e+04f,  2.115321979e+04f,  2.111017592e+04f,  2.112089964e+04f,
                2.098052089e+04f,  2.093775858e+04f,  2.094840607e+04f,  2.080885583e+04f,  2.076637436e+04f,  2.077694598e+04f,
                2.142402974e+04f,  2.138061724e+04f,  2.139141759e+04f,  2.124960775e+04f,  2.120647794e+04f,  2.121720170e+04f,
                2.107622287e+04f,  2.103337504e+04f,  2.104402256e+04f,  2.090387510e+04f,  2.086130851e+04f,  2.087188016e+04f,
                2.152110694e+04f,  2.147760810e+04f,  2.148840849e+04f,  2.134599571e+04f,  2.130277997e+04f,  2.131350376e+04f,
                2.117192485e+04f,  2.112899149e+04f,  2.113963905e+04f,  2.099889437e+04f,  2.095624266e+04f,  2.096681435e+04f,
                2.161818415e+04f,  2.157459896e+04f,  2.158539938e+04f,  2.144238367e+04f,  2.139908200e+04f,  2.140980582e+04f,
                2.126762683e+04f,  2.122460795e+04f,  2.123525553e+04f,  2.109391363e+04f,  2.105117681e+04f,  2.106174853e+04f,
                2.171526136e+04f,  2.167158983e+04f,  2.168239028e+04f,  2.153877163e+04f,  2.149538402e+04f,  2.150610788e+04f,
                2.136332881e+04f,  2.132022440e+04f,  2.133087202e+04f,  2.118893290e+04f,  2.114611097e+04f,  2.115668271e+04f,
                5.734259592e+04f,  5.726723660e+04f,  5.727804954e+04f,  5.691315320e+04f,  5.683822763e+04f,  5.684896397e+04f,
                5.648595599e+04f,  5.641146344e+04f,  5.642212354e+04f,  5.606100428e+04f,  5.598694402e+04f,  5.599752825e+04f,
                5.743967312e+04f,  5.736422746e+04f,  5.737504043e+04f,  5.700954116e+04f,  5.693452965e+04f,  5.694526603e+04f,
                5.658165797e+04f,  5.650707989e+04f,  5.651774003e+04f,  5.615602354e+04f,  5.608187817e+04f,  5.609246244e+04f,
                5.753675033e+04f,  5.746121832e+04f,  5.747203133e+04f,  5.710592912e+04f,  5.703083168e+04f,  5.704156809e+04f,
                5.667735995e+04f,  5.660269635e+04f,  5.661335652e+04f,  5.625104281e+04f,  5.617681232e+04f,  5.618739662e+04f,
                5.763382754e+04f,  5.755820919e+04f,  5.756902223e+04f,  5.720231708e+04f,  5.712713371e+04f,  5.713787015e+04f,
                5.677306193e+04f,  5.669831280e+04f,  5.670897301e+04f,  5.634606208e+04f,  5.627174647e+04f,  5.628233081e+04f,
                5.773090474e+04f,  5.765520005e+04f,  5.766601312e+04f,  5.729870505e+04f,  5.722343573e+04f,  5.723417221e+04f,
                5.686876391e+04f,  5.679392926e+04f,  5.680458950e+04f,  5.644108135e+04f,  5.636668062e+04f,  5.637726499e+04f,
                5.802213636e+04f,  5.794617264e+04f,  5.795698582e+04f,  5.758786893e+04f,  5.751234181e+04f,  5.752307839e+04f,
                5.715586986e+04f,  5.708077862e+04f,  5.709143897e+04f,  5.672613915e+04f,  5.665148307e+04f,  5.666206754e+04f,
                5.811921357e+04f,  5.804316350e+04f,  5.805397671e+04f,  5.768425689e+04f,  5.760864384e+04f,  5.761938045e+04f,
                5.725157184e+04f,  5.717639508e+04f,  5.718705546e+04f,  5.682115842e+04f,  5.674641722e+04f,  5.675700173e+04f,
                5.821629077e+04f,  5.814015436e+04f,  5.815096761e+04f,  5.778064485e+04f,  5.770494586e+04f,  5.771568251e+04f,
                5.734727382e+04f,  5.727201153e+04f,  5.728267194e+04f,  5.691617769e+04f,  5.684135137e+04f,  5.685193591e+04f,
                5.831336798e+04f,  5.823714523e+04f,  5.824795851e+04f,  5.787703281e+04f,  5.780124789e+04f,  5.781198457e+04f,
                5.744297580e+04f,  5.736762799e+04f,  5.737828843e+04f,  5.701119696e+04f,  5.693628552e+04f,  5.694687009e+04f,
                5.841044518e+04f,  5.833413609e+04f,  5.834494940e+04f,  5.797342077e+04f,  5.789754992e+04f,  5.790828663e+04f,
                5.753867778e+04f,  5.746324444e+04f,  5.747390492e+04f,  5.710621622e+04f,  5.703121967e+04f,  5.704180428e+04f,
                5.870167680e+04f,  5.862510868e+04f,  5.863592210e+04f,  5.826258465e+04f,  5.818645599e+04f,  5.819719281e+04f,
                5.782578373e+04f,  5.775009381e+04f,  5.776075439e+04f,  5.739127403e+04f,  5.731602212e+04f,  5.732660683e+04f,
                5.879875401e+04f,  5.872209954e+04f,  5.873291299e+04f,  5.835897261e+04f,  5.828275802e+04f,  5.829349487e+04f,
                5.792148571e+04f,  5.784571026e+04f,  5.785637088e+04f,  5.748629330e+04f,  5.741095627e+04f,  5.742154101e+04f,
                5.889583121e+04f,  5.881909041e+04f,  5.882990389e+04f,  5.845536057e+04f,  5.837906005e+04f,  5.838979693e+04f,
                5.801718769e+04f,  5.794132672e+04f,  5.795198737e+04f,  5.758131256e+04f,  5.750589042e+04f,  5.751647520e+04f,
                5.899290842e+04f,  5.891608127e+04f,  5.892689479e+04f,  5.855174853e+04f,  5.847536207e+04f,  5.848609899e+04f,
                5.811288967e+04f,  5.803694317e+04f,  5.804760386e+04f,  5.767633183e+04f,  5.760082457e+04f,  5.761140938e+04f,
                5.908998562e+04f,  5.901307213e+04f,  5.902388568e+04f,  5.864813649e+04f,  5.857166410e+04f,  5.858240105e+04f,
                5.820859165e+04f,  5.813255963e+04f,  5.814322035e+04f,  5.777135110e+04f,  5.769575872e+04f,  5.770634357e+04f,
                5.938121724e+04f,  5.930404472e+04f,  5.931485838e+04f,  5.893730038e+04f,  5.886057018e+04f,  5.887130723e+04f,
                5.849569760e+04f,  5.841940899e+04f,  5.843006981e+04f,  5.805640890e+04f,  5.798056117e+04f,  5.799114612e+04f,
                5.947829445e+04f,  5.940103558e+04f,  5.941184927e+04f,  5.903368834e+04f,  5.895687220e+04f,  5.896760929e+04f,
                5.859139958e+04f,  5.851502545e+04f,  5.852568630e+04f,  5.815142817e+04f,  5.807549532e+04f,  5.808608030e+04f,
                5.957537165e+04f,  5.949802645e+04f,  5.950884017e+04f,  5.913007630e+04f,  5.905317423e+04f,  5.906391135e+04f,
                5.868710156e+04f,  5.861064190e+04f,  5.862130279e+04f,  5.824644744e+04f,  5.817042947e+04f,  5.818101449e+04f,
                5.967244886e+04f,  5.959501731e+04f,  5.960583107e+04f,  5.922646426e+04f,  5.914947626e+04f,  5.916021341e+04f,
                5.878280354e+04f,  5.870625836e+04f,  5.871691928e+04f,  5.834146671e+04f,  5.826536362e+04f,  5.827594867e+04f,
                5.976952607e+04f,  5.969200817e+04f,  5.970282196e+04f,  5.932285222e+04f,  5.924577828e+04f,  5.925651547e+04f,
                5.887850552e+04f,  5.880187481e+04f,  5.881253577e+04f,  5.843648598e+04f,  5.836029777e+04f,  5.837088285e+04f,
                6.277891945e+04f,  6.269872493e+04f,  6.270953978e+04f,  6.231087900e+04f,  6.223114109e+04f,  6.224187934e+04f,
                6.184526695e+04f,  6.176598492e+04f,  6.177664693e+04f,  6.138208329e+04f,  6.130325642e+04f,  6.131384256e+04f,
                6.287599665e+04f,  6.279571579e+04f,  6.280653067e+04f,  6.240726696e+04f,  6.232744312e+04f,  6.233818140e+04f,
                6.194096893e+04f,  6.186160138e+04f,  6.187226342e+04f,  6.147710255e+04f,  6.139819057e+04f,  6.140877674e+04f,
                6.297307386e+04f,  6.289270666e+04f,  6.290352157e+04f,  6.250365492e+04f,  6.242374515e+04f,  6.243448346e+04f,
                6.203667091e+04f,  6.195721783e+04f,  6.196787991e+04f,  6.157212182e+04f,  6.149312472e+04f,  6.150371093e+04f,
                6.307015107e+04f,  6.298969752e+04f,  6.300051247e+04f,  6.260004288e+04f,  6.252004717e+04f,  6.253078552e+04f,
                6.213237289e+04f,  6.205283429e+04f,  6.206349640e+04f,  6.166714109e+04f,  6.158805887e+04f,  6.159864511e+04f,
                6.316722827e+04f,  6.308668838e+04f,  6.309750336e+04f,  6.269643084e+04f,  6.261634920e+04f,  6.262708758e+04f,
                6.222807487e+04f,  6.214845074e+04f,  6.215911289e+04f,  6.176216036e+04f,  6.168299302e+04f,  6.169357930e+04f,
                6.345845989e+04f,  6.337766097e+04f,  6.338847606e+04f,  6.298559473e+04f,  6.290525528e+04f,  6.291599376e+04f,
                6.251518082e+04f,  6.243530011e+04f,  6.244596236e+04f,  6.204721816e+04f,  6.196779547e+04f,  6.197838185e+04f,
                6.355553709e+04f,  6.347465184e+04f,  6.348546695e+04f,  6.308198269e+04f,  6.300155730e+04f,  6.301229582e+04f,
                6.261088280e+04f,  6.253091656e+04f,  6.254157885e+04f,  6.214223743e+04f,  6.206272962e+04f,  6.207331603e+04f,
                6.365261430e+04f,  6.357164270e+04f,  6.358245785e+04f,  6.317837065e+04f,  6.309785933e+04f,  6.310859788e+04f,
                6.270658478e+04f,  6.262653302e+04f,  6.263719534e+04f,  6.223725670e+04f,  6.215766377e+04f,  6.216825022e+04f,
                6.374969151e+04f,  6.366863356e+04f,  6.367944875e+04f,  6.327475861e+04f,  6.319416136e+04f,  6.320489994e+04f,
                6.280228676e+04f,  6.272214947e+04f,  6.273281183e+04f,  6.233227597e+04f,  6.225259792e+04f,  6.226318440e+04f,
                6.384676871e+04f,  6.376562443e+04f,  6.377643964e+04f,  6.337114657e+04f,  6.329046338e+04f,  6.330120200e+04f,
                6.289798874e+04f,  6.281776593e+04f,  6.282842832e+04f,  6.242729524e+04f,  6.234753207e+04f,  6.235811858e+04f,
                6.413800033e+04f,  6.405659701e+04f,  6.406741234e+04f,  6.366031045e+04f,  6.357936946e+04f,  6.359010818e+04f,
                6.318509469e+04f,  6.310461529e+04f,  6.311527778e+04f,  6.271235304e+04f,  6.263233452e+04f,  6.264292114e+04f,
                6.423507754e+04f,  6.415358788e+04f,  6.416440323e+04f,  6.375669841e+04f,  6.367567149e+04f,  6.368641024e+04f,
                6.328079667e+04f,  6.320023175e+04f,  6.321089427e+04f,  6.280737231e+04f,  6.272726867e+04f,  6.273785532e+04f,
                6.433215474e+04f,  6.425057874e+04f,  6.426139413e+04f,  6.385308637e+04f,  6.377197351e+04f,  6.378271230e+04f,
                6.337649865e+04f,  6.329584820e+04f,  6.330651076e+04f,  6.290239158e+04f,  6.282220282e+04f,  6.283278950e+04f,
                6.442923195e+04f,  6.434756960e+04f,  6.435838503e+04f,  6.394947433e+04f,  6.386827554e+04f,  6.387901436e+04f,
                6.347220063e+04f,  6.339146466e+04f,  6.340212725e+04f,  6.299741084e+04f,  6.291713697e+04f,  6.292772369e+04f,
                6.452630915e+04f,  6.444456047e+04f,  6.445537592e+04f,  6.404586229e+04f,  6.396457756e+04f,  6.397531642e+04f,
                6.356790261e+04f,  6.348708112e+04f,  6.349774374e+04f,  6.309243011e+04f,  6.301207112e+04f,  6.302265787e+04f,
                6.481754077e+04f,  6.473553306e+04f,  6.474634862e+04f,  6.433502618e+04f,  6.425348364e+04f,  6.426422260e+04f,
                6.385500856e+04f,  6.377393048e+04f,  6.378459321e+04f,  6.337748792e+04f,  6.329687357e+04f,  6.330746042e+04f,
                6.491461798e+04f,  6.483252392e+04f,  6.484333951e+04f,  6.443141414e+04f,  6.434978567e+04f,  6.436052466e+04f,
                6.395071054e+04f,  6.386954694e+04f,  6.388020970e+04f,  6.347250718e+04f,  6.339180772e+04f,  6.340239461e+04f,
                6.501169518e+04f,  6.492951478e+04f,  6.494033041e+04f,  6.452780210e+04f,  6.444608770e+04f,  6.445682672e+04f,
                6.404641252e+04f,  6.396516339e+04f,  6.397582618e+04f,  6.356752645e+04f,  6.348674187e+04f,  6.349732879e+04f,
                6.510877239e+04f,  6.502650565e+04f,  6.503732131e+04f,  6.462419006e+04f,  6.454238972e+04f,  6.455312878e+04f,
                6.414211450e+04f,  6.406077985e+04f,  6.407144267e+04f,  6.366254572e+04f,  6.358167602e+04f,  6.359226298e+04f,
                6.520584960e+04f,  6.512349651e+04f,  6.513431220e+04f,  6.472057802e+04f,  6.463869175e+04f,  6.464943084e+04f,
                6.423781648e+04f,  6.415639630e+04f,  6.416705916e+04f,  6.375756499e+04f,  6.367661017e+04f,  6.368719716e+04f,
                6.821524298e+04f,  6.813021327e+04f,  6.814103002e+04f,  6.770860480e+04f,  6.762405456e+04f,  6.763479471e+04f,
                6.720457791e+04f,  6.712050641e+04f,  6.713117033e+04f,  6.670316230e+04f,  6.661956882e+04f,  6.663015686e+04f,
                6.831232018e+04f,  6.822720413e+04f,  6.823802091e+04f,  6.780499276e+04f,  6.772035659e+04f,  6.773109677e+04f,
                6.730027989e+04f,  6.721612287e+04f,  6.722678682e+04f,  6.679818157e+04f,  6.671450297e+04f,  6.672509105e+04f,
                6.840939739e+04f,  6.832419499e+04f,  6.833501181e+04f,  6.790138072e+04f,  6.781665861e+04f,  6.782739883e+04f,
                6.739598187e+04f,  6.731173932e+04f,  6.732240331e+04f,  6.689320083e+04f,  6.680943712e+04f,  6.682002523e+04f,
                6.850647460e+04f,  6.842118586e+04f,  6.843200271e+04f,  6.799776868e+04f,  6.791296064e+04f,  6.792370089e+04f,
                6.749168385e+04f,  6.740735578e+04f,  6.741801979e+04f,  6.698822010e+04f,  6.690437127e+04f,  6.691495942e+04f,
                6.860355180e+04f,  6.851817672e+04f,  6.852899360e+04f,  6.809415664e+04f,  6.800926266e+04f,  6.802000295e+04f,
                6.758738583e+04f,  6.750297223e+04f,  6.751363628e+04f,  6.708323937e+04f,  6.699930542e+04f,  6.700989360e+04f,
                6.889478342e+04f,  6.880914931e+04f,  6.881996630e+04f,  6.838332052e+04f,  6.829816874e+04f,  6.830890913e+04f,
                6.787449178e+04f,  6.778982160e+04f,  6.780048575e+04f,  6.736829717e+04f,  6.728410787e+04f,  6.729469615e+04f,
                6.899186062e+04f,  6.890614017e+04f,  6.891695719e+04f,  6.847970849e+04f,  6.839447077e+04f,  6.840521119e+04f,
                6.797019376e+04f,  6.788543805e+04f,  6.789610224e+04f,  6.746331644e+04f,  6.737904202e+04f,  6.738963034e+04f,
                6.908893783e+04f,  6.900313103e+04f,  6.901394809e+04f,  6.857609645e+04f,  6.849077279e+04f,  6.850151325e+04f,
                6.806589574e+04f,  6.798105451e+04f,  6.799171873e+04f,  6.755833571e+04f,  6.747397617e+04f,  6.748456452e+04f,
                6.918601504e+04f,  6.910012190e+04f,  6.911093899e+04f,  6.867248441e+04f,  6.858707482e+04f,  6.859781531e+04f,
                6.816159772e+04f,  6.807667096e+04f,  6.808733522e+04f,  6.765335498e+04f,  6.756891032e+04f,  6.757949870e+04f,
                6.928309224e+04f,  6.919711276e+04f,  6.920792989e+04f,  6.876887237e+04f,  6.868337685e+04f,  6.869411737e+04f,
                6.825729970e+04f,  6.817228742e+04f,  6.818295171e+04f,  6.774837425e+04f,  6.766384447e+04f,  6.767443289e+04f,
                6.957432386e+04f,  6.948808535e+04f,  6.949890258e+04f,  6.905803625e+04f,  6.897228293e+04f,  6.898302355e+04f,
                6.854440565e+04f,  6.845913678e+04f,  6.846980118e+04f,  6.803343205e+04f,  6.794864692e+04f,  6.795923544e+04f,
                6.967140107e+04f,  6.958507621e+04f,  6.959589347e+04f,  6.915442421e+04f,  6.906858495e+04f,  6.907932561e+04f,
                6.864010763e+04f,  6.855475324e+04f,  6.856541766e+04f,  6.812845132e+04f,  6.804358107e+04f,  6.805416963e+04f,
                6.976847827e+04f,  6.968206708e+04f,  6.969288437e+04f,  6.925081217e+04f,  6.916488698e+04f,  6.917562767e+04f,
                6.873580961e+04f,  6.865036969e+04f,  6.866103415e+04f,  6.822347059e+04f,  6.813851522e+04f,  6.814910381e+04f,
                6.986555548e+04f,  6.977905794e+04f,  6.978987527e+04f,  6.934720013e+04f,  6.926118900e+04f,  6.927192973e+04f,
                6.883151159e+04f,  6.874598615e+04f,  6.875665064e+04f,  6.831848985e+04f,  6.823344937e+04f,  6.824403799e+04f,
                6.996263268e+04f,  6.987604880e+04f,  6.988686617e+04f,  6.944358809e+04f,  6.935749103e+04f,  6.936823179e+04f,
                6.892721357e+04f,  6.884160260e+04f,  6.885226713e+04f,  6.841350912e+04f,  6.832838352e+04f,  6.833897218e+04f,
                7.025386430e+04f,  7.016702139e+04f,  7.017783886e+04f,  6.973275197e+04f,  6.964639711e+04f,  6.965713797e+04f,
                6.921431952e+04f,  6.912845197e+04f,  6.913911660e+04f,  6.869856693e+04f,  6.861318597e+04f,  6.862377473e+04f,
                7.035094151e+04f,  7.026401226e+04f,  7.027482975e+04f,  6.982913993e+04f,  6.974269913e+04f,  6.975344004e+04f,
                6.931002150e+04f,  6.922406842e+04f,  6.923473309e+04f,  6.879358619e+04f,  6.870812012e+04f,  6.871870891e+04f,
                7.044801871e+04f,  7.036100312e+04f,  7.037182065e+04f,  6.992552790e+04f,  6.983900116e+04f,  6.984974210e+04f,
                6.940572348e+04f,  6.931968488e+04f,  6.933034958e+04f,  6.888860546e+04f,  6.880305427e+04f,  6.881364310e+04f,
                7.054509592e+04f,  7.045799398e+04f,  7.046881155e+04f,  7.002191586e+04f,  6.993530319e+04f,  6.994604416e+04f,
                6.950142546e+04f,  6.941530133e+04f,  6.942596607e+04f,  6.898362473e+04f,  6.889798842e+04f,  6.890857728e+04f,
                7.064217312e+04f,  7.055498484e+04f,  7.056580245e+04f,  7.011830382e+04f,  7.003160521e+04f,  7.004234622e+04f,
                6.959712744e+04f,  6.951091779e+04f,  6.952158256e+04f,  6.907864400e+04f,  6.899292257e+04f,  6.900351147e+04f,
                1.062695077e+05f,  1.061506316e+05f,  1.061614617e+05f,  1.054926854e+05f,  1.053744488e+05f,  1.053852023e+05f,
                1.047197546e+05f,  1.046021568e+05f,  1.046128341e+05f,  1.039507154e+05f,  1.038337556e+05f,  1.038443570e+05f,
                1.063665849e+05f,  1.062476225e+05f,  1.062584526e+05f,  1.055890734e+05f,  1.054707508e+05f,  1.054815044e+05f,
                1.048154566e+05f,  1.046977733e+05f,  1.047084506e+05f,  1.040457346e+05f,  1.039286898e+05f,  1.039392912e+05f,
                1.064636621e+05f,  1.063446133e+05f,  1.063554435e+05f,  1.056854613e+05f,  1.055670529e+05f,  1.055778064e+05f,
                1.049111586e+05f,  1.047933897e+05f,  1.048040671e+05f,  1.041407539e+05f,  1.040236239e+05f,  1.040342254e+05f,
                1.065607393e+05f,  1.064416042e+05f,  1.064524344e+05f,  1.057818493e+05f,  1.056633549e+05f,  1.056741085e+05f,
                1.050068606e+05f,  1.048890062e+05f,  1.048996835e+05f,  1.042357732e+05f,  1.041185581e+05f,  1.041291596e+05f,
                1.066578165e+05f,  1.065385951e+05f,  1.065494253e+05f,  1.058782372e+05f,  1.057596569e+05f,  1.057704105e+05f,
                1.051025625e+05f,  1.049846226e+05f,  1.049953000e+05f,  1.043307924e+05f,  1.042134922e+05f,  1.042240937e+05f,
                1.069490481e+05f,  1.068295677e+05f,  1.068403980e+05f,  1.061674011e+05f,  1.060485630e+05f,  1.060593167e+05f,
                1.053896685e+05f,  1.052714720e+05f,  1.052821495e+05f,  1.046158502e+05f,  1.044982947e+05f,  1.045088963e+05f,
                1.070461253e+05f,  1.069265585e+05f,  1.069373889e+05f,  1.062637891e+05f,  1.061448650e+05f,  1.061556188e+05f,
                1.054853705e+05f,  1.053670885e+05f,  1.053777660e+05f,  1.047108695e+05f,  1.045932288e+05f,  1.046038305e+05f,
                1.071432025e+05f,  1.070235494e+05f,  1.070343798e+05f,  1.063601770e+05f,  1.062411671e+05f,  1.062519208e+05f,
                1.055810725e+05f,  1.054627049e+05f,  1.054733825e+05f,  1.048058888e+05f,  1.046881630e+05f,  1.046987647e+05f,
                1.072402797e+05f,  1.071205402e+05f,  1.071313707e+05f,  1.064565650e+05f,  1.063374691e+05f,  1.063482229e+05f,
                1.056767744e+05f,  1.055583214e+05f,  1.055689990e+05f,  1.049009081e+05f,  1.047830971e+05f,  1.047936988e+05f,
                1.073373570e+05f,  1.072175311e+05f,  1.072283616e+05f,  1.065529530e+05f,  1.064337711e+05f,  1.064445250e+05f,
                1.057724764e+05f,  1.056539378e+05f,  1.056646155e+05f,  1.049959273e+05f,  1.048780313e+05f,  1.048886330e+05f,
                1.076285886e+05f,  1.075085037e+05f,  1.075193343e+05f,  1.068421168e+05f,  1.067226772e+05f,  1.067334311e+05f,
                1.060595824e+05f,  1.059407872e+05f,  1.059514649e+05f,  1.052809851e+05f,  1.051628337e+05f,  1.051734356e+05f,
                1.077256658e+05f,  1.076054946e+05f,  1.076163252e+05f,  1.069385048e+05f,  1.068189792e+05f,  1.068297332e+05f,
                1.061552843e+05f,  1.060364036e+05f,  1.060470814e+05f,  1.053760044e+05f,  1.052577679e+05f,  1.052683698e+05f,
                1.078227430e+05f,  1.077024854e+05f,  1.077133161e+05f,  1.070348928e+05f,  1.069152812e+05f,  1.069260353e+05f,
                1.062509863e+05f,  1.061320201e+05f,  1.061426979e+05f,  1.054710237e+05f,  1.053527020e+05f,  1.053633039e+05f,
                1.079198202e+05f,  1.077994763e+05f,  1.078103070e+05f,  1.071312807e+05f,  1.070115833e+05f,  1.070223373e+05f,
                1.063466883e+05f,  1.062276366e+05f,  1.062383144e+05f,  1.055660429e+05f,  1.054476362e+05f,  1.054582381e+05f,
                1.080168974e+05f,  1.078964671e+05f,  1.079072978e+05f,  1.072276687e+05f,  1.071078853e+05f,  1.071186394e+05f,
                1.064423903e+05f,  1.063232530e+05f,  1.063339309e+05f,  1.056610622e+05f,  1.055425703e+05f,  1.055531723e+05f,
                1.083081290e+05f,  1.081874397e+05f,  1.081982705e+05f,  1.075168326e+05f,  1.073967914e+05f,  1.074075456e+05f,
                1.067294962e+05f,  1.066101024e+05f,  1.066207803e+05f,  1.059461200e+05f,  1.058273728e+05f,  1.058379749e+05f,
                1.084052062e+05f,  1.082844306e+05f,  1.082952614e+05f,  1.076132205e+05f,  1.074930934e+05f,  1.075038476e+05f,
                1.068251982e+05f,  1.067057188e+05f,  1.067163968e+05f,  1.060411393e+05f,  1.059223069e+05f,  1.059329091e+05f,
                1.085022834e+05f,  1.083814215e+05f,  1.083922523e+05f,  1.077096085e+05f,  1.075893954e+05f,  1.076001497e+05f,
                1.069209002e+05f,  1.068013353e+05f,  1.068120133e+05f,  1.061361585e+05f,  1.060172411e+05f,  1.060278432e+05f,
                1.085993606e+05f,  1.084784123e+05f,  1.084892432e+05f,  1.078059964e+05f,  1.076856974e+05f,  1.076964517e+05f,
                1.070166022e+05f,  1.068969517e+05f,  1.069076298e+05f,  1.062311778e+05f,  1.061121752e+05f,  1.061227774e+05f,
                1.086964378e+05f,  1.085754032e+05f,  1.085862341e+05f,  1.079023844e+05f,  1.077819995e+05f,  1.077927538e+05f,
                1.071123042e+05f,  1.069925682e+05f,  1.070032463e+05f,  1.063261971e+05f,  1.062071094e+05f,  1.062177116e+05f,
                1.117058312e+05f,  1.115821199e+05f,  1.115929519e+05f,  1.108904112e+05f,  1.107673623e+05f,  1.107781177e+05f,
                1.100790656e+05f,  1.099566783e+05f,  1.099673575e+05f,  1.092717944e+05f,  1.091500680e+05f,  1.091606713e+05f,
                1.118029084e+05f,  1.116791108e+05f,  1.116899428e+05f,  1.109867991e+05f,  1.108636643e+05f,  1.108744197e+05f,
                1.101747676e+05f,  1.100522948e+05f,  1.100629740e+05f,  1.093668137e+05f,  1.092450022e+05f,  1.092556055e+05f,
                1.118999856e+05f,  1.117761017e+05f,  1.117869337e+05f,  1.110831871e+05f,  1.109599663e+05f,  1.109707218e+05f,
                1.102704695e+05f,  1.101479112e+05f,  1.101585904e+05f,  1.094618329e+05f,  1.093399363e+05f,  1.093505397e+05f,
                1.119970628e+05f,  1.118730925e+05f,  1.118839246e+05f,  1.111795751e+05f,  1.110562684e+05f,  1.110670239e+05f,
                1.103661715e+05f,  1.102435277e+05f,  1.102542069e+05f,  1.095568522e+05f,  1.094348705e+05f,  1.094454739e+05f,
                1.120941400e+05f,  1.119700834e+05f,  1.119809155e+05f,  1.112759630e+05f,  1.111525704e+05f,  1.111633259e+05f,
                1.104618735e+05f,  1.103391441e+05f,  1.103498234e+05f,  1.096518715e+05f,  1.095298046e+05f,  1.095404080e+05f,
                1.123853717e+05f,  1.122610560e+05f,  1.122718882e+05f,  1.115651269e+05f,  1.114414765e+05f,  1.114522321e+05f,
                1.107489794e+05f,  1.106259935e+05f,  1.106366729e+05f,  1.099369293e+05f,  1.098146071e+05f,  1.098252106e+05f,
                1.124824489e+05f,  1.123580469e+05f,  1.123688791e+05f,  1.116615149e+05f,  1.115377785e+05f,  1.115485342e+05f,
                1.108446814e+05f,  1.107216099e+05f,  1.107322894e+05f,  1.100319485e+05f,  1.099095412e+05f,  1.099201448e+05f,
                1.125795261e+05f,  1.124550377e+05f,  1.124658700e+05f,  1.117579028e+05f,  1.116340805e+05f,  1.116448362e+05f,
                1.109403834e+05f,  1.108172264e+05f,  1.108279059e+05f,  1.101269678e+05f,  1.100044754e+05f,  1.100150790e+05f,
                1.126766033e+05f,  1.125520286e+05f,  1.125628609e+05f,  1.118542908e+05f,  1.117303825e+05f,  1.117411383e+05f,
                1.110360854e+05f,  1.109128429e+05f,  1.109235224e+05f,  1.102219871e+05f,  1.100994095e+05f,  1.101100132e+05f,
                1.127736805e+05f,  1.126490194e+05f,  1.126598518e+05f,  1.119506788e+05f,  1.118266846e+05f,  1.118374403e+05f,
                1.111317874e+05f,  1.110084593e+05f,  1.110191388e+05f,  1.103170063e+05f,  1.101943437e+05f,  1.102049473e+05f,
                1.130649121e+05f,  1.129399920e+05f,  1.129508245e+05f,  1.122398426e+05f,  1.121155906e+05f,  1.121263465e+05f,
                1.114188933e+05f,  1.112953087e+05f,  1.113059883e+05f,  1.106020641e+05f,  1.104791461e+05f,  1.104897499e+05f,
                1.131619893e+05f,  1.130369829e+05f,  1.130478154e+05f,  1.123362306e+05f,  1.122118927e+05f,  1.122226486e+05f,
                1.115145953e+05f,  1.113909251e+05f,  1.114016048e+05f,  1.106970834e+05f,  1.105740803e+05f,  1.105846841e+05f,
                1.132590665e+05f,  1.131339738e+05f,  1.131448063e+05f,  1.124326186e+05f,  1.123081947e+05f,  1.123189506e+05f,
                1.116102973e+05f,  1.114865416e+05f,  1.114972213e+05f,  1.107921027e+05f,  1.106690144e+05f,  1.106796183e+05f,
                1.133561437e+05f,  1.132309646e+05f,  1.132417972e+05f,  1.125290065e+05f,  1.124044967e+05f,  1.124152527e+05f,
                1.117059993e+05f,  1.115821580e+05f,  1.115928378e+05f,  1.108871219e+05f,  1.107639486e+05f,  1.107745524e+05f,
                1.134532209e+05f,  1.133279555e+05f,  1.133387881e+05f,  1.126253945e+05f,  1.125007988e+05f,  1.125115548e+05f,
                1.118017012e+05f,  1.116777745e+05f,  1.116884543e+05f,  1.109821412e+05f,  1.108588827e+05f,  1.108694866e+05f,
                1.137444525e+05f,  1.136189281e+05f,  1.136297608e+05f,  1.129145584e+05f,  1.127897048e+05f,  1.128004609e+05f,
                1.120888072e+05f,  1.119646239e+05f,  1.119753037e+05f,  1.112671990e+05f,  1.111436852e+05f,  1.111542892e+05f,
                1.138415297e+05f,  1.137159189e+05f,  1.137267517e+05f,  1.130109463e+05f,  1.128860069e+05f,  1.128967630e+05f,
                1.121845092e+05f,  1.120602403e+05f,  1.120709202e+05f,  1.113622183e+05f,  1.112386193e+05f,  1.112492234e+05f,
                1.139386070e+05f,  1.138129098e+05f,  1.138237426e+05f,  1.131073343e+05f,  1.129823089e+05f,  1.129930651e+05f,
                1.122802112e+05f,  1.121558568e+05f,  1.121665367e+05f,  1.114572375e+05f,  1.113335535e+05f,  1.113441575e+05f,
                1.140356842e+05f,  1.139099007e+05f,  1.139207335e+05f,  1.132037222e+05f,  1.130786109e+05f,  1.130893671e+05f,
                1.123759131e+05f,  1.122514732e+05f,  1.122621532e+05f,  1.115522568e+05f,  1.114284876e+05f,  1.114390917e+05f,
                1.141327614e+05f,  1.140068915e+05f,  1.140177244e+05f,  1.133001102e+05f,  1.131749129e+05f,  1.131856692e+05f,
                1.124716151e+05f,  1.123470897e+05f,  1.123577697e+05f,  1.116472761e+05f,  1.115234218e+05f,  1.115340259e+05f,
                1.171421547e+05f,  1.170136083e+05f,  1.170244422e+05f,  1.162881370e+05f,  1.161602757e+05f,  1.161710330e+05f,
                1.154383765e+05f,  1.153111998e+05f,  1.153218809e+05f,  1.145928734e+05f,  1.144663804e+05f,  1.144769856e+05f,
                1.172392320e+05f,  1.171105991e+05f,  1.171214331e+05f,  1.163845249e+05f,  1.162565778e+05f,  1.162673351e+05f,
                1.155340785e+05f,  1.154068163e+05f,  1.154174973e+05f,  1.146878927e+05f,  1.145613146e+05f,  1.145719198e+05f,
                1.173363092e+05f,  1.172075900e+05f,  1.172184240e+05f,  1.164809129e+05f,  1.163528798e+05f,  1.163636372e+05f,
                1.156297805e+05f,  1.155024327e+05f,  1.155131138e+05f,  1.147829119e+05f,  1.146562487e+05f,  1.146668540e+05f,
                1.174333864e+05f,  1.173045809e+05f,  1.173154149e+05f,  1.165773009e+05f,  1.164491818e+05f,  1.164599392e+05f,
                1.157254825e+05f,  1.155980492e+05f,  1.156087303e+05f,  1.148779312e+05f,  1.147511829e+05f,  1.147617882e+05f,
                1.175304636e+05f,  1.174015717e+05f,  1.174124058e+05f,  1.166736888e+05f,  1.165454839e+05f,  1.165562413e+05f,
                1.158211845e+05f,  1.156936656e+05f,  1.157043468e+05f,  1.149729505e+05f,  1.148461170e+05f,  1.148567224e+05f,
                1.178216952e+05f,  1.176925443e+05f,  1.177033785e+05f,  1.169628527e+05f,  1.168343899e+05f,  1.168451475e+05f,
                1.161082904e+05f,  1.159805150e+05f,  1.159911963e+05f,  1.152580083e+05f,  1.151309195e+05f,  1.151415249e+05f,
                1.179187724e+05f,  1.177895352e+05f,  1.178003694e+05f,  1.170592407e+05f,  1.169306920e+05f,  1.169414495e+05f,
                1.162039924e+05f,  1.160761314e+05f,  1.160868128e+05f,  1.153530275e+05f,  1.152258536e+05f,  1.152364591e+05f,
                1.180158496e+05f,  1.178865261e+05f,  1.178973603e+05f,  1.171556286e+05f,  1.170269940e+05f,  1.170377516e+05f,
                1.162996944e+05f,  1.161717479e+05f,  1.161824293e+05f,  1.154480468e+05f,  1.153207878e+05f,  1.153313933e+05f,
                1.181129268e+05f,  1.179835169e+05f,  1.179943512e+05f,  1.172520166e+05f,  1.171232960e+05f,  1.171340536e+05f,
                1.163953964e+05f,  1.162673643e+05f,  1.162780458e+05f,  1.155430661e+05f,  1.154157219e+05f,  1.154263275e+05f,
                1.182100040e+05f,  1.180805078e+05f,  1.180913420e+05f,  1.173484046e+05f,  1.172195980e+05f,  1.172303557e+05f,
                1.164910983e+05f,  1.163629808e+05f,  1.163736622e+05f,  1.156380853e+05f,  1.155106561e+05f,  1.155212616e+05f,
                1.185012356e+05f,  1.183714804e+05f,  1.183823147e+05f,  1.176375684e+05f,  1.175085041e+05f,  1.175192619e+05f,
                1.167782043e+05f,  1.166498302e+05f,  1.166605117e+05f,  1.159231431e+05f,  1.157954585e+05f,  1.158060642e+05f,
                1.185983128e+05f,  1.184684712e+05f,  1.184793056e+05f,  1.177339564e+05f,  1.176048061e+05f,  1.176155639e+05f,
                1.168739063e+05f,  1.167454466e+05f,  1.167561282e+05f,  1.160181624e+05f,  1.158903927e+05f,  1.159009984e+05f,
                1.186953900e+05f,  1.185654621e+05f,  1.185762965e+05f,  1.178303444e+05f,  1.177011082e+05f,  1.177118660e+05f,
                1.169696082e+05f,  1.168410631e+05f,  1.168517447e+05f,  1.161131817e+05f,  1.159853268e+05f,  1.159959326e+05f,
                1.187924672e+05f,  1.186624530e+05f,  1.186732874e+05f,  1.179267323e+05f,  1.177974102e+05f,  1.178081681e+05f,
                1.170653102e+05f,  1.169366795e+05f,  1.169473612e+05f,  1.162082010e+05f,  1.160802610e+05f,  1.160908667e+05f,
                1.188895445e+05f,  1.187594438e+05f,  1.187702783e+05f,  1.180231203e+05f,  1.178937122e+05f,  1.179044701e+05f,
                1.171610122e+05f,  1.170322960e+05f,  1.170429777e+05f,  1.163032202e+05f,  1.161751951e+05f,  1.161858009e+05f,
                1.191807761e+05f,  1.190504164e+05f,  1.190612510e+05f,  1.183122842e+05f,  1.181826183e+05f,  1.181933763e+05f,
                1.174481181e+05f,  1.173191454e+05f,  1.173298271e+05f,  1.165882780e+05f,  1.164599976e+05f,  1.164706035e+05f,
                1.192778533e+05f,  1.191474073e+05f,  1.191582419e+05f,  1.184086721e+05f,  1.182789203e+05f,  1.182896784e+05f,
                1.175438201e+05f,  1.174147618e+05f,  1.174254436e+05f,  1.166832973e+05f,  1.165549317e+05f,  1.165655377e+05f,
                1.193749305e+05f,  1.192443981e+05f,  1.192552328e+05f,  1.185050601e+05f,  1.183752224e+05f,  1.183859804e+05f,
                1.176395221e+05f,  1.175103783e+05f,  1.175210601e+05f,  1.167783166e+05f,  1.166498659e+05f,  1.166604718e+05f,
                1.194720077e+05f,  1.193413890e+05f,  1.193522237e+05f,  1.186014480e+05f,  1.184715244e+05f,  1.184822825e+05f,
                1.177352241e+05f,  1.176059947e+05f,  1.176166766e+05f,  1.168733358e+05f,  1.167448000e+05f,  1.167554060e+05f,
                1.195690849e+05f,  1.194383799e+05f,  1.194492146e+05f,  1.186978360e+05f,  1.185678264e+05f,  1.185785846e+05f,
                1.178309261e+05f,  1.177016112e+05f,  1.177122931e+05f,  1.169683551e+05f,  1.168397342e+05f,  1.168503402e+05f,
            };

            float[] y_actual = y.ToArray();

            AssertError.Tolerance(y_expect, y_actual, 1e-7f, 1e-5f, $"mismatch value {inchannels},{outchannels},{kwidth},{kheight},{kdepth},{inwidth},{inheight},{indepth},{batch}");
        }
    }
}
