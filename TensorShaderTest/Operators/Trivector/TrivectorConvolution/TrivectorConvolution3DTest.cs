using System;
using System.Diagnostics;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using TensorShader.Operators.TrivectorConvolution;

namespace TensorShaderTest.Operators.Trivector {
    [TestClass]
    public class TrivectorConvolution3DTest {
        [TestMethod]
        public void ExecuteTest() {
            float max_err = 0;

            foreach (int batch in new int[] { 1, 2 }) {
                foreach (int inchannels in new int[] { 3, 6, 9, 12 }) {
                    foreach (int outchannels in new int[] { 3, 6, 9, 12 }) {
                        foreach ((int kwidth, int kheight, int kdepth) in new (int, int, int)[] { (1, 1, 1), (3, 3, 3), (5, 5, 5), (1, 3, 5), (3, 5, 1), (5, 1, 3) }) {
                            foreach ((int inwidth, int inheight, int indepth) in new (int, int, int)[] { (13, 13, 13), (17, 17, 17), (19, 19, 19), (17, 19, 13), (13, 17, 19), (19, 13, 17) }) {
                                int outwidth = inwidth - kwidth + 1, outheight = inheight - kheight + 1, outdepth = indepth - kdepth + 1;

                                float[] xval = (new float[inwidth * inheight * indepth * inchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
                                float[] wval = (new float[kwidth * kheight * kdepth * inchannels * outchannels / 9 * 4]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

                                Trivector[] xcval = (new Trivector[xval.Length / 3])
                                    .Select((_, idx) => new Trivector(xval[idx * 3], xval[idx * 3 + 1], xval[idx * 3 + 2])).ToArray();

                                Quaternion.Quaternion[] wcval = (new Quaternion.Quaternion[wval.Length / 4])
                                    .Select((_, idx) => new Quaternion.Quaternion(wval[idx * 4], wval[idx * 4 + 1], wval[idx * 4 + 2], wval[idx * 4 + 3])).ToArray();

                                TrivectorMap3D x = new TrivectorMap3D(inchannels / 3, inwidth, inheight, indepth, batch, xcval);
                                Quaternion.QuaternionFilter3D w = new Quaternion.QuaternionFilter3D(inchannels / 3, outchannels / 3, kwidth, kheight, kdepth, wcval);

                                TrivectorMap3D y = Reference(x, w, kwidth, kheight, kdepth);

                                OverflowCheckedTensor x_tensor = new OverflowCheckedTensor(Shape.Map3D(inchannels, inwidth, inheight, indepth, batch), xval);
                                OverflowCheckedTensor w_tensor = new OverflowCheckedTensor(Shape.Kernel3D(inchannels / 3 * 4, outchannels / 3, kwidth, kheight, kdepth), wval);

                                OverflowCheckedTensor y_tensor = new OverflowCheckedTensor(Shape.Map3D(outchannels, outwidth, outheight, outdepth, batch));

                                TrivectorConvolution3D ope = new TrivectorConvolution3D(inwidth, inheight, indepth, inchannels, outchannels, kwidth, kheight, kdepth, gradmode: false, batch);

                                ope.Execute(x_tensor, w_tensor, y_tensor);

                                float[] y_expect = y.ToArray();
                                float[] y_actual = y_tensor.State;

                                CollectionAssert.AreEqual(xval, x_tensor.State);
                                CollectionAssert.AreEqual(wval, w_tensor.State);

                                AssertError.Tolerance(y_expect, y_actual, 1e-7f, 1e-5f, ref max_err, $"mismatch value {inchannels},{outchannels},{kwidth},{kheight},{kdepth},{inwidth},{inheight},{indepth},{batch}");

                                Console.WriteLine($"pass: {inchannels},{outchannels},{kwidth},{kheight},{kdepth},{inwidth},{inheight},{indepth},{batch}");
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
                foreach (int batch in new int[] { 1, 2 }) {
                    foreach (int inchannels in new int[] { 3, 6, 9, 12 }) {
                        foreach (int outchannels in new int[] { 3, 6, 9, 12 }) {
                            foreach ((int kwidth, int kheight, int kdepth) in new (int, int, int)[] { (1, 1, 1), (3, 3, 3), (5, 5, 5), (1, 3, 5), (3, 5, 1), (5, 1, 3) }) {
                                foreach ((int inwidth, int inheight, int indepth) in new (int, int, int)[] { (13, 13, 13), (17, 17, 17), (19, 19, 19), (17, 19, 13), (13, 17, 19), (19, 13, 17) }) {
                                    int outwidth = inwidth - kwidth + 1, outheight = inheight - kheight + 1, outdepth = indepth - kdepth + 1;

                                    float[] xval = (new float[inwidth * inheight * indepth * inchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
                                    float[] wval = (new float[kwidth * kheight * kdepth * inchannels * outchannels / 9 * 4]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

                                    OverflowCheckedTensor x_tensor = new OverflowCheckedTensor(Shape.Map3D(inchannels, inwidth, inheight, indepth, batch), xval);
                                    OverflowCheckedTensor w_tensor = new OverflowCheckedTensor(Shape.Kernel3D(inchannels / 3 * 4, outchannels / 3, kwidth, kheight, kdepth), wval);

                                    OverflowCheckedTensor y_tensor = new OverflowCheckedTensor(Shape.Map3D(outchannels, outwidth, outheight, outdepth, batch));

                                    TrivectorConvolution3D ope = new TrivectorConvolution3D(inwidth, inheight, indepth, inchannels, outchannels, kwidth, kheight, kdepth, gradmode, batch);

                                    ope.Execute(x_tensor, w_tensor, y_tensor);

                                    CollectionAssert.AreEqual(xval, x_tensor.State);
                                    CollectionAssert.AreEqual(wval, w_tensor.State);

                                    y_tensor.CheckOverflow();

                                    Console.WriteLine($"pass: {inchannels},{outchannels},{kwidth},{kheight},{kdepth},{inwidth},{inheight},{indepth},{batch},{gradmode}");
                                }
                            }
                        }
                    }
                }
            }
        }

        [TestMethod]
        public void SpeedTest() {
            int inwidth = 32, inheight = 32, indepth = 32, inchannels = 33, outchannels = 33, ksize = 3;
            int outwidth = inwidth - ksize + 1, outheight = inheight - ksize + 1, outdepth = indepth - ksize + 1;

            OverflowCheckedTensor x_tensor = new OverflowCheckedTensor(Shape.Map3D(inchannels, inwidth, inheight, indepth));
            OverflowCheckedTensor w_tensor = new OverflowCheckedTensor(Shape.Kernel3D(inchannels / 3 * 4, outchannels / 3, ksize, ksize, ksize));

            OverflowCheckedTensor y_tensor = new OverflowCheckedTensor(Shape.Map3D(outchannels, outwidth, outheight, outdepth));

            TrivectorConvolution3D ope = new TrivectorConvolution3D(inwidth, inheight, indepth, inchannels, outchannels, ksize, ksize, ksize);

            Stopwatch sw = new Stopwatch();
            sw.Start();

            ope.Execute(x_tensor, w_tensor, y_tensor);
            ope.Execute(x_tensor, w_tensor, y_tensor);
            ope.Execute(x_tensor, w_tensor, y_tensor);
            ope.Execute(x_tensor, w_tensor, y_tensor);

            sw.Stop();

            Console.WriteLine($"{sw.ElapsedMilliseconds / 4} msec");
        }

        public static TrivectorMap3D Reference(TrivectorMap3D x, Quaternion.QuaternionFilter3D w, int kwidth, int kheight, int kdepth) {
            int inchannels = x.Channels, outchannels = w.OutChannels, batch = x.Batch;
            int inw = x.Width, inh = x.Height, ind = x.Depth;
            int outw = inw - kwidth + 1, outh = inh - kheight + 1, outd = ind - kdepth + 1;

            TrivectorMap3D y = new TrivectorMap3D(outchannels, outw, outh, outd, batch);

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

            TrivectorMap3D x = new TrivectorMap3D(inchannels / 3, inwidth, inheight, indepth, batch, xcval);
            Quaternion.QuaternionFilter3D w = new Quaternion.QuaternionFilter3D(inchannels / 3, outchannels / 3, kwidth, kheight, kdepth, wcval);

            TrivectorMap3D y = Reference(x, w, kwidth, kheight, kdepth);

            float[] y_expect = {
                8.415684149e+03f,  8.383841578e+03f,  8.394637372e+03f,  8.333621015e+03f,  8.302006438e+03f,  8.312725634e+03f,
                8.252157358e+03f,  8.220770050e+03f,  8.231413011e+03f,  8.171293179e+03f,  8.140132414e+03f,  8.150699502e+03f,
                8.609838561e+03f,  8.577823305e+03f,  8.588619166e+03f,  8.526396936e+03f,  8.494610490e+03f,  8.505329754e+03f,
                8.443561321e+03f,  8.412002960e+03f,  8.422645989e+03f,  8.361331715e+03f,  8.330000714e+03f,  8.340567870e+03f,
                8.803992973e+03f,  8.771805031e+03f,  8.782600961e+03f,  8.719172857e+03f,  8.687214543e+03f,  8.697933875e+03f,
                8.634965283e+03f,  8.603235870e+03f,  8.613878967e+03f,  8.551370251e+03f,  8.519869014e+03f,  8.530436238e+03f,
                9.774765032e+03f,  9.741713662e+03f,  9.752509932e+03f,  9.683052464e+03f,  9.650234804e+03f,  9.660954476e+03f,
                9.591985098e+03f,  9.559400422e+03f,  9.570043859e+03f,  9.501562931e+03f,  9.469210514e+03f,  9.479778079e+03f,
                9.968919444e+03f,  9.935695388e+03f,  9.946491726e+03f,  9.875828386e+03f,  9.842838857e+03f,  9.853558597e+03f,
                9.783389060e+03f,  9.750633332e+03f,  9.761276837e+03f,  9.691601467e+03f,  9.659078814e+03f,  9.669646447e+03f,
                1.016307386e+04f,  1.012967711e+04f,  1.014047352e+04f,  1.006860431e+04f,  1.003544291e+04f,  1.004616272e+04f,
                9.974793023e+03f,  9.941866242e+03f,  9.952509815e+03f,  9.881640003e+03f,  9.848947114e+03f,  9.859514815e+03f,
                1.928833121e+04f,  1.924681825e+04f,  1.925761785e+04f,  1.912907261e+04f,  1.908783337e+04f,  1.909855638e+04f,
                1.897077928e+04f,  1.892981302e+04f,  1.894045980e+04f,  1.881345120e+04f,  1.877275721e+04f,  1.878332811e+04f,
                1.948248562e+04f,  1.944079998e+04f,  1.945159965e+04f,  1.932184853e+04f,  1.928043742e+04f,  1.929116050e+04f,
                1.916218324e+04f,  1.912104593e+04f,  1.913169277e+04f,  1.900348974e+04f,  1.896262551e+04f,  1.897319648e+04f,
                1.967664003e+04f,  1.963478170e+04f,  1.964558144e+04f,  1.951462445e+04f,  1.947304147e+04f,  1.948376462e+04f,
                1.935358720e+04f,  1.931227884e+04f,  1.932292575e+04f,  1.919352827e+04f,  1.915249381e+04f,  1.916306485e+04f,
                2.064741209e+04f,  2.060469033e+04f,  2.061549041e+04f,  2.047850406e+04f,  2.043606174e+04f,  2.044678522e+04f,
                2.031060702e+04f,  2.026844340e+04f,  2.027909064e+04f,  2.014372095e+04f,  2.010183531e+04f,  2.011240669e+04f,
                2.084156650e+04f,  2.079867206e+04f,  2.080947221e+04f,  2.067127998e+04f,  2.062866579e+04f,  2.063938934e+04f,
                2.050201098e+04f,  2.045967631e+04f,  2.047032362e+04f,  2.033375949e+04f,  2.029170361e+04f,  2.030227506e+04f,
                2.103572092e+04f,  2.099265379e+04f,  2.100345400e+04f,  2.086405590e+04f,  2.082126984e+04f,  2.083199346e+04f,
                2.069341494e+04f,  2.065090922e+04f,  2.066155660e+04f,  2.052379803e+04f,  2.048157191e+04f,  2.049214343e+04f,
                5.734259592e+04f,  5.726723660e+04f,  5.727804954e+04f,  5.691315320e+04f,  5.683822763e+04f,  5.684896397e+04f,
                5.648595599e+04f,  5.641146344e+04f,  5.642212354e+04f,  5.606100428e+04f,  5.598694402e+04f,  5.599752825e+04f,
                5.753675033e+04f,  5.746121832e+04f,  5.747203133e+04f,  5.710592912e+04f,  5.703083168e+04f,  5.704156809e+04f,
                5.667735995e+04f,  5.660269635e+04f,  5.661335652e+04f,  5.625104281e+04f,  5.617681232e+04f,  5.618739662e+04f,
                5.773090474e+04f,  5.765520005e+04f,  5.766601312e+04f,  5.729870505e+04f,  5.722343573e+04f,  5.723417221e+04f,
                5.686876391e+04f,  5.679392926e+04f,  5.680458950e+04f,  5.644108135e+04f,  5.636668062e+04f,  5.637726499e+04f,
                5.870167680e+04f,  5.862510868e+04f,  5.863592210e+04f,  5.826258465e+04f,  5.818645599e+04f,  5.819719281e+04f,
                5.782578373e+04f,  5.775009381e+04f,  5.776075439e+04f,  5.739127403e+04f,  5.731602212e+04f,  5.732660683e+04f,
                5.889583121e+04f,  5.881909041e+04f,  5.882990389e+04f,  5.845536057e+04f,  5.837906005e+04f,  5.838979693e+04f,
                5.801718769e+04f,  5.794132672e+04f,  5.795198737e+04f,  5.758131256e+04f,  5.750589042e+04f,  5.751647520e+04f,
                5.908998562e+04f,  5.901307213e+04f,  5.902388568e+04f,  5.864813649e+04f,  5.857166410e+04f,  5.858240105e+04f,
                5.820859165e+04f,  5.813255963e+04f,  5.814322035e+04f,  5.777135110e+04f,  5.769575872e+04f,  5.770634357e+04f,
                6.821524298e+04f,  6.813021327e+04f,  6.814103002e+04f,  6.770860480e+04f,  6.762405456e+04f,  6.763479471e+04f,
                6.720457791e+04f,  6.712050641e+04f,  6.713117033e+04f,  6.670316230e+04f,  6.661956882e+04f,  6.663015686e+04f,
                6.840939739e+04f,  6.832419499e+04f,  6.833501181e+04f,  6.790138072e+04f,  6.781665861e+04f,  6.782739883e+04f,
                6.739598187e+04f,  6.731173932e+04f,  6.732240331e+04f,  6.689320083e+04f,  6.680943712e+04f,  6.682002523e+04f,
                6.860355180e+04f,  6.851817672e+04f,  6.852899360e+04f,  6.809415664e+04f,  6.800926266e+04f,  6.802000295e+04f,
                6.758738583e+04f,  6.750297223e+04f,  6.751363628e+04f,  6.708323937e+04f,  6.699930542e+04f,  6.700989360e+04f,
                6.957432386e+04f,  6.948808535e+04f,  6.949890258e+04f,  6.905803625e+04f,  6.897228293e+04f,  6.898302355e+04f,
                6.854440565e+04f,  6.845913678e+04f,  6.846980118e+04f,  6.803343205e+04f,  6.794864692e+04f,  6.795923544e+04f,
                6.976847827e+04f,  6.968206708e+04f,  6.969288437e+04f,  6.925081217e+04f,  6.916488698e+04f,  6.917562767e+04f,
                6.873580961e+04f,  6.865036969e+04f,  6.866103415e+04f,  6.822347059e+04f,  6.813851522e+04f,  6.814910381e+04f,
                6.996263268e+04f,  6.987604880e+04f,  6.988686617e+04f,  6.944358809e+04f,  6.935749103e+04f,  6.936823179e+04f,
                6.892721357e+04f,  6.884160260e+04f,  6.885226713e+04f,  6.841350912e+04f,  6.832838352e+04f,  6.833897218e+04f,
                1.062695077e+05f,  1.061506316e+05f,  1.061614617e+05f,  1.054926854e+05f,  1.053744488e+05f,  1.053852023e+05f,
                1.047197546e+05f,  1.046021568e+05f,  1.046128341e+05f,  1.039507154e+05f,  1.038337556e+05f,  1.038443570e+05f,
                1.064636621e+05f,  1.063446133e+05f,  1.063554435e+05f,  1.056854613e+05f,  1.055670529e+05f,  1.055778064e+05f,
                1.049111586e+05f,  1.047933897e+05f,  1.048040671e+05f,  1.041407539e+05f,  1.040236239e+05f,  1.040342254e+05f,
                1.066578165e+05f,  1.065385951e+05f,  1.065494253e+05f,  1.058782372e+05f,  1.057596569e+05f,  1.057704105e+05f,
                1.051025625e+05f,  1.049846226e+05f,  1.049953000e+05f,  1.043307924e+05f,  1.042134922e+05f,  1.042240937e+05f,
                1.076285886e+05f,  1.075085037e+05f,  1.075193343e+05f,  1.068421168e+05f,  1.067226772e+05f,  1.067334311e+05f,
                1.060595824e+05f,  1.059407872e+05f,  1.059514649e+05f,  1.052809851e+05f,  1.051628337e+05f,  1.051734356e+05f,
                1.078227430e+05f,  1.077024854e+05f,  1.077133161e+05f,  1.070348928e+05f,  1.069152812e+05f,  1.069260353e+05f,
                1.062509863e+05f,  1.061320201e+05f,  1.061426979e+05f,  1.054710237e+05f,  1.053527020e+05f,  1.053633039e+05f,
                1.080168974e+05f,  1.078964671e+05f,  1.079072978e+05f,  1.072276687e+05f,  1.071078853e+05f,  1.071186394e+05f,
                1.064423903e+05f,  1.063232530e+05f,  1.063339309e+05f,  1.056610622e+05f,  1.055425703e+05f,  1.055531723e+05f,
                1.171421547e+05f,  1.170136083e+05f,  1.170244422e+05f,  1.162881370e+05f,  1.161602757e+05f,  1.161710330e+05f,
                1.154383765e+05f,  1.153111998e+05f,  1.153218809e+05f,  1.145928734e+05f,  1.144663804e+05f,  1.144769856e+05f,
                1.173363092e+05f,  1.172075900e+05f,  1.172184240e+05f,  1.164809129e+05f,  1.163528798e+05f,  1.163636372e+05f,
                1.156297805e+05f,  1.155024327e+05f,  1.155131138e+05f,  1.147829119e+05f,  1.146562487e+05f,  1.146668540e+05f,
                1.175304636e+05f,  1.174015717e+05f,  1.174124058e+05f,  1.166736888e+05f,  1.165454839e+05f,  1.165562413e+05f,
                1.158211845e+05f,  1.156936656e+05f,  1.157043468e+05f,  1.149729505e+05f,  1.148461170e+05f,  1.148567224e+05f,
                1.185012356e+05f,  1.183714804e+05f,  1.183823147e+05f,  1.176375684e+05f,  1.175085041e+05f,  1.175192619e+05f,
                1.167782043e+05f,  1.166498302e+05f,  1.166605117e+05f,  1.159231431e+05f,  1.157954585e+05f,  1.158060642e+05f,
                1.186953900e+05f,  1.185654621e+05f,  1.185762965e+05f,  1.178303444e+05f,  1.177011082e+05f,  1.177118660e+05f,
                1.169696082e+05f,  1.168410631e+05f,  1.168517447e+05f,  1.161131817e+05f,  1.159853268e+05f,  1.159959326e+05f,
                1.188895445e+05f,  1.187594438e+05f,  1.187702783e+05f,  1.180231203e+05f,  1.178937122e+05f,  1.179044701e+05f,
                1.171610122e+05f,  1.170322960e+05f,  1.170429777e+05f,  1.163032202e+05f,  1.161751951e+05f,  1.161858009e+05f,
            };

            float[] y_actual = y.ToArray();

            AssertError.Tolerance(y_expect, y_actual, 1e-7f, 1e-5f, $"mismatch value {inchannels},{outchannels},{kwidth},{kheight},{kdepth},{inwidth},{inheight},{indepth},{batch}");
        }
    }
}
