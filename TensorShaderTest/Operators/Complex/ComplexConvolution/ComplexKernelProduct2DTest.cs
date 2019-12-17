using System;
using System.Diagnostics;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using TensorShader.Operators.ComplexConvolution;

namespace TensorShaderTest.Operators.Complex {
    [TestClass]
    public class ComplexKernelProduct2DTest {
        [TestMethod]
        public void ExecuteTest() {
            float max_err = 0;

            foreach (int batch in new int[] { 1, 2, 3 }) {
                foreach (int inchannels in new int[] { 2, 4, 10, 20 }) {
                    foreach (int outchannels in new int[] { 6, 14 }) {
                        foreach (int kheight in new int[] { 1, 3, 5 }) {
                            foreach (int kwidth in new int[] { 1, 3, 5 }) {
                                foreach (int inwidth in new int[] { 8, 9, 13, 17 }) {
                                    foreach (int inheight in new int[] { 8, 9, 19, 23 }) {
                                        int outwidth = inwidth - kwidth + 1, outheight = inheight - kheight + 1;

                                        float[] xval = (new float[inwidth * inheight * inchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
                                        float[] yval = (new float[outwidth * outheight * outchannels * batch]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

                                        System.Numerics.Complex[] xcval = (new System.Numerics.Complex[xval.Length / 2])
                                            .Select((_, idx) => new System.Numerics.Complex(xval[idx * 2], xval[idx * 2 + 1])).ToArray();

                                        System.Numerics.Complex[] ycval = (new System.Numerics.Complex[yval.Length / 2])
                                            .Select((_, idx) => new System.Numerics.Complex(yval[idx * 2], yval[idx * 2 + 1])).ToArray();

                                        ComplexMap2D x = new ComplexMap2D(inchannels / 2, inwidth, inheight, batch, xcval);
                                        ComplexMap2D y = new ComplexMap2D(outchannels / 2, outwidth, outheight, batch, ycval);

                                        ComplexFilter2D gw = Reference(x, y, kwidth, kheight);

                                        OverflowCheckedTensor x_tensor = new OverflowCheckedTensor(Shape.Map2D(inchannels, inwidth, inheight, batch), xval);
                                        OverflowCheckedTensor y_tensor = new OverflowCheckedTensor(Shape.Map2D(outchannels, outwidth, outheight, batch), yval);

                                        OverflowCheckedTensor gw_tensor = new OverflowCheckedTensor(Shape.Kernel2D(inchannels, outchannels / 2, kwidth, kheight));

                                        ComplexKernelProduct2D ope = new ComplexKernelProduct2D(inwidth, inheight, inchannels, outchannels, kwidth, kheight, transpose: false, batch);

                                        ope.Execute(x_tensor, y_tensor, gw_tensor);

                                        float[] gw_expect = gw.ToArray();
                                        float[] gw_actual = gw_tensor.State;

                                        CollectionAssert.AreEqual(xval, x_tensor.State);
                                        CollectionAssert.AreEqual(yval, y_tensor.State);

                                        AssertError.Tolerance(gw_expect, gw_actual, 1e-7f, 1e-5f, ref max_err, $"mismatch value {inchannels},{outchannels},{kwidth},{kheight},{inwidth},{inheight},{batch}");

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
        public void OverflowTest() {
            foreach (bool transpose in new bool[] { false, true }) {
                foreach (int batch in new int[] { 1, 2, 3 }) {
                    foreach (int inchannels in new int[] { 2, 4, 10, 20 }) {
                        foreach (int outchannels in new int[] { 6, 14 }) {
                            foreach (int kheight in new int[] { 1, 3, 5 }) {
                                foreach (int kwidth in new int[] { 1, 3, 5 }) {
                                    foreach (int inwidth in new int[] { 8, 9, 13, 17 }) {
                                        foreach (int inheight in new int[] { 8, 9, 19, 23 }) {
                                            int outwidth = inwidth - kwidth + 1, outheight = inheight - kheight + 1;

                                            float[] xval = (new float[inwidth * inheight * inchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
                                            float[] yval = (new float[outwidth * outheight * outchannels * batch]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

                                            OverflowCheckedTensor x_tensor = new OverflowCheckedTensor(Shape.Map2D(inchannels, inwidth, inheight, batch), xval);
                                            OverflowCheckedTensor y_tensor = new OverflowCheckedTensor(Shape.Map2D(outchannels, outwidth, outheight, batch), yval);

                                            OverflowCheckedTensor gw_tensor = new OverflowCheckedTensor(Shape.Kernel2D(inchannels, outchannels / 2, kwidth, kheight));

                                            ComplexKernelProduct2D ope = new ComplexKernelProduct2D(inwidth, inheight, inchannels, outchannels, kwidth, kheight, transpose, batch);

                                            ope.Execute(x_tensor, y_tensor, gw_tensor);

                                            CollectionAssert.AreEqual(xval, x_tensor.State);
                                            CollectionAssert.AreEqual(yval, y_tensor.State);

                                            gw_tensor.CheckOverflow();

                                            Console.WriteLine($"pass: {inchannels},{outchannels},{kwidth},{kheight},{inwidth},{inheight},{batch},{transpose}");

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
            int inwidth = 512, inheight = 512, inchannels = 32, outchannels = 32, ksize = 3;
            int outwidth = inwidth - ksize + 1, outheight = inheight - ksize + 1;

            OverflowCheckedTensor x_tensor = new OverflowCheckedTensor(Shape.Map2D(inchannels, inwidth, inheight));
            OverflowCheckedTensor y_tensor = new OverflowCheckedTensor(Shape.Map2D(outchannels, outwidth, outheight));

            OverflowCheckedTensor gw_tensor = new OverflowCheckedTensor(Shape.Kernel2D(inchannels, outchannels / 2, ksize, ksize));

            ComplexKernelProduct2D ope = new ComplexKernelProduct2D(inwidth, inheight, inchannels, outchannels, ksize, ksize);

            Stopwatch sw = new Stopwatch();
            sw.Start();

            ope.Execute(x_tensor, y_tensor, gw_tensor);
            ope.Execute(x_tensor, y_tensor, gw_tensor);
            ope.Execute(x_tensor, y_tensor, gw_tensor);
            ope.Execute(x_tensor, y_tensor, gw_tensor);

            sw.Stop();

            Console.WriteLine($"{sw.ElapsedMilliseconds / 4} msec");
        }

        public static ComplexFilter2D Reference(ComplexMap2D x, ComplexMap2D gy, int kwidth, int kheight) {
            int inchannels = x.Channels, outchannels = gy.Channels, batch = x.Batch;
            int inw = x.Width, inh = x.Height, outw = gy.Width, outh = gy.Height;

            if (outw != inw - kwidth + 1 || outh != inh - kheight + 1) {
                throw new ArgumentException("mismatch shape");
            }

            ComplexFilter2D w = new ComplexFilter2D(inchannels, outchannels, kwidth, kheight);

            Func<System.Numerics.Complex, System.Numerics.Complex, System.Numerics.Complex> mul_grad = (z1, z2) => {
                return new System.Numerics.Complex(z1.Real * z2.Real + z1.Imaginary * z2.Imaginary, z1.Imaginary * z2.Real - z1.Real * z2.Imaginary);
            };

            for (int kx, ky = 0; ky < kheight; ky++) {
                for (kx = 0; kx < kwidth; kx++) {
                    for (int th = 0; th < batch; th++) {
                        for (int inch, outch = 0; outch < outchannels; outch++) {
                            for (inch = 0; inch < inchannels; inch++) {
                                System.Numerics.Complex sum = 0;

                                for (int ix, iy = ky, ox, oy = 0; oy < outh; iy++, oy++) {
                                    for (ix = kx, ox = 0; ox < outw; ix++, ox++) {
                                        sum += mul_grad(gy[outch, ox, oy, th], x[inch, ix, iy, th]);
                                    }
                                }

                                w[inch, outch, kx, ky] += sum;
                            }
                        }

                    }
                }
            }

            return w;
        }

        [TestMethod]
        public void ReferenceTest() {
            int inchannels = 6, outchannels = 8, kwidth = 3, kheight = 5, inwidth = 13, inheight = 17;
            int outwidth = inwidth - kwidth + 1, outheight = inheight - kheight + 1, batch = 1;

            float[] xval = (new float[inwidth * inheight * inchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
            float[] yval = (new float[outwidth * outheight * outchannels * batch]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

            System.Numerics.Complex[] xcval = (new System.Numerics.Complex[xval.Length / 2])
                .Select((_, idx) => new System.Numerics.Complex(xval[idx * 2], xval[idx * 2 + 1])).ToArray();

            System.Numerics.Complex[] ycval = (new System.Numerics.Complex[yval.Length / 2])
                .Select((_, idx) => new System.Numerics.Complex(yval[idx * 2], yval[idx * 2 + 1])).ToArray();

            ComplexMap2D x = new ComplexMap2D(inchannels / 2, inwidth, inheight, batch, xcval);
            ComplexMap2D y = new ComplexMap2D(outchannels / 2, outwidth, outheight, batch, ycval);

            ComplexFilter2D gw = Reference(x, y, kwidth, kheight);

            float[] gw_expect = {
                4.600008000e+00f,  -2.809800000e-02f,  4.628652000e+00f,  -2.818200000e-02f,  4.657296000e+00f,  -2.826600000e-02f,
                4.516260000e+00f,  -2.801400000e-02f,  4.544568000e+00f,  -2.809800000e-02f,  4.572876000e+00f,  -2.818200000e-02f,
                4.432512000e+00f,  -2.793000000e-02f,  4.460484000e+00f,  -2.801400000e-02f,  4.488456000e+00f,  -2.809800000e-02f,
                4.348764000e+00f,  -2.784600000e-02f,  4.376400000e+00f,  -2.793000000e-02f,  4.404036000e+00f,  -2.801400000e-02f,
                4.685940000e+00f,  -2.835000000e-02f,  4.714584000e+00f,  -2.843400000e-02f,  4.743228000e+00f,  -2.851800000e-02f,
                4.601184000e+00f,  -2.826600000e-02f,  4.629492000e+00f,  -2.835000000e-02f,  4.657800000e+00f,  -2.843400000e-02f,
                4.516428000e+00f,  -2.818200000e-02f,  4.544400000e+00f,  -2.826600000e-02f,  4.572372000e+00f,  -2.835000000e-02f,
                4.431672000e+00f,  -2.809800000e-02f,  4.459308000e+00f,  -2.818200000e-02f,  4.486944000e+00f,  -2.826600000e-02f,
                4.771872000e+00f,  -2.860200000e-02f,  4.800516000e+00f,  -2.868600000e-02f,  4.829160000e+00f,  -2.877000000e-02f,
                4.686108000e+00f,  -2.851800000e-02f,  4.714416000e+00f,  -2.860200000e-02f,  4.742724000e+00f,  -2.868600000e-02f,
                4.600344000e+00f,  -2.843400000e-02f,  4.628316000e+00f,  -2.851800000e-02f,  4.656288000e+00f,  -2.860200000e-02f,
                4.514580000e+00f,  -2.835000000e-02f,  4.542216000e+00f,  -2.843400000e-02f,  4.569852000e+00f,  -2.851800000e-02f,
                5.717124000e+00f,  -3.137400000e-02f,  5.745768000e+00f,  -3.145800000e-02f,  5.774412000e+00f,  -3.154200000e-02f,
                5.620272000e+00f,  -3.129000000e-02f,  5.648580000e+00f,  -3.137400000e-02f,  5.676888000e+00f,  -3.145800000e-02f,
                5.523420000e+00f,  -3.120600000e-02f,  5.551392000e+00f,  -3.129000000e-02f,  5.579364000e+00f,  -3.137400000e-02f,
                5.426568000e+00f,  -3.112200000e-02f,  5.454204000e+00f,  -3.120600000e-02f,  5.481840000e+00f,  -3.129000000e-02f,
                5.803056000e+00f,  -3.162600000e-02f,  5.831700000e+00f,  -3.171000000e-02f,  5.860344000e+00f,  -3.179400000e-02f,
                5.705196000e+00f,  -3.154200000e-02f,  5.733504000e+00f,  -3.162600000e-02f,  5.761812000e+00f,  -3.171000000e-02f,
                5.607336000e+00f,  -3.145800000e-02f,  5.635308000e+00f,  -3.154200000e-02f,  5.663280000e+00f,  -3.162600000e-02f,
                5.509476000e+00f,  -3.137400000e-02f,  5.537112000e+00f,  -3.145800000e-02f,  5.564748000e+00f,  -3.154200000e-02f,
                5.888988000e+00f,  -3.187800000e-02f,  5.917632000e+00f,  -3.196200000e-02f,  5.946276000e+00f,  -3.204600000e-02f,
                5.790120000e+00f,  -3.179400000e-02f,  5.818428000e+00f,  -3.187800000e-02f,  5.846736000e+00f,  -3.196200000e-02f,
                5.691252000e+00f,  -3.171000000e-02f,  5.719224000e+00f,  -3.179400000e-02f,  5.747196000e+00f,  -3.187800000e-02f,
                5.592384000e+00f,  -3.162600000e-02f,  5.620020000e+00f,  -3.171000000e-02f,  5.647656000e+00f,  -3.179400000e-02f,
                6.834240000e+00f,  -3.465000000e-02f,  6.862884000e+00f,  -3.473400000e-02f,  6.891528000e+00f,  -3.481800000e-02f,
                6.724284000e+00f,  -3.456600000e-02f,  6.752592000e+00f,  -3.465000000e-02f,  6.780900000e+00f,  -3.473400000e-02f,
                6.614328000e+00f,  -3.448200000e-02f,  6.642300000e+00f,  -3.456600000e-02f,  6.670272000e+00f,  -3.465000000e-02f,
                6.504372000e+00f,  -3.439800000e-02f,  6.532008000e+00f,  -3.448200000e-02f,  6.559644000e+00f,  -3.456600000e-02f,
                6.920172000e+00f,  -3.490200000e-02f,  6.948816000e+00f,  -3.498600000e-02f,  6.977460000e+00f,  -3.507000000e-02f,
                6.809208000e+00f,  -3.481800000e-02f,  6.837516000e+00f,  -3.490200000e-02f,  6.865824000e+00f,  -3.498600000e-02f,
                6.698244000e+00f,  -3.473400000e-02f,  6.726216000e+00f,  -3.481800000e-02f,  6.754188000e+00f,  -3.490200000e-02f,
                6.587280000e+00f,  -3.465000000e-02f,  6.614916000e+00f,  -3.473400000e-02f,  6.642552000e+00f,  -3.481800000e-02f,
                7.006104000e+00f,  -3.515400000e-02f,  7.034748000e+00f,  -3.523800000e-02f,  7.063392000e+00f,  -3.532200000e-02f,
                6.894132000e+00f,  -3.507000000e-02f,  6.922440000e+00f,  -3.515400000e-02f,  6.950748000e+00f,  -3.523800000e-02f,
                6.782160000e+00f,  -3.498600000e-02f,  6.810132000e+00f,  -3.507000000e-02f,  6.838104000e+00f,  -3.515400000e-02f,
                6.670188000e+00f,  -3.490200000e-02f,  6.697824000e+00f,  -3.498600000e-02f,  6.725460000e+00f,  -3.507000000e-02f,
                7.951356000e+00f,  -3.792600000e-02f,  7.980000000e+00f,  -3.801000000e-02f,  8.008644000e+00f,  -3.809400000e-02f,
                7.828296000e+00f,  -3.784200000e-02f,  7.856604000e+00f,  -3.792600000e-02f,  7.884912000e+00f,  -3.801000000e-02f,
                7.705236000e+00f,  -3.775800000e-02f,  7.733208000e+00f,  -3.784200000e-02f,  7.761180000e+00f,  -3.792600000e-02f,
                7.582176000e+00f,  -3.767400000e-02f,  7.609812000e+00f,  -3.775800000e-02f,  7.637448000e+00f,  -3.784200000e-02f,
                8.037288000e+00f,  -3.817800000e-02f,  8.065932000e+00f,  -3.826200000e-02f,  8.094576000e+00f,  -3.834600000e-02f,
                7.913220000e+00f,  -3.809400000e-02f,  7.941528000e+00f,  -3.817800000e-02f,  7.969836000e+00f,  -3.826200000e-02f,
                7.789152000e+00f,  -3.801000000e-02f,  7.817124000e+00f,  -3.809400000e-02f,  7.845096000e+00f,  -3.817800000e-02f,
                7.665084000e+00f,  -3.792600000e-02f,  7.692720000e+00f,  -3.801000000e-02f,  7.720356000e+00f,  -3.809400000e-02f,
                8.123220000e+00f,  -3.843000000e-02f,  8.151864000e+00f,  -3.851400000e-02f,  8.180508000e+00f,  -3.859800000e-02f,
                7.998144000e+00f,  -3.834600000e-02f,  8.026452000e+00f,  -3.843000000e-02f,  8.054760000e+00f,  -3.851400000e-02f,
                7.873068000e+00f,  -3.826200000e-02f,  7.901040000e+00f,  -3.834600000e-02f,  7.929012000e+00f,  -3.843000000e-02f,
                7.747992000e+00f,  -3.817800000e-02f,  7.775628000e+00f,  -3.826200000e-02f,  7.803264000e+00f,  -3.834600000e-02f,
                9.068472000e+00f,  -4.120200000e-02f,  9.097116000e+00f,  -4.128600000e-02f,  9.125760000e+00f,  -4.137000000e-02f,
                8.932308000e+00f,  -4.111800000e-02f,  8.960616000e+00f,  -4.120200000e-02f,  8.988924000e+00f,  -4.128600000e-02f,
                8.796144000e+00f,  -4.103400000e-02f,  8.824116000e+00f,  -4.111800000e-02f,  8.852088000e+00f,  -4.120200000e-02f,
                8.659980000e+00f,  -4.095000000e-02f,  8.687616000e+00f,  -4.103400000e-02f,  8.715252000e+00f,  -4.111800000e-02f,
                9.154404000e+00f,  -4.145400000e-02f,  9.183048000e+00f,  -4.153800000e-02f,  9.211692000e+00f,  -4.162200000e-02f,
                9.017232000e+00f,  -4.137000000e-02f,  9.045540000e+00f,  -4.145400000e-02f,  9.073848000e+00f,  -4.153800000e-02f,
                8.880060000e+00f,  -4.128600000e-02f,  8.908032000e+00f,  -4.137000000e-02f,  8.936004000e+00f,  -4.145400000e-02f,
                8.742888000e+00f,  -4.120200000e-02f,  8.770524000e+00f,  -4.128600000e-02f,  8.798160000e+00f,  -4.137000000e-02f,
                9.240336000e+00f,  -4.170600000e-02f,  9.268980000e+00f,  -4.179000000e-02f,  9.297624000e+00f,  -4.187400000e-02f,
                9.102156000e+00f,  -4.162200000e-02f,  9.130464000e+00f,  -4.170600000e-02f,  9.158772000e+00f,  -4.179000000e-02f,
                8.963976000e+00f,  -4.153800000e-02f,  8.991948000e+00f,  -4.162200000e-02f,  9.019920000e+00f,  -4.170600000e-02f,
                8.825796000e+00f,  -4.145400000e-02f,  8.853432000e+00f,  -4.153800000e-02f,  8.881068000e+00f,  -4.162200000e-02f,
            };

            float[] gw_actual = gw.ToArray();

            AssertError.Tolerance(gw_expect, gw_actual, 1e-7f, 1e-5f, $"mismatch value {inchannels},{outchannels},{kwidth},{kheight},{inwidth},{inheight},{batch}");
        }
    }
}
