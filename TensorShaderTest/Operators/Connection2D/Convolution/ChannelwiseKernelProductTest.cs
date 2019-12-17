using System;
using System.Diagnostics;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using TensorShader.Operators.Connection2D;

namespace TensorShaderTest.Operators.Connection2D {
    [TestClass]
    public class ChannelwiseKernelProductTest {
        [TestMethod]
        public void ExecuteTest() {
            float max_err = 0;

            foreach (int batch in new int[] { 1, 2 }) {
                foreach (int channels in new int[] { 1, 2, 3, 4, 5, 10, 15, 20 }) {
                    foreach (int kheight in new int[] { 1, 3, 5 }) {
                        foreach (int kwidth in new int[] { 1, 3, 5 }) {
                            foreach (int inwidth in new int[] { 8, 9, 13, 17 }) {
                                foreach (int inheight in new int[] { 8, 9, 19, 23 }) {
                                    int outwidth = inwidth - kwidth + 1, outheight = inheight - kheight + 1;

                                    float[] xval = (new float[inwidth * inheight * channels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
                                    float[] gyval = (new float[outwidth * outheight * channels * batch]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

                                    Map2D x = new Map2D(channels, inwidth, inheight, batch, xval);
                                    Map2D gy = new Map2D(channels, outwidth, outheight, batch, gyval);

                                    Filter2D gw = Reference(x, gy, kwidth, kheight);

                                    OverflowCheckedTensor x_tensor = new OverflowCheckedTensor(Shape.Map2D(channels, inwidth, inheight, batch), xval);
                                    OverflowCheckedTensor gy_tensor = new OverflowCheckedTensor(Shape.Map2D(channels, outwidth, outheight, batch), gyval);

                                    OverflowCheckedTensor gw_tensor = new OverflowCheckedTensor(Shape.Kernel2D(channels, 1, kwidth, kheight));

                                    ChannelwiseKernelProduct ope = new ChannelwiseKernelProduct(inwidth, inheight, channels, kwidth, kheight, batch);

                                    ope.Execute(x_tensor, gy_tensor, gw_tensor);

                                    float[] gw_expect = gw.ToArray();
                                    float[] gw_actual = gw_tensor.State;

                                    CollectionAssert.AreEqual(xval, x_tensor.State);
                                    CollectionAssert.AreEqual(gyval, gy_tensor.State);

                                    AssertError.Tolerance(gw_expect, gw_actual, 1e-7f, 1e-5f, ref max_err, $"mismatch value {channels},{kwidth},{kheight},{inwidth},{inheight},{batch}");

                                    Console.WriteLine($"pass: {channels},{kwidth},{kheight},{inwidth},{inheight},{batch}");

                                }
                            }
                        }
                    }
                }
            }

            Console.WriteLine($"maxerr:{max_err}");
        }

        [TestMethod]
        public void SpeedTest() {
            int inwidth = 512, inheight = 512, channels = 32, ksize = 3;
            int outwidth = inwidth - ksize + 1, outheight = inheight - ksize + 1;

            OverflowCheckedTensor x_tensor = new OverflowCheckedTensor(Shape.Map2D(channels, inwidth, inheight));
            OverflowCheckedTensor gy_tensor = new OverflowCheckedTensor(Shape.Map2D(channels, outwidth, outheight));

            OverflowCheckedTensor gw_tensor = new OverflowCheckedTensor(Shape.Kernel2D(channels, 1, ksize, ksize));

            ChannelwiseKernelProduct ope = new ChannelwiseKernelProduct(inwidth, inheight, channels, ksize, ksize);

            Stopwatch sw = new Stopwatch();
            sw.Start();

            ope.Execute(x_tensor, gy_tensor, gw_tensor);
            ope.Execute(x_tensor, gy_tensor, gw_tensor);
            ope.Execute(x_tensor, gy_tensor, gw_tensor);
            ope.Execute(x_tensor, gy_tensor, gw_tensor);

            sw.Stop();

            Console.WriteLine($"{sw.ElapsedMilliseconds / 4} msec");
        }

        public static Filter2D Reference(Map2D x, Map2D gy, int kwidth, int kheight) {
            int channels = x.Channels, batch = x.Batch;
            int inw = x.Width, inh = x.Height, outw = gy.Width, outh = gy.Height;

            if (outw != inw - kwidth + 1 || outh != inh - kheight + 1) {
                throw new ArgumentException("mismatch shape");
            }

            Filter2D w = new Filter2D(channels, 1, kwidth, kheight);

            for (int kx, ky = 0; ky < kheight; ky++) {
                for (kx = 0; kx < kwidth; kx++) {
                    for (int th = 0; th < batch; th++) {
                        for (int ch = 0; ch < channels; ch++) {
                            double sum = 0;

                            for (int ix, iy = ky, ox, oy = 0; oy < outh; iy++, oy++) {
                                for (ix = kx, ox = 0; ox < outw; ix++, ox++) {
                                    sum += x[ch, ix, iy, th] * gy[ch, ox, oy, th];
                                }
                            }

                            w[ch, 0, kx, ky] += sum;
                        }

                    }
                }
            }

            return w;
        }

        public static Filter2D OptimizedReference(Map2D x, Map2D gy, int kwidth, int kheight) {
            int channels = x.Channels, batch = x.Batch;
            int inw = x.Width, inh = x.Height, outw = gy.Width, outh = gy.Height;

            if (outw < inw - kwidth + 1 || outh < inh - kheight + 1) {
                throw new ArgumentException("mismatch shape");
            }

            Filter2D w = new Filter2D(channels, 1, kwidth, kheight);

            for (int kx, ky = 0; ky < kheight; ky++) {
                for (kx = 0; kx < kwidth; kx++) {
                    for (int th = 0; th < batch; th++) {
                        for (int ch = 0; ch < channels; ch++) {
                            int filter_idx = ch + (kx + ky * kwidth) * channels;
                            int inmap_org = ch + (kx + ky * inw) * channels + th * inw * inh * channels;
                            int outmap_idx = ch + th * outw * outh * channels;

                            double sum = 0;

                            for (int ox, oy = 0; oy < outh; oy++) {
                                int inmap_idx = inmap_org;

                                for (ox = 0; ox < outw; ox++) {
                                    sum += x[inmap_idx] * gy[outmap_idx];

                                    inmap_idx += channels;
                                    outmap_idx += channels;
                                }

                                inmap_org += channels * inw;
                            }

                            w[filter_idx] += sum;
                        }
                    }
                }
            }

            return w;
        }

        [TestMethod]
        public void ReferenceTest() {
            int channels = 7, kwidth = 3, kheight = 5, inwidth = 13, inheight = 17;
            int outwidth = inwidth - kwidth + 1, outheight = inheight - kheight + 1;

            float[] xval = (new float[inwidth * inheight * channels]).Select((_, idx) => idx * 1e-3f).ToArray();
            float[] gyval = (new float[outwidth * outheight * channels]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

            Map2D x = new Map2D(channels, inwidth, inheight, 1, xval);
            Map2D gy = new Map2D(channels, outwidth, outheight, 1, gyval);

            Filter2D gw = Reference(x, gy, kwidth, kheight);

            float[] gw_expect = {
                2.3519020e+00f,  2.3337370e+00f,  2.3154880e+00f,  2.2971550e+00f,  2.2787380e+00f,  2.2602370e+00f,  2.2416520e+00f,
                2.3958550e+00f,  2.3773960e+00f,  2.3588530e+00f,  2.3402260e+00f,  2.3215150e+00f,  2.3027200e+00f,  2.2838410e+00f,
                2.4398080e+00f,  2.4210550e+00f,  2.4022180e+00f,  2.3832970e+00f,  2.3642920e+00f,  2.3452030e+00f,  2.3260300e+00f,
                2.9232910e+00f,  2.9013040e+00f,  2.8792330e+00f,  2.8570780e+00f,  2.8348390e+00f,  2.8125160e+00f,  2.7901090e+00f,
                2.9672440e+00f,  2.9449630e+00f,  2.9225980e+00f,  2.9001490e+00f,  2.8776160e+00f,  2.8549990e+00f,  2.8322980e+00f,
                3.0111970e+00f,  2.9886220e+00f,  2.9659630e+00f,  2.9432200e+00f,  2.9203930e+00f,  2.8974820e+00f,  2.8744870e+00f,
                3.4946800e+00f,  3.4688710e+00f,  3.4429780e+00f,  3.4170010e+00f,  3.3909400e+00f,  3.3647950e+00f,  3.3385660e+00f,
                3.5386330e+00f,  3.5125300e+00f,  3.4863430e+00f,  3.4600720e+00f,  3.4337170e+00f,  3.4072780e+00f,  3.3807550e+00f,
                3.5825860e+00f,  3.5561890e+00f,  3.5297080e+00f,  3.5031430e+00f,  3.4764940e+00f,  3.4497610e+00f,  3.4229440e+00f,
                4.0660690e+00f,  4.0364380e+00f,  4.0067230e+00f,  3.9769240e+00f,  3.9470410e+00f,  3.9170740e+00f,  3.8870230e+00f,
                4.1100220e+00f,  4.0800970e+00f,  4.0500880e+00f,  4.0199950e+00f,  3.9898180e+00f,  3.9595570e+00f,  3.9292120e+00f,
                4.1539750e+00f,  4.1237560e+00f,  4.0934530e+00f,  4.0630660e+00f,  4.0325950e+00f,  4.0020400e+00f,  3.9714010e+00f,
                4.6374580e+00f,  4.6040050e+00f,  4.5704680e+00f,  4.5368470e+00f,  4.5031420e+00f,  4.4693530e+00f,  4.4354800e+00f,
                4.6814110e+00f,  4.6476640e+00f,  4.6138330e+00f,  4.5799180e+00f,  4.5459190e+00f,  4.5118360e+00f,  4.4776690e+00f,
                4.7253640e+00f,  4.6913230e+00f,  4.6571980e+00f,  4.6229890e+00f,  4.5886960e+00f,  4.5543190e+00f,  4.5198580e+00f,
            };

            float[] gw_actual = gw.ToArray();

            AssertError.Tolerance(gw_expect, gw_actual, 1e-7f, 1e-5f, $"mismatch value {channels},{kwidth},{kheight},{inwidth},{inheight}");
        }

        [TestMethod]
        public void OptimizeTest() {
            float max_err = 0;

            foreach (int batch in new int[] { 1, 2 }) {
                foreach (int channels in new int[] { 1, 2, 3, 4, 5, 10, 15, 20 }) {
                    foreach (int kheight in new int[] { 1, 3, 5 }) {
                        foreach (int kwidth in new int[] { 1, 3, 5 }) {
                            foreach (int inwidth in new int[] { 8, 9, 13, 17 }) {
                                foreach (int inheight in new int[] { 8, 9, 19, 23 }) {
                                    int outwidth = inwidth - kwidth + 1, outheight = inheight - kheight + 1;

                                    float[] xval = (new float[inwidth * inheight * channels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
                                    float[] gyval = (new float[outwidth * outheight * channels * batch]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

                                    Map2D x = new Map2D(channels, inwidth, inheight, batch, xval);
                                    Map2D gy = new Map2D(channels, outwidth, outheight, batch, gyval);

                                    Filter2D gw = Reference(x, gy, kwidth, kheight);
                                    Filter2D gw_optimized = OptimizedReference(x, gy, kwidth, kheight);

                                    float[] gw_expect = gw.ToArray();
                                    float[] gw_actual = gw_optimized.ToArray();

                                    AssertError.Tolerance(gw_expect, gw_actual, 1e-7f, 1e-5f, ref max_err, $"mismatch value {channels},{kwidth},{kheight},{inwidth},{inheight},{batch}");

                                    Console.WriteLine($"pass: {channels},{kwidth},{kheight},{inwidth},{inheight},{batch}");
                                }
                            }
                        }
                    }
                }
            }

            Console.WriteLine($"maxerr:{max_err}");
        }
    }
}
