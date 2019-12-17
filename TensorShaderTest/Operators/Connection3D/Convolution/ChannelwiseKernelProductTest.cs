using System;
using System.Diagnostics;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using TensorShader.Operators.Connection3D;

namespace TensorShaderTest.Operators.Connection3D {
    [TestClass]
    public class ChannelwiseKernelProductTest {
        [TestMethod]
        public void ExecuteTest() {
            float max_err = 0;

            foreach (int batch in new int[] { 1, 2 }) {
                foreach (int channels in new int[] { 1, 2, 3, 4, 5, 10, 15, 20 }) {
                    foreach ((int kwidth, int kheight, int kdepth) in new (int, int, int)[] { (1, 1, 1), (3, 3, 3), (5, 5, 5), (1, 3, 5), (3, 5, 1), (5, 1, 3) }) {
                        foreach ((int stride, int inwidth, int inheight, int indepth) in new (int, int, int, int)[] { (1, 13, 13, 13), (2, 17, 17, 17), (3, 19, 19, 19), (1, 17, 19, 13), (2, 13, 17, 19), (3, 19, 13, 17) }) {
                            int outwidth = inwidth - kwidth + 1, outheight = inheight - kheight + 1, outdepth = indepth - kdepth + 1;

                            float[] xval = (new float[inwidth * inheight * indepth * channels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
                            float[] gyval = (new float[outwidth * outheight * outdepth * channels * batch]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

                            Map3D x = new Map3D(channels, inwidth, inheight, indepth, batch, xval);
                            Map3D gy = new Map3D(channels, outwidth, outheight, outdepth, batch, gyval);

                            Filter3D gw = Reference(x, gy, kwidth, kheight, kdepth, stride);

                            OverflowCheckedTensor x_tensor = new OverflowCheckedTensor(Shape.Map3D(channels, inwidth, inheight, indepth, batch), xval);
                            OverflowCheckedTensor gy_tensor = new OverflowCheckedTensor(Shape.Map3D(channels, outwidth, outheight, outdepth, batch), gyval);

                            OverflowCheckedTensor gw_tensor = new OverflowCheckedTensor(Shape.Kernel3D(channels, 1, kwidth, kheight, kdepth));

                            ChannelwiseKernelProduct ope = new ChannelwiseKernelProduct(inwidth, inheight, indepth, channels, kwidth, kheight, kdepth, stride, batch);

                            ope.Execute(x_tensor, gy_tensor, gw_tensor);

                            float[] gw_expect = gw.ToArray();
                            float[] gw_actual = gw_tensor.State;

                            CollectionAssert.AreEqual(xval, x_tensor.State);
                            CollectionAssert.AreEqual(gyval, gy_tensor.State);

                            AssertError.Tolerance(gw_expect, gw_actual, 1e-7f, 1e-5f, ref max_err, $"mismatch value {channels},{kwidth},{kheight},{kdepth},{stride},{inwidth},{inheight},{indepth},{batch}");

                            Console.WriteLine($"pass: {channels},{kwidth},{kheight},{kdepth},{stride},{inwidth},{inheight},{indepth},{batch}");
                        }
                    }
                }
            }

            Console.WriteLine($"maxerr:{max_err}");
        }

        [TestMethod]
        public void SpeedTest() {
            int inwidth = 64, inheight = 64, indepth = 64, channels = 32, ksize = 3, stride = 2;
            int outwidth = (inwidth - ksize) / stride + 1, outheight = (inheight - ksize) / stride + 1, outdepth = (indepth - ksize) / stride + 1;

            OverflowCheckedTensor x_tensor = new OverflowCheckedTensor(Shape.Map3D(channels, inwidth, inheight, indepth));
            OverflowCheckedTensor gy_tensor = new OverflowCheckedTensor(Shape.Map3D(channels, outwidth, outheight, outdepth));

            OverflowCheckedTensor gw_tensor = new OverflowCheckedTensor(Shape.Kernel3D(channels, 1, ksize, ksize, ksize));

            ChannelwiseKernelProduct ope = new ChannelwiseKernelProduct(inwidth, inheight, indepth, channels, ksize, ksize, ksize, stride);

            Stopwatch sw = new Stopwatch();
            sw.Start();

            ope.Execute(x_tensor, gy_tensor, gw_tensor);
            ope.Execute(x_tensor, gy_tensor, gw_tensor);
            ope.Execute(x_tensor, gy_tensor, gw_tensor);
            ope.Execute(x_tensor, gy_tensor, gw_tensor);

            sw.Stop();

            Console.WriteLine($"{sw.ElapsedMilliseconds / 4} msec");
        }

        public static Filter3D Reference(Map3D x, Map3D gy, int kwidth, int kheight, int kdepth, int stride) {
            int channels = x.Channels, batch = x.Batch;
            int inw = x.Width, inh = x.Height, ind = x.Depth, outw = gy.Width, outh = gy.Height, outd = gy.Depth;

            if (outw != inw - kwidth + 1 || outh != inh - kheight + 1 || outd != (ind - kdepth) / stride + 1) {
                throw new ArgumentException("mismatch shape");
            }

            Filter3D w = new Filter3D(channels, 1, kwidth, kheight, kdepth);

            for (int kx, ky, kz = 0; kz < kdepth; kz++) {
                for (ky = 0; ky < kheight; ky++) {
                    for (kx = 0; kx < kwidth; kx++) {
                        for (int th = 0; th < batch; th++) {
                            for (int ch = 0; ch < channels; ch++) {
                                double sum = 0;

                                for (int ix, iy, iz = kz, ox, oy, oz = 0; oz < outd; iz += stride, oz++) {
                                    for (iy = ky, oy = 0; oy < outh; iy += stride, oy++) {
                                        for (ix = kx, ox = 0; ox < outw; ix += stride, ox++) {
                                            sum += x[ch, ix, iy, iz, th] * gy[ch, ox, oy, oz, th];
                                        }
                                    }
                                }

                                w[ch, 0, kx, ky, kz] += sum;
                            }
                        }
                    }
                }
            }

            return w;
        }

        public static Filter3D OptimizedReference(Map3D x, Map3D gy, int kwidth, int kheight, int kdepth, int stride) {
            int channels = x.Channels, batch = x.Batch;
            int inw = x.Width, inh = x.Height, ind = x.Depth, outw = gy.Width, outh = gy.Height, outd = gy.Depth;

            if (outw != inw - kwidth + 1 || outh != inh - kheight + 1 || outd != (ind - kdepth) / stride + 1) {
                throw new ArgumentException("mismatch shape");
            }

            Filter3D w = new Filter3D(channels, 1, kwidth, kheight, kdepth);

            for (int kx, ky, kz = 0; kz < kdepth; kz++) {
                for (ky = 0; ky < kheight; ky++) {
                    for (kx = 0; kx < kwidth; kx++) {
                        for (int th = 0; th < batch; th++) {
                            for (int ch = 0; ch < channels; ch++) {
                                int filter_idx = ch + (kx + ky * kwidth + kz * kwidth * kheight) * channels;
                                int inmap_org = ch + (kx + ky * inw + kz * inw * inh) * channels + th * inw * inh * ind * channels;
                                int outmap_idx = ch + th * outw * outh * outd * channels;

                                double sum = 0;

                                for (int ox, oy, oz = 0; oz < outd; oz++) {
                                    int inmap_car = inmap_org;

                                    for (oy = 0; oy < outh; oy++) {
                                        int inmap_idx = inmap_car;

                                        for (ox = 0; ox < outw; ox++) {
                                            sum += x[inmap_idx] * gy[outmap_idx];

                                            inmap_idx += channels * stride;
                                            outmap_idx += channels;
                                        }

                                        inmap_car += channels * inw * stride;
                                    }

                                    inmap_org += channels * inw * inh * stride;
                                }

                                w[filter_idx] += sum;
                            }
                        }
                    }
                }
            }

            return w;
        }

        [TestMethod]
        public void ReferenceTest() {
            int channels = 2, kwidth = 3, kheight = 5, kdepth = 7, stride = 2, inwidth = 13, inheight = 12, indepth = 11;
            int outwidth = inwidth - kwidth + 1, outheight = inheight - kheight + 1, outdepth = indepth - kdepth + 1;

            float[] xval = (new float[inwidth * inheight * indepth * channels]).Select((_, idx) => idx * 1e-3f).ToArray();
            float[] gyval = (new float[outwidth * outheight * outdepth * channels]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

            Map3D x = new Map3D(channels, inwidth, inheight, indepth, 1, xval);
            Map3D gy = new Map3D(channels, outwidth, outheight, outdepth, 1, gyval);

            Filter3D gw = Reference(x, gy, kwidth, kheight, kdepth, stride);

            float[] gw_expect = {
                2.19547200e+00f,  2.14931989e+00f,  2.20584011e+00f,  2.15954399e+00f,  2.21620798e+00f,  2.16976810e+00f,  2.33025599e+00f,  2.28223205e+00f,  2.34062409e+00f,  2.29245591e+00f,
                2.35099196e+00f,  2.30268002e+00f,  2.46503997e+00f,  2.41514397e+00f,  2.47540808e+00f,  2.42536807e+00f,  2.48577595e+00f,  2.43559194e+00f,  2.59982395e+00f,  2.54805589e+00f,
                2.61019206e+00f,  2.55827999e+00f,  2.62055993e+00f,  2.56850410e+00f,  2.73460793e+00f,  2.68096805e+00f,  2.74497604e+00f,  2.69119191e+00f,  2.75534391e+00f,  2.70141602e+00f,
                3.81288004e+00f,  3.74426389e+00f,  3.82324791e+00f,  3.75448799e+00f,  3.83361602e+00f,  3.76471210e+00f,  3.94766402e+00f,  3.87717605e+00f,  3.95803189e+00f,  3.88739991e+00f,
                3.96840000e+00f,  3.89762402e+00f,  4.08244801e+00f,  4.01008797e+00f,  4.09281588e+00f,  4.02031183e+00f,  4.10318422e+00f,  4.03053617e+00f,  4.21723223e+00f,  4.14300013e+00f,
                4.22760010e+00f,  4.15322399e+00f,  4.23796797e+00f,  4.16344786e+00f,  4.35201597e+00f,  4.27591181e+00f,  4.36238384e+00f,  4.28613615e+00f,  4.37275219e+00f,  4.29636002e+00f,
                5.43028784e+00f,  5.33920813e+00f,  5.44065619e+00f,  5.34943199e+00f,  5.45102406e+00f,  5.35965586e+00f,  5.56507206e+00f,  5.47211981e+00f,  5.57543993e+00f,  5.48234415e+00f,
                5.58580780e+00f,  5.49256802e+00f,  5.69985580e+00f,  5.60503197e+00f,  5.71022415e+00f,  5.61525583e+00f,  5.72059202e+00f,  5.62548018e+00f,  5.83464003e+00f,  5.73794413e+00f,
                5.84500790e+00f,  5.74816799e+00f,  5.85537577e+00f,  5.75839186e+00f,  5.96942377e+00f,  5.87085581e+00f,  5.97979212e+00f,  5.88108015e+00f,  5.99015999e+00f,  5.89130402e+00f,
                7.04769611e+00f,  6.93415213e+00f,  7.05806398e+00f,  6.94437599e+00f,  7.06843185e+00f,  6.95459986e+00f,  7.18247986e+00f,  7.06706381e+00f,  7.19284821e+00f,  7.07728815e+00f,
                7.20321608e+00f,  7.08751202e+00f,  7.31726408e+00f,  7.19997597e+00f,  7.32763195e+00f,  7.21019983e+00f,  7.33799982e+00f,  7.22042418e+00f,  7.45204782e+00f,  7.33288813e+00f,
                7.46241617e+00f,  7.34311199e+00f,  7.47278404e+00f,  7.35333586e+00f,  7.58683205e+00f,  7.46579981e+00f,  7.59719992e+00f,  7.47602415e+00f,  7.60756779e+00f,  7.48624802e+00f,
                8.66510391e+00f,  8.52909565e+00f,  8.67547226e+00f,  8.53931999e+00f,  8.68583965e+00f,  8.54954433e+00f,  8.79988766e+00f,  8.66200829e+00f,  8.81025600e+00f,  8.67223167e+00f,
                8.82062435e+00f,  8.68245602e+00f,  8.93467236e+00f,  8.79491997e+00f,  8.94503975e+00f,  8.80514431e+00f,  8.95540810e+00f,  8.81536770e+00f,  9.06945610e+00f,  8.92783165e+00f,
                9.07982445e+00f,  8.93805599e+00f,  9.09019184e+00f,  8.94828033e+00f,  9.20423985e+00f,  9.06074429e+00f,  9.21460819e+00f,  9.07096767e+00f,  9.22497559e+00f,  9.08119202e+00f,
                1.02825117e+01f,  1.01240396e+01f,  1.02928801e+01f,  1.01342640e+01f,  1.03032484e+01f,  1.01444883e+01f,  1.04172964e+01f,  1.02569523e+01f,  1.04276638e+01f,  1.02671757e+01f,
                1.04380322e+01f,  1.02774000e+01f,  1.05520802e+01f,  1.03898640e+01f,  1.05624475e+01f,  1.04000883e+01f,  1.05728159e+01f,  1.04103117e+01f,  1.06868639e+01f,  1.05227757e+01f,
                1.06972322e+01f,  1.05330000e+01f,  1.07075996e+01f,  1.05432243e+01f,  1.08216476e+01f,  1.06556883e+01f,  1.08320160e+01f,  1.06659117e+01f,  1.08423843e+01f,  1.06761360e+01f,
                1.18999205e+01f,  1.17189837e+01f,  1.19102879e+01f,  1.17292080e+01f,  1.19206562e+01f,  1.17394323e+01f,  1.20347042e+01f,  1.18518963e+01f,  1.20450716e+01f,  1.18621197e+01f,
                1.20554399e+01f,  1.18723440e+01f,  1.21694880e+01f,  1.19848080e+01f,  1.21798563e+01f,  1.19950323e+01f,  1.21902237e+01f,  1.20052557e+01f,  1.23042717e+01f,  1.21177197e+01f,
                1.23146400e+01f,  1.21279440e+01f,  1.23250084e+01f,  1.21381683e+01f,  1.24390564e+01f,  1.22506323e+01f,  1.24494238e+01f,  1.22608557e+01f,  1.24597921e+01f,  1.22710800e+01f,
            };

            float[] gw_actual = gw.ToArray();

            AssertError.Tolerance(gw_expect, gw_actual, 1e-7f, 1e-5f, $"mismatch value {channels},{kwidth},{kheight},{kdepth},{stride},{inwidth},{inheight},{indepth}");
        }

        [TestMethod]
        public void OptimizeTest() {
            float max_err = 0;

            foreach (int batch in new int[] { 1, 2 }) {
                foreach (int channels in new int[] { 1, 2, 3, 4, 5, 10, 15, 20 }) {
                    foreach ((int kwidth, int kheight, int kdepth) in new (int, int, int)[] { (1, 1, 1), (3, 3, 3), (5, 5, 5), (1, 3, 5), (3, 5, 1), (5, 1, 3) }) {
                        foreach ((int stride, int inwidth, int inheight, int indepth) in new (int, int, int, int)[] { (1, 13, 13, 13), (2, 17, 17, 17), (3, 19, 19, 19), (1, 17, 19, 13), (2, 13, 17, 19), (3, 19, 13, 17) }) {
                            int outwidth = inwidth - kwidth + 1, outheight = inheight - kheight + 1, outdepth = indepth - kdepth + 1;

                            float[] xval = (new float[inwidth * inheight * indepth * channels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
                            float[] gyval = (new float[outwidth * outheight * outdepth * channels * batch]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

                            Map3D x = new Map3D(channels, inwidth, inheight, indepth, batch, xval);
                            Map3D gy = new Map3D(channels, outwidth, outheight, outdepth, batch, gyval);

                            Filter3D gw = Reference(x, gy, kwidth, kheight, kdepth, stride);
                            Filter3D gw_optimized = OptimizedReference(x, gy, kwidth, kheight, kdepth, stride);

                            float[] gw_expect = gw.ToArray();
                            float[] gw_actual = gw_optimized.ToArray();

                            AssertError.Tolerance(gw_expect, gw_actual, 1e-7f, 1e-5f, ref max_err, $"mismatch value {channels},{kwidth},{kheight},{kdepth},{stride},{inwidth},{inheight},{indepth},{batch}");

                            Console.WriteLine($"pass: {channels},{kwidth},{kheight},{kdepth},{stride},{inwidth},{inheight},{indepth},{batch}");
                        }
                    }
                }
            }

            Console.WriteLine($"maxerr:{max_err}");
        }
    }
}
