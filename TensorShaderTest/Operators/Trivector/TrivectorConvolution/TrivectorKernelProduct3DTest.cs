using System;
using System.Diagnostics;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using TensorShader.Operators.TrivectorConvolution;

namespace TensorShaderTest.Operators.Trivector {
    [TestClass]
    public class TrivectorKernelProduct3DTest {
        [TestMethod]
        public void ExecuteTest() {
            float max_err = 0;

            foreach (int batch in new int[] { 1, 2 }) {
                foreach (int inchannels in new int[] { 3, 6, 9, 12 }) {
                    foreach (int outchannels in new int[] { 3, 6, 9, 12 }) {
                        foreach ((int kwidth, int kheight, int kdepth) in new (int, int, int)[] { (1, 1, 1), (3, 3, 3), (5, 5, 5), (1, 3, 5), (3, 5, 1), (5, 1, 3) }) {
                            foreach ((int stride, int inwidth, int inheight, int indepth) in new (int, int, int, int)[] { (1, 13, 13, 13), (2, 17, 17, 17), (3, 19, 19, 19), (1, 17, 19, 13), (2, 13, 17, 19), (3, 19, 13, 17) }) {
                                int outwidth = (inwidth - kwidth) / stride + 1, outheight = (inheight - kheight) / stride + 1, outdepth = (indepth - kdepth) / stride + 1;

                                float[] xval = (new float[inwidth * inheight * indepth * inchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
                                float[] yval = (new float[outwidth * outheight * outdepth * outchannels * batch]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();
                                float[] wval = (new float[kwidth * kheight * kdepth * inchannels * outchannels / 9 * 4]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

                                Trivector[] xcval = (new Trivector[xval.Length / 3])
                                    .Select((_, idx) => new Trivector(xval[idx * 3], xval[idx * 3 + 1], xval[idx * 3 + 2])).ToArray();

                                Trivector[] ycval = (new Trivector[yval.Length / 3])
                                    .Select((_, idx) => new Trivector(yval[idx * 3], yval[idx * 3 + 1], yval[idx * 3 + 2])).ToArray();

                                Quaternion.Quaternion[] wcval = (new Quaternion.Quaternion[wval.Length / 4])
                                    .Select((_, idx) => new Quaternion.Quaternion(wval[idx * 4], wval[idx * 4 + 1], wval[idx * 4 + 2], wval[idx * 4 + 3])).ToArray();

                                TrivectorMap3D x = new TrivectorMap3D(inchannels / 3, inwidth, inheight, indepth, batch, xcval);
                                TrivectorMap3D y = new TrivectorMap3D(outchannels / 3, outwidth, outheight, outdepth, batch, ycval);
                                Quaternion.QuaternionFilter3D w = new Quaternion.QuaternionFilter3D(inchannels / 3, outchannels / 3, kwidth, kheight, kdepth, wcval);

                                Quaternion.QuaternionFilter3D gw = Reference(x, y, w, kwidth, kheight, kdepth, stride);

                                OverflowCheckedTensor x_tensor = new OverflowCheckedTensor(Shape.Map3D(inchannels, inwidth, inheight, indepth, batch), xval);
                                OverflowCheckedTensor y_tensor = new OverflowCheckedTensor(Shape.Map3D(outchannels, outwidth, outheight, outdepth, batch), yval);
                                OverflowCheckedTensor w_tensor = new OverflowCheckedTensor(Shape.Kernel3D(inchannels / 3 * 4, outchannels / 3, kwidth, kheight, kdepth), wval);

                                OverflowCheckedTensor gw_tensor = new OverflowCheckedTensor(Shape.Kernel3D(inchannels / 3 * 4, outchannels / 3, kwidth, kheight, kdepth));

                                TrivectorKernelProduct3D ope = new TrivectorKernelProduct3D(inwidth, inheight, indepth, inchannels, outchannels, kwidth, kheight, kdepth, stride, transpose: false, batch);

                                ope.Execute(x_tensor, y_tensor, w_tensor, gw_tensor);

                                float[] gw_expect = gw.ToArray();
                                float[] gw_actual = gw_tensor.State;

                                CollectionAssert.AreEqual(xval, x_tensor.State);
                                CollectionAssert.AreEqual(yval, y_tensor.State);
                                CollectionAssert.AreEqual(wval, w_tensor.State);

                                AssertError.Tolerance(gw_expect, gw_actual, 1e-6f, 1e-4f, ref max_err, $"mismatch value {inchannels},{outchannels},{kwidth},{kheight},{kdepth},{stride},{inwidth},{inheight},{indepth},{batch}");

                                Console.WriteLine($"pass: {inchannels},{outchannels},{kwidth},{kheight},{kdepth},{stride},{inwidth},{inheight},{indepth},{batch}");

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
                foreach (int batch in new int[] { 1, 2 }) {
                    foreach (int inchannels in new int[] { 3, 6, 9, 12 }) {
                        foreach (int outchannels in new int[] { 3, 6, 9, 12 }) {
                            foreach ((int kwidth, int kheight, int kdepth) in new (int, int, int)[] { (1, 1, 1), (3, 3, 3), (5, 5, 5), (1, 3, 5), (3, 5, 1), (5, 1, 3) }) {
                                foreach ((int stride, int inwidth, int inheight, int indepth) in new (int, int, int, int)[] { (1, 13, 13, 13), (2, 17, 17, 17), (3, 19, 19, 19), (1, 17, 19, 13), (2, 13, 17, 19), (3, 19, 13, 17) }) {
                                    int outwidth = (inwidth - kwidth) / stride + 1, outheight = (inheight - kheight) / stride + 1, outdepth = (indepth - kdepth) / stride + 1;

                                    float[] xval = (new float[inwidth * inheight * indepth * inchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
                                    float[] yval = (new float[outwidth * outheight * outdepth * outchannels * batch]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();
                                    float[] wval = (new float[kwidth * kheight * kdepth * inchannels * outchannels / 9 * 4]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

                                    OverflowCheckedTensor x_tensor = new OverflowCheckedTensor(Shape.Map3D(inchannels, inwidth, inheight, indepth, batch), xval);
                                    OverflowCheckedTensor y_tensor = new OverflowCheckedTensor(Shape.Map3D(outchannels, outwidth, outheight, outdepth, batch), yval);
                                    OverflowCheckedTensor w_tensor = new OverflowCheckedTensor(Shape.Kernel3D(inchannels / 3 * 4, outchannels / 3, kwidth, kheight, kdepth), wval);

                                    OverflowCheckedTensor gw_tensor = new OverflowCheckedTensor(Shape.Kernel3D(inchannels / 3 * 4, outchannels / 3, kwidth, kheight, kdepth));

                                    TrivectorKernelProduct3D ope = new TrivectorKernelProduct3D(inwidth, inheight, indepth, inchannels, outchannels, kwidth, kheight, kdepth, stride, transpose, batch);

                                    ope.Execute(x_tensor, y_tensor, w_tensor, gw_tensor);

                                    CollectionAssert.AreEqual(xval, x_tensor.State);
                                    CollectionAssert.AreEqual(yval, y_tensor.State);
                                    CollectionAssert.AreEqual(wval, w_tensor.State);

                                    gw_tensor.CheckOverflow();

                                    Console.WriteLine($"pass: {inchannels},{outchannels},{kwidth},{kheight},{kdepth},{stride},{inwidth},{inheight},{indepth},{batch},{transpose}");
                                }
                            }
                        }
                    }
                }
            }
        }

        [TestMethod]
        public void SpeedTest() {
            int inwidth = 32, inheight = 32, indepth = 32, inchannels = 33, outchannels = 33, ksize = 3, stride = 2;
            int outwidth = (inwidth - ksize) / stride + 1, outheight = (inheight - ksize) / stride + 1, outdepth = (indepth - ksize) / stride + 1;

            OverflowCheckedTensor x_tensor = new OverflowCheckedTensor(Shape.Map3D(inchannels, inwidth, inheight, indepth));
            OverflowCheckedTensor y_tensor = new OverflowCheckedTensor(Shape.Map3D(outchannels, outwidth, outheight, outdepth));
            OverflowCheckedTensor w_tensor = new OverflowCheckedTensor(Shape.Kernel3D(inchannels / 3 * 4, outchannels / 3, ksize, ksize, ksize));

            OverflowCheckedTensor gw_tensor = new OverflowCheckedTensor(Shape.Kernel3D(inchannels / 3 * 4, outchannels / 3, ksize, ksize, ksize));

            TrivectorKernelProduct3D ope = new TrivectorKernelProduct3D(inwidth, inheight, indepth, inchannels, outchannels, ksize, ksize, ksize, stride);

            Stopwatch sw = new Stopwatch();
            sw.Start();

            ope.Execute(x_tensor, y_tensor, w_tensor, gw_tensor);
            ope.Execute(x_tensor, y_tensor, w_tensor, gw_tensor);
            ope.Execute(x_tensor, y_tensor, w_tensor, gw_tensor);
            ope.Execute(x_tensor, y_tensor, w_tensor, gw_tensor);

            sw.Stop();

            Console.WriteLine($"{sw.ElapsedMilliseconds / 4} msec");
        }

        public static Quaternion.QuaternionFilter3D Reference(TrivectorMap3D x, TrivectorMap3D gy, Quaternion.QuaternionFilter3D w, int kwidth, int kheight, int kdepth, int stride) {
            int inchannels = x.Channels, outchannels = gy.Channels, batch = x.Batch;
            int inw = x.Width, inh = x.Height, ind = x.Depth, outw = gy.Width, outh = gy.Height, outd = gy.Depth;

            if (outw != (inw - kwidth) / stride + 1 || outh != (inh - kheight) / stride + 1 || outd != (ind - kdepth) / stride + 1) {
                throw new ArgumentException("mismatch shape");
            }

            Quaternion.QuaternionFilter3D gw = new Quaternion.QuaternionFilter3D(inchannels, outchannels, kwidth, kheight, kdepth);

            for (int kx, ky, kz = 0; kz < kdepth; kz++) {
                for (ky = 0; ky < kheight; ky++) {
                    for (kx = 0; kx < kwidth; kx++) {
                        for (int th = 0; th < batch; th++) {
                            for (int inch, outch = 0; outch < outchannels; outch++) {
                                for (inch = 0; inch < inchannels; inch++) {
                                    Quaternion.Quaternion sum = 0;
                                    Quaternion.Quaternion q = w[inch, outch, kx, ky, kz];

                                    for (int ix, iy, iz = kz, ox, oy, oz = 0; oz < outd; iz += stride, oz++) {
                                        for (iy = ky, oy = 0; oy < outh; iy += stride, oy++) {
                                            for (ix = kx, ox = 0; ox < outw; ix += stride, ox++) {
                                                sum += Trivector.MulQGrad(x[inch, ix, iy, iz, th], gy[outch, ox, oy, oz, th], q);
                                            }
                                        }
                                    }

                                    gw[inch, outch, kx, ky, kz] += sum;
                                }
                            }

                        }
                    }
                }
            }

            return gw;
        }
    }
}
