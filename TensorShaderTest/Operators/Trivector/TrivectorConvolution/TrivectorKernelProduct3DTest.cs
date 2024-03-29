using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;
using TensorShader;
using TensorShader.Operators.TrivectorConvolution;
using TensorShaderCudaBackend.API;

namespace TensorShaderTest.Operators.Trivector {
    [TestClass]
    public class TrivectorKernelProduct3DTest {
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
                            float[] yval = (new float[outwidth * outheight * outdepth * outchannels * batch]).Select((_, idx) => (float)random.NextDouble() * 1e-2f).ToArray();
                            float[] wval = (new float[kwidth * kheight * kdepth * inchannels * outchannels / 9 * 4]).Select((_, idx) => (float)random.NextDouble() * 1e-2f).ToArray();

                            Trivector[] xcval = (new Trivector[xval.Length / 3])
                                .Select((_, idx) => new Trivector(xval[idx * 3], xval[idx * 3 + 1], xval[idx * 3 + 2])).ToArray();

                            Trivector[] ycval = (new Trivector[yval.Length / 3])
                                .Select((_, idx) => new Trivector(yval[idx * 3], yval[idx * 3 + 1], yval[idx * 3 + 2])).ToArray();

                            Quaternion.Quaternion[] wcval = (new Quaternion.Quaternion[wval.Length / 4])
                                .Select((_, idx) => new Quaternion.Quaternion(wval[idx * 4], wval[idx * 4 + 1], wval[idx * 4 + 2], wval[idx * 4 + 3])).ToArray();

                            TrivectorMap3D x = new(inchannels / 3, inwidth, inheight, indepth, batch, xcval);
                            TrivectorMap3D y = new(outchannels / 3, outwidth, outheight, outdepth, batch, ycval);
                            Quaternion.QuaternionFilter3D w = new(inchannels / 3, outchannels / 3, kwidth, kheight, kdepth, wcval);

                            Quaternion.QuaternionFilter3D gw = Reference(x, y, w, kwidth, kheight, kdepth);

                            OverflowCheckedTensor x_tensor = new(Shape.Map3D(inchannels, inwidth, inheight, indepth, batch), xval);
                            OverflowCheckedTensor y_tensor = new(Shape.Map3D(outchannels, outwidth, outheight, outdepth, batch), yval);
                            OverflowCheckedTensor w_tensor = new(Shape.Kernel3D(inchannels / 3 * 4, outchannels / 3, kwidth, kheight, kdepth), wval);

                            OverflowCheckedTensor gw_tensor = new(Shape.Kernel3D(inchannels / 3 * 4, outchannels / 3, kwidth, kheight, kdepth));

                            TrivectorKernelProduct3D ope = new(inwidth, inheight, indepth, inchannels, outchannels, kwidth, kheight, kdepth, transpose: false, batch);

                            ope.Execute(x_tensor, y_tensor, w_tensor, gw_tensor);

                            float[] gw_expect = gw.ToArray();
                            float[] gw_actual = gw_tensor.State.Value;

                            CollectionAssert.AreEqual(xval, x_tensor.State.Value);
                            CollectionAssert.AreEqual(yval, y_tensor.State.Value);
                            CollectionAssert.AreEqual(wval, w_tensor.State.Value);

                            AssertError.Tolerance(gw_expect, gw_actual, 1e-5f, 1e-3f, ref max_err, $"mismatch value {inchannels},{outchannels},{kwidth},{kheight},{kdepth},{inwidth},{inheight},{indepth},{batch}");

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
                            float[] yval = (new float[outwidth * outheight * outdepth * outchannels * batch]).Select((_, idx) => (float)random.NextDouble() * 1e-2f).ToArray();
                            float[] wval = (new float[kwidth * kheight * kdepth * inchannels * outchannels / 9 * 4]).Select((_, idx) => (float)random.NextDouble() * 1e-2f).ToArray();

                            Trivector[] xcval = (new Trivector[xval.Length / 3])
                                .Select((_, idx) => new Trivector(xval[idx * 3], xval[idx * 3 + 1], xval[idx * 3 + 2])).ToArray();

                            Trivector[] ycval = (new Trivector[yval.Length / 3])
                                .Select((_, idx) => new Trivector(yval[idx * 3], yval[idx * 3 + 1], yval[idx * 3 + 2])).ToArray();

                            Quaternion.Quaternion[] wcval = (new Quaternion.Quaternion[wval.Length / 4])
                                .Select((_, idx) => new Quaternion.Quaternion(wval[idx * 4], wval[idx * 4 + 1], wval[idx * 4 + 2], wval[idx * 4 + 3])).ToArray();

                            TrivectorMap3D x = new(inchannels / 3, inwidth, inheight, indepth, batch, xcval);
                            TrivectorMap3D y = new(outchannels / 3, outwidth, outheight, outdepth, batch, ycval);
                            Quaternion.QuaternionFilter3D w = new(inchannels / 3, outchannels / 3, kwidth, kheight, kdepth, wcval);

                            Quaternion.QuaternionFilter3D gw = Reference(x, y, w, kwidth, kheight, kdepth);

                            OverflowCheckedTensor x_tensor = new(Shape.Map3D(inchannels, inwidth, inheight, indepth, batch), xval);
                            OverflowCheckedTensor y_tensor = new(Shape.Map3D(outchannels, outwidth, outheight, outdepth, batch), yval);
                            OverflowCheckedTensor w_tensor = new(Shape.Kernel3D(inchannels / 3 * 4, outchannels / 3, kwidth, kheight, kdepth), wval);

                            OverflowCheckedTensor gw_tensor = new(Shape.Kernel3D(inchannels / 3 * 4, outchannels / 3, kwidth, kheight, kdepth));

                            TrivectorKernelProduct3D ope = new(inwidth, inheight, indepth, inchannels, outchannels, kwidth, kheight, kdepth, transpose: false, batch);

                            ope.Execute(x_tensor, y_tensor, w_tensor, gw_tensor);

                            float[] gw_expect = gw.ToArray();
                            float[] gw_actual = gw_tensor.State.Value;

                            CollectionAssert.AreEqual(xval, x_tensor.State.Value);
                            CollectionAssert.AreEqual(yval, y_tensor.State.Value);
                            CollectionAssert.AreEqual(wval, w_tensor.State.Value);

                            AssertError.Tolerance(gw_expect, gw_actual, 1e-6f, 1e-4f, ref max_err, $"mismatch value {inchannels},{outchannels},{kwidth},{kheight},{kdepth},{inwidth},{inheight},{indepth},{batch}");

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
            float[] yval = (new float[outwidth * outheight * outdepth * outchannels * batch]).Select((_, idx) => (float)random.NextDouble() * 1e-2f).ToArray();
            float[] wval = (new float[kwidth * kheight * kdepth * inchannels * outchannels / 9 * 4]).Select((_, idx) => (float)random.NextDouble() * 1e-2f).ToArray();

            Trivector[] xcval = (new Trivector[xval.Length / 3])
                .Select((_, idx) => new Trivector(xval[idx * 3], xval[idx * 3 + 1], xval[idx * 3 + 2])).ToArray();

            Trivector[] ycval = (new Trivector[yval.Length / 3])
                .Select((_, idx) => new Trivector(yval[idx * 3], yval[idx * 3 + 1], yval[idx * 3 + 2])).ToArray();

            Quaternion.Quaternion[] wcval = (new Quaternion.Quaternion[wval.Length / 4])
                .Select((_, idx) => new Quaternion.Quaternion(wval[idx * 4], wval[idx * 4 + 1], wval[idx * 4 + 2], wval[idx * 4 + 3])).ToArray();

            TrivectorMap3D x = new(inchannels / 3, inwidth, inheight, indepth, batch, xcval);
            TrivectorMap3D y = new(outchannels / 3, outwidth, outheight, outdepth, batch, ycval);
            Quaternion.QuaternionFilter3D w = new(inchannels / 3, outchannels / 3, kwidth, kheight, kdepth, wcval);

            Quaternion.QuaternionFilter3D gw = Reference(x, y, w, kwidth, kheight, kdepth);

            OverflowCheckedTensor x_tensor = new(Shape.Map3D(inchannels, inwidth, inheight, indepth, batch), xval);
            OverflowCheckedTensor y_tensor = new(Shape.Map3D(outchannels, outwidth, outheight, outdepth, batch), yval);
            OverflowCheckedTensor w_tensor = new(Shape.Kernel3D(inchannels / 3 * 4, outchannels / 3, kwidth, kheight, kdepth), wval);

            OverflowCheckedTensor gw_tensor = new(Shape.Kernel3D(inchannels / 3 * 4, outchannels / 3, kwidth, kheight, kdepth));

            TrivectorKernelProduct3D ope = new(inwidth, inheight, indepth, inchannels, outchannels, kwidth, kheight, kdepth, transpose: false, batch);

            ope.Execute(x_tensor, y_tensor, w_tensor, gw_tensor);

            float[] gw_expect = gw.ToArray();
            float[] gw_actual = gw_tensor.State.Value;

            CollectionAssert.AreEqual(xval, x_tensor.State.Value);
            CollectionAssert.AreEqual(yval, y_tensor.State.Value);
            CollectionAssert.AreEqual(wval, w_tensor.State.Value);

            AssertError.Tolerance(gw_expect, gw_actual, 1e-6f, 1e-4f, ref max_err, $"mismatch value {inchannels},{outchannels},{kwidth},{kheight},{kdepth},{inwidth},{inheight},{indepth},{batch}");

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
            OverflowCheckedTensor y_tensor = new(Shape.Map3D(outchannels, outwidth, outheight, outdepth));
            OverflowCheckedTensor w_tensor = new(Shape.Kernel3D(inchannels / 3 * 4, outchannels / 3, ksize, ksize, ksize));

            OverflowCheckedTensor gw_tensor = new(Shape.Kernel3D(inchannels / 3 * 4, outchannels / 3, ksize, ksize, ksize));

            TrivectorKernelProduct3D ope = new(inwidth, inheight, indepth, inchannels, outchannels, ksize, ksize, ksize);

            Cuda.Profiler.Initialize("../../../../profiler.nvsetting", "../../nvprofiles/trivector_kernelproduct_3d_fp.nvvp");
            Cuda.Profiler.Start();

            ope.Execute(x_tensor, y_tensor, w_tensor, gw_tensor);

            Cuda.Profiler.Stop();
        }

        [TestMethod]
        public void SpeedFFPTest() {
            TensorShaderCudaBackend.Environment.CudnnEnabled = false;
            TensorShaderCudaBackend.Environment.Precision = TensorShaderCudaBackend.Environment.PrecisionMode.FloatFloat;

            int inwidth = 32, inheight = 32, indepth = 32, inchannels = 33, outchannels = 33, ksize = 3;
            int outwidth = inwidth - ksize + 1, outheight = inheight - ksize + 1, outdepth = indepth - ksize + 1;

            OverflowCheckedTensor x_tensor = new(Shape.Map3D(inchannels, inwidth, inheight, indepth));
            OverflowCheckedTensor y_tensor = new(Shape.Map3D(outchannels, outwidth, outheight, outdepth));
            OverflowCheckedTensor w_tensor = new(Shape.Kernel3D(inchannels / 3 * 4, outchannels / 3, ksize, ksize, ksize));

            OverflowCheckedTensor gw_tensor = new(Shape.Kernel3D(inchannels / 3 * 4, outchannels / 3, ksize, ksize, ksize));

            TrivectorKernelProduct3D ope = new(inwidth, inheight, indepth, inchannels, outchannels, ksize, ksize, ksize);

            Cuda.Profiler.Initialize("../../../../profiler.nvsetting", "../../nvprofiles/trivector_kernelproduct_3d_ffp.nvvp");
            Cuda.Profiler.Start();

            ope.Execute(x_tensor, y_tensor, w_tensor, gw_tensor);

            Cuda.Profiler.Stop();
        }

        public static Quaternion.QuaternionFilter3D Reference(TrivectorMap3D x, TrivectorMap3D gy, Quaternion.QuaternionFilter3D w, int kwidth, int kheight, int kdepth) {
            int inchannels = x.Channels, outchannels = gy.Channels, batch = x.Batch;
            int inw = x.Width, inh = x.Height, ind = x.Depth, outw = gy.Width, outh = gy.Height, outd = gy.Depth;

            if (outw != inw - kwidth + 1 || outh != inh - kheight + 1 || outd != ind - kdepth + 1) {
                throw new ArgumentException("mismatch shape");
            }

            Quaternion.QuaternionFilter3D gw = new(inchannels, outchannels, kwidth, kheight, kdepth);

            for (int kx, ky, kz = 0; kz < kdepth; kz++) {
                for (ky = 0; ky < kheight; ky++) {
                    for (kx = 0; kx < kwidth; kx++) {
                        for (int th = 0; th < batch; th++) {
                            for (int ix, iy, iz = kz, ox, oy, oz = 0; oz < outd; iz++, oz++) {
                                for (iy = ky, oy = 0; oy < outh; iy++, oy++) {
                                    for (ix = kx, ox = 0; ox < outw; ix++, ox++) {
                                        for (int inch, outch = 0; outch < outchannels; outch++) {
                                            for (inch = 0; inch < inchannels; inch++) {
                                                gw[inch, outch, kx, ky, kz] += Trivector.MulQGrad(x[inch, ix, iy, iz, th], gy[outch, ox, oy, oz, th], w[inch, outch, kx, ky, kz]);
                                            }
                                        }
                                    }
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
