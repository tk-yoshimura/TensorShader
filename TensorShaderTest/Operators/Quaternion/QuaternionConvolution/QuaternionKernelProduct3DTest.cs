using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;
using TensorShader;
using TensorShader.Operators.QuaternionConvolution;
using TensorShaderCudaBackend.API;

namespace TensorShaderTest.Operators.Quaternion {
    [TestClass]
    public class QuaternionKernelProduct3DTest {
        [TestMethod]
        public void ExecuteFPTest() {
            TensorShaderCudaBackend.Environment.Precision = TensorShaderCudaBackend.Environment.PrecisionMode.Float;
            TensorShaderCudaBackend.Environment.CudnnEnabled = false;

            Random random = new(1234);

            float max_err = 0;

            foreach (int batch in new int[] { 1, 2 }) {
                foreach ((int inchannels, int outchannels) in new (int, int)[] { (4, 4), (4, 8), (8, 4), (8, 20), (20, 32), (32, 8), (32, 36), (36, 36) }) {
                    foreach ((int kwidth, int kheight, int kdepth) in new (int, int, int)[] { (1, 1, 1), (3, 3, 3), (5, 5, 5), (1, 3, 5), (3, 5, 1), (5, 1, 3) }) {
                        foreach ((int inwidth, int inheight, int indepth) in new (int, int, int)[] { (kwidth, kheight, kdepth), (kwidth * 2, kheight * 2, kdepth * 2), (13, 13, 13), (17, 17, 17), (19, 19, 19), (17, 19, 13), (13, 17, 19), (19, 13, 17) }) {

                            int outwidth = inwidth - kwidth + 1, outheight = inheight - kheight + 1, outdepth = indepth - kdepth + 1;

                            float[] xval = (new float[inwidth * inheight * indepth * inchannels * batch]).Select((_, idx) => (float)random.NextDouble() * 1e-2f).ToArray();
                            float[] yval = (new float[outwidth * outheight * outdepth * outchannels * batch]).Select((_, idx) => (float)random.NextDouble() * 1e-2f).ToArray();

                            Quaternion[] xcval = (new Quaternion[xval.Length / 4])
                                .Select((_, idx) => new Quaternion(xval[idx * 4], xval[idx * 4 + 1], xval[idx * 4 + 2], xval[idx * 4 + 3])).ToArray();

                            Quaternion[] ycval = (new Quaternion[yval.Length / 4])
                                .Select((_, idx) => new Quaternion(yval[idx * 4], yval[idx * 4 + 1], yval[idx * 4 + 2], yval[idx * 4 + 3])).ToArray();

                            QuaternionMap3D x = new(inchannels / 4, inwidth, inheight, indepth, batch, xcval);
                            QuaternionMap3D y = new(outchannels / 4, outwidth, outheight, outdepth, batch, ycval);

                            QuaternionFilter3D gw = Reference(x, y, kwidth, kheight, kdepth);

                            OverflowCheckedTensor x_tensor = new(Shape.Map3D(inchannels, inwidth, inheight, indepth, batch), xval);
                            OverflowCheckedTensor y_tensor = new(Shape.Map3D(outchannels, outwidth, outheight, outdepth, batch), yval);

                            OverflowCheckedTensor gw_tensor = new(Shape.Kernel3D(inchannels, outchannels / 4, kwidth, kheight, kdepth));

                            QuaternionKernelProduct3D ope = new(inwidth, inheight, indepth, inchannels, outchannels, kwidth, kheight, kdepth, transpose: false, batch);

                            ope.Execute(x_tensor, y_tensor, gw_tensor);

                            float[] gw_expect = gw.ToArray();
                            float[] gw_actual = gw_tensor.State.Value;

                            CollectionAssert.AreEqual(xval, x_tensor.State.Value);
                            CollectionAssert.AreEqual(yval, y_tensor.State.Value);

                            AssertError.Tolerance(gw_expect, gw_actual, 1e-6f, 1e-4f, ref max_err, $"mismatch value {inchannels},{outchannels},{kwidth},{kheight},{kdepth},{inwidth},{inheight},{indepth},{batch}");

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
                foreach ((int inchannels, int outchannels) in new (int, int)[] { (4, 4), (4, 8), (8, 4), (8, 20), (20, 32), (32, 8), (32, 36), (36, 36) }) {
                    foreach ((int kwidth, int kheight, int kdepth) in new (int, int, int)[] { (1, 1, 1), (3, 3, 3), (5, 5, 5), (1, 3, 5), (3, 5, 1), (5, 1, 3) }) {
                        foreach ((int inwidth, int inheight, int indepth) in new (int, int, int)[] { (kwidth, kheight, kdepth), (kwidth * 2, kheight * 2, kdepth * 2), (13, 13, 13), (17, 17, 17), (19, 19, 19), (17, 19, 13), (13, 17, 19), (19, 13, 17) }) {

                            int outwidth = inwidth - kwidth + 1, outheight = inheight - kheight + 1, outdepth = indepth - kdepth + 1;

                            float[] xval = (new float[inwidth * inheight * indepth * inchannels * batch]).Select((_, idx) => (float)random.NextDouble() * 1e-2f).ToArray();
                            float[] yval = (new float[outwidth * outheight * outdepth * outchannels * batch]).Select((_, idx) => (float)random.NextDouble() * 1e-2f).ToArray();

                            Quaternion[] xcval = (new Quaternion[xval.Length / 4])
                                .Select((_, idx) => new Quaternion(xval[idx * 4], xval[idx * 4 + 1], xval[idx * 4 + 2], xval[idx * 4 + 3])).ToArray();

                            Quaternion[] ycval = (new Quaternion[yval.Length / 4])
                                .Select((_, idx) => new Quaternion(yval[idx * 4], yval[idx * 4 + 1], yval[idx * 4 + 2], yval[idx * 4 + 3])).ToArray();

                            QuaternionMap3D x = new(inchannels / 4, inwidth, inheight, indepth, batch, xcval);
                            QuaternionMap3D y = new(outchannels / 4, outwidth, outheight, outdepth, batch, ycval);

                            QuaternionFilter3D gw = Reference(x, y, kwidth, kheight, kdepth);

                            OverflowCheckedTensor x_tensor = new(Shape.Map3D(inchannels, inwidth, inheight, indepth, batch), xval);
                            OverflowCheckedTensor y_tensor = new(Shape.Map3D(outchannels, outwidth, outheight, outdepth, batch), yval);

                            OverflowCheckedTensor gw_tensor = new(Shape.Kernel3D(inchannels, outchannels / 4, kwidth, kheight, kdepth));

                            QuaternionKernelProduct3D ope = new(inwidth, inheight, indepth, inchannels, outchannels, kwidth, kheight, kdepth, transpose: false, batch);

                            ope.Execute(x_tensor, y_tensor, gw_tensor);

                            float[] gw_expect = gw.ToArray();
                            float[] gw_actual = gw_tensor.State.Value;

                            CollectionAssert.AreEqual(xval, x_tensor.State.Value);
                            CollectionAssert.AreEqual(yval, y_tensor.State.Value);

                            AssertError.Tolerance(gw_expect, gw_actual, 1e-7f, 1e-5f, ref max_err, $"mismatch value {inchannels},{outchannels},{kwidth},{kheight},{kdepth},{inwidth},{inheight},{indepth},{batch}");

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
            int inchannels = 196, outchannels = 200;
            int kwidth = 7, kheight = 5, kdepth = 3;
            int inwidth = 125, inheight = 196, indepth = 4;
            int outwidth = inwidth - kwidth + 1, outheight = inheight - kheight + 1, outdepth = indepth - kdepth + 1;

            float[] xval = (new float[inwidth * inheight * indepth * inchannels * batch]).Select((_, idx) => (float)random.NextDouble() * 1e-2f).ToArray();
            float[] yval = (new float[outwidth * outheight * outdepth * outchannels * batch]).Select((_, idx) => (float)random.NextDouble() * 1e-2f).ToArray();

            Quaternion[] xcval = (new Quaternion[xval.Length / 4])
                .Select((_, idx) => new Quaternion(xval[idx * 4], xval[idx * 4 + 1], xval[idx * 4 + 2], xval[idx * 4 + 3])).ToArray();

            Quaternion[] ycval = (new Quaternion[yval.Length / 4])
                .Select((_, idx) => new Quaternion(yval[idx * 4], yval[idx * 4 + 1], yval[idx * 4 + 2], yval[idx * 4 + 3])).ToArray();

            QuaternionMap3D x = new(inchannels / 4, inwidth, inheight, indepth, batch, xcval);
            QuaternionMap3D y = new(outchannels / 4, outwidth, outheight, outdepth, batch, ycval);

            QuaternionFilter3D gw = Reference(x, y, kwidth, kheight, kdepth);

            OverflowCheckedTensor x_tensor = new(Shape.Map3D(inchannels, inwidth, inheight, indepth, batch), xval);
            OverflowCheckedTensor y_tensor = new(Shape.Map3D(outchannels, outwidth, outheight, outdepth, batch), yval);

            OverflowCheckedTensor gw_tensor = new(Shape.Kernel3D(inchannels, outchannels / 4, kwidth, kheight, kdepth));

            QuaternionKernelProduct3D ope = new(inwidth, inheight, indepth, inchannels, outchannels, kwidth, kheight, kdepth, transpose: false, batch);

            ope.Execute(x_tensor, y_tensor, gw_tensor);

            float[] gw_expect = gw.ToArray();
            float[] gw_actual = gw_tensor.State.Value;

            CollectionAssert.AreEqual(xval, x_tensor.State.Value);
            CollectionAssert.AreEqual(yval, y_tensor.State.Value);

            AssertError.Tolerance(gw_expect, gw_actual, 1e-7f, 1e-5f, ref max_err, $"mismatch value {inchannels},{outchannels},{kwidth},{kheight},{kdepth},{inwidth},{inheight},{indepth},{batch}");

            Console.WriteLine($"pass: {inchannels},{outchannels},{kwidth},{kheight},{kdepth},{inwidth},{inheight},{indepth},{batch}");

            Console.WriteLine($"maxerr:{max_err}");
        }

        [TestMethod]
        public void SpeedFPTest() {
            TensorShaderCudaBackend.Environment.Precision = TensorShaderCudaBackend.Environment.PrecisionMode.Float;
            TensorShaderCudaBackend.Environment.CudnnEnabled = false;

            int inwidth = 32, inheight = 32, indepth = 32, inchannels = 32, outchannels = 32, ksize = 3;
            int outwidth = inwidth - ksize + 1, outheight = inheight - ksize + 1, outdepth = indepth - ksize + 1;

            OverflowCheckedTensor x_tensor = new(Shape.Map3D(inchannels, inwidth, inheight, indepth));
            OverflowCheckedTensor y_tensor = new(Shape.Map3D(outchannels, outwidth, outheight, outdepth));

            OverflowCheckedTensor gw_tensor = new(Shape.Kernel3D(inchannels, outchannels / 4, ksize, ksize, ksize));

            QuaternionKernelProduct3D ope = new(inwidth, inheight, indepth, inchannels, outchannels, ksize, ksize, ksize);

            Cuda.Profiler.Initialize("../../../../profiler.nvsetting", "../../nvprofiles/quaternion_kernelproduct_3d_fp.nvvp");
            Cuda.Profiler.Start();

            ope.Execute(x_tensor, y_tensor, gw_tensor);

            Cuda.Profiler.Stop();
        }

        [TestMethod]
        public void SpeedFFPTest() {
            TensorShaderCudaBackend.Environment.CudnnEnabled = false;
            TensorShaderCudaBackend.Environment.Precision = TensorShaderCudaBackend.Environment.PrecisionMode.FloatFloat;

            int inwidth = 32, inheight = 32, indepth = 32, inchannels = 32, outchannels = 32, ksize = 3;
            int outwidth = inwidth - ksize + 1, outheight = inheight - ksize + 1, outdepth = indepth - ksize + 1;

            OverflowCheckedTensor x_tensor = new(Shape.Map3D(inchannels, inwidth, inheight, indepth));
            OverflowCheckedTensor y_tensor = new(Shape.Map3D(outchannels, outwidth, outheight, outdepth));

            OverflowCheckedTensor gw_tensor = new(Shape.Kernel3D(inchannels, outchannels / 4, ksize, ksize, ksize));

            QuaternionKernelProduct3D ope = new(inwidth, inheight, indepth, inchannels, outchannels, ksize, ksize, ksize);

            Cuda.Profiler.Initialize("../../../../profiler.nvsetting", "../../nvprofiles/quaternion_kernelproduct_3d_ffp.nvvp");
            Cuda.Profiler.Start();

            ope.Execute(x_tensor, y_tensor, gw_tensor);

            Cuda.Profiler.Stop();
        }

        public static QuaternionFilter3D Reference(QuaternionMap3D x, QuaternionMap3D gy, int kwidth, int kheight, int kdepth) {
            int inchannels = x.Channels, outchannels = gy.Channels, batch = x.Batch;
            int inw = x.Width, inh = x.Height, ind = x.Depth, outw = gy.Width, outh = gy.Height, outd = gy.Depth;

            if (outw != inw - kwidth + 1 || outh != inh - kheight + 1 || outd != ind - kdepth + 1) {
                throw new ArgumentException("mismatch shape");
            }

            QuaternionFilter3D w = new(inchannels, outchannels, kwidth, kheight, kdepth);

            for (int kx, ky, kz = 0; kz < kdepth; kz++) {
                for (ky = 0; ky < kheight; ky++) {
                    for (kx = 0; kx < kwidth; kx++) {
                        for (int th = 0; th < batch; th++) {
                            for (int ix, iy, iz = kz, ox, oy, oz = 0; oz < outd; iz++, oz++) {
                                for (iy = ky, oy = 0; oy < outh; iy++, oy++) {
                                    for (ix = kx, ox = 0; ox < outw; ix++, ox++) {
                                        for (int inch, outch = 0; outch < outchannels; outch++) {
                                            for (inch = 0; inch < inchannels; inch++) {
                                                w[inch, outch, kx, ky, kz] += Quaternion.MulGrad(gy[outch, ox, oy, oz, th], x[inch, ix, iy, iz, th]);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            return w;
        }

        [TestMethod]
        public void ReferenceTest() {
            int inchannels = 8, outchannels = 12, kwidth = 3, kheight = 5, kdepth = 7, inwidth = 13, inheight = 12, indepth = 11;
            int outwidth = inwidth - kwidth + 1, outheight = inheight - kheight + 1, outdepth = indepth - kdepth + 1, batch = 1;

            float[] xval = (new float[inwidth * inheight * indepth * inchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
            float[] yval = (new float[outwidth * outheight * outdepth * outchannels * batch]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

            Quaternion[] xcval = (new Quaternion[xval.Length / 4])
                .Select((_, idx) => new Quaternion(xval[idx * 4], xval[idx * 4 + 1], xval[idx * 4 + 2], xval[idx * 4 + 3])).ToArray();

            Quaternion[] ycval = (new Quaternion[yval.Length / 4])
                .Select((_, idx) => new Quaternion(yval[idx * 4], yval[idx * 4 + 1], yval[idx * 4 + 2], yval[idx * 4 + 3])).ToArray();

            QuaternionMap3D x = new(inchannels / 4, inwidth, inheight, indepth, batch, xcval);
            QuaternionMap3D y = new(outchannels / 4, outwidth, outheight, outdepth, batch, ycval);

            QuaternionFilter3D gw = Reference(x, y, kwidth, kheight, kdepth);

            float[] gw_expect = {
                8.731898560e+03f,  -4.547473509e-13f,  -9.759200000e+00f,  -4.879600000e+00f,  8.750508800e+03f,  4.547473509e-13f,
                -9.766240000e+00f,  -4.883120000e+00f,  8.711472000e+03f,  1.364242053e-12f,  -9.752160000e+00f,  -4.876080000e+00f,
                8.730054080e+03f,  -1.364242053e-12f,  -9.759200000e+00f,  -4.879600000e+00f,  8.691045440e+03f,  4.547473509e-13f,
                -9.745120000e+00f,  -4.872560000e+00f,  8.709599360e+03f,  -0.000000000e+00f,  -9.752160000e+00f,  -4.876080000e+00f,
                8.769119040e+03f,  -0.000000000e+00f,  -9.773280000e+00f,  -4.886640000e+00f,  8.787729280e+03f,  4.547473509e-13f,
                -9.780320000e+00f,  -4.890160000e+00f,  8.748636160e+03f,  -0.000000000e+00f,  -9.766240000e+00f,  -4.883120000e+00f,
                8.767218240e+03f,  -9.094947018e-13f,  -9.773280000e+00f,  -4.886640000e+00f,  8.728153280e+03f,  -4.547473509e-13f,
                -9.759200000e+00f,  -4.879600000e+00f,  8.746707200e+03f,  -0.000000000e+00f,  -9.766240000e+00f,  -4.883120000e+00f,
                8.806339520e+03f,  4.547473509e-13f,  -9.787360000e+00f,  -4.893680000e+00f,  8.824949760e+03f,  -4.547473509e-13f,
                -9.794400000e+00f,  -4.897200000e+00f,  8.785800320e+03f,  -4.547473509e-13f,  -9.780320000e+00f,  -4.890160000e+00f,
                8.804382400e+03f,  9.094947018e-13f,  -9.787360000e+00f,  -4.893680000e+00f,  8.765261120e+03f,  -0.000000000e+00f,
                -9.773280000e+00f,  -4.886640000e+00f,  8.783815040e+03f,  -0.000000000e+00f,  -9.780320000e+00f,  -4.890160000e+00f,
                9.215764800e+03f,  4.547473509e-13f,  -9.942240000e+00f,  -4.971120000e+00f,  9.234375040e+03f,  4.547473509e-13f,
                -9.949280000e+00f,  -4.974640000e+00f,  9.194606080e+03f,  -4.547473509e-13f,  -9.935200000e+00f,  -4.967600000e+00f,
                9.213188160e+03f,  -4.547473509e-13f,  -9.942240000e+00f,  -4.971120000e+00f,  9.173447360e+03f,  -0.000000000e+00f,
                -9.928160000e+00f,  -4.964080000e+00f,  9.192001280e+03f,  -0.000000000e+00f,  -9.935200000e+00f,  -4.967600000e+00f,
                9.252985280e+03f,  -4.547473509e-13f,  -9.956320000e+00f,  -4.978160000e+00f,  9.271595520e+03f,  -0.000000000e+00f,
                -9.963360000e+00f,  -4.981680000e+00f,  9.231770240e+03f,  1.818989404e-12f,  -9.949280000e+00f,  -4.974640000e+00f,
                9.250352320e+03f,  -9.094947018e-13f,  -9.956320000e+00f,  -4.978160000e+00f,  9.210555200e+03f,  -0.000000000e+00f,
                -9.942240000e+00f,  -4.971120000e+00f,  9.229109120e+03f,  -9.094947018e-13f,  -9.949280000e+00f,  -4.974640000e+00f,
                9.290205760e+03f,  9.094947018e-13f,  -9.970400000e+00f,  -4.985200000e+00f,  9.308816000e+03f,  -9.094947018e-13f,
                -9.977440000e+00f,  -4.988720000e+00f,  9.268934400e+03f,  9.094947018e-13f,  -9.963360000e+00f,  -4.981680000e+00f,
                9.287516480e+03f,  9.094947018e-13f,  -9.970400000e+00f,  -4.985200000e+00f,  9.247663040e+03f,  -0.000000000e+00f,
                -9.956320000e+00f,  -4.978160000e+00f,  9.266216960e+03f,  -0.000000000e+00f,  -9.963360000e+00f,  -4.981680000e+00f,
                9.699631040e+03f,  4.547473509e-13f,  -1.012528000e+01f,  -5.062640000e+00f,  9.718241280e+03f,  -9.094947018e-13f,
                -1.013232000e+01f,  -5.066160000e+00f,  9.677740160e+03f,  4.547473509e-13f,  -1.011824000e+01f,  -5.059120000e+00f,
                9.696322240e+03f,  -0.000000000e+00f,  -1.012528000e+01f,  -5.062640000e+00f,  9.655849280e+03f,  -4.547473509e-13f,
                -1.011120000e+01f,  -5.055600000e+00f,  9.674403200e+03f,  -4.547473509e-13f,  -1.011824000e+01f,  -5.059120000e+00f,
                9.736851520e+03f,  -0.000000000e+00f,  -1.013936000e+01f,  -5.069680000e+00f,  9.755461760e+03f,  -4.547473509e-13f,
                -1.014640000e+01f,  -5.073200000e+00f,  9.714904320e+03f,  -9.094947018e-13f,  -1.013232000e+01f,  -5.066160000e+00f,
                9.733486400e+03f,  -4.547473509e-13f,  -1.013936000e+01f,  -5.069680000e+00f,  9.692957120e+03f,  -4.547473509e-13f,
                -1.012528000e+01f,  -5.062640000e+00f,  9.711511040e+03f,  -4.547473509e-13f,  -1.013232000e+01f,  -5.066160000e+00f,
                9.774072000e+03f,  -9.094947018e-13f,  -1.015344000e+01f,  -5.076720000e+00f,  9.792682240e+03f,  4.547473509e-13f,
                -1.016048000e+01f,  -5.080240000e+00f,  9.752068480e+03f,  -9.094947018e-13f,  -1.014640000e+01f,  -5.073200000e+00f,
                9.770650560e+03f,  9.094947018e-13f,  -1.015344000e+01f,  -5.076720000e+00f,  9.730064960e+03f,  4.547473509e-13f,
                -1.013936000e+01f,  -5.069680000e+00f,  9.748618880e+03f,  -0.000000000e+00f,  -1.014640000e+01f,  -5.073200000e+00f,
                1.018349728e+04f,  -4.547473509e-13f,  -1.030832000e+01f,  -5.154160000e+00f,  1.020210752e+04f,  -1.364242053e-12f,
                -1.031536000e+01f,  -5.157680000e+00f,  1.016087424e+04f,  9.094947018e-13f,  -1.030128000e+01f,  -5.150640000e+00f,
                1.017945632e+04f,  4.547473509e-13f,  -1.030832000e+01f,  -5.154160000e+00f,  1.013825120e+04f,  -4.547473509e-13f,
                -1.029424000e+01f,  -5.147120000e+00f,  1.015680512e+04f,  4.547473509e-13f,  -1.030128000e+01f,  -5.150640000e+00f,
                1.022071776e+04f,  4.547473509e-13f,  -1.032240000e+01f,  -5.161200000e+00f,  1.023932800e+04f,  4.547473509e-13f,
                -1.032944000e+01f,  -5.164720000e+00f,  1.019803840e+04f,  -0.000000000e+00f,  -1.031536000e+01f,  -5.157680000e+00f,
                1.021662048e+04f,  -9.094947018e-13f,  -1.032240000e+01f,  -5.161200000e+00f,  1.017535904e+04f,  -4.547473509e-13f,
                -1.030832000e+01f,  -5.154160000e+00f,  1.019391296e+04f,  9.094947018e-13f,  -1.031536000e+01f,  -5.157680000e+00f,
                1.025793824e+04f,  -0.000000000e+00f,  -1.033648000e+01f,  -5.168240000e+00f,  1.027654848e+04f,  -4.547473509e-13f,
                -1.034352000e+01f,  -5.171760000e+00f,  1.023520256e+04f,  4.547473509e-13f,  -1.032944000e+01f,  -5.164720000e+00f,
                1.025378464e+04f,  -1.364242053e-12f,  -1.033648000e+01f,  -5.168240000e+00f,  1.021246688e+04f,  4.547473509e-13f,
                -1.032240000e+01f,  -5.161200000e+00f,  1.023102080e+04f,  -4.547473509e-13f,  -1.032944000e+01f,  -5.164720000e+00f,
                1.066736352e+04f,  4.547473509e-13f,  -1.049136000e+01f,  -5.245680000e+00f,  1.068597376e+04f,  9.094947018e-13f,
                -1.049840000e+01f,  -5.249200000e+00f,  1.064400832e+04f,  -0.000000000e+00f,  -1.048432000e+01f,  -5.242160000e+00f,
                1.066259040e+04f,  -0.000000000e+00f,  -1.049136000e+01f,  -5.245680000e+00f,  1.062065312e+04f,  -0.000000000e+00f,
                -1.047728000e+01f,  -5.238640000e+00f,  1.063920704e+04f,  1.364242053e-12f,  -1.048432000e+01f,  -5.242160000e+00f,
                1.070458400e+04f,  1.818989404e-12f,  -1.050544000e+01f,  -5.252720000e+00f,  1.072319424e+04f,  -9.094947018e-13f,
                -1.051248000e+01f,  -5.256240000e+00f,  1.068117248e+04f,  -9.094947018e-13f,  -1.049840000e+01f,  -5.249200000e+00f,
                1.069975456e+04f,  -9.094947018e-13f,  -1.050544000e+01f,  -5.252720000e+00f,  1.065776096e+04f,  4.547473509e-13f,
                -1.049136000e+01f,  -5.245680000e+00f,  1.067631488e+04f,  -0.000000000e+00f,  -1.049840000e+01f,  -5.249200000e+00f,
                1.074180448e+04f,  4.547473509e-13f,  -1.051952000e+01f,  -5.259760000e+00f,  1.076041472e+04f,  4.547473509e-13f,
                -1.052656000e+01f,  -5.263280000e+00f,  1.071833664e+04f,  4.547473509e-13f,  -1.051248000e+01f,  -5.256240000e+00f,
                1.073691872e+04f,  -4.547473509e-13f,  -1.051952000e+01f,  -5.259760000e+00f,  1.069486880e+04f,  4.547473509e-13f,
                -1.050544000e+01f,  -5.252720000e+00f,  1.071342272e+04f,  4.547473509e-13f,  -1.051248000e+01f,  -5.256240000e+00f,
                1.453829344e+04f,  -1.364242053e-12f,  -1.195568000e+01f,  -5.977840000e+00f,  1.455690368e+04f,  9.094947018e-13f,
                -1.196272000e+01f,  -5.981360000e+00f,  1.450908096e+04f,  -9.094947018e-13f,  -1.194864000e+01f,  -5.974320000e+00f,
                1.452766304e+04f,  -3.183231456e-12f,  -1.195568000e+01f,  -5.977840000e+00f,  1.447986848e+04f,  -0.000000000e+00f,
                -1.194160000e+01f,  -5.970800000e+00f,  1.449842240e+04f,  1.364242053e-12f,  -1.194864000e+01f,  -5.974320000e+00f,
                1.457551392e+04f,  -9.094947018e-13f,  -1.196976000e+01f,  -5.984880000e+00f,  1.459412416e+04f,  9.094947018e-13f,
                -1.197680000e+01f,  -5.988400000e+00f,  1.454624512e+04f,  2.728484105e-12f,  -1.196272000e+01f,  -5.981360000e+00f,
                1.456482720e+04f,  1.818989404e-12f,  -1.196976000e+01f,  -5.984880000e+00f,  1.451697632e+04f,  -1.364242053e-12f,
                -1.195568000e+01f,  -5.977840000e+00f,  1.453553024e+04f,  -4.547473509e-13f,  -1.196272000e+01f,  -5.981360000e+00f,
                1.461273440e+04f,  -1.818989404e-12f,  -1.198384000e+01f,  -5.991920000e+00f,  1.463134464e+04f,  -9.094947018e-13f,
                -1.199088000e+01f,  -5.995440000e+00f,  1.458340928e+04f,  1.364242053e-12f,  -1.197680000e+01f,  -5.988400000e+00f,
                1.460199136e+04f,  9.094947018e-13f,  -1.198384000e+01f,  -5.991920000e+00f,  1.455408416e+04f,  4.547473509e-13f,
                -1.196976000e+01f,  -5.984880000e+00f,  1.457263808e+04f,  9.094947018e-13f,  -1.197680000e+01f,  -5.988400000e+00f,
                1.502215968e+04f,  9.094947018e-13f,  -1.213872000e+01f,  -6.069360000e+00f,  1.504076992e+04f,  -0.000000000e+00f,
                -1.214576000e+01f,  -6.072880000e+00f,  1.499221504e+04f,  1.364242053e-12f,  -1.213168000e+01f,  -6.065840000e+00f,
                1.501079712e+04f,  2.728484105e-12f,  -1.213872000e+01f,  -6.069360000e+00f,  1.496227040e+04f,  -0.000000000e+00f,
                -1.212464000e+01f,  -6.062320000e+00f,  1.498082432e+04f,  -9.094947018e-13f,  -1.213168000e+01f,  -6.065840000e+00f,
                1.505938016e+04f,  2.273736754e-12f,  -1.215280000e+01f,  -6.076400000e+00f,  1.507799040e+04f,  2.273736754e-12f,
                -1.215984000e+01f,  -6.079920000e+00f,  1.502937920e+04f,  1.818989404e-12f,  -1.214576000e+01f,  -6.072880000e+00f,
                1.504796128e+04f,  -1.818989404e-12f,  -1.215280000e+01f,  -6.076400000e+00f,  1.499937824e+04f,  -0.000000000e+00f,
                -1.213872000e+01f,  -6.069360000e+00f,  1.501793216e+04f,  -9.094947018e-13f,  -1.214576000e+01f,  -6.072880000e+00f,
                1.509660064e+04f,  -4.547473509e-13f,  -1.216688000e+01f,  -6.083440000e+00f,  1.511521088e+04f,  4.547473509e-13f,
                -1.217392000e+01f,  -6.086960000e+00f,  1.506654336e+04f,  9.094947018e-13f,  -1.215984000e+01f,  -6.079920000e+00f,
                1.508512544e+04f,  4.547473509e-13f,  -1.216688000e+01f,  -6.083440000e+00f,  1.503648608e+04f,  4.547473509e-13f,
                -1.215280000e+01f,  -6.076400000e+00f,  1.505504000e+04f,  -9.094947018e-13f,  -1.215984000e+01f,  -6.079920000e+00f,
                1.550602592e+04f,  -9.094947018e-13f,  -1.232176000e+01f,  -6.160880000e+00f,  1.552463616e+04f,  -1.818989404e-12f,
                -1.232880000e+01f,  -6.164400000e+00f,  1.547534912e+04f,  2.273736754e-12f,  -1.231472000e+01f,  -6.157360000e+00f,
                1.549393120e+04f,  -1.364242053e-12f,  -1.232176000e+01f,  -6.160880000e+00f,  1.544467232e+04f,  4.547473509e-13f,
                -1.230768000e+01f,  -6.153840000e+00f,  1.546322624e+04f,  -9.094947018e-13f,  -1.231472000e+01f,  -6.157360000e+00f,
                1.554324640e+04f,  -1.364242053e-12f,  -1.233584000e+01f,  -6.167920000e+00f,  1.556185664e+04f,  -1.818989404e-12f,
                -1.234288000e+01f,  -6.171440000e+00f,  1.551251328e+04f,  -3.183231456e-12f,  -1.232880000e+01f,  -6.164400000e+00f,
                1.553109536e+04f,  -4.092726158e-12f,  -1.233584000e+01f,  -6.167920000e+00f,  1.548178016e+04f,  4.547473509e-13f,
                -1.232176000e+01f,  -6.160880000e+00f,  1.550033408e+04f,  -0.000000000e+00f,  -1.232880000e+01f,  -6.164400000e+00f,
                1.558046688e+04f,  -4.547473509e-13f,  -1.234992000e+01f,  -6.174960000e+00f,  1.559907712e+04f,  9.094947018e-13f,
                -1.235696000e+01f,  -6.178480000e+00f,  1.554967744e+04f,  1.364242053e-12f,  -1.234288000e+01f,  -6.171440000e+00f,
                1.556825952e+04f,  -1.818989404e-12f,  -1.234992000e+01f,  -6.174960000e+00f,  1.551888800e+04f,  4.547473509e-13f,
                -1.233584000e+01f,  -6.167920000e+00f,  1.553744192e+04f,  9.094947018e-13f,  -1.234288000e+01f,  -6.171440000e+00f,
                1.598989216e+04f,  9.094947018e-13f,  -1.250480000e+01f,  -6.252400000e+00f,  1.600850240e+04f,  -0.000000000e+00f,
                -1.251184000e+01f,  -6.255920000e+00f,  1.595848320e+04f,  -4.547473509e-13f,  -1.249776000e+01f,  -6.248880000e+00f,
                1.597706528e+04f,  -9.094947018e-13f,  -1.250480000e+01f,  -6.252400000e+00f,  1.592707424e+04f,  9.094947018e-13f,
                -1.249072000e+01f,  -6.245360000e+00f,  1.594562816e+04f,  -9.094947018e-13f,  -1.249776000e+01f,  -6.248880000e+00f,
                1.602711264e+04f,  9.094947018e-13f,  -1.251888000e+01f,  -6.259440000e+00f,  1.604572288e+04f,  1.364242053e-12f,
                -1.252592000e+01f,  -6.262960000e+00f,  1.599564736e+04f,  -4.547473509e-13f,  -1.251184000e+01f,  -6.255920000e+00f,
                1.601422944e+04f,  -4.547473509e-13f,  -1.251888000e+01f,  -6.259440000e+00f,  1.596418208e+04f,  4.547473509e-13f,
                -1.250480000e+01f,  -6.252400000e+00f,  1.598273600e+04f,  -4.547473509e-13f,  -1.251184000e+01f,  -6.255920000e+00f,
                1.606433312e+04f,  -0.000000000e+00f,  -1.253296000e+01f,  -6.266480000e+00f,  1.608294336e+04f,  -4.547473509e-13f,
                -1.254000000e+01f,  -6.270000000e+00f,  1.603281152e+04f,  1.818989404e-12f,  -1.252592000e+01f,  -6.262960000e+00f,
                1.605139360e+04f,  -3.637978807e-12f,  -1.253296000e+01f,  -6.266480000e+00f,  1.600128992e+04f,  1.364242053e-12f,
                -1.251888000e+01f,  -6.259440000e+00f,  1.601984384e+04f,  4.547473509e-13f,  -1.252592000e+01f,  -6.262960000e+00f,
                1.647375840e+04f,  -1.818989404e-12f,  -1.268784000e+01f,  -6.343920000e+00f,  1.649236864e+04f,  -0.000000000e+00f,
                -1.269488000e+01f,  -6.347440000e+00f,  1.644161728e+04f,  -0.000000000e+00f,  -1.268080000e+01f,  -6.340400000e+00f,
                1.646019936e+04f,  -0.000000000e+00f,  -1.268784000e+01f,  -6.343920000e+00f,  1.640947616e+04f,  -1.818989404e-12f,
                -1.267376000e+01f,  -6.336880000e+00f,  1.642803008e+04f,  -0.000000000e+00f,  -1.268080000e+01f,  -6.340400000e+00f,
                1.651097888e+04f,  9.094947018e-13f,  -1.270192000e+01f,  -6.350960000e+00f,  1.652958912e+04f,  9.094947018e-13f,
                -1.270896000e+01f,  -6.354480000e+00f,  1.647878144e+04f,  1.818989404e-12f,  -1.269488000e+01f,  -6.347440000e+00f,
                1.649736352e+04f,  3.637978807e-12f,  -1.270192000e+01f,  -6.350960000e+00f,  1.644658400e+04f,  9.094947018e-13f,
                -1.268784000e+01f,  -6.343920000e+00f,  1.646513792e+04f,  -9.094947018e-13f,  -1.269488000e+01f,  -6.347440000e+00f,
                1.654819936e+04f,  -9.094947018e-13f,  -1.271600000e+01f,  -6.358000000e+00f,  1.656680960e+04f,  -0.000000000e+00f,
                -1.272304000e+01f,  -6.361520000e+00f,  1.651594560e+04f,  9.094947018e-13f,  -1.270896000e+01f,  -6.354480000e+00f,
                1.653452768e+04f,  1.818989404e-12f,  -1.271600000e+01f,  -6.358000000e+00f,  1.648369184e+04f,  -9.094947018e-13f,
                -1.270192000e+01f,  -6.350960000e+00f,  1.650224576e+04f,  -0.000000000e+00f,  -1.270896000e+01f,  -6.354480000e+00f,
                2.034468832e+04f,  -9.094947018e-13f,  -1.415216000e+01f,  -7.076080000e+00f,  2.036329856e+04f,  -9.094947018e-13f,
                -1.415920000e+01f,  -7.079600000e+00f,  2.030668992e+04f,  -9.094947018e-13f,  -1.414512000e+01f,  -7.072560000e+00f,
                2.032527200e+04f,  9.094947018e-13f,  -1.415216000e+01f,  -7.076080000e+00f,  2.026869152e+04f,  9.094947018e-13f,
                -1.413808000e+01f,  -7.069040000e+00f,  2.028724544e+04f,  -0.000000000e+00f,  -1.414512000e+01f,  -7.072560000e+00f,
                2.038190880e+04f,  1.818989404e-12f,  -1.416624000e+01f,  -7.083120000e+00f,  2.040051904e+04f,  -0.000000000e+00f,
                -1.417328000e+01f,  -7.086640000e+00f,  2.034385408e+04f,  -3.637978807e-12f,  -1.415920000e+01f,  -7.079600000e+00f,
                2.036243616e+04f,  -9.094947018e-13f,  -1.416624000e+01f,  -7.083120000e+00f,  2.030579936e+04f,  -9.094947018e-13f,
                -1.415216000e+01f,  -7.076080000e+00f,  2.032435328e+04f,  9.094947018e-13f,  -1.415920000e+01f,  -7.079600000e+00f,
                2.041912928e+04f,  9.094947018e-13f,  -1.418032000e+01f,  -7.090160000e+00f,  2.043773952e+04f,  -9.094947018e-13f,
                -1.418736000e+01f,  -7.093680000e+00f,  2.038101824e+04f,  1.818989404e-12f,  -1.417328000e+01f,  -7.086640000e+00f,
                2.039960032e+04f,  -9.094947018e-13f,  -1.418032000e+01f,  -7.090160000e+00f,  2.034290720e+04f,  -0.000000000e+00f,
                -1.416624000e+01f,  -7.083120000e+00f,  2.036146112e+04f,  -1.818989404e-12f,  -1.417328000e+01f,  -7.086640000e+00f,
                2.082855456e+04f,  -4.547473509e-12f,  -1.433520000e+01f,  -7.167600000e+00f,  2.084716480e+04f,  -0.000000000e+00f,
                -1.434224000e+01f,  -7.171120000e+00f,  2.078982400e+04f,  -0.000000000e+00f,  -1.432816000e+01f,  -7.164080000e+00f,
                2.080840608e+04f,  -9.094947018e-13f,  -1.433520000e+01f,  -7.167600000e+00f,  2.075109344e+04f,  9.094947018e-13f,
                -1.432112000e+01f,  -7.160560000e+00f,  2.076964736e+04f,  9.094947018e-13f,  -1.432816000e+01f,  -7.164080000e+00f,
                2.086577504e+04f,  -1.818989404e-12f,  -1.434928000e+01f,  -7.174640000e+00f,  2.088438528e+04f,  -9.094947018e-13f,
                -1.435632000e+01f,  -7.178160000e+00f,  2.082698816e+04f,  9.094947018e-13f,  -1.434224000e+01f,  -7.171120000e+00f,
                2.084557024e+04f,  9.094947018e-13f,  -1.434928000e+01f,  -7.174640000e+00f,  2.078820128e+04f,  -0.000000000e+00f,
                -1.433520000e+01f,  -7.167600000e+00f,  2.080675520e+04f,  1.818989404e-12f,  -1.434224000e+01f,  -7.171120000e+00f,
                2.090299552e+04f,  -9.094947018e-13f,  -1.436336000e+01f,  -7.181680000e+00f,  2.092160576e+04f,  1.818989404e-12f,
                -1.437040000e+01f,  -7.185200000e+00f,  2.086415232e+04f,  -9.094947018e-13f,  -1.435632000e+01f,  -7.178160000e+00f,
                2.088273440e+04f,  -1.818989404e-12f,  -1.436336000e+01f,  -7.181680000e+00f,  2.082530912e+04f,  -0.000000000e+00f,
                -1.434928000e+01f,  -7.174640000e+00f,  2.084386304e+04f,  -1.818989404e-12f,  -1.435632000e+01f,  -7.178160000e+00f,
                2.131242080e+04f,  1.818989404e-12f,  -1.451824000e+01f,  -7.259120000e+00f,  2.133103104e+04f,  -1.818989404e-12f,
                -1.452528000e+01f,  -7.262640000e+00f,  2.127295808e+04f,  -1.818989404e-12f,  -1.451120000e+01f,  -7.255600000e+00f,
                2.129154016e+04f,  -9.094947018e-13f,  -1.451824000e+01f,  -7.259120000e+00f,  2.123349536e+04f,  9.094947018e-13f,
                -1.450416000e+01f,  -7.252080000e+00f,  2.125204928e+04f,  9.094947018e-13f,  -1.451120000e+01f,  -7.255600000e+00f,
                2.134964128e+04f,  1.818989404e-12f,  -1.453232000e+01f,  -7.266160000e+00f,  2.136825152e+04f,  -9.094947018e-13f,
                -1.453936000e+01f,  -7.269680000e+00f,  2.131012224e+04f,  -1.818989404e-12f,  -1.452528000e+01f,  -7.262640000e+00f,
                2.132870432e+04f,  -0.000000000e+00f,  -1.453232000e+01f,  -7.266160000e+00f,  2.127060320e+04f,  -9.094947018e-13f,
                -1.451824000e+01f,  -7.259120000e+00f,  2.128915712e+04f,  -0.000000000e+00f,  -1.452528000e+01f,  -7.262640000e+00f,
                2.138686176e+04f,  9.094947018e-13f,  -1.454640000e+01f,  -7.273200000e+00f,  2.140547200e+04f,  9.094947018e-13f,
                -1.455344000e+01f,  -7.276720000e+00f,  2.134728640e+04f,  1.818989404e-12f,  -1.453936000e+01f,  -7.269680000e+00f,
                2.136586848e+04f,  9.094947018e-13f,  -1.454640000e+01f,  -7.273200000e+00f,  2.130771104e+04f,  9.094947018e-13f,
                -1.453232000e+01f,  -7.266160000e+00f,  2.132626496e+04f,  -0.000000000e+00f,  -1.453936000e+01f,  -7.269680000e+00f,
                2.179628704e+04f,  9.094947018e-13f,  -1.470128000e+01f,  -7.350640000e+00f,  2.181489728e+04f,  1.818989404e-12f,
                -1.470832000e+01f,  -7.354160000e+00f,  2.175609216e+04f,  -9.094947018e-13f,  -1.469424000e+01f,  -7.347120000e+00f,
                2.177467424e+04f,  -3.637978807e-12f,  -1.470128000e+01f,  -7.350640000e+00f,  2.171589728e+04f,  1.818989404e-12f,
                -1.468720000e+01f,  -7.343600000e+00f,  2.173445120e+04f,  9.094947018e-13f,  -1.469424000e+01f,  -7.347120000e+00f,
                2.183350752e+04f,  9.094947018e-13f,  -1.471536000e+01f,  -7.357680000e+00f,  2.185211776e+04f,  -9.094947018e-13f,
                -1.472240000e+01f,  -7.361200000e+00f,  2.179325632e+04f,  -2.728484105e-12f,  -1.470832000e+01f,  -7.354160000e+00f,
                2.181183840e+04f,  1.818989404e-12f,  -1.471536000e+01f,  -7.357680000e+00f,  2.175300512e+04f,  -9.094947018e-13f,
                -1.470128000e+01f,  -7.350640000e+00f,  2.177155904e+04f,  -9.094947018e-13f,  -1.470832000e+01f,  -7.354160000e+00f,
                2.187072800e+04f,  1.818989404e-12f,  -1.472944000e+01f,  -7.364720000e+00f,  2.188933824e+04f,  9.094947018e-13f,
                -1.473648000e+01f,  -7.368240000e+00f,  2.183042048e+04f,  -1.818989404e-12f,  -1.472240000e+01f,  -7.361200000e+00f,
                2.184900256e+04f,  -9.094947018e-13f,  -1.472944000e+01f,  -7.364720000e+00f,  2.179011296e+04f,  9.094947018e-13f,
                -1.471536000e+01f,  -7.357680000e+00f,  2.180866688e+04f,  1.818989404e-12f,  -1.472240000e+01f,  -7.361200000e+00f,
                2.228015328e+04f,  -0.000000000e+00f,  -1.488432000e+01f,  -7.442160000e+00f,  2.229876352e+04f,  -0.000000000e+00f,
                -1.489136000e+01f,  -7.445680000e+00f,  2.223922624e+04f,  -5.456968211e-12f,  -1.487728000e+01f,  -7.438640000e+00f,
                2.225780832e+04f,  -0.000000000e+00f,  -1.488432000e+01f,  -7.442160000e+00f,  2.219829920e+04f,  9.094947018e-13f,
                -1.487024000e+01f,  -7.435120000e+00f,  2.221685312e+04f,  9.094947018e-13f,  -1.487728000e+01f,  -7.438640000e+00f,
                2.231737376e+04f,  -0.000000000e+00f,  -1.489840000e+01f,  -7.449200000e+00f,  2.233598400e+04f,  2.728484105e-12f,
                -1.490544000e+01f,  -7.452720000e+00f,  2.227639040e+04f,  -0.000000000e+00f,  -1.489136000e+01f,  -7.445680000e+00f,
                2.229497248e+04f,  -9.094947018e-13f,  -1.489840000e+01f,  -7.449200000e+00f,  2.223540704e+04f,  9.094947018e-13f,
                -1.488432000e+01f,  -7.442160000e+00f,  2.225396096e+04f,  -0.000000000e+00f,  -1.489136000e+01f,  -7.445680000e+00f,
                2.235459424e+04f,  -9.094947018e-13f,  -1.491248000e+01f,  -7.456240000e+00f,  2.237320448e+04f,  -9.094947018e-13f,
                -1.491952000e+01f,  -7.459760000e+00f,  2.231355456e+04f,  -9.094947018e-13f,  -1.490544000e+01f,  -7.452720000e+00f,
                2.233213664e+04f,  9.094947018e-13f,  -1.491248000e+01f,  -7.456240000e+00f,  2.227251488e+04f,  1.818989404e-12f,
                -1.489840000e+01f,  -7.449200000e+00f,  2.229106880e+04f,  -0.000000000e+00f,  -1.490544000e+01f,  -7.452720000e+00f,
                2.615108320e+04f,  3.637978807e-12f,  -1.634864000e+01f,  -8.174320000e+00f,  2.616969344e+04f,  -0.000000000e+00f,
                -1.635568000e+01f,  -8.177840000e+00f,  2.610429888e+04f,  5.456968211e-12f,  -1.634160000e+01f,  -8.170800000e+00f,
                2.612288096e+04f,  -0.000000000e+00f,  -1.634864000e+01f,  -8.174320000e+00f,  2.605751456e+04f,  -1.818989404e-12f,
                -1.633456000e+01f,  -8.167280000e+00f,  2.607606848e+04f,  -0.000000000e+00f,  -1.634160000e+01f,  -8.170800000e+00f,
                2.618830368e+04f,  2.728484105e-12f,  -1.636272000e+01f,  -8.181360000e+00f,  2.620691392e+04f,  9.094947018e-13f,
                -1.636976000e+01f,  -8.184880000e+00f,  2.614146304e+04f,  -1.818989404e-12f,  -1.635568000e+01f,  -8.177840000e+00f,
                2.616004512e+04f,  -5.456968211e-12f,  -1.636272000e+01f,  -8.181360000e+00f,  2.609462240e+04f,  1.818989404e-12f,
                -1.634864000e+01f,  -8.174320000e+00f,  2.611317632e+04f,  -0.000000000e+00f,  -1.635568000e+01f,  -8.177840000e+00f,
                2.622552416e+04f,  -2.728484105e-12f,  -1.637680000e+01f,  -8.188400000e+00f,  2.624413440e+04f,  -3.637978807e-12f,
                -1.638384000e+01f,  -8.191920000e+00f,  2.617862720e+04f,  -1.818989404e-12f,  -1.636976000e+01f,  -8.184880000e+00f,
                2.619720928e+04f,  -3.637978807e-12f,  -1.637680000e+01f,  -8.188400000e+00f,  2.613173024e+04f,  -0.000000000e+00f,
                -1.636272000e+01f,  -8.181360000e+00f,  2.615028416e+04f,  -0.000000000e+00f,  -1.636976000e+01f,  -8.184880000e+00f,
                2.663494944e+04f,  2.728484105e-12f,  -1.653168000e+01f,  -8.265840000e+00f,  2.665355968e+04f,  -0.000000000e+00f,
                -1.653872000e+01f,  -8.269360000e+00f,  2.658743296e+04f,  -9.094947018e-13f,  -1.652464000e+01f,  -8.262320000e+00f,
                2.660601504e+04f,  3.637978807e-12f,  -1.653168000e+01f,  -8.265840000e+00f,  2.653991648e+04f,  -0.000000000e+00f,
                -1.651760000e+01f,  -8.258800000e+00f,  2.655847040e+04f,  -0.000000000e+00f,  -1.652464000e+01f,  -8.262320000e+00f,
                2.667216992e+04f,  -0.000000000e+00f,  -1.654576000e+01f,  -8.272880000e+00f,  2.669078016e+04f,  9.094947018e-13f,
                -1.655280000e+01f,  -8.276400000e+00f,  2.662459712e+04f,  9.094947018e-13f,  -1.653872000e+01f,  -8.269360000e+00f,
                2.664317920e+04f,  -9.094947018e-13f,  -1.654576000e+01f,  -8.272880000e+00f,  2.657702432e+04f,  -0.000000000e+00f,
                -1.653168000e+01f,  -8.265840000e+00f,  2.659557824e+04f,  -0.000000000e+00f,  -1.653872000e+01f,  -8.269360000e+00f,
                2.670939040e+04f,  -1.818989404e-12f,  -1.655984000e+01f,  -8.279920000e+00f,  2.672800064e+04f,  -1.818989404e-12f,
                -1.656688000e+01f,  -8.283440000e+00f,  2.666176128e+04f,  3.637978807e-12f,  -1.655280000e+01f,  -8.276400000e+00f,
                2.668034336e+04f,  -9.094947018e-13f,  -1.655984000e+01f,  -8.279920000e+00f,  2.661413216e+04f,  -9.094947018e-13f,
                -1.654576000e+01f,  -8.272880000e+00f,  2.663268608e+04f,  -1.818989404e-12f,  -1.655280000e+01f,  -8.276400000e+00f,
                2.711881568e+04f,  1.818989404e-12f,  -1.671472000e+01f,  -8.357360000e+00f,  2.713742592e+04f,  -9.094947018e-13f,
                -1.672176000e+01f,  -8.360880000e+00f,  2.707056704e+04f,  1.818989404e-12f,  -1.670768000e+01f,  -8.353840000e+00f,
                2.708914912e+04f,  2.728484105e-12f,  -1.671472000e+01f,  -8.357360000e+00f,  2.702231840e+04f,  9.094947018e-13f,
                -1.670064000e+01f,  -8.350320000e+00f,  2.704087232e+04f,  9.094947018e-13f,  -1.670768000e+01f,  -8.353840000e+00f,
                2.715603616e+04f,  2.728484105e-12f,  -1.672880000e+01f,  -8.364400000e+00f,  2.717464640e+04f,  -9.094947018e-13f,
                -1.673584000e+01f,  -8.367920000e+00f,  2.710773120e+04f,  1.818989404e-12f,  -1.672176000e+01f,  -8.360880000e+00f,
                2.712631328e+04f,  -2.728484105e-12f,  -1.672880000e+01f,  -8.364400000e+00f,  2.705942624e+04f,  -0.000000000e+00f,
                -1.671472000e+01f,  -8.357360000e+00f,  2.707798016e+04f,  9.094947018e-13f,  -1.672176000e+01f,  -8.360880000e+00f,
                2.719325664e+04f,  -9.094947018e-13f,  -1.674288000e+01f,  -8.371440000e+00f,  2.721186688e+04f,  -0.000000000e+00f,
                -1.674992000e+01f,  -8.374960000e+00f,  2.714489536e+04f,  -5.456968211e-12f,  -1.673584000e+01f,  -8.367920000e+00f,
                2.716347744e+04f,  -0.000000000e+00f,  -1.674288000e+01f,  -8.371440000e+00f,  2.709653408e+04f,  -0.000000000e+00f,
                -1.672880000e+01f,  -8.364400000e+00f,  2.711508800e+04f,  9.094947018e-13f,  -1.673584000e+01f,  -8.367920000e+00f,
                2.760268192e+04f,  9.094947018e-13f,  -1.689776000e+01f,  -8.448880000e+00f,  2.762129216e+04f,  1.818989404e-12f,
                -1.690480000e+01f,  -8.452400000e+00f,  2.755370112e+04f,  -3.637978807e-12f,  -1.689072000e+01f,  -8.445360000e+00f,
                2.757228320e+04f,  9.094947018e-13f,  -1.689776000e+01f,  -8.448880000e+00f,  2.750472032e+04f,  1.818989404e-12f,
                -1.688368000e+01f,  -8.441840000e+00f,  2.752327424e+04f,  -0.000000000e+00f,  -1.689072000e+01f,  -8.445360000e+00f,
                2.763990240e+04f,  9.094947018e-13f,  -1.691184000e+01f,  -8.455920000e+00f,  2.765851264e+04f,  -1.818989404e-12f,
                -1.691888000e+01f,  -8.459440000e+00f,  2.759086528e+04f,  -0.000000000e+00f,  -1.690480000e+01f,  -8.452400000e+00f,
                2.760944736e+04f,  -1.818989404e-12f,  -1.691184000e+01f,  -8.455920000e+00f,  2.754182816e+04f,  -1.818989404e-12f,
                -1.689776000e+01f,  -8.448880000e+00f,  2.756038208e+04f,  9.094947018e-13f,  -1.690480000e+01f,  -8.452400000e+00f,
                2.767712288e+04f,  1.818989404e-12f,  -1.692592000e+01f,  -8.462960000e+00f,  2.769573312e+04f,  -9.094947018e-13f,
                -1.693296000e+01f,  -8.466480000e+00f,  2.762802944e+04f,  4.547473509e-12f,  -1.691888000e+01f,  -8.459440000e+00f,
                2.764661152e+04f,  9.094947018e-13f,  -1.692592000e+01f,  -8.462960000e+00f,  2.757893600e+04f,  -0.000000000e+00f,
                -1.691184000e+01f,  -8.455920000e+00f,  2.759748992e+04f,  1.818989404e-12f,  -1.691888000e+01f,  -8.459440000e+00f,
                2.808654816e+04f,  5.456968211e-12f,  -1.708080000e+01f,  -8.540400000e+00f,  2.810515840e+04f,  5.456968211e-12f,
                -1.708784000e+01f,  -8.543920000e+00f,  2.803683520e+04f,  1.818989404e-12f,  -1.707376000e+01f,  -8.536880000e+00f,
                2.805541728e+04f,  -0.000000000e+00f,  -1.708080000e+01f,  -8.540400000e+00f,  2.798712224e+04f,  9.094947018e-13f,
                -1.706672000e+01f,  -8.533360000e+00f,  2.800567616e+04f,  -9.094947018e-13f,  -1.707376000e+01f,  -8.536880000e+00f,
                2.812376864e+04f,  -9.094947018e-13f,  -1.709488000e+01f,  -8.547440000e+00f,  2.814237888e+04f,  3.637978807e-12f,
                -1.710192000e+01f,  -8.550960000e+00f,  2.807399936e+04f,  -3.637978807e-12f,  -1.708784000e+01f,  -8.543920000e+00f,
                2.809258144e+04f,  9.094947018e-13f,  -1.709488000e+01f,  -8.547440000e+00f,  2.802423008e+04f,  -0.000000000e+00f,
                -1.708080000e+01f,  -8.540400000e+00f,  2.804278400e+04f,  -9.094947018e-13f,  -1.708784000e+01f,  -8.543920000e+00f,
                2.816098912e+04f,  -3.637978807e-12f,  -1.710896000e+01f,  -8.554480000e+00f,  2.817959936e+04f,  -9.094947018e-13f,
                -1.711600000e+01f,  -8.558000000e+00f,  2.811116352e+04f,  -9.094947018e-13f,  -1.710192000e+01f,  -8.550960000e+00f,
                2.812974560e+04f,  3.637978807e-12f,  -1.710896000e+01f,  -8.554480000e+00f,  2.806133792e+04f,  -0.000000000e+00f,
                -1.709488000e+01f,  -8.547440000e+00f,  2.807989184e+04f,  -0.000000000e+00f,  -1.710192000e+01f,  -8.550960000e+00f,
                3.195747808e+04f,  -3.637978807e-12f,  -1.854512000e+01f,  -9.272560000e+00f,  3.197608832e+04f,  3.637978807e-12f,
                -1.855216000e+01f,  -9.276080000e+00f,  3.190190784e+04f,  -1.818989404e-12f,  -1.853808000e+01f,  -9.269040000e+00f,
                3.192048992e+04f,  1.818989404e-12f,  -1.854512000e+01f,  -9.272560000e+00f,  3.184633760e+04f,  -9.094947018e-13f,
                -1.853104000e+01f,  -9.265520000e+00f,  3.186489152e+04f,  1.818989404e-12f,  -1.853808000e+01f,  -9.269040000e+00f,
                3.199469856e+04f,  1.818989404e-12f,  -1.855920000e+01f,  -9.279600000e+00f,  3.201330880e+04f,  -0.000000000e+00f,
                -1.856624000e+01f,  -9.283120000e+00f,  3.193907200e+04f,  -0.000000000e+00f,  -1.855216000e+01f,  -9.276080000e+00f,
                3.195765408e+04f,  -2.728484105e-12f,  -1.855920000e+01f,  -9.279600000e+00f,  3.188344544e+04f,  -0.000000000e+00f,
                -1.854512000e+01f,  -9.272560000e+00f,  3.190199936e+04f,  -9.094947018e-13f,  -1.855216000e+01f,  -9.276080000e+00f,
                3.203191904e+04f,  9.094947018e-13f,  -1.857328000e+01f,  -9.286640000e+00f,  3.205052928e+04f,  3.637978807e-12f,
                -1.858032000e+01f,  -9.290160000e+00f,  3.197623616e+04f,  1.818989404e-12f,  -1.856624000e+01f,  -9.283120000e+00f,
                3.199481824e+04f,  9.094947018e-13f,  -1.857328000e+01f,  -9.286640000e+00f,  3.192055328e+04f,  -2.728484105e-12f,
                -1.855920000e+01f,  -9.279600000e+00f,  3.193910720e+04f,  -0.000000000e+00f,  -1.856624000e+01f,  -9.283120000e+00f,
                3.244134432e+04f,  -2.728484105e-12f,  -1.872816000e+01f,  -9.364080000e+00f,  3.245995456e+04f,  2.728484105e-12f,
                -1.873520000e+01f,  -9.367600000e+00f,  3.238504192e+04f,  2.728484105e-12f,  -1.872112000e+01f,  -9.360560000e+00f,
                3.240362400e+04f,  5.456968211e-12f,  -1.872816000e+01f,  -9.364080000e+00f,  3.232873952e+04f,  9.094947018e-13f,
                -1.871408000e+01f,  -9.357040000e+00f,  3.234729344e+04f,  -0.000000000e+00f,  -1.872112000e+01f,  -9.360560000e+00f,
                3.247856480e+04f,  3.637978807e-12f,  -1.874224000e+01f,  -9.371120000e+00f,  3.249717504e+04f,  -9.094947018e-13f,
                -1.874928000e+01f,  -9.374640000e+00f,  3.242220608e+04f,  -3.637978807e-12f,  -1.873520000e+01f,  -9.367600000e+00f,
                3.244078816e+04f,  9.094947018e-13f,  -1.874224000e+01f,  -9.371120000e+00f,  3.236584736e+04f,  -1.818989404e-12f,
                -1.872816000e+01f,  -9.364080000e+00f,  3.238440128e+04f,  1.818989404e-12f,  -1.873520000e+01f,  -9.367600000e+00f,
                3.251578528e+04f,  -8.185452316e-12f,  -1.875632000e+01f,  -9.378160000e+00f,  3.253439552e+04f,  -2.728484105e-12f,
                -1.876336000e+01f,  -9.381680000e+00f,  3.245937024e+04f,  -9.094947018e-13f,  -1.874928000e+01f,  -9.374640000e+00f,
                3.247795232e+04f,  9.094947018e-13f,  -1.875632000e+01f,  -9.378160000e+00f,  3.240295520e+04f,  -0.000000000e+00f,
                -1.874224000e+01f,  -9.371120000e+00f,  3.242150912e+04f,  9.094947018e-13f,  -1.874928000e+01f,  -9.374640000e+00f,
                3.292521056e+04f,  -0.000000000e+00f,  -1.891120000e+01f,  -9.455600000e+00f,  3.294382080e+04f,  1.818989404e-12f,
                -1.891824000e+01f,  -9.459120000e+00f,  3.286817600e+04f,  -0.000000000e+00f,  -1.890416000e+01f,  -9.452080000e+00f,
                3.288675808e+04f,  1.818989404e-12f,  -1.891120000e+01f,  -9.455600000e+00f,  3.281114144e+04f,  -3.637978807e-12f,
                -1.889712000e+01f,  -9.448560000e+00f,  3.282969536e+04f,  1.818989404e-12f,  -1.890416000e+01f,  -9.452080000e+00f,
                3.296243104e+04f,  3.637978807e-12f,  -1.892528000e+01f,  -9.462640000e+00f,  3.298104128e+04f,  1.818989404e-12f,
                -1.893232000e+01f,  -9.466160000e+00f,  3.290534016e+04f,  -0.000000000e+00f,  -1.891824000e+01f,  -9.459120000e+00f,
                3.292392224e+04f,  -3.637978807e-12f,  -1.892528000e+01f,  -9.462640000e+00f,  3.284824928e+04f,  1.818989404e-12f,
                -1.891120000e+01f,  -9.455600000e+00f,  3.286680320e+04f,  1.818989404e-12f,  -1.891824000e+01f,  -9.459120000e+00f,
                3.299965152e+04f,  1.818989404e-12f,  -1.893936000e+01f,  -9.469680000e+00f,  3.301826176e+04f,  3.637978807e-12f,
                -1.894640000e+01f,  -9.473200000e+00f,  3.294250432e+04f,  -7.275957614e-12f,  -1.893232000e+01f,  -9.466160000e+00f,
                3.296108640e+04f,  -0.000000000e+00f,  -1.893936000e+01f,  -9.469680000e+00f,  3.288535712e+04f,  -1.818989404e-12f,
                -1.892528000e+01f,  -9.462640000e+00f,  3.290391104e+04f,  -0.000000000e+00f,  -1.893232000e+01f,  -9.466160000e+00f,
                3.340907680e+04f,  -0.000000000e+00f,  -1.909424000e+01f,  -9.547120000e+00f,  3.342768704e+04f,  -5.456968211e-12f,
                -1.910128000e+01f,  -9.550640000e+00f,  3.335131008e+04f,  3.637978807e-12f,  -1.908720000e+01f,  -9.543600000e+00f,
                3.336989216e+04f,  5.456968211e-12f,  -1.909424000e+01f,  -9.547120000e+00f,  3.329354336e+04f,  -0.000000000e+00f,
                -1.908016000e+01f,  -9.540080000e+00f,  3.331209728e+04f,  -0.000000000e+00f,  -1.908720000e+01f,  -9.543600000e+00f,
                3.344629728e+04f,  3.637978807e-12f,  -1.910832000e+01f,  -9.554160000e+00f,  3.346490752e+04f,  -5.456968211e-12f,
                -1.911536000e+01f,  -9.557680000e+00f,  3.338847424e+04f,  -0.000000000e+00f,  -1.910128000e+01f,  -9.550640000e+00f,
                3.340705632e+04f,  1.818989404e-12f,  -1.910832000e+01f,  -9.554160000e+00f,  3.333065120e+04f,  -1.818989404e-12f,
                -1.909424000e+01f,  -9.547120000e+00f,  3.334920512e+04f,  1.818989404e-12f,  -1.910128000e+01f,  -9.550640000e+00f,
                3.348351776e+04f,  -1.818989404e-12f,  -1.912240000e+01f,  -9.561200000e+00f,  3.350212800e+04f,  3.637978807e-12f,
                -1.912944000e+01f,  -9.564720000e+00f,  3.342563840e+04f,  -0.000000000e+00f,  -1.911536000e+01f,  -9.557680000e+00f,
                3.344422048e+04f,  1.818989404e-12f,  -1.912240000e+01f,  -9.561200000e+00f,  3.336775904e+04f,  -0.000000000e+00f,
                -1.910832000e+01f,  -9.554160000e+00f,  3.338631296e+04f,  -1.818989404e-12f,  -1.911536000e+01f,  -9.557680000e+00f,
                3.389294304e+04f,  -1.818989404e-12f,  -1.927728000e+01f,  -9.638640000e+00f,  3.391155328e+04f,  -0.000000000e+00f,
                -1.928432000e+01f,  -9.642160000e+00f,  3.383444416e+04f,  -1.818989404e-12f,  -1.927024000e+01f,  -9.635120000e+00f,
                3.385302624e+04f,  1.818989404e-12f,  -1.927728000e+01f,  -9.638640000e+00f,  3.377594528e+04f,  -0.000000000e+00f,
                -1.926320000e+01f,  -9.631600000e+00f,  3.379449920e+04f,  -1.818989404e-12f,  -1.927024000e+01f,  -9.635120000e+00f,
                3.393016352e+04f,  -0.000000000e+00f,  -1.929136000e+01f,  -9.645680000e+00f,  3.394877376e+04f,  1.818989404e-12f,
                -1.929840000e+01f,  -9.649200000e+00f,  3.387160832e+04f,  -1.818989404e-12f,  -1.928432000e+01f,  -9.642160000e+00f,
                3.389019040e+04f,  -1.818989404e-12f,  -1.929136000e+01f,  -9.645680000e+00f,  3.381305312e+04f,  -1.818989404e-12f,
                -1.927728000e+01f,  -9.638640000e+00f,  3.383160704e+04f,  -0.000000000e+00f,  -1.928432000e+01f,  -9.642160000e+00f,
                3.396738400e+04f,  -3.637978807e-12f,  -1.930544000e+01f,  -9.652720000e+00f,  3.398599424e+04f,  -7.275957614e-12f,
                -1.931248000e+01f,  -9.656240000e+00f,  3.390877248e+04f,  1.818989404e-12f,  -1.929840000e+01f,  -9.649200000e+00f,
                3.392735456e+04f,  1.818989404e-12f,  -1.930544000e+01f,  -9.652720000e+00f,  3.385016096e+04f,  -0.000000000e+00f,
                -1.929136000e+01f,  -9.645680000e+00f,  3.386871488e+04f,  -0.000000000e+00f,  -1.929840000e+01f,  -9.649200000e+00f,
                3.776387296e+04f,  -0.000000000e+00f,  -2.074160000e+01f,  -1.037080000e+01f,  3.778248320e+04f,  9.094947018e-12f,
                -2.074864000e+01f,  -1.037432000e+01f,  3.769951680e+04f,  -3.637978807e-12f,  -2.073456000e+01f,  -1.036728000e+01f,
                3.771809888e+04f,  -5.456968211e-12f,  -2.074160000e+01f,  -1.037080000e+01f,  3.763516064e+04f,  -1.818989404e-12f,
                -2.072752000e+01f,  -1.036376000e+01f,  3.765371456e+04f,  1.818989404e-12f,  -2.073456000e+01f,  -1.036728000e+01f,
                3.780109344e+04f,  1.818989404e-12f,  -2.075568000e+01f,  -1.037784000e+01f,  3.781970368e+04f,  1.818989404e-12f,
                -2.076272000e+01f,  -1.038136000e+01f,  3.773668096e+04f,  -3.637978807e-12f,  -2.074864000e+01f,  -1.037432000e+01f,
                3.775526304e+04f,  -0.000000000e+00f,  -2.075568000e+01f,  -1.037784000e+01f,  3.767226848e+04f,  -0.000000000e+00f,
                -2.074160000e+01f,  -1.037080000e+01f,  3.769082240e+04f,  3.637978807e-12f,  -2.074864000e+01f,  -1.037432000e+01f,
                3.783831392e+04f,  -1.818989404e-12f,  -2.076976000e+01f,  -1.038488000e+01f,  3.785692416e+04f,  -3.637978807e-12f,
                -2.077680000e+01f,  -1.038840000e+01f,  3.777384512e+04f,  -3.637978807e-12f,  -2.076272000e+01f,  -1.038136000e+01f,
                3.779242720e+04f,  3.637978807e-12f,  -2.076976000e+01f,  -1.038488000e+01f,  3.770937632e+04f,  -0.000000000e+00f,
                -2.075568000e+01f,  -1.037784000e+01f,  3.772793024e+04f,  3.637978807e-12f,  -2.076272000e+01f,  -1.038136000e+01f,
                3.824773920e+04f,  3.637978807e-12f,  -2.092464000e+01f,  -1.046232000e+01f,  3.826634944e+04f,  -0.000000000e+00f,
                -2.093168000e+01f,  -1.046584000e+01f,  3.818265088e+04f,  -1.818989404e-12f,  -2.091760000e+01f,  -1.045880000e+01f,
                3.820123296e+04f,  -0.000000000e+00f,  -2.092464000e+01f,  -1.046232000e+01f,  3.811756256e+04f,  -1.818989404e-12f,
                -2.091056000e+01f,  -1.045528000e+01f,  3.813611648e+04f,  -0.000000000e+00f,  -2.091760000e+01f,  -1.045880000e+01f,
                3.828495968e+04f,  3.637978807e-12f,  -2.093872000e+01f,  -1.046936000e+01f,  3.830356992e+04f,  -1.818989404e-12f,
                -2.094576000e+01f,  -1.047288000e+01f,  3.821981504e+04f,  1.818989404e-12f,  -2.093168000e+01f,  -1.046584000e+01f,
                3.823839712e+04f,  -0.000000000e+00f,  -2.093872000e+01f,  -1.046936000e+01f,  3.815467040e+04f,  -0.000000000e+00f,
                -2.092464000e+01f,  -1.046232000e+01f,  3.817322432e+04f,  1.818989404e-12f,  -2.093168000e+01f,  -1.046584000e+01f,
                3.832218016e+04f,  -5.456968211e-12f,  -2.095280000e+01f,  -1.047640000e+01f,  3.834079040e+04f,  -9.094947018e-12f,
                -2.095984000e+01f,  -1.047992000e+01f,  3.825697920e+04f,  -3.637978807e-12f,  -2.094576000e+01f,  -1.047288000e+01f,
                3.827556128e+04f,  -3.637978807e-12f,  -2.095280000e+01f,  -1.047640000e+01f,  3.819177824e+04f,  3.637978807e-12f,
                -2.093872000e+01f,  -1.046936000e+01f,  3.821033216e+04f,  1.818989404e-12f,  -2.094576000e+01f,  -1.047288000e+01f,
                3.873160544e+04f,  1.818989404e-12f,  -2.110768000e+01f,  -1.055384000e+01f,  3.875021568e+04f,  1.818989404e-12f,
                -2.111472000e+01f,  -1.055736000e+01f,  3.866578496e+04f,  -1.818989404e-12f,  -2.110064000e+01f,  -1.055032000e+01f,
                3.868436704e+04f,  5.456968211e-12f,  -2.110768000e+01f,  -1.055384000e+01f,  3.859996448e+04f,  1.818989404e-12f,
                -2.109360000e+01f,  -1.054680000e+01f,  3.861851840e+04f,  1.818989404e-12f,  -2.110064000e+01f,  -1.055032000e+01f,
                3.876882592e+04f,  -1.818989404e-12f,  -2.112176000e+01f,  -1.056088000e+01f,  3.878743616e+04f,  -3.637978807e-12f,
                -2.112880000e+01f,  -1.056440000e+01f,  3.870294912e+04f,  1.818989404e-12f,  -2.111472000e+01f,  -1.055736000e+01f,
                3.872153120e+04f,  -3.637978807e-12f,  -2.112176000e+01f,  -1.056088000e+01f,  3.863707232e+04f,  -1.818989404e-12f,
                -2.110768000e+01f,  -1.055384000e+01f,  3.865562624e+04f,  1.818989404e-12f,  -2.111472000e+01f,  -1.055736000e+01f,
                3.880604640e+04f,  3.637978807e-12f,  -2.113584000e+01f,  -1.056792000e+01f,  3.882465664e+04f,  1.818989404e-12f,
                -2.114288000e+01f,  -1.057144000e+01f,  3.874011328e+04f,  -0.000000000e+00f,  -2.112880000e+01f,  -1.056440000e+01f,
                3.875869536e+04f,  1.818989404e-12f,  -2.113584000e+01f,  -1.056792000e+01f,  3.867418016e+04f,  1.818989404e-12f,
                -2.112176000e+01f,  -1.056088000e+01f,  3.869273408e+04f,  -1.818989404e-12f,  -2.112880000e+01f,  -1.056440000e+01f,
                3.921547168e+04f,  -5.456968211e-12f,  -2.129072000e+01f,  -1.064536000e+01f,  3.923408192e+04f,  -0.000000000e+00f,
                -2.129776000e+01f,  -1.064888000e+01f,  3.914891904e+04f,  -0.000000000e+00f,  -2.128368000e+01f,  -1.064184000e+01f,
                3.916750112e+04f,  -0.000000000e+00f,  -2.129072000e+01f,  -1.064536000e+01f,  3.908236640e+04f,  -0.000000000e+00f,
                -2.127664000e+01f,  -1.063832000e+01f,  3.910092032e+04f,  -3.637978807e-12f,  -2.128368000e+01f,  -1.064184000e+01f,
                3.925269216e+04f,  5.456968211e-12f,  -2.130480000e+01f,  -1.065240000e+01f,  3.927130240e+04f,  1.818989404e-12f,
                -2.131184000e+01f,  -1.065592000e+01f,  3.918608320e+04f,  1.818989404e-12f,  -2.129776000e+01f,  -1.064888000e+01f,
                3.920466528e+04f,  3.637978807e-12f,  -2.130480000e+01f,  -1.065240000e+01f,  3.911947424e+04f,  -3.637978807e-12f,
                -2.129072000e+01f,  -1.064536000e+01f,  3.913802816e+04f,  1.818989404e-12f,  -2.129776000e+01f,  -1.064888000e+01f,
                3.928991264e+04f,  5.456968211e-12f,  -2.131888000e+01f,  -1.065944000e+01f,  3.930852288e+04f,  1.818989404e-12f,
                -2.132592000e+01f,  -1.066296000e+01f,  3.922324736e+04f,  1.818989404e-12f,  -2.131184000e+01f,  -1.065592000e+01f,
                3.924182944e+04f,  7.275957614e-12f,  -2.131888000e+01f,  -1.065944000e+01f,  3.915658208e+04f,  1.818989404e-12f,
                -2.130480000e+01f,  -1.065240000e+01f,  3.917513600e+04f,  1.818989404e-12f,  -2.131184000e+01f,  -1.065592000e+01f,
                3.969933792e+04f,  5.456968211e-12f,  -2.147376000e+01f,  -1.073688000e+01f,  3.971794816e+04f,  1.818989404e-12f,
                -2.148080000e+01f,  -1.074040000e+01f,  3.963205312e+04f,  -0.000000000e+00f,  -2.146672000e+01f,  -1.073336000e+01f,
                3.965063520e+04f,  -1.818989404e-12f,  -2.147376000e+01f,  -1.073688000e+01f,  3.956476832e+04f,  1.818989404e-12f,
                -2.145968000e+01f,  -1.072984000e+01f,  3.958332224e+04f,  1.818989404e-12f,  -2.146672000e+01f,  -1.073336000e+01f,
                3.973655840e+04f,  5.456968211e-12f,  -2.148784000e+01f,  -1.074392000e+01f,  3.975516864e+04f,  1.818989404e-12f,
                -2.149488000e+01f,  -1.074744000e+01f,  3.966921728e+04f,  -0.000000000e+00f,  -2.148080000e+01f,  -1.074040000e+01f,
                3.968779936e+04f,  1.818989404e-12f,  -2.148784000e+01f,  -1.074392000e+01f,  3.960187616e+04f,  -0.000000000e+00f,
                -2.147376000e+01f,  -1.073688000e+01f,  3.962043008e+04f,  1.818989404e-12f,  -2.148080000e+01f,  -1.074040000e+01f,
                3.977377888e+04f,  3.637978807e-12f,  -2.150192000e+01f,  -1.075096000e+01f,  3.979238912e+04f,  -1.818989404e-12f,
                -2.150896000e+01f,  -1.075448000e+01f,  3.970638144e+04f,  -5.456968211e-12f,  -2.149488000e+01f,  -1.074744000e+01f,
                3.972496352e+04f,  5.456968211e-12f,  -2.150192000e+01f,  -1.075096000e+01f,  3.963898400e+04f,  1.818989404e-12f,
                -2.148784000e+01f,  -1.074392000e+01f,  3.965753792e+04f,  -0.000000000e+00f,  -2.149488000e+01f,  -1.074744000e+01f,
                4.357026784e+04f,  -1.818989404e-12f,  -2.293808000e+01f,  -1.146904000e+01f,  4.358887808e+04f,  7.275957614e-12f,
                -2.294512000e+01f,  -1.147256000e+01f,  4.349712576e+04f,  3.637978807e-12f,  -2.293104000e+01f,  -1.146552000e+01f,
                4.351570784e+04f,  -5.456968211e-12f,  -2.293808000e+01f,  -1.146904000e+01f,  4.342398368e+04f,  1.818989404e-12f,
                -2.292400000e+01f,  -1.146200000e+01f,  4.344253760e+04f,  -5.456968211e-12f,  -2.293104000e+01f,  -1.146552000e+01f,
                4.360748832e+04f,  7.275957614e-12f,  -2.295216000e+01f,  -1.147608000e+01f,  4.362609856e+04f,  -7.275957614e-12f,
                -2.295920000e+01f,  -1.147960000e+01f,  4.353428992e+04f,  -7.275957614e-12f,  -2.294512000e+01f,  -1.147256000e+01f,
                4.355287200e+04f,  -3.637978807e-12f,  -2.295216000e+01f,  -1.147608000e+01f,  4.346109152e+04f,  7.275957614e-12f,
                -2.293808000e+01f,  -1.146904000e+01f,  4.347964544e+04f,  -1.818989404e-12f,  -2.294512000e+01f,  -1.147256000e+01f,
                4.364470880e+04f,  5.456968211e-12f,  -2.296624000e+01f,  -1.148312000e+01f,  4.366331904e+04f,  3.637978807e-12f,
                -2.297328000e+01f,  -1.148664000e+01f,  4.357145408e+04f,  5.456968211e-12f,  -2.295920000e+01f,  -1.147960000e+01f,
                4.359003616e+04f,  -1.818989404e-12f,  -2.296624000e+01f,  -1.148312000e+01f,  4.349819936e+04f,  3.637978807e-12f,
                -2.295216000e+01f,  -1.147608000e+01f,  4.351675328e+04f,  -1.818989404e-12f,  -2.295920000e+01f,  -1.147960000e+01f,
                4.405413408e+04f,  -1.818989404e-12f,  -2.312112000e+01f,  -1.156056000e+01f,  4.407274432e+04f,  3.637978807e-12f,
                -2.312816000e+01f,  -1.156408000e+01f,  4.398025984e+04f,  1.818989404e-12f,  -2.311408000e+01f,  -1.155704000e+01f,
                4.399884192e+04f,  5.456968211e-12f,  -2.312112000e+01f,  -1.156056000e+01f,  4.390638560e+04f,  -3.637978807e-12f,
                -2.310704000e+01f,  -1.155352000e+01f,  4.392493952e+04f,  -0.000000000e+00f,  -2.311408000e+01f,  -1.155704000e+01f,
                4.409135456e+04f,  -0.000000000e+00f,  -2.313520000e+01f,  -1.156760000e+01f,  4.410996480e+04f,  9.094947018e-12f,
                -2.314224000e+01f,  -1.157112000e+01f,  4.401742400e+04f,  -7.275957614e-12f,  -2.312816000e+01f,  -1.156408000e+01f,
                4.403600608e+04f,  1.818989404e-12f,  -2.313520000e+01f,  -1.156760000e+01f,  4.394349344e+04f,  5.456968211e-12f,
                -2.312112000e+01f,  -1.156056000e+01f,  4.396204736e+04f,  -1.818989404e-12f,  -2.312816000e+01f,  -1.156408000e+01f,
                4.412857504e+04f,  -1.818989404e-12f,  -2.314928000e+01f,  -1.157464000e+01f,  4.414718528e+04f,  1.818989404e-12f,
                -2.315632000e+01f,  -1.157816000e+01f,  4.405458816e+04f,  3.637978807e-12f,  -2.314224000e+01f,  -1.157112000e+01f,
                4.407317024e+04f,  1.818989404e-12f,  -2.314928000e+01f,  -1.157464000e+01f,  4.398060128e+04f,  1.818989404e-12f,
                -2.313520000e+01f,  -1.156760000e+01f,  4.399915520e+04f,  -1.818989404e-12f,  -2.314224000e+01f,  -1.157112000e+01f,
                4.453800032e+04f,  1.818989404e-12f,  -2.330416000e+01f,  -1.165208000e+01f,  4.455661056e+04f,  -5.456968211e-12f,
                -2.331120000e+01f,  -1.165560000e+01f,  4.446339392e+04f,  1.818989404e-12f,  -2.329712000e+01f,  -1.164856000e+01f,
                4.448197600e+04f,  5.456968211e-12f,  -2.330416000e+01f,  -1.165208000e+01f,  4.438878752e+04f,  -0.000000000e+00f,
                -2.329008000e+01f,  -1.164504000e+01f,  4.440734144e+04f,  -0.000000000e+00f,  -2.329712000e+01f,  -1.164856000e+01f,
                4.457522080e+04f,  3.637978807e-12f,  -2.331824000e+01f,  -1.165912000e+01f,  4.459383104e+04f,  1.818989404e-12f,
                -2.332528000e+01f,  -1.166264000e+01f,  4.450055808e+04f,  -3.637978807e-12f,  -2.331120000e+01f,  -1.165560000e+01f,
                4.451914016e+04f,  -1.818989404e-12f,  -2.331824000e+01f,  -1.165912000e+01f,  4.442589536e+04f,  -1.818989404e-12f,
                -2.330416000e+01f,  -1.165208000e+01f,  4.444444928e+04f,  1.818989404e-12f,  -2.331120000e+01f,  -1.165560000e+01f,
                4.461244128e+04f,  1.818989404e-12f,  -2.333232000e+01f,  -1.166616000e+01f,  4.463105152e+04f,  7.275957614e-12f,
                -2.333936000e+01f,  -1.166968000e+01f,  4.453772224e+04f,  -1.818989404e-12f,  -2.332528000e+01f,  -1.166264000e+01f,
                4.455630432e+04f,  -1.818989404e-12f,  -2.333232000e+01f,  -1.166616000e+01f,  4.446300320e+04f,  -0.000000000e+00f,
                -2.331824000e+01f,  -1.165912000e+01f,  4.448155712e+04f,  5.456968211e-12f,  -2.332528000e+01f,  -1.166264000e+01f,
                4.502186656e+04f,  -9.094947018e-12f,  -2.348720000e+01f,  -1.174360000e+01f,  4.504047680e+04f,  -0.000000000e+00f,
                -2.349424000e+01f,  -1.174712000e+01f,  4.494652800e+04f,  -5.456968211e-12f,  -2.348016000e+01f,  -1.174008000e+01f,
                4.496511008e+04f,  9.094947018e-12f,  -2.348720000e+01f,  -1.174360000e+01f,  4.487118944e+04f,  -1.818989404e-12f,
                -2.347312000e+01f,  -1.173656000e+01f,  4.488974336e+04f,  -1.818989404e-12f,  -2.348016000e+01f,  -1.174008000e+01f,
                4.505908704e+04f,  1.818989404e-12f,  -2.350128000e+01f,  -1.175064000e+01f,  4.507769728e+04f,  -3.637978807e-12f,
                -2.350832000e+01f,  -1.175416000e+01f,  4.498369216e+04f,  -0.000000000e+00f,  -2.349424000e+01f,  -1.174712000e+01f,
                4.500227424e+04f,  -5.456968211e-12f,  -2.350128000e+01f,  -1.175064000e+01f,  4.490829728e+04f,  3.637978807e-12f,
                -2.348720000e+01f,  -1.174360000e+01f,  4.492685120e+04f,  -1.818989404e-12f,  -2.349424000e+01f,  -1.174712000e+01f,
                4.509630752e+04f,  -1.818989404e-12f,  -2.351536000e+01f,  -1.175768000e+01f,  4.511491776e+04f,  -0.000000000e+00f,
                -2.352240000e+01f,  -1.176120000e+01f,  4.502085632e+04f,  3.637978807e-12f,  -2.350832000e+01f,  -1.175416000e+01f,
                4.503943840e+04f,  7.275957614e-12f,  -2.351536000e+01f,  -1.175768000e+01f,  4.494540512e+04f,  3.637978807e-12f,
                -2.350128000e+01f,  -1.175064000e+01f,  4.496395904e+04f,  -0.000000000e+00f,  -2.350832000e+01f,  -1.175416000e+01f,
                4.550573280e+04f,  -9.094947018e-12f,  -2.367024000e+01f,  -1.183512000e+01f,  4.552434304e+04f,  -3.637978807e-12f,
                -2.367728000e+01f,  -1.183864000e+01f,  4.542966208e+04f,  -1.818989404e-12f,  -2.366320000e+01f,  -1.183160000e+01f,
                4.544824416e+04f,  -5.456968211e-12f,  -2.367024000e+01f,  -1.183512000e+01f,  4.535359136e+04f,  1.818989404e-12f,
                -2.365616000e+01f,  -1.182808000e+01f,  4.537214528e+04f,  1.818989404e-12f,  -2.366320000e+01f,  -1.183160000e+01f,
                4.554295328e+04f,  -5.456968211e-12f,  -2.368432000e+01f,  -1.184216000e+01f,  4.556156352e+04f,  -0.000000000e+00f,
                -2.369136000e+01f,  -1.184568000e+01f,  4.546682624e+04f,  -1.818989404e-12f,  -2.367728000e+01f,  -1.183864000e+01f,
                4.548540832e+04f,  5.456968211e-12f,  -2.368432000e+01f,  -1.184216000e+01f,  4.539069920e+04f,  -1.818989404e-12f,
                -2.367024000e+01f,  -1.183512000e+01f,  4.540925312e+04f,  -1.818989404e-12f,  -2.367728000e+01f,  -1.183864000e+01f,
                4.558017376e+04f,  -1.818989404e-12f,  -2.369840000e+01f,  -1.184920000e+01f,  4.559878400e+04f,  5.456968211e-12f,
                -2.370544000e+01f,  -1.185272000e+01f,  4.550399040e+04f,  -0.000000000e+00f,  -2.369136000e+01f,  -1.184568000e+01f,
                4.552257248e+04f,  -3.637978807e-12f,  -2.369840000e+01f,  -1.184920000e+01f,  4.542780704e+04f,  1.818989404e-12f,
                -2.368432000e+01f,  -1.184216000e+01f,  4.544636096e+04f,  -1.818989404e-12f,  -2.369136000e+01f,  -1.184568000e+01f,
            };

            float[] gw_actual = gw.ToArray();

            AssertError.Tolerance(gw_expect, gw_actual, 1e-4f, 1e-4f, $"mismatch value {inchannels},{outchannels},{kwidth},{kheight},{kdepth},{inwidth},{inheight},{indepth},{batch}"); /*many fma tolerance*/
        }
    }
}
