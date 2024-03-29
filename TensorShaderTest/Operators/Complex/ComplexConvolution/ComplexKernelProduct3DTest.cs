using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;
using TensorShader;
using TensorShader.Operators.ComplexConvolution;
using TensorShaderCudaBackend.API;

namespace TensorShaderTest.Operators.Complex {
    [TestClass]
    public class ComplexKernelProduct3DTest {
        [TestMethod]
        public void ExecuteFPTest() {
            TensorShaderCudaBackend.Environment.Precision = TensorShaderCudaBackend.Environment.PrecisionMode.Float;
            TensorShaderCudaBackend.Environment.CudnnEnabled = false;

            Random random = new(1234);

            float max_err = 0;

            foreach (int batch in new int[] { 1, 2 }) {
                foreach ((int inchannels, int outchannels) in new (int, int)[] { (2, 2), (4, 2), (2, 4), (4, 10), (10, 4), (10, 20), (20, 32), (32, 10), (32, 34), (34, 34) }) {
                    foreach ((int kwidth, int kheight, int kdepth) in new (int, int, int)[] { (1, 1, 1), (3, 3, 3), (5, 5, 5), (1, 3, 5), (3, 5, 1), (5, 1, 3) }) {
                        foreach ((int inwidth, int inheight, int indepth) in new (int, int, int)[] { (kwidth, kheight, kdepth), (kwidth * 2, kheight * 2, kdepth * 2), (13, 13, 13), (17, 17, 17), (19, 19, 19), (17, 19, 13), (13, 17, 19), (19, 13, 17) }) {

                            int outwidth = inwidth - kwidth + 1, outheight = inheight - kheight + 1, outdepth = indepth - kdepth + 1;

                            float[] xval = (new float[inwidth * inheight * indepth * inchannels * batch]).Select((_, idx) => (float)random.NextDouble() * 1e-2f).ToArray();
                            float[] yval = (new float[outwidth * outheight * outdepth * outchannels * batch]).Select((_, idx) => (float)random.NextDouble() * 1e-2f).ToArray();

                            System.Numerics.Complex[] xcval = (new System.Numerics.Complex[xval.Length / 2])
                                .Select((_, idx) => new System.Numerics.Complex(xval[idx * 2], xval[idx * 2 + 1])).ToArray();

                            System.Numerics.Complex[] ycval = (new System.Numerics.Complex[yval.Length / 2])
                                .Select((_, idx) => new System.Numerics.Complex(yval[idx * 2], yval[idx * 2 + 1])).ToArray();

                            ComplexMap3D x = new(inchannels / 2, inwidth, inheight, indepth, batch, xcval);
                            ComplexMap3D y = new(outchannels / 2, outwidth, outheight, outdepth, batch, ycval);

                            ComplexFilter3D gw = Reference(x, y, kwidth, kheight, kdepth);

                            OverflowCheckedTensor x_tensor = new(Shape.Map3D(inchannels, inwidth, inheight, indepth, batch), xval);
                            OverflowCheckedTensor y_tensor = new(Shape.Map3D(outchannels, outwidth, outheight, outdepth, batch), yval);

                            OverflowCheckedTensor gw_tensor = new(Shape.Kernel3D(inchannels, outchannels / 2, kwidth, kheight, kdepth));

                            ComplexKernelProduct3D ope = new(inwidth, inheight, indepth, inchannels, outchannels, kwidth, kheight, kdepth, transpose: false, batch);

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
                foreach ((int inchannels, int outchannels) in new (int, int)[] { (2, 2), (4, 2), (2, 4), (4, 10), (10, 4), (10, 20), (20, 32), (32, 10), (32, 34), (34, 34) }) {
                    foreach ((int kwidth, int kheight, int kdepth) in new (int, int, int)[] { (1, 1, 1), (3, 3, 3), (5, 5, 5), (1, 3, 5), (3, 5, 1), (5, 1, 3) }) {
                        foreach ((int inwidth, int inheight, int indepth) in new (int, int, int)[] { (kwidth, kheight, kdepth), (kwidth * 2, kheight * 2, kdepth * 2), (13, 13, 13), (17, 17, 17), (19, 19, 19), (17, 19, 13), (13, 17, 19), (19, 13, 17) }) {

                            int outwidth = inwidth - kwidth + 1, outheight = inheight - kheight + 1, outdepth = indepth - kdepth + 1;

                            float[] xval = (new float[inwidth * inheight * indepth * inchannels * batch]).Select((_, idx) => (float)random.NextDouble() * 1e-2f).ToArray();
                            float[] yval = (new float[outwidth * outheight * outdepth * outchannels * batch]).Select((_, idx) => (float)random.NextDouble() * 1e-2f).ToArray();

                            System.Numerics.Complex[] xcval = (new System.Numerics.Complex[xval.Length / 2])
                                .Select((_, idx) => new System.Numerics.Complex(xval[idx * 2], xval[idx * 2 + 1])).ToArray();

                            System.Numerics.Complex[] ycval = (new System.Numerics.Complex[yval.Length / 2])
                                .Select((_, idx) => new System.Numerics.Complex(yval[idx * 2], yval[idx * 2 + 1])).ToArray();

                            ComplexMap3D x = new(inchannels / 2, inwidth, inheight, indepth, batch, xcval);
                            ComplexMap3D y = new(outchannels / 2, outwidth, outheight, outdepth, batch, ycval);

                            ComplexFilter3D gw = Reference(x, y, kwidth, kheight, kdepth);

                            OverflowCheckedTensor x_tensor = new(Shape.Map3D(inchannels, inwidth, inheight, indepth, batch), xval);
                            OverflowCheckedTensor y_tensor = new(Shape.Map3D(outchannels, outwidth, outheight, outdepth, batch), yval);

                            OverflowCheckedTensor gw_tensor = new(Shape.Kernel3D(inchannels, outchannels / 2, kwidth, kheight, kdepth));

                            ComplexKernelProduct3D ope = new(inwidth, inheight, indepth, inchannels, outchannels, kwidth, kheight, kdepth, transpose: false, batch);

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
            int inchannels = 98, outchannels = 100;
            int kwidth = 7, kheight = 5, kdepth = 3;
            int inwidth = 125, inheight = 196, indepth = 4;
            int outwidth = inwidth - kwidth + 1, outheight = inheight - kheight + 1, outdepth = indepth - kdepth + 1;

            float[] xval = (new float[inwidth * inheight * indepth * inchannels * batch]).Select((_, idx) => (float)random.NextDouble() * 1e-2f).ToArray();
            float[] yval = (new float[outwidth * outheight * outdepth * outchannels * batch]).Select((_, idx) => (float)random.NextDouble() * 1e-2f).ToArray();

            System.Numerics.Complex[] xcval = (new System.Numerics.Complex[xval.Length / 2])
                .Select((_, idx) => new System.Numerics.Complex(xval[idx * 2], xval[idx * 2 + 1])).ToArray();

            System.Numerics.Complex[] ycval = (new System.Numerics.Complex[yval.Length / 2])
                .Select((_, idx) => new System.Numerics.Complex(yval[idx * 2], yval[idx * 2 + 1])).ToArray();

            ComplexMap3D x = new(inchannels / 2, inwidth, inheight, indepth, batch, xcval);
            ComplexMap3D y = new(outchannels / 2, outwidth, outheight, outdepth, batch, ycval);

            ComplexFilter3D gw = Reference(x, y, kwidth, kheight, kdepth);

            OverflowCheckedTensor x_tensor = new(Shape.Map3D(inchannels, inwidth, inheight, indepth, batch), xval);
            OverflowCheckedTensor y_tensor = new(Shape.Map3D(outchannels, outwidth, outheight, outdepth, batch), yval);

            OverflowCheckedTensor gw_tensor = new(Shape.Kernel3D(inchannels, outchannels / 2, kwidth, kheight, kdepth));

            ComplexKernelProduct3D ope = new(inwidth, inheight, indepth, inchannels, outchannels, kwidth, kheight, kdepth, transpose: false, batch);

            ope.Execute(x_tensor, y_tensor, gw_tensor);

            float[] gw_expect = gw.ToArray();
            float[] gw_actual = gw_tensor.State.Value;

            CollectionAssert.AreEqual(xval, x_tensor.State.Value);
            CollectionAssert.AreEqual(yval, y_tensor.State.Value);

            AssertError.Tolerance(gw_expect, gw_actual, 1e-6f, 1e-4f, ref max_err, $"mismatch value {inchannels},{outchannels},{kwidth},{kheight},{kdepth},{inwidth},{inheight},{indepth},{batch}");

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

            OverflowCheckedTensor gw_tensor = new(Shape.Kernel3D(inchannels, outchannels / 2, ksize, ksize, ksize));

            ComplexKernelProduct3D ope = new(inwidth, inheight, indepth, inchannels, outchannels, ksize, ksize, ksize);

            Cuda.Profiler.Initialize("../../../../profiler.nvsetting", "../../nvprofiles/complex_kernelproduct_3d_fp.nvvp");
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

            OverflowCheckedTensor gw_tensor = new(Shape.Kernel3D(inchannels, outchannels / 2, ksize, ksize, ksize));

            ComplexKernelProduct3D ope = new(inwidth, inheight, indepth, inchannels, outchannels, ksize, ksize, ksize);

            Cuda.Profiler.Initialize("../../../../profiler.nvsetting", "../../nvprofiles/complex_kernelproduct_3d_ffp.nvvp");
            Cuda.Profiler.Start();

            ope.Execute(x_tensor, y_tensor, gw_tensor);

            Cuda.Profiler.Stop();
        }

        public static ComplexFilter3D Reference(ComplexMap3D x, ComplexMap3D gy, int kwidth, int kheight, int kdepth) {
            int inchannels = x.Channels, outchannels = gy.Channels, batch = x.Batch;
            int inw = x.Width, inh = x.Height, ind = x.Depth, outw = gy.Width, outh = gy.Height, outd = gy.Depth;

            if (outw != inw - kwidth + 1 || outh != inh - kheight + 1 || outd != ind - kdepth + 1) {
                throw new ArgumentException("mismatch shape");
            }

            ComplexFilter3D w = new(inchannels, outchannels, kwidth, kheight, kdepth);

            Func<System.Numerics.Complex, System.Numerics.Complex, System.Numerics.Complex> mul_grad = (z1, z2) => {
                return new System.Numerics.Complex(z1.Real * z2.Real + z1.Imaginary * z2.Imaginary, z1.Imaginary * z2.Real - z1.Real * z2.Imaginary);
            };

            for (int kx, ky, kz = 0; kz < kdepth; kz++) {
                for (ky = 0; ky < kheight; ky++) {
                    for (kx = 0; kx < kwidth; kx++) {
                        for (int th = 0; th < batch; th++) {
                            for (int ix, iy, iz = kz, ox, oy, oz = 0; oz < outd; iz++, oz++) {
                                for (iy = ky, oy = 0; oy < outh; iy++, oy++) {
                                    for (ix = kx, ox = 0; ox < outw; ix++, ox++) {
                                        for (int inch, outch = 0; outch < outchannels; outch++) {
                                            for (inch = 0; inch < inchannels; inch++) {
                                                w[inch, outch, kx, ky, kz] += mul_grad(gy[outch, ox, oy, oz, th], x[inch, ix, iy, iz, th]);
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
            int inchannels = 6, outchannels = 8, kwidth = 3, kheight = 5, kdepth = 7, inwidth = 13, inheight = 12, indepth = 11;
            int outwidth = inwidth - kwidth + 1, outheight = inheight - kheight + 1, outdepth = indepth - kdepth + 1, batch = 1;

            float[] xval = (new float[inwidth * inheight * indepth * inchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
            float[] yval = (new float[outwidth * outheight * outdepth * outchannels * batch]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

            System.Numerics.Complex[] xcval = (new System.Numerics.Complex[xval.Length / 2])
                .Select((_, idx) => new System.Numerics.Complex(xval[idx * 2], xval[idx * 2 + 1])).ToArray();

            System.Numerics.Complex[] ycval = (new System.Numerics.Complex[yval.Length / 2])
                .Select((_, idx) => new System.Numerics.Complex(yval[idx * 2], yval[idx * 2 + 1])).ToArray();

            ComplexMap3D x = new(inchannels / 2, inwidth, inheight, indepth, batch, xcval);
            ComplexMap3D y = new(outchannels / 2, outwidth, outheight, outdepth, batch, ycval);

            ComplexFilter3D gw = Reference(x, y, kwidth, kheight, kdepth);

            float[] gw_expect = {
                2.182324760e+03f,  -1.732720000e+00f,  2.185426760e+03f,  -1.733600000e+00f,  2.188528760e+03f,  -1.734480000e+00f,
                2.178495880e+03f,  -1.731840000e+00f,  2.181594360e+03f,  -1.732720000e+00f,  2.184692840e+03f,  -1.733600000e+00f,
                2.174667000e+03f,  -1.730960000e+00f,  2.177761960e+03f,  -1.731840000e+00f,  2.180856920e+03f,  -1.732720000e+00f,
                2.170838120e+03f,  -1.730080000e+00f,  2.173929560e+03f,  -1.730960000e+00f,  2.177021000e+03f,  -1.731840000e+00f,
                2.191630760e+03f,  -1.735360000e+00f,  2.194732760e+03f,  -1.736240000e+00f,  2.197834760e+03f,  -1.737120000e+00f,
                2.187791320e+03f,  -1.734480000e+00f,  2.190889800e+03f,  -1.735360000e+00f,  2.193988280e+03f,  -1.736240000e+00f,
                2.183951880e+03f,  -1.733600000e+00f,  2.187046840e+03f,  -1.734480000e+00f,  2.190141800e+03f,  -1.735360000e+00f,
                2.180112440e+03f,  -1.732720000e+00f,  2.183203880e+03f,  -1.733600000e+00f,  2.186295320e+03f,  -1.734480000e+00f,
                2.200936760e+03f,  -1.738000000e+00f,  2.204038760e+03f,  -1.738880000e+00f,  2.207140760e+03f,  -1.739760000e+00f,
                2.197086760e+03f,  -1.737120000e+00f,  2.200185240e+03f,  -1.738000000e+00f,  2.203283720e+03f,  -1.738880000e+00f,
                2.193236760e+03f,  -1.736240000e+00f,  2.196331720e+03f,  -1.737120000e+00f,  2.199426680e+03f,  -1.738000000e+00f,
                2.189386760e+03f,  -1.735360000e+00f,  2.192478200e+03f,  -1.736240000e+00f,  2.195569640e+03f,  -1.737120000e+00f,
                2.303302760e+03f,  -1.767040000e+00f,  2.306404760e+03f,  -1.767920000e+00f,  2.309506760e+03f,  -1.768800000e+00f,
                2.299336600e+03f,  -1.766160000e+00f,  2.302435080e+03f,  -1.767040000e+00f,  2.305533560e+03f,  -1.767920000e+00f,
                2.295370440e+03f,  -1.765280000e+00f,  2.298465400e+03f,  -1.766160000e+00f,  2.301560360e+03f,  -1.767040000e+00f,
                2.291404280e+03f,  -1.764400000e+00f,  2.294495720e+03f,  -1.765280000e+00f,  2.297587160e+03f,  -1.766160000e+00f,
                2.312608760e+03f,  -1.769680000e+00f,  2.315710760e+03f,  -1.770560000e+00f,  2.318812760e+03f,  -1.771440000e+00f,
                2.308632040e+03f,  -1.768800000e+00f,  2.311730520e+03f,  -1.769680000e+00f,  2.314829000e+03f,  -1.770560000e+00f,
                2.304655320e+03f,  -1.767920000e+00f,  2.307750280e+03f,  -1.768800000e+00f,  2.310845240e+03f,  -1.769680000e+00f,
                2.300678600e+03f,  -1.767040000e+00f,  2.303770040e+03f,  -1.767920000e+00f,  2.306861480e+03f,  -1.768800000e+00f,
                2.321914760e+03f,  -1.772320000e+00f,  2.325016760e+03f,  -1.773200000e+00f,  2.328118760e+03f,  -1.774080000e+00f,
                2.317927480e+03f,  -1.771440000e+00f,  2.321025960e+03f,  -1.772320000e+00f,  2.324124440e+03f,  -1.773200000e+00f,
                2.313940200e+03f,  -1.770560000e+00f,  2.317035160e+03f,  -1.771440000e+00f,  2.320130120e+03f,  -1.772320000e+00f,
                2.309952920e+03f,  -1.769680000e+00f,  2.313044360e+03f,  -1.770560000e+00f,  2.316135800e+03f,  -1.771440000e+00f,
                2.424280760e+03f,  -1.801360000e+00f,  2.427382760e+03f,  -1.802240000e+00f,  2.430484760e+03f,  -1.803120000e+00f,
                2.420177320e+03f,  -1.800480000e+00f,  2.423275800e+03f,  -1.801360000e+00f,  2.426374280e+03f,  -1.802240000e+00f,
                2.416073880e+03f,  -1.799600000e+00f,  2.419168840e+03f,  -1.800480000e+00f,  2.422263800e+03f,  -1.801360000e+00f,
                2.411970440e+03f,  -1.798720000e+00f,  2.415061880e+03f,  -1.799600000e+00f,  2.418153320e+03f,  -1.800480000e+00f,
                2.433586760e+03f,  -1.804000000e+00f,  2.436688760e+03f,  -1.804880000e+00f,  2.439790760e+03f,  -1.805760000e+00f,
                2.429472760e+03f,  -1.803120000e+00f,  2.432571240e+03f,  -1.804000000e+00f,  2.435669720e+03f,  -1.804880000e+00f,
                2.425358760e+03f,  -1.802240000e+00f,  2.428453720e+03f,  -1.803120000e+00f,  2.431548680e+03f,  -1.804000000e+00f,
                2.421244760e+03f,  -1.801360000e+00f,  2.424336200e+03f,  -1.802240000e+00f,  2.427427640e+03f,  -1.803120000e+00f,
                2.442892760e+03f,  -1.806640000e+00f,  2.445994760e+03f,  -1.807520000e+00f,  2.449096760e+03f,  -1.808400000e+00f,
                2.438768200e+03f,  -1.805760000e+00f,  2.441866680e+03f,  -1.806640000e+00f,  2.444965160e+03f,  -1.807520000e+00f,
                2.434643640e+03f,  -1.804880000e+00f,  2.437738600e+03f,  -1.805760000e+00f,  2.440833560e+03f,  -1.806640000e+00f,
                2.430519080e+03f,  -1.804000000e+00f,  2.433610520e+03f,  -1.804880000e+00f,  2.436701960e+03f,  -1.805760000e+00f,
                2.545258760e+03f,  -1.835680000e+00f,  2.548360760e+03f,  -1.836560000e+00f,  2.551462760e+03f,  -1.837440000e+00f,
                2.541018040e+03f,  -1.834800000e+00f,  2.544116520e+03f,  -1.835680000e+00f,  2.547215000e+03f,  -1.836560000e+00f,
                2.536777320e+03f,  -1.833920000e+00f,  2.539872280e+03f,  -1.834800000e+00f,  2.542967240e+03f,  -1.835680000e+00f,
                2.532536600e+03f,  -1.833040000e+00f,  2.535628040e+03f,  -1.833920000e+00f,  2.538719480e+03f,  -1.834800000e+00f,
                2.554564760e+03f,  -1.838320000e+00f,  2.557666760e+03f,  -1.839200000e+00f,  2.560768760e+03f,  -1.840080000e+00f,
                2.550313480e+03f,  -1.837440000e+00f,  2.553411960e+03f,  -1.838320000e+00f,  2.556510440e+03f,  -1.839200000e+00f,
                2.546062200e+03f,  -1.836560000e+00f,  2.549157160e+03f,  -1.837440000e+00f,  2.552252120e+03f,  -1.838320000e+00f,
                2.541810920e+03f,  -1.835680000e+00f,  2.544902360e+03f,  -1.836560000e+00f,  2.547993800e+03f,  -1.837440000e+00f,
                2.563870760e+03f,  -1.840960000e+00f,  2.566972760e+03f,  -1.841840000e+00f,  2.570074760e+03f,  -1.842720000e+00f,
                2.559608920e+03f,  -1.840080000e+00f,  2.562707400e+03f,  -1.840960000e+00f,  2.565805880e+03f,  -1.841840000e+00f,
                2.555347080e+03f,  -1.839200000e+00f,  2.558442040e+03f,  -1.840080000e+00f,  2.561537000e+03f,  -1.840960000e+00f,
                2.551085240e+03f,  -1.838320000e+00f,  2.554176680e+03f,  -1.839200000e+00f,  2.557268120e+03f,  -1.840080000e+00f,
                2.666236760e+03f,  -1.870000000e+00f,  2.669338760e+03f,  -1.870880000e+00f,  2.672440760e+03f,  -1.871760000e+00f,
                2.661858760e+03f,  -1.869120000e+00f,  2.664957240e+03f,  -1.870000000e+00f,  2.668055720e+03f,  -1.870880000e+00f,
                2.657480760e+03f,  -1.868240000e+00f,  2.660575720e+03f,  -1.869120000e+00f,  2.663670680e+03f,  -1.870000000e+00f,
                2.653102760e+03f,  -1.867360000e+00f,  2.656194200e+03f,  -1.868240000e+00f,  2.659285640e+03f,  -1.869120000e+00f,
                2.675542760e+03f,  -1.872640000e+00f,  2.678644760e+03f,  -1.873520000e+00f,  2.681746760e+03f,  -1.874400000e+00f,
                2.671154200e+03f,  -1.871760000e+00f,  2.674252680e+03f,  -1.872640000e+00f,  2.677351160e+03f,  -1.873520000e+00f,
                2.666765640e+03f,  -1.870880000e+00f,  2.669860600e+03f,  -1.871760000e+00f,  2.672955560e+03f,  -1.872640000e+00f,
                2.662377080e+03f,  -1.870000000e+00f,  2.665468520e+03f,  -1.870880000e+00f,  2.668559960e+03f,  -1.871760000e+00f,
                2.684848760e+03f,  -1.875280000e+00f,  2.687950760e+03f,  -1.876160000e+00f,  2.691052760e+03f,  -1.877040000e+00f,
                2.680449640e+03f,  -1.874400000e+00f,  2.683548120e+03f,  -1.875280000e+00f,  2.686646600e+03f,  -1.876160000e+00f,
                2.676050520e+03f,  -1.873520000e+00f,  2.679145480e+03f,  -1.874400000e+00f,  2.682240440e+03f,  -1.875280000e+00f,
                2.671651400e+03f,  -1.872640000e+00f,  2.674742840e+03f,  -1.873520000e+00f,  2.677834280e+03f,  -1.874400000e+00f,
                3.634060760e+03f,  -2.144560000e+00f,  3.637162760e+03f,  -2.145440000e+00f,  3.640264760e+03f,  -2.146320000e+00f,
                3.628584520e+03f,  -2.143680000e+00f,  3.631683000e+03f,  -2.144560000e+00f,  3.634781480e+03f,  -2.145440000e+00f,
                3.623108280e+03f,  -2.142800000e+00f,  3.626203240e+03f,  -2.143680000e+00f,  3.629298200e+03f,  -2.144560000e+00f,
                3.617632040e+03f,  -2.141920000e+00f,  3.620723480e+03f,  -2.142800000e+00f,  3.623814920e+03f,  -2.143680000e+00f,
                3.643366760e+03f,  -2.147200000e+00f,  3.646468760e+03f,  -2.148080000e+00f,  3.649570760e+03f,  -2.148960000e+00f,
                3.637879960e+03f,  -2.146320000e+00f,  3.640978440e+03f,  -2.147200000e+00f,  3.644076920e+03f,  -2.148080000e+00f,
                3.632393160e+03f,  -2.145440000e+00f,  3.635488120e+03f,  -2.146320000e+00f,  3.638583080e+03f,  -2.147200000e+00f,
                3.626906360e+03f,  -2.144560000e+00f,  3.629997800e+03f,  -2.145440000e+00f,  3.633089240e+03f,  -2.146320000e+00f,
                3.652672760e+03f,  -2.149840000e+00f,  3.655774760e+03f,  -2.150720000e+00f,  3.658876760e+03f,  -2.151600000e+00f,
                3.647175400e+03f,  -2.148960000e+00f,  3.650273880e+03f,  -2.149840000e+00f,  3.653372360e+03f,  -2.150720000e+00f,
                3.641678040e+03f,  -2.148080000e+00f,  3.644773000e+03f,  -2.148960000e+00f,  3.647867960e+03f,  -2.149840000e+00f,
                3.636180680e+03f,  -2.147200000e+00f,  3.639272120e+03f,  -2.148080000e+00f,  3.642363560e+03f,  -2.148960000e+00f,
                3.755038760e+03f,  -2.178880000e+00f,  3.758140760e+03f,  -2.179760000e+00f,  3.761242760e+03f,  -2.180640000e+00f,
                3.749425240e+03f,  -2.178000000e+00f,  3.752523720e+03f,  -2.178880000e+00f,  3.755622200e+03f,  -2.179760000e+00f,
                3.743811720e+03f,  -2.177120000e+00f,  3.746906680e+03f,  -2.178000000e+00f,  3.750001640e+03f,  -2.178880000e+00f,
                3.738198200e+03f,  -2.176240000e+00f,  3.741289640e+03f,  -2.177120000e+00f,  3.744381080e+03f,  -2.178000000e+00f,
                3.764344760e+03f,  -2.181520000e+00f,  3.767446760e+03f,  -2.182400000e+00f,  3.770548760e+03f,  -2.183280000e+00f,
                3.758720680e+03f,  -2.180640000e+00f,  3.761819160e+03f,  -2.181520000e+00f,  3.764917640e+03f,  -2.182400000e+00f,
                3.753096600e+03f,  -2.179760000e+00f,  3.756191560e+03f,  -2.180640000e+00f,  3.759286520e+03f,  -2.181520000e+00f,
                3.747472520e+03f,  -2.178880000e+00f,  3.750563960e+03f,  -2.179760000e+00f,  3.753655400e+03f,  -2.180640000e+00f,
                3.773650760e+03f,  -2.184160000e+00f,  3.776752760e+03f,  -2.185040000e+00f,  3.779854760e+03f,  -2.185920000e+00f,
                3.768016120e+03f,  -2.183280000e+00f,  3.771114600e+03f,  -2.184160000e+00f,  3.774213080e+03f,  -2.185040000e+00f,
                3.762381480e+03f,  -2.182400000e+00f,  3.765476440e+03f,  -2.183280000e+00f,  3.768571400e+03f,  -2.184160000e+00f,
                3.756746840e+03f,  -2.181520000e+00f,  3.759838280e+03f,  -2.182400000e+00f,  3.762929720e+03f,  -2.183280000e+00f,
                3.876016760e+03f,  -2.213200000e+00f,  3.879118760e+03f,  -2.214080000e+00f,  3.882220760e+03f,  -2.214960000e+00f,
                3.870265960e+03f,  -2.212320000e+00f,  3.873364440e+03f,  -2.213200000e+00f,  3.876462920e+03f,  -2.214080000e+00f,
                3.864515160e+03f,  -2.211440000e+00f,  3.867610120e+03f,  -2.212320000e+00f,  3.870705080e+03f,  -2.213200000e+00f,
                3.858764360e+03f,  -2.210560000e+00f,  3.861855800e+03f,  -2.211440000e+00f,  3.864947240e+03f,  -2.212320000e+00f,
                3.885322760e+03f,  -2.215840000e+00f,  3.888424760e+03f,  -2.216720000e+00f,  3.891526760e+03f,  -2.217600000e+00f,
                3.879561400e+03f,  -2.214960000e+00f,  3.882659880e+03f,  -2.215840000e+00f,  3.885758360e+03f,  -2.216720000e+00f,
                3.873800040e+03f,  -2.214080000e+00f,  3.876895000e+03f,  -2.214960000e+00f,  3.879989960e+03f,  -2.215840000e+00f,
                3.868038680e+03f,  -2.213200000e+00f,  3.871130120e+03f,  -2.214080000e+00f,  3.874221560e+03f,  -2.214960000e+00f,
                3.894628760e+03f,  -2.218480000e+00f,  3.897730760e+03f,  -2.219360000e+00f,  3.900832760e+03f,  -2.220240000e+00f,
                3.888856840e+03f,  -2.217600000e+00f,  3.891955320e+03f,  -2.218480000e+00f,  3.895053800e+03f,  -2.219360000e+00f,
                3.883084920e+03f,  -2.216720000e+00f,  3.886179880e+03f,  -2.217600000e+00f,  3.889274840e+03f,  -2.218480000e+00f,
                3.877313000e+03f,  -2.215840000e+00f,  3.880404440e+03f,  -2.216720000e+00f,  3.883495880e+03f,  -2.217600000e+00f,
                3.996994760e+03f,  -2.247520000e+00f,  4.000096760e+03f,  -2.248400000e+00f,  4.003198760e+03f,  -2.249280000e+00f,
                3.991106680e+03f,  -2.246640000e+00f,  3.994205160e+03f,  -2.247520000e+00f,  3.997303640e+03f,  -2.248400000e+00f,
                3.985218600e+03f,  -2.245760000e+00f,  3.988313560e+03f,  -2.246640000e+00f,  3.991408520e+03f,  -2.247520000e+00f,
                3.979330520e+03f,  -2.244880000e+00f,  3.982421960e+03f,  -2.245760000e+00f,  3.985513400e+03f,  -2.246640000e+00f,
                4.006300760e+03f,  -2.250160000e+00f,  4.009402760e+03f,  -2.251040000e+00f,  4.012504760e+03f,  -2.251920000e+00f,
                4.000402120e+03f,  -2.249280000e+00f,  4.003500600e+03f,  -2.250160000e+00f,  4.006599080e+03f,  -2.251040000e+00f,
                3.994503480e+03f,  -2.248400000e+00f,  3.997598440e+03f,  -2.249280000e+00f,  4.000693400e+03f,  -2.250160000e+00f,
                3.988604840e+03f,  -2.247520000e+00f,  3.991696280e+03f,  -2.248400000e+00f,  3.994787720e+03f,  -2.249280000e+00f,
                4.015606760e+03f,  -2.252800000e+00f,  4.018708760e+03f,  -2.253680000e+00f,  4.021810760e+03f,  -2.254560000e+00f,
                4.009697560e+03f,  -2.251920000e+00f,  4.012796040e+03f,  -2.252800000e+00f,  4.015894520e+03f,  -2.253680000e+00f,
                4.003788360e+03f,  -2.251040000e+00f,  4.006883320e+03f,  -2.251920000e+00f,  4.009978280e+03f,  -2.252800000e+00f,
                3.997879160e+03f,  -2.250160000e+00f,  4.000970600e+03f,  -2.251040000e+00f,  4.004062040e+03f,  -2.251920000e+00f,
                4.117972760e+03f,  -2.281840000e+00f,  4.121074760e+03f,  -2.282720000e+00f,  4.124176760e+03f,  -2.283600000e+00f,
                4.111947400e+03f,  -2.280960000e+00f,  4.115045880e+03f,  -2.281840000e+00f,  4.118144360e+03f,  -2.282720000e+00f,
                4.105922040e+03f,  -2.280080000e+00f,  4.109017000e+03f,  -2.280960000e+00f,  4.112111960e+03f,  -2.281840000e+00f,
                4.099896680e+03f,  -2.279200000e+00f,  4.102988120e+03f,  -2.280080000e+00f,  4.106079560e+03f,  -2.280960000e+00f,
                4.127278760e+03f,  -2.284480000e+00f,  4.130380760e+03f,  -2.285360000e+00f,  4.133482760e+03f,  -2.286240000e+00f,
                4.121242840e+03f,  -2.283600000e+00f,  4.124341320e+03f,  -2.284480000e+00f,  4.127439800e+03f,  -2.285360000e+00f,
                4.115206920e+03f,  -2.282720000e+00f,  4.118301880e+03f,  -2.283600000e+00f,  4.121396840e+03f,  -2.284480000e+00f,
                4.109171000e+03f,  -2.281840000e+00f,  4.112262440e+03f,  -2.282720000e+00f,  4.115353880e+03f,  -2.283600000e+00f,
                4.136584760e+03f,  -2.287120000e+00f,  4.139686760e+03f,  -2.288000000e+00f,  4.142788760e+03f,  -2.288880000e+00f,
                4.130538280e+03f,  -2.286240000e+00f,  4.133636760e+03f,  -2.287120000e+00f,  4.136735240e+03f,  -2.288000000e+00f,
                4.124491800e+03f,  -2.285360000e+00f,  4.127586760e+03f,  -2.286240000e+00f,  4.130681720e+03f,  -2.287120000e+00f,
                4.118445320e+03f,  -2.284480000e+00f,  4.121536760e+03f,  -2.285360000e+00f,  4.124628200e+03f,  -2.286240000e+00f,
                5.085796760e+03f,  -2.556400000e+00f,  5.088898760e+03f,  -2.557280000e+00f,  5.092000760e+03f,  -2.558160000e+00f,
                5.078673160e+03f,  -2.555520000e+00f,  5.081771640e+03f,  -2.556400000e+00f,  5.084870120e+03f,  -2.557280000e+00f,
                5.071549560e+03f,  -2.554640000e+00f,  5.074644520e+03f,  -2.555520000e+00f,  5.077739480e+03f,  -2.556400000e+00f,
                5.064425960e+03f,  -2.553760000e+00f,  5.067517400e+03f,  -2.554640000e+00f,  5.070608840e+03f,  -2.555520000e+00f,
                5.095102760e+03f,  -2.559040000e+00f,  5.098204760e+03f,  -2.559920000e+00f,  5.101306760e+03f,  -2.560800000e+00f,
                5.087968600e+03f,  -2.558160000e+00f,  5.091067080e+03f,  -2.559040000e+00f,  5.094165560e+03f,  -2.559920000e+00f,
                5.080834440e+03f,  -2.557280000e+00f,  5.083929400e+03f,  -2.558160000e+00f,  5.087024360e+03f,  -2.559040000e+00f,
                5.073700280e+03f,  -2.556400000e+00f,  5.076791720e+03f,  -2.557280000e+00f,  5.079883160e+03f,  -2.558160000e+00f,
                5.104408760e+03f,  -2.561680000e+00f,  5.107510760e+03f,  -2.562560000e+00f,  5.110612760e+03f,  -2.563440000e+00f,
                5.097264040e+03f,  -2.560800000e+00f,  5.100362520e+03f,  -2.561680000e+00f,  5.103461000e+03f,  -2.562560000e+00f,
                5.090119320e+03f,  -2.559920000e+00f,  5.093214280e+03f,  -2.560800000e+00f,  5.096309240e+03f,  -2.561680000e+00f,
                5.082974600e+03f,  -2.559040000e+00f,  5.086066040e+03f,  -2.559920000e+00f,  5.089157480e+03f,  -2.560800000e+00f,
                5.206774760e+03f,  -2.590720000e+00f,  5.209876760e+03f,  -2.591600000e+00f,  5.212978760e+03f,  -2.592480000e+00f,
                5.199513880e+03f,  -2.589840000e+00f,  5.202612360e+03f,  -2.590720000e+00f,  5.205710840e+03f,  -2.591600000e+00f,
                5.192253000e+03f,  -2.588960000e+00f,  5.195347960e+03f,  -2.589840000e+00f,  5.198442920e+03f,  -2.590720000e+00f,
                5.184992120e+03f,  -2.588080000e+00f,  5.188083560e+03f,  -2.588960000e+00f,  5.191175000e+03f,  -2.589840000e+00f,
                5.216080760e+03f,  -2.593360000e+00f,  5.219182760e+03f,  -2.594240000e+00f,  5.222284760e+03f,  -2.595120000e+00f,
                5.208809320e+03f,  -2.592480000e+00f,  5.211907800e+03f,  -2.593360000e+00f,  5.215006280e+03f,  -2.594240000e+00f,
                5.201537880e+03f,  -2.591600000e+00f,  5.204632840e+03f,  -2.592480000e+00f,  5.207727800e+03f,  -2.593360000e+00f,
                5.194266440e+03f,  -2.590720000e+00f,  5.197357880e+03f,  -2.591600000e+00f,  5.200449320e+03f,  -2.592480000e+00f,
                5.225386760e+03f,  -2.596000000e+00f,  5.228488760e+03f,  -2.596880000e+00f,  5.231590760e+03f,  -2.597760000e+00f,
                5.218104760e+03f,  -2.595120000e+00f,  5.221203240e+03f,  -2.596000000e+00f,  5.224301720e+03f,  -2.596880000e+00f,
                5.210822760e+03f,  -2.594240000e+00f,  5.213917720e+03f,  -2.595120000e+00f,  5.217012680e+03f,  -2.596000000e+00f,
                5.203540760e+03f,  -2.593360000e+00f,  5.206632200e+03f,  -2.594240000e+00f,  5.209723640e+03f,  -2.595120000e+00f,
                5.327752760e+03f,  -2.625040000e+00f,  5.330854760e+03f,  -2.625920000e+00f,  5.333956760e+03f,  -2.626800000e+00f,
                5.320354600e+03f,  -2.624160000e+00f,  5.323453080e+03f,  -2.625040000e+00f,  5.326551560e+03f,  -2.625920000e+00f,
                5.312956440e+03f,  -2.623280000e+00f,  5.316051400e+03f,  -2.624160000e+00f,  5.319146360e+03f,  -2.625040000e+00f,
                5.305558280e+03f,  -2.622400000e+00f,  5.308649720e+03f,  -2.623280000e+00f,  5.311741160e+03f,  -2.624160000e+00f,
                5.337058760e+03f,  -2.627680000e+00f,  5.340160760e+03f,  -2.628560000e+00f,  5.343262760e+03f,  -2.629440000e+00f,
                5.329650040e+03f,  -2.626800000e+00f,  5.332748520e+03f,  -2.627680000e+00f,  5.335847000e+03f,  -2.628560000e+00f,
                5.322241320e+03f,  -2.625920000e+00f,  5.325336280e+03f,  -2.626800000e+00f,  5.328431240e+03f,  -2.627680000e+00f,
                5.314832600e+03f,  -2.625040000e+00f,  5.317924040e+03f,  -2.625920000e+00f,  5.321015480e+03f,  -2.626800000e+00f,
                5.346364760e+03f,  -2.630320000e+00f,  5.349466760e+03f,  -2.631200000e+00f,  5.352568760e+03f,  -2.632080000e+00f,
                5.338945480e+03f,  -2.629440000e+00f,  5.342043960e+03f,  -2.630320000e+00f,  5.345142440e+03f,  -2.631200000e+00f,
                5.331526200e+03f,  -2.628560000e+00f,  5.334621160e+03f,  -2.629440000e+00f,  5.337716120e+03f,  -2.630320000e+00f,
                5.324106920e+03f,  -2.627680000e+00f,  5.327198360e+03f,  -2.628560000e+00f,  5.330289800e+03f,  -2.629440000e+00f,
                5.448730760e+03f,  -2.659360000e+00f,  5.451832760e+03f,  -2.660240000e+00f,  5.454934760e+03f,  -2.661120000e+00f,
                5.441195320e+03f,  -2.658480000e+00f,  5.444293800e+03f,  -2.659360000e+00f,  5.447392280e+03f,  -2.660240000e+00f,
                5.433659880e+03f,  -2.657600000e+00f,  5.436754840e+03f,  -2.658480000e+00f,  5.439849800e+03f,  -2.659360000e+00f,
                5.426124440e+03f,  -2.656720000e+00f,  5.429215880e+03f,  -2.657600000e+00f,  5.432307320e+03f,  -2.658480000e+00f,
                5.458036760e+03f,  -2.662000000e+00f,  5.461138760e+03f,  -2.662880000e+00f,  5.464240760e+03f,  -2.663760000e+00f,
                5.450490760e+03f,  -2.661120000e+00f,  5.453589240e+03f,  -2.662000000e+00f,  5.456687720e+03f,  -2.662880000e+00f,
                5.442944760e+03f,  -2.660240000e+00f,  5.446039720e+03f,  -2.661120000e+00f,  5.449134680e+03f,  -2.662000000e+00f,
                5.435398760e+03f,  -2.659360000e+00f,  5.438490200e+03f,  -2.660240000e+00f,  5.441581640e+03f,  -2.661120000e+00f,
                5.467342760e+03f,  -2.664640000e+00f,  5.470444760e+03f,  -2.665520000e+00f,  5.473546760e+03f,  -2.666400000e+00f,
                5.459786200e+03f,  -2.663760000e+00f,  5.462884680e+03f,  -2.664640000e+00f,  5.465983160e+03f,  -2.665520000e+00f,
                5.452229640e+03f,  -2.662880000e+00f,  5.455324600e+03f,  -2.663760000e+00f,  5.458419560e+03f,  -2.664640000e+00f,
                5.444673080e+03f,  -2.662000000e+00f,  5.447764520e+03f,  -2.662880000e+00f,  5.450855960e+03f,  -2.663760000e+00f,
                5.569708760e+03f,  -2.693680000e+00f,  5.572810760e+03f,  -2.694560000e+00f,  5.575912760e+03f,  -2.695440000e+00f,
                5.562036040e+03f,  -2.692800000e+00f,  5.565134520e+03f,  -2.693680000e+00f,  5.568233000e+03f,  -2.694560000e+00f,
                5.554363320e+03f,  -2.691920000e+00f,  5.557458280e+03f,  -2.692800000e+00f,  5.560553240e+03f,  -2.693680000e+00f,
                5.546690600e+03f,  -2.691040000e+00f,  5.549782040e+03f,  -2.691920000e+00f,  5.552873480e+03f,  -2.692800000e+00f,
                5.579014760e+03f,  -2.696320000e+00f,  5.582116760e+03f,  -2.697200000e+00f,  5.585218760e+03f,  -2.698080000e+00f,
                5.571331480e+03f,  -2.695440000e+00f,  5.574429960e+03f,  -2.696320000e+00f,  5.577528440e+03f,  -2.697200000e+00f,
                5.563648200e+03f,  -2.694560000e+00f,  5.566743160e+03f,  -2.695440000e+00f,  5.569838120e+03f,  -2.696320000e+00f,
                5.555964920e+03f,  -2.693680000e+00f,  5.559056360e+03f,  -2.694560000e+00f,  5.562147800e+03f,  -2.695440000e+00f,
                5.588320760e+03f,  -2.698960000e+00f,  5.591422760e+03f,  -2.699840000e+00f,  5.594524760e+03f,  -2.700720000e+00f,
                5.580626920e+03f,  -2.698080000e+00f,  5.583725400e+03f,  -2.698960000e+00f,  5.586823880e+03f,  -2.699840000e+00f,
                5.572933080e+03f,  -2.697200000e+00f,  5.576028040e+03f,  -2.698080000e+00f,  5.579123000e+03f,  -2.698960000e+00f,
                5.565239240e+03f,  -2.696320000e+00f,  5.568330680e+03f,  -2.697200000e+00f,  5.571422120e+03f,  -2.698080000e+00f,
                6.537532760e+03f,  -2.968240000e+00f,  6.540634760e+03f,  -2.969120000e+00f,  6.543736760e+03f,  -2.970000000e+00f,
                6.528761800e+03f,  -2.967360000e+00f,  6.531860280e+03f,  -2.968240000e+00f,  6.534958760e+03f,  -2.969120000e+00f,
                6.519990840e+03f,  -2.966480000e+00f,  6.523085800e+03f,  -2.967360000e+00f,  6.526180760e+03f,  -2.968240000e+00f,
                6.511219880e+03f,  -2.965600000e+00f,  6.514311320e+03f,  -2.966480000e+00f,  6.517402760e+03f,  -2.967360000e+00f,
                6.546838760e+03f,  -2.970880000e+00f,  6.549940760e+03f,  -2.971760000e+00f,  6.553042760e+03f,  -2.972640000e+00f,
                6.538057240e+03f,  -2.970000000e+00f,  6.541155720e+03f,  -2.970880000e+00f,  6.544254200e+03f,  -2.971760000e+00f,
                6.529275720e+03f,  -2.969120000e+00f,  6.532370680e+03f,  -2.970000000e+00f,  6.535465640e+03f,  -2.970880000e+00f,
                6.520494200e+03f,  -2.968240000e+00f,  6.523585640e+03f,  -2.969120000e+00f,  6.526677080e+03f,  -2.970000000e+00f,
                6.556144760e+03f,  -2.973520000e+00f,  6.559246760e+03f,  -2.974400000e+00f,  6.562348760e+03f,  -2.975280000e+00f,
                6.547352680e+03f,  -2.972640000e+00f,  6.550451160e+03f,  -2.973520000e+00f,  6.553549640e+03f,  -2.974400000e+00f,
                6.538560600e+03f,  -2.971760000e+00f,  6.541655560e+03f,  -2.972640000e+00f,  6.544750520e+03f,  -2.973520000e+00f,
                6.529768520e+03f,  -2.970880000e+00f,  6.532859960e+03f,  -2.971760000e+00f,  6.535951400e+03f,  -2.972640000e+00f,
                6.658510760e+03f,  -3.002560000e+00f,  6.661612760e+03f,  -3.003440000e+00f,  6.664714760e+03f,  -3.004320000e+00f,
                6.649602520e+03f,  -3.001680000e+00f,  6.652701000e+03f,  -3.002560000e+00f,  6.655799480e+03f,  -3.003440000e+00f,
                6.640694280e+03f,  -3.000800000e+00f,  6.643789240e+03f,  -3.001680000e+00f,  6.646884200e+03f,  -3.002560000e+00f,
                6.631786040e+03f,  -2.999920000e+00f,  6.634877480e+03f,  -3.000800000e+00f,  6.637968920e+03f,  -3.001680000e+00f,
                6.667816760e+03f,  -3.005200000e+00f,  6.670918760e+03f,  -3.006080000e+00f,  6.674020760e+03f,  -3.006960000e+00f,
                6.658897960e+03f,  -3.004320000e+00f,  6.661996440e+03f,  -3.005200000e+00f,  6.665094920e+03f,  -3.006080000e+00f,
                6.649979160e+03f,  -3.003440000e+00f,  6.653074120e+03f,  -3.004320000e+00f,  6.656169080e+03f,  -3.005200000e+00f,
                6.641060360e+03f,  -3.002560000e+00f,  6.644151800e+03f,  -3.003440000e+00f,  6.647243240e+03f,  -3.004320000e+00f,
                6.677122760e+03f,  -3.007840000e+00f,  6.680224760e+03f,  -3.008720000e+00f,  6.683326760e+03f,  -3.009600000e+00f,
                6.668193400e+03f,  -3.006960000e+00f,  6.671291880e+03f,  -3.007840000e+00f,  6.674390360e+03f,  -3.008720000e+00f,
                6.659264040e+03f,  -3.006080000e+00f,  6.662359000e+03f,  -3.006960000e+00f,  6.665453960e+03f,  -3.007840000e+00f,
                6.650334680e+03f,  -3.005200000e+00f,  6.653426120e+03f,  -3.006080000e+00f,  6.656517560e+03f,  -3.006960000e+00f,
                6.779488760e+03f,  -3.036880000e+00f,  6.782590760e+03f,  -3.037760000e+00f,  6.785692760e+03f,  -3.038640000e+00f,
                6.770443240e+03f,  -3.036000000e+00f,  6.773541720e+03f,  -3.036880000e+00f,  6.776640200e+03f,  -3.037760000e+00f,
                6.761397720e+03f,  -3.035120000e+00f,  6.764492680e+03f,  -3.036000000e+00f,  6.767587640e+03f,  -3.036880000e+00f,
                6.752352200e+03f,  -3.034240000e+00f,  6.755443640e+03f,  -3.035120000e+00f,  6.758535080e+03f,  -3.036000000e+00f,
                6.788794760e+03f,  -3.039520000e+00f,  6.791896760e+03f,  -3.040400000e+00f,  6.794998760e+03f,  -3.041280000e+00f,
                6.779738680e+03f,  -3.038640000e+00f,  6.782837160e+03f,  -3.039520000e+00f,  6.785935640e+03f,  -3.040400000e+00f,
                6.770682600e+03f,  -3.037760000e+00f,  6.773777560e+03f,  -3.038640000e+00f,  6.776872520e+03f,  -3.039520000e+00f,
                6.761626520e+03f,  -3.036880000e+00f,  6.764717960e+03f,  -3.037760000e+00f,  6.767809400e+03f,  -3.038640000e+00f,
                6.798100760e+03f,  -3.042160000e+00f,  6.801202760e+03f,  -3.043040000e+00f,  6.804304760e+03f,  -3.043920000e+00f,
                6.789034120e+03f,  -3.041280000e+00f,  6.792132600e+03f,  -3.042160000e+00f,  6.795231080e+03f,  -3.043040000e+00f,
                6.779967480e+03f,  -3.040400000e+00f,  6.783062440e+03f,  -3.041280000e+00f,  6.786157400e+03f,  -3.042160000e+00f,
                6.770900840e+03f,  -3.039520000e+00f,  6.773992280e+03f,  -3.040400000e+00f,  6.777083720e+03f,  -3.041280000e+00f,
                6.900466760e+03f,  -3.071200000e+00f,  6.903568760e+03f,  -3.072080000e+00f,  6.906670760e+03f,  -3.072960000e+00f,
                6.891283960e+03f,  -3.070320000e+00f,  6.894382440e+03f,  -3.071200000e+00f,  6.897480920e+03f,  -3.072080000e+00f,
                6.882101160e+03f,  -3.069440000e+00f,  6.885196120e+03f,  -3.070320000e+00f,  6.888291080e+03f,  -3.071200000e+00f,
                6.872918360e+03f,  -3.068560000e+00f,  6.876009800e+03f,  -3.069440000e+00f,  6.879101240e+03f,  -3.070320000e+00f,
                6.909772760e+03f,  -3.073840000e+00f,  6.912874760e+03f,  -3.074720000e+00f,  6.915976760e+03f,  -3.075600000e+00f,
                6.900579400e+03f,  -3.072960000e+00f,  6.903677880e+03f,  -3.073840000e+00f,  6.906776360e+03f,  -3.074720000e+00f,
                6.891386040e+03f,  -3.072080000e+00f,  6.894481000e+03f,  -3.072960000e+00f,  6.897575960e+03f,  -3.073840000e+00f,
                6.882192680e+03f,  -3.071200000e+00f,  6.885284120e+03f,  -3.072080000e+00f,  6.888375560e+03f,  -3.072960000e+00f,
                6.919078760e+03f,  -3.076480000e+00f,  6.922180760e+03f,  -3.077360000e+00f,  6.925282760e+03f,  -3.078240000e+00f,
                6.909874840e+03f,  -3.075600000e+00f,  6.912973320e+03f,  -3.076480000e+00f,  6.916071800e+03f,  -3.077360000e+00f,
                6.900670920e+03f,  -3.074720000e+00f,  6.903765880e+03f,  -3.075600000e+00f,  6.906860840e+03f,  -3.076480000e+00f,
                6.891467000e+03f,  -3.073840000e+00f,  6.894558440e+03f,  -3.074720000e+00f,  6.897649880e+03f,  -3.075600000e+00f,
                7.021444760e+03f,  -3.105520000e+00f,  7.024546760e+03f,  -3.106400000e+00f,  7.027648760e+03f,  -3.107280000e+00f,
                7.012124680e+03f,  -3.104640000e+00f,  7.015223160e+03f,  -3.105520000e+00f,  7.018321640e+03f,  -3.106400000e+00f,
                7.002804600e+03f,  -3.103760000e+00f,  7.005899560e+03f,  -3.104640000e+00f,  7.008994520e+03f,  -3.105520000e+00f,
                6.993484520e+03f,  -3.102880000e+00f,  6.996575960e+03f,  -3.103760000e+00f,  6.999667400e+03f,  -3.104640000e+00f,
                7.030750760e+03f,  -3.108160000e+00f,  7.033852760e+03f,  -3.109040000e+00f,  7.036954760e+03f,  -3.109920000e+00f,
                7.021420120e+03f,  -3.107280000e+00f,  7.024518600e+03f,  -3.108160000e+00f,  7.027617080e+03f,  -3.109040000e+00f,
                7.012089480e+03f,  -3.106400000e+00f,  7.015184440e+03f,  -3.107280000e+00f,  7.018279400e+03f,  -3.108160000e+00f,
                7.002758840e+03f,  -3.105520000e+00f,  7.005850280e+03f,  -3.106400000e+00f,  7.008941720e+03f,  -3.107280000e+00f,
                7.040056760e+03f,  -3.110800000e+00f,  7.043158760e+03f,  -3.111680000e+00f,  7.046260760e+03f,  -3.112560000e+00f,
                7.030715560e+03f,  -3.109920000e+00f,  7.033814040e+03f,  -3.110800000e+00f,  7.036912520e+03f,  -3.111680000e+00f,
                7.021374360e+03f,  -3.109040000e+00f,  7.024469320e+03f,  -3.109920000e+00f,  7.027564280e+03f,  -3.110800000e+00f,
                7.012033160e+03f,  -3.108160000e+00f,  7.015124600e+03f,  -3.109040000e+00f,  7.018216040e+03f,  -3.109920000e+00f,
                7.989268760e+03f,  -3.380080000e+00f,  7.992370760e+03f,  -3.380960000e+00f,  7.995472760e+03f,  -3.381840000e+00f,
                7.978850440e+03f,  -3.379200000e+00f,  7.981948920e+03f,  -3.380080000e+00f,  7.985047400e+03f,  -3.380960000e+00f,
                7.968432120e+03f,  -3.378320000e+00f,  7.971527080e+03f,  -3.379200000e+00f,  7.974622040e+03f,  -3.380080000e+00f,
                7.958013800e+03f,  -3.377440000e+00f,  7.961105240e+03f,  -3.378320000e+00f,  7.964196680e+03f,  -3.379200000e+00f,
                7.998574760e+03f,  -3.382720000e+00f,  8.001676760e+03f,  -3.383600000e+00f,  8.004778760e+03f,  -3.384480000e+00f,
                7.988145880e+03f,  -3.381840000e+00f,  7.991244360e+03f,  -3.382720000e+00f,  7.994342840e+03f,  -3.383600000e+00f,
                7.977717000e+03f,  -3.380960000e+00f,  7.980811960e+03f,  -3.381840000e+00f,  7.983906920e+03f,  -3.382720000e+00f,
                7.967288120e+03f,  -3.380080000e+00f,  7.970379560e+03f,  -3.380960000e+00f,  7.973471000e+03f,  -3.381840000e+00f,
                8.007880760e+03f,  -3.385360000e+00f,  8.010982760e+03f,  -3.386240000e+00f,  8.014084760e+03f,  -3.387120000e+00f,
                7.997441320e+03f,  -3.384480000e+00f,  8.000539800e+03f,  -3.385360000e+00f,  8.003638280e+03f,  -3.386240000e+00f,
                7.987001880e+03f,  -3.383600000e+00f,  7.990096840e+03f,  -3.384480000e+00f,  7.993191800e+03f,  -3.385360000e+00f,
                7.976562440e+03f,  -3.382720000e+00f,  7.979653880e+03f,  -3.383600000e+00f,  7.982745320e+03f,  -3.384480000e+00f,
                8.110246760e+03f,  -3.414400000e+00f,  8.113348760e+03f,  -3.415280000e+00f,  8.116450760e+03f,  -3.416160000e+00f,
                8.099691160e+03f,  -3.413520000e+00f,  8.102789640e+03f,  -3.414400000e+00f,  8.105888120e+03f,  -3.415280000e+00f,
                8.089135560e+03f,  -3.412640000e+00f,  8.092230520e+03f,  -3.413520000e+00f,  8.095325480e+03f,  -3.414400000e+00f,
                8.078579960e+03f,  -3.411760000e+00f,  8.081671400e+03f,  -3.412640000e+00f,  8.084762840e+03f,  -3.413520000e+00f,
                8.119552760e+03f,  -3.417040000e+00f,  8.122654760e+03f,  -3.417920000e+00f,  8.125756760e+03f,  -3.418800000e+00f,
                8.108986600e+03f,  -3.416160000e+00f,  8.112085080e+03f,  -3.417040000e+00f,  8.115183560e+03f,  -3.417920000e+00f,
                8.098420440e+03f,  -3.415280000e+00f,  8.101515400e+03f,  -3.416160000e+00f,  8.104610360e+03f,  -3.417040000e+00f,
                8.087854280e+03f,  -3.414400000e+00f,  8.090945720e+03f,  -3.415280000e+00f,  8.094037160e+03f,  -3.416160000e+00f,
                8.128858760e+03f,  -3.419680000e+00f,  8.131960760e+03f,  -3.420560000e+00f,  8.135062760e+03f,  -3.421440000e+00f,
                8.118282040e+03f,  -3.418800000e+00f,  8.121380520e+03f,  -3.419680000e+00f,  8.124479000e+03f,  -3.420560000e+00f,
                8.107705320e+03f,  -3.417920000e+00f,  8.110800280e+03f,  -3.418800000e+00f,  8.113895240e+03f,  -3.419680000e+00f,
                8.097128600e+03f,  -3.417040000e+00f,  8.100220040e+03f,  -3.417920000e+00f,  8.103311480e+03f,  -3.418800000e+00f,
                8.231224760e+03f,  -3.448720000e+00f,  8.234326760e+03f,  -3.449600000e+00f,  8.237428760e+03f,  -3.450480000e+00f,
                8.220531880e+03f,  -3.447840000e+00f,  8.223630360e+03f,  -3.448720000e+00f,  8.226728840e+03f,  -3.449600000e+00f,
                8.209839000e+03f,  -3.446960000e+00f,  8.212933960e+03f,  -3.447840000e+00f,  8.216028920e+03f,  -3.448720000e+00f,
                8.199146120e+03f,  -3.446080000e+00f,  8.202237560e+03f,  -3.446960000e+00f,  8.205329000e+03f,  -3.447840000e+00f,
                8.240530760e+03f,  -3.451360000e+00f,  8.243632760e+03f,  -3.452240000e+00f,  8.246734760e+03f,  -3.453120000e+00f,
                8.229827320e+03f,  -3.450480000e+00f,  8.232925800e+03f,  -3.451360000e+00f,  8.236024280e+03f,  -3.452240000e+00f,
                8.219123880e+03f,  -3.449600000e+00f,  8.222218840e+03f,  -3.450480000e+00f,  8.225313800e+03f,  -3.451360000e+00f,
                8.208420440e+03f,  -3.448720000e+00f,  8.211511880e+03f,  -3.449600000e+00f,  8.214603320e+03f,  -3.450480000e+00f,
                8.249836760e+03f,  -3.454000000e+00f,  8.252938760e+03f,  -3.454880000e+00f,  8.256040760e+03f,  -3.455760000e+00f,
                8.239122760e+03f,  -3.453120000e+00f,  8.242221240e+03f,  -3.454000000e+00f,  8.245319720e+03f,  -3.454880000e+00f,
                8.228408760e+03f,  -3.452240000e+00f,  8.231503720e+03f,  -3.453120000e+00f,  8.234598680e+03f,  -3.454000000e+00f,
                8.217694760e+03f,  -3.451360000e+00f,  8.220786200e+03f,  -3.452240000e+00f,  8.223877640e+03f,  -3.453120000e+00f,
                8.352202760e+03f,  -3.483040000e+00f,  8.355304760e+03f,  -3.483920000e+00f,  8.358406760e+03f,  -3.484800000e+00f,
                8.341372600e+03f,  -3.482160000e+00f,  8.344471080e+03f,  -3.483040000e+00f,  8.347569560e+03f,  -3.483920000e+00f,
                8.330542440e+03f,  -3.481280000e+00f,  8.333637400e+03f,  -3.482160000e+00f,  8.336732360e+03f,  -3.483040000e+00f,
                8.319712280e+03f,  -3.480400000e+00f,  8.322803720e+03f,  -3.481280000e+00f,  8.325895160e+03f,  -3.482160000e+00f,
                8.361508760e+03f,  -3.485680000e+00f,  8.364610760e+03f,  -3.486560000e+00f,  8.367712760e+03f,  -3.487440000e+00f,
                8.350668040e+03f,  -3.484800000e+00f,  8.353766520e+03f,  -3.485680000e+00f,  8.356865000e+03f,  -3.486560000e+00f,
                8.339827320e+03f,  -3.483920000e+00f,  8.342922280e+03f,  -3.484800000e+00f,  8.346017240e+03f,  -3.485680000e+00f,
                8.328986600e+03f,  -3.483040000e+00f,  8.332078040e+03f,  -3.483920000e+00f,  8.335169480e+03f,  -3.484800000e+00f,
                8.370814760e+03f,  -3.488320000e+00f,  8.373916760e+03f,  -3.489200000e+00f,  8.377018760e+03f,  -3.490080000e+00f,
                8.359963480e+03f,  -3.487440000e+00f,  8.363061960e+03f,  -3.488320000e+00f,  8.366160440e+03f,  -3.489200000e+00f,
                8.349112200e+03f,  -3.486560000e+00f,  8.352207160e+03f,  -3.487440000e+00f,  8.355302120e+03f,  -3.488320000e+00f,
                8.338260920e+03f,  -3.485680000e+00f,  8.341352360e+03f,  -3.486560000e+00f,  8.344443800e+03f,  -3.487440000e+00f,
                8.473180760e+03f,  -3.517360000e+00f,  8.476282760e+03f,  -3.518240000e+00f,  8.479384760e+03f,  -3.519120000e+00f,
                8.462213320e+03f,  -3.516480000e+00f,  8.465311800e+03f,  -3.517360000e+00f,  8.468410280e+03f,  -3.518240000e+00f,
                8.451245880e+03f,  -3.515600000e+00f,  8.454340840e+03f,  -3.516480000e+00f,  8.457435800e+03f,  -3.517360000e+00f,
                8.440278440e+03f,  -3.514720000e+00f,  8.443369880e+03f,  -3.515600000e+00f,  8.446461320e+03f,  -3.516480000e+00f,
                8.482486760e+03f,  -3.520000000e+00f,  8.485588760e+03f,  -3.520880000e+00f,  8.488690760e+03f,  -3.521760000e+00f,
                8.471508760e+03f,  -3.519120000e+00f,  8.474607240e+03f,  -3.520000000e+00f,  8.477705720e+03f,  -3.520880000e+00f,
                8.460530760e+03f,  -3.518240000e+00f,  8.463625720e+03f,  -3.519120000e+00f,  8.466720680e+03f,  -3.520000000e+00f,
                8.449552760e+03f,  -3.517360000e+00f,  8.452644200e+03f,  -3.518240000e+00f,  8.455735640e+03f,  -3.519120000e+00f,
                8.491792760e+03f,  -3.522640000e+00f,  8.494894760e+03f,  -3.523520000e+00f,  8.497996760e+03f,  -3.524400000e+00f,
                8.480804200e+03f,  -3.521760000e+00f,  8.483902680e+03f,  -3.522640000e+00f,  8.487001160e+03f,  -3.523520000e+00f,
                8.469815640e+03f,  -3.520880000e+00f,  8.472910600e+03f,  -3.521760000e+00f,  8.476005560e+03f,  -3.522640000e+00f,
                8.458827080e+03f,  -3.520000000e+00f,  8.461918520e+03f,  -3.520880000e+00f,  8.465009960e+03f,  -3.521760000e+00f,
                9.441004760e+03f,  -3.791920000e+00f,  9.444106760e+03f,  -3.792800000e+00f,  9.447208760e+03f,  -3.793680000e+00f,
                9.428939080e+03f,  -3.791040000e+00f,  9.432037560e+03f,  -3.791920000e+00f,  9.435136040e+03f,  -3.792800000e+00f,
                9.416873400e+03f,  -3.790160000e+00f,  9.419968360e+03f,  -3.791040000e+00f,  9.423063320e+03f,  -3.791920000e+00f,
                9.404807720e+03f,  -3.789280000e+00f,  9.407899160e+03f,  -3.790160000e+00f,  9.410990600e+03f,  -3.791040000e+00f,
                9.450310760e+03f,  -3.794560000e+00f,  9.453412760e+03f,  -3.795440000e+00f,  9.456514760e+03f,  -3.796320000e+00f,
                9.438234520e+03f,  -3.793680000e+00f,  9.441333000e+03f,  -3.794560000e+00f,  9.444431480e+03f,  -3.795440000e+00f,
                9.426158280e+03f,  -3.792800000e+00f,  9.429253240e+03f,  -3.793680000e+00f,  9.432348200e+03f,  -3.794560000e+00f,
                9.414082040e+03f,  -3.791920000e+00f,  9.417173480e+03f,  -3.792800000e+00f,  9.420264920e+03f,  -3.793680000e+00f,
                9.459616760e+03f,  -3.797200000e+00f,  9.462718760e+03f,  -3.798080000e+00f,  9.465820760e+03f,  -3.798960000e+00f,
                9.447529960e+03f,  -3.796320000e+00f,  9.450628440e+03f,  -3.797200000e+00f,  9.453726920e+03f,  -3.798080000e+00f,
                9.435443160e+03f,  -3.795440000e+00f,  9.438538120e+03f,  -3.796320000e+00f,  9.441633080e+03f,  -3.797200000e+00f,
                9.423356360e+03f,  -3.794560000e+00f,  9.426447800e+03f,  -3.795440000e+00f,  9.429539240e+03f,  -3.796320000e+00f,
                9.561982760e+03f,  -3.826240000e+00f,  9.565084760e+03f,  -3.827120000e+00f,  9.568186760e+03f,  -3.828000000e+00f,
                9.549779800e+03f,  -3.825360000e+00f,  9.552878280e+03f,  -3.826240000e+00f,  9.555976760e+03f,  -3.827120000e+00f,
                9.537576840e+03f,  -3.824480000e+00f,  9.540671800e+03f,  -3.825360000e+00f,  9.543766760e+03f,  -3.826240000e+00f,
                9.525373880e+03f,  -3.823600000e+00f,  9.528465320e+03f,  -3.824480000e+00f,  9.531556760e+03f,  -3.825360000e+00f,
                9.571288760e+03f,  -3.828880000e+00f,  9.574390760e+03f,  -3.829760000e+00f,  9.577492760e+03f,  -3.830640000e+00f,
                9.559075240e+03f,  -3.828000000e+00f,  9.562173720e+03f,  -3.828880000e+00f,  9.565272200e+03f,  -3.829760000e+00f,
                9.546861720e+03f,  -3.827120000e+00f,  9.549956680e+03f,  -3.828000000e+00f,  9.553051640e+03f,  -3.828880000e+00f,
                9.534648200e+03f,  -3.826240000e+00f,  9.537739640e+03f,  -3.827120000e+00f,  9.540831080e+03f,  -3.828000000e+00f,
                9.580594760e+03f,  -3.831520000e+00f,  9.583696760e+03f,  -3.832400000e+00f,  9.586798760e+03f,  -3.833280000e+00f,
                9.568370680e+03f,  -3.830640000e+00f,  9.571469160e+03f,  -3.831520000e+00f,  9.574567640e+03f,  -3.832400000e+00f,
                9.556146600e+03f,  -3.829760000e+00f,  9.559241560e+03f,  -3.830640000e+00f,  9.562336520e+03f,  -3.831520000e+00f,
                9.543922520e+03f,  -3.828880000e+00f,  9.547013960e+03f,  -3.829760000e+00f,  9.550105400e+03f,  -3.830640000e+00f,
                9.682960760e+03f,  -3.860560000e+00f,  9.686062760e+03f,  -3.861440000e+00f,  9.689164760e+03f,  -3.862320000e+00f,
                9.670620520e+03f,  -3.859680000e+00f,  9.673719000e+03f,  -3.860560000e+00f,  9.676817480e+03f,  -3.861440000e+00f,
                9.658280280e+03f,  -3.858800000e+00f,  9.661375240e+03f,  -3.859680000e+00f,  9.664470200e+03f,  -3.860560000e+00f,
                9.645940040e+03f,  -3.857920000e+00f,  9.649031480e+03f,  -3.858800000e+00f,  9.652122920e+03f,  -3.859680000e+00f,
                9.692266760e+03f,  -3.863200000e+00f,  9.695368760e+03f,  -3.864080000e+00f,  9.698470760e+03f,  -3.864960000e+00f,
                9.679915960e+03f,  -3.862320000e+00f,  9.683014440e+03f,  -3.863200000e+00f,  9.686112920e+03f,  -3.864080000e+00f,
                9.667565160e+03f,  -3.861440000e+00f,  9.670660120e+03f,  -3.862320000e+00f,  9.673755080e+03f,  -3.863200000e+00f,
                9.655214360e+03f,  -3.860560000e+00f,  9.658305800e+03f,  -3.861440000e+00f,  9.661397240e+03f,  -3.862320000e+00f,
                9.701572760e+03f,  -3.865840000e+00f,  9.704674760e+03f,  -3.866720000e+00f,  9.707776760e+03f,  -3.867600000e+00f,
                9.689211400e+03f,  -3.864960000e+00f,  9.692309880e+03f,  -3.865840000e+00f,  9.695408360e+03f,  -3.866720000e+00f,
                9.676850040e+03f,  -3.864080000e+00f,  9.679945000e+03f,  -3.864960000e+00f,  9.683039960e+03f,  -3.865840000e+00f,
                9.664488680e+03f,  -3.863200000e+00f,  9.667580120e+03f,  -3.864080000e+00f,  9.670671560e+03f,  -3.864960000e+00f,
                9.803938760e+03f,  -3.894880000e+00f,  9.807040760e+03f,  -3.895760000e+00f,  9.810142760e+03f,  -3.896640000e+00f,
                9.791461240e+03f,  -3.894000000e+00f,  9.794559720e+03f,  -3.894880000e+00f,  9.797658200e+03f,  -3.895760000e+00f,
                9.778983720e+03f,  -3.893120000e+00f,  9.782078680e+03f,  -3.894000000e+00f,  9.785173640e+03f,  -3.894880000e+00f,
                9.766506200e+03f,  -3.892240000e+00f,  9.769597640e+03f,  -3.893120000e+00f,  9.772689080e+03f,  -3.894000000e+00f,
                9.813244760e+03f,  -3.897520000e+00f,  9.816346760e+03f,  -3.898400000e+00f,  9.819448760e+03f,  -3.899280000e+00f,
                9.800756680e+03f,  -3.896640000e+00f,  9.803855160e+03f,  -3.897520000e+00f,  9.806953640e+03f,  -3.898400000e+00f,
                9.788268600e+03f,  -3.895760000e+00f,  9.791363560e+03f,  -3.896640000e+00f,  9.794458520e+03f,  -3.897520000e+00f,
                9.775780520e+03f,  -3.894880000e+00f,  9.778871960e+03f,  -3.895760000e+00f,  9.781963400e+03f,  -3.896640000e+00f,
                9.822550760e+03f,  -3.900160000e+00f,  9.825652760e+03f,  -3.901040000e+00f,  9.828754760e+03f,  -3.901920000e+00f,
                9.810052120e+03f,  -3.899280000e+00f,  9.813150600e+03f,  -3.900160000e+00f,  9.816249080e+03f,  -3.901040000e+00f,
                9.797553480e+03f,  -3.898400000e+00f,  9.800648440e+03f,  -3.899280000e+00f,  9.803743400e+03f,  -3.900160000e+00f,
                9.785054840e+03f,  -3.897520000e+00f,  9.788146280e+03f,  -3.898400000e+00f,  9.791237720e+03f,  -3.899280000e+00f,
                9.924916760e+03f,  -3.929200000e+00f,  9.928018760e+03f,  -3.930080000e+00f,  9.931120760e+03f,  -3.930960000e+00f,
                9.912301960e+03f,  -3.928320000e+00f,  9.915400440e+03f,  -3.929200000e+00f,  9.918498920e+03f,  -3.930080000e+00f,
                9.899687160e+03f,  -3.927440000e+00f,  9.902782120e+03f,  -3.928320000e+00f,  9.905877080e+03f,  -3.929200000e+00f,
                9.887072360e+03f,  -3.926560000e+00f,  9.890163800e+03f,  -3.927440000e+00f,  9.893255240e+03f,  -3.928320000e+00f,
                9.934222760e+03f,  -3.931840000e+00f,  9.937324760e+03f,  -3.932720000e+00f,  9.940426760e+03f,  -3.933600000e+00f,
                9.921597400e+03f,  -3.930960000e+00f,  9.924695880e+03f,  -3.931840000e+00f,  9.927794360e+03f,  -3.932720000e+00f,
                9.908972040e+03f,  -3.930080000e+00f,  9.912067000e+03f,  -3.930960000e+00f,  9.915161960e+03f,  -3.931840000e+00f,
                9.896346680e+03f,  -3.929200000e+00f,  9.899438120e+03f,  -3.930080000e+00f,  9.902529560e+03f,  -3.930960000e+00f,
                9.943528760e+03f,  -3.934480000e+00f,  9.946630760e+03f,  -3.935360000e+00f,  9.949732760e+03f,  -3.936240000e+00f,
                9.930892840e+03f,  -3.933600000e+00f,  9.933991320e+03f,  -3.934480000e+00f,  9.937089800e+03f,  -3.935360000e+00f,
                9.918256920e+03f,  -3.932720000e+00f,  9.921351880e+03f,  -3.933600000e+00f,  9.924446840e+03f,  -3.934480000e+00f,
                9.905621000e+03f,  -3.931840000e+00f,  9.908712440e+03f,  -3.932720000e+00f,  9.911803880e+03f,  -3.933600000e+00f,
                1.089274076e+04f,  -4.203760000e+00f,  1.089584276e+04f,  -4.204640000e+00f,  1.089894476e+04f,  -4.205520000e+00f,
                1.087902772e+04f,  -4.202880000e+00f,  1.088212620e+04f,  -4.203760000e+00f,  1.088522468e+04f,  -4.204640000e+00f,
                1.086531468e+04f,  -4.202000000e+00f,  1.086840964e+04f,  -4.202880000e+00f,  1.087150460e+04f,  -4.203760000e+00f,
                1.085160164e+04f,  -4.201120000e+00f,  1.085469308e+04f,  -4.202000000e+00f,  1.085778452e+04f,  -4.202880000e+00f,
                1.090204676e+04f,  -4.206400000e+00f,  1.090514876e+04f,  -4.207280000e+00f,  1.090825076e+04f,  -4.208160000e+00f,
                1.088832316e+04f,  -4.205520000e+00f,  1.089142164e+04f,  -4.206400000e+00f,  1.089452012e+04f,  -4.207280000e+00f,
                1.087459956e+04f,  -4.204640000e+00f,  1.087769452e+04f,  -4.205520000e+00f,  1.088078948e+04f,  -4.206400000e+00f,
                1.086087596e+04f,  -4.203760000e+00f,  1.086396740e+04f,  -4.204640000e+00f,  1.086705884e+04f,  -4.205520000e+00f,
                1.091135276e+04f,  -4.209040000e+00f,  1.091445476e+04f,  -4.209920000e+00f,  1.091755676e+04f,  -4.210800000e+00f,
                1.089761860e+04f,  -4.208160000e+00f,  1.090071708e+04f,  -4.209040000e+00f,  1.090381556e+04f,  -4.209920000e+00f,
                1.088388444e+04f,  -4.207280000e+00f,  1.088697940e+04f,  -4.208160000e+00f,  1.089007436e+04f,  -4.209040000e+00f,
                1.087015028e+04f,  -4.206400000e+00f,  1.087324172e+04f,  -4.207280000e+00f,  1.087633316e+04f,  -4.208160000e+00f,
                1.101371876e+04f,  -4.238080000e+00f,  1.101682076e+04f,  -4.238960000e+00f,  1.101992276e+04f,  -4.239840000e+00f,
                1.099986844e+04f,  -4.237200000e+00f,  1.100296692e+04f,  -4.238080000e+00f,  1.100606540e+04f,  -4.238960000e+00f,
                1.098601812e+04f,  -4.236320000e+00f,  1.098911308e+04f,  -4.237200000e+00f,  1.099220804e+04f,  -4.238080000e+00f,
                1.097216780e+04f,  -4.235440000e+00f,  1.097525924e+04f,  -4.236320000e+00f,  1.097835068e+04f,  -4.237200000e+00f,
                1.102302476e+04f,  -4.240720000e+00f,  1.102612676e+04f,  -4.241600000e+00f,  1.102922876e+04f,  -4.242480000e+00f,
                1.100916388e+04f,  -4.239840000e+00f,  1.101226236e+04f,  -4.240720000e+00f,  1.101536084e+04f,  -4.241600000e+00f,
                1.099530300e+04f,  -4.238960000e+00f,  1.099839796e+04f,  -4.239840000e+00f,  1.100149292e+04f,  -4.240720000e+00f,
                1.098144212e+04f,  -4.238080000e+00f,  1.098453356e+04f,  -4.238960000e+00f,  1.098762500e+04f,  -4.239840000e+00f,
                1.103233076e+04f,  -4.243360000e+00f,  1.103543276e+04f,  -4.244240000e+00f,  1.103853476e+04f,  -4.245120000e+00f,
                1.101845932e+04f,  -4.242480000e+00f,  1.102155780e+04f,  -4.243360000e+00f,  1.102465628e+04f,  -4.244240000e+00f,
                1.100458788e+04f,  -4.241600000e+00f,  1.100768284e+04f,  -4.242480000e+00f,  1.101077780e+04f,  -4.243360000e+00f,
                1.099071644e+04f,  -4.240720000e+00f,  1.099380788e+04f,  -4.241600000e+00f,  1.099689932e+04f,  -4.242480000e+00f,
                1.113469676e+04f,  -4.272400000e+00f,  1.113779876e+04f,  -4.273280000e+00f,  1.114090076e+04f,  -4.274160000e+00f,
                1.112070916e+04f,  -4.271520000e+00f,  1.112380764e+04f,  -4.272400000e+00f,  1.112690612e+04f,  -4.273280000e+00f,
                1.110672156e+04f,  -4.270640000e+00f,  1.110981652e+04f,  -4.271520000e+00f,  1.111291148e+04f,  -4.272400000e+00f,
                1.109273396e+04f,  -4.269760000e+00f,  1.109582540e+04f,  -4.270640000e+00f,  1.109891684e+04f,  -4.271520000e+00f,
                1.114400276e+04f,  -4.275040000e+00f,  1.114710476e+04f,  -4.275920000e+00f,  1.115020676e+04f,  -4.276800000e+00f,
                1.113000460e+04f,  -4.274160000e+00f,  1.113310308e+04f,  -4.275040000e+00f,  1.113620156e+04f,  -4.275920000e+00f,
                1.111600644e+04f,  -4.273280000e+00f,  1.111910140e+04f,  -4.274160000e+00f,  1.112219636e+04f,  -4.275040000e+00f,
                1.110200828e+04f,  -4.272400000e+00f,  1.110509972e+04f,  -4.273280000e+00f,  1.110819116e+04f,  -4.274160000e+00f,
                1.115330876e+04f,  -4.277680000e+00f,  1.115641076e+04f,  -4.278560000e+00f,  1.115951276e+04f,  -4.279440000e+00f,
                1.113930004e+04f,  -4.276800000e+00f,  1.114239852e+04f,  -4.277680000e+00f,  1.114549700e+04f,  -4.278560000e+00f,
                1.112529132e+04f,  -4.275920000e+00f,  1.112838628e+04f,  -4.276800000e+00f,  1.113148124e+04f,  -4.277680000e+00f,
                1.111128260e+04f,  -4.275040000e+00f,  1.111437404e+04f,  -4.275920000e+00f,  1.111746548e+04f,  -4.276800000e+00f,
                1.125567476e+04f,  -4.306720000e+00f,  1.125877676e+04f,  -4.307600000e+00f,  1.126187876e+04f,  -4.308480000e+00f,
                1.124154988e+04f,  -4.305840000e+00f,  1.124464836e+04f,  -4.306720000e+00f,  1.124774684e+04f,  -4.307600000e+00f,
                1.122742500e+04f,  -4.304960000e+00f,  1.123051996e+04f,  -4.305840000e+00f,  1.123361492e+04f,  -4.306720000e+00f,
                1.121330012e+04f,  -4.304080000e+00f,  1.121639156e+04f,  -4.304960000e+00f,  1.121948300e+04f,  -4.305840000e+00f,
                1.126498076e+04f,  -4.309360000e+00f,  1.126808276e+04f,  -4.310240000e+00f,  1.127118476e+04f,  -4.311120000e+00f,
                1.125084532e+04f,  -4.308480000e+00f,  1.125394380e+04f,  -4.309360000e+00f,  1.125704228e+04f,  -4.310240000e+00f,
                1.123670988e+04f,  -4.307600000e+00f,  1.123980484e+04f,  -4.308480000e+00f,  1.124289980e+04f,  -4.309360000e+00f,
                1.122257444e+04f,  -4.306720000e+00f,  1.122566588e+04f,  -4.307600000e+00f,  1.122875732e+04f,  -4.308480000e+00f,
                1.127428676e+04f,  -4.312000000e+00f,  1.127738876e+04f,  -4.312880000e+00f,  1.128049076e+04f,  -4.313760000e+00f,
                1.126014076e+04f,  -4.311120000e+00f,  1.126323924e+04f,  -4.312000000e+00f,  1.126633772e+04f,  -4.312880000e+00f,
                1.124599476e+04f,  -4.310240000e+00f,  1.124908972e+04f,  -4.311120000e+00f,  1.125218468e+04f,  -4.312000000e+00f,
                1.123184876e+04f,  -4.309360000e+00f,  1.123494020e+04f,  -4.310240000e+00f,  1.123803164e+04f,  -4.311120000e+00f,
                1.137665276e+04f,  -4.341040000e+00f,  1.137975476e+04f,  -4.341920000e+00f,  1.138285676e+04f,  -4.342800000e+00f,
                1.136239060e+04f,  -4.340160000e+00f,  1.136548908e+04f,  -4.341040000e+00f,  1.136858756e+04f,  -4.341920000e+00f,
                1.134812844e+04f,  -4.339280000e+00f,  1.135122340e+04f,  -4.340160000e+00f,  1.135431836e+04f,  -4.341040000e+00f,
                1.133386628e+04f,  -4.338400000e+00f,  1.133695772e+04f,  -4.339280000e+00f,  1.134004916e+04f,  -4.340160000e+00f,
                1.138595876e+04f,  -4.343680000e+00f,  1.138906076e+04f,  -4.344560000e+00f,  1.139216276e+04f,  -4.345440000e+00f,
                1.137168604e+04f,  -4.342800000e+00f,  1.137478452e+04f,  -4.343680000e+00f,  1.137788300e+04f,  -4.344560000e+00f,
                1.135741332e+04f,  -4.341920000e+00f,  1.136050828e+04f,  -4.342800000e+00f,  1.136360324e+04f,  -4.343680000e+00f,
                1.134314060e+04f,  -4.341040000e+00f,  1.134623204e+04f,  -4.341920000e+00f,  1.134932348e+04f,  -4.342800000e+00f,
                1.139526476e+04f,  -4.346320000e+00f,  1.139836676e+04f,  -4.347200000e+00f,  1.140146876e+04f,  -4.348080000e+00f,
                1.138098148e+04f,  -4.345440000e+00f,  1.138407996e+04f,  -4.346320000e+00f,  1.138717844e+04f,  -4.347200000e+00f,
                1.136669820e+04f,  -4.344560000e+00f,  1.136979316e+04f,  -4.345440000e+00f,  1.137288812e+04f,  -4.346320000e+00f,
                1.135241492e+04f,  -4.343680000e+00f,  1.135550636e+04f,  -4.344560000e+00f,  1.135859780e+04f,  -4.345440000e+00f,

            };

            float[] gw_actual = gw.ToArray();

            AssertError.Tolerance(gw_expect, gw_actual, 1e-7f, 1e-5f, $"mismatch value {inchannels},{outchannels},{kwidth},{kheight},{kdepth},{inwidth},{inheight},{indepth},{batch}");
        }
    }
}
