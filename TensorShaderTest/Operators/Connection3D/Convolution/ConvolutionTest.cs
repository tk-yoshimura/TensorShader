using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;
using TensorShader;
using TensorShader.Operators.Connection3D;
using TensorShaderCudaBackend.API;

namespace TensorShaderTest.Operators.Connection3D {
    [TestClass]
    public class ConvolutionTest {
        [TestMethod]
        public void ExecuteFPTest() {
            TensorShaderCudaBackend.Environment.Precision = TensorShaderCudaBackend.Environment.PrecisionMode.Float;
            TensorShaderCudaBackend.Environment.CudnnEnabled = false;

            float max_err = 0;

            foreach (int batch in new int[] { 1, 2 }) {
                foreach ((int inchannels, int outchannels) in new (int, int)[] { (1, 1), (2, 1), (1, 2), (2, 3), (3, 1), (3, 4), (4, 5), (5, 3), (5, 10), (10, 15), (15, 5), (15, 20), (20, 32), (32, 15), (32, 33), (33, 33) }) {
                    foreach ((int kwidth, int kheight, int kdepth) in new (int, int, int)[] { (1, 1, 1), (3, 3, 3), (5, 5, 5), (1, 3, 5), (3, 5, 1), (5, 1, 3) }) {
                        foreach ((int inwidth, int inheight, int indepth) in new (int, int, int)[] { (kwidth, kheight, kdepth), (kwidth * 2, kheight * 2, kdepth * 2), (13, 13, 13), (17, 17, 17), (19, 19, 19), (17, 19, 13), (13, 17, 19), (19, 13, 17) }) {

                            int outwidth = inwidth - kwidth + 1, outheight = inheight - kheight + 1, outdepth = indepth - kdepth + 1;

                            float[] xval = (new float[inwidth * inheight * indepth * inchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
                            float[] wval = (new float[kwidth * kheight * kdepth * inchannels * outchannels]).Select((_, idx) => (idx + 1) * 1e-3f).Reverse().ToArray();

                            Map3D x = new(inchannels, inwidth, inheight, indepth, batch, xval);
                            Filter3D w = new(inchannels, outchannels, kwidth, kheight, kdepth, wval);

                            Map3D y = Reference(x, w, kwidth, kheight, kdepth);

                            OverflowCheckedTensor x_tensor = new(Shape.Map3D(inchannels, inwidth, inheight, indepth, batch), xval);
                            OverflowCheckedTensor w_tensor = new(Shape.Kernel3D(inchannels, outchannels, kwidth, kheight, kdepth), wval);

                            OverflowCheckedTensor y_tensor = new(Shape.Map3D(outchannels, outwidth, outheight, outdepth, batch));

                            Convolution ope = new(inwidth, inheight, indepth, inchannels, outchannels, kwidth, kheight, kdepth, batch);

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

            float max_err = 0;

            foreach (int batch in new int[] { 1, 2 }) {
                foreach ((int inchannels, int outchannels) in new (int, int)[] { (1, 1), (2, 1), (1, 2), (2, 3), (3, 1), (3, 4), (4, 5), (5, 3), (5, 10), (10, 15), (15, 5), (15, 20), (20, 32), (32, 15), (32, 33), (33, 33) }) {
                    foreach ((int kwidth, int kheight, int kdepth) in new (int, int, int)[] { (1, 1, 1), (3, 3, 3), (5, 5, 5), (1, 3, 5), (3, 5, 1), (5, 1, 3) }) {
                        foreach ((int inwidth, int inheight, int indepth) in new (int, int, int)[] { (kwidth, kheight, kdepth), (kwidth * 2, kheight * 2, kdepth * 2), (13, 13, 13), (17, 17, 17), (19, 19, 19), (17, 19, 13), (13, 17, 19), (19, 13, 17) }) {

                            int outwidth = inwidth - kwidth + 1, outheight = inheight - kheight + 1, outdepth = indepth - kdepth + 1;

                            float[] xval = (new float[inwidth * inheight * indepth * inchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
                            float[] wval = (new float[kwidth * kheight * kdepth * inchannels * outchannels]).Select((_, idx) => (idx + 1) * 1e-3f).Reverse().ToArray();

                            Map3D x = new(inchannels, inwidth, inheight, indepth, batch, xval);
                            Filter3D w = new(inchannels, outchannels, kwidth, kheight, kdepth, wval);

                            Map3D y = Reference(x, w, kwidth, kheight, kdepth);

                            OverflowCheckedTensor x_tensor = new(Shape.Map3D(inchannels, inwidth, inheight, indepth, batch), xval);
                            OverflowCheckedTensor w_tensor = new(Shape.Kernel3D(inchannels, outchannels, kwidth, kheight, kdepth), wval);

                            OverflowCheckedTensor y_tensor = new(Shape.Map3D(outchannels, outwidth, outheight, outdepth, batch));

                            Convolution ope = new(inwidth, inheight, indepth, inchannels, outchannels, kwidth, kheight, kdepth, batch);

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
        public void ExecuteCudnnTest() {
            if (!TensorShaderCudaBackend.Environment.CudnnExists) {
                Console.WriteLine("test was skipped. Cudnn library not exists.");
                Assert.Inconclusive();
            }

            TensorShaderCudaBackend.Environment.Precision = TensorShaderCudaBackend.Environment.PrecisionMode.Float;
            TensorShaderCudaBackend.Environment.CudnnEnabled = true;

            float max_err = 0;

            foreach (int batch in new int[] { 1, 2 }) {
                foreach ((int inchannels, int outchannels) in new (int, int)[] { (1, 1), (2, 1), (1, 2), (2, 3), (3, 1), (3, 4), (4, 5), (5, 3), (5, 10), (10, 15), (15, 5), (15, 20), (20, 32), (32, 15), (32, 33), (33, 33) }) {
                    foreach ((int kwidth, int kheight, int kdepth) in new (int, int, int)[] { (1, 1, 1), (3, 3, 3), (5, 5, 5), (1, 3, 5), (3, 5, 1), (5, 1, 3) }) {
                        foreach ((int inwidth, int inheight, int indepth) in new (int, int, int)[] { (kwidth, kheight, kdepth), (kwidth * 2, kheight * 2, kdepth * 2), (13, 13, 13), (17, 17, 17), (19, 19, 19), (17, 19, 13), (13, 17, 19), (19, 13, 17) }) {

                            int outwidth = inwidth - kwidth + 1, outheight = inheight - kheight + 1, outdepth = indepth - kdepth + 1;

                            float[] xval = (new float[inwidth * inheight * indepth * inchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
                            float[] wval = (new float[kwidth * kheight * kdepth * inchannels * outchannels]).Select((_, idx) => (idx + 1) * 1e-3f).Reverse().ToArray();

                            Map3D x = new(inchannels, inwidth, inheight, indepth, batch, xval);
                            Filter3D w = new(inchannels, outchannels, kwidth, kheight, kdepth, wval);

                            Map3D y = Reference(x, w, kwidth, kheight, kdepth);

                            OverflowCheckedTensor x_tensor = new(Shape.Map3D(inchannels, inwidth, inheight, indepth, batch), xval);
                            OverflowCheckedTensor w_tensor = new(Shape.Kernel3D(inchannels, outchannels, kwidth, kheight, kdepth), wval);

                            OverflowCheckedTensor y_tensor = new(Shape.Map3D(outchannels, outwidth, outheight, outdepth, batch));

                            Convolution ope = new(inwidth, inheight, indepth, inchannels, outchannels, kwidth, kheight, kdepth, batch);

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
        public void LargeMapTest() {
            TensorShaderCudaBackend.Environment.CudnnEnabled = false;
            TensorShaderCudaBackend.Environment.Precision = TensorShaderCudaBackend.Environment.PrecisionMode.FloatFloat;

            float max_err = 0;

            Random random = new(1234);

            int batch = 3;
            int inchannels = 49, outchannels = 50;
            int kwidth = 7, kheight = 5, kdepth = 3;
            int inwidth = 125, inheight = 196, indepth = 4;
            int outwidth = inwidth - kwidth + 1, outheight = inheight - kheight + 1, outdepth = indepth - kdepth + 1;

            float[] xval = (new float[inwidth * inheight * indepth * inchannels * batch]).Select((_, idx) => (float)random.NextDouble() * 1e-2f).ToArray();
            float[] wval = (new float[kwidth * kheight * kdepth * inchannels * outchannels]).Select((_, idx) => (float)random.NextDouble() * 1e-2f).ToArray();

            Map3D x = new(inchannels, inwidth, inheight, indepth, batch, xval);
            Filter3D w = new(inchannels, outchannels, kwidth, kheight, kdepth, wval);

            Map3D y = Reference(x, w, kwidth, kheight, kdepth);

            OverflowCheckedTensor x_tensor = new(Shape.Map3D(inchannels, inwidth, inheight, indepth, batch), xval);
            OverflowCheckedTensor w_tensor = new(Shape.Kernel3D(inchannels, outchannels, kwidth, kheight, kdepth), wval);

            OverflowCheckedTensor y_tensor = new(Shape.Map3D(outchannels, outwidth, outheight, outdepth, batch));

            Convolution ope = new(inwidth, inheight, indepth, inchannels, outchannels, kwidth, kheight, kdepth, batch);

            ope.Execute(x_tensor, w_tensor, y_tensor);

            float[] y_expect = y.ToArray();
            float[] y_actual = y_tensor.State.Value;

            CollectionAssert.AreEqual(xval, x_tensor.State.Value);
            CollectionAssert.AreEqual(wval, w_tensor.State.Value);

            AssertError.Tolerance(y_expect, y_actual, 1e-7f, 1e-5f, ref max_err, $"mismatch value {inchannels},{outchannels},{kwidth},{kheight},{kdepth},{inwidth},{inheight},{kdepth},{batch}");

            Console.WriteLine($"pass: {inchannels},{outchannels},{kwidth},{kheight},{kdepth},{inwidth},{inheight},{kdepth},{batch}");

            Console.WriteLine($"maxerr:{max_err}");
        }

        [TestMethod]
        public void SpeedFPTest() {
            TensorShaderCudaBackend.Environment.Precision = TensorShaderCudaBackend.Environment.PrecisionMode.Float;
            TensorShaderCudaBackend.Environment.CudnnEnabled = false;

            int inwidth = 64, inheight = 64, indepth = 64, inchannels = 31, outchannels = 31, ksize = 3;
            int outwidth = inwidth - ksize + 1, outheight = inheight - ksize + 1, outdepth = indepth - ksize + 1;

            OverflowCheckedTensor x_tensor = new(Shape.Map3D(inchannels, inwidth, inheight, indepth));
            OverflowCheckedTensor w_tensor = new(Shape.Kernel3D(inchannels, outchannels, ksize, ksize, ksize));

            OverflowCheckedTensor y_tensor = new(Shape.Map3D(outchannels, outwidth, outheight, outdepth));

            Convolution ope = new(inwidth, inheight, indepth, inchannels, outchannels, ksize, ksize, ksize);

            Cuda.Profiler.Initialize("../../../../profiler.nvsetting", "../../nvprofiles/convolution_3d_fp.nvvp");
            Cuda.Profiler.Start();

            ope.Execute(x_tensor, w_tensor, y_tensor);

            Cuda.Profiler.Stop();
        }

        [TestMethod]
        public void SpeedFFPTest() {
            TensorShaderCudaBackend.Environment.CudnnEnabled = false;
            TensorShaderCudaBackend.Environment.Precision = TensorShaderCudaBackend.Environment.PrecisionMode.FloatFloat;

            int inwidth = 64, inheight = 64, indepth = 64, inchannels = 31, outchannels = 31, ksize = 3;
            int outwidth = inwidth - ksize + 1, outheight = inheight - ksize + 1, outdepth = indepth - ksize + 1;

            OverflowCheckedTensor x_tensor = new(Shape.Map3D(inchannels, inwidth, inheight, indepth));
            OverflowCheckedTensor w_tensor = new(Shape.Kernel3D(inchannels, outchannels, ksize, ksize, ksize));

            OverflowCheckedTensor y_tensor = new(Shape.Map3D(outchannels, outwidth, outheight, outdepth));

            Convolution ope = new(inwidth, inheight, indepth, inchannels, outchannels, ksize, ksize, ksize);

            Cuda.Profiler.Initialize("../../../../profiler.nvsetting", "../../nvprofiles/convolution_3d_ffp.nvvp");
            Cuda.Profiler.Start();

            ope.Execute(x_tensor, w_tensor, y_tensor);

            Cuda.Profiler.Stop();
        }

        [TestMethod]
        public void SpeedCudnnTest() {
            if (!TensorShaderCudaBackend.Environment.CudnnExists) {
                Console.WriteLine("test was skipped. Cudnn library not exists.");
                Assert.Inconclusive();
            }

            TensorShaderCudaBackend.Environment.Precision = TensorShaderCudaBackend.Environment.PrecisionMode.Float;
            TensorShaderCudaBackend.Environment.CudnnEnabled = true;

            int inwidth = 64, inheight = 64, indepth = 64, inchannels = 31, outchannels = 31, ksize = 3;
            int outwidth = inwidth - ksize + 1, outheight = inheight - ksize + 1, outdepth = indepth - ksize + 1;

            OverflowCheckedTensor x_tensor = new(Shape.Map3D(inchannels, inwidth, inheight, indepth));
            OverflowCheckedTensor w_tensor = new(Shape.Kernel3D(inchannels, outchannels, ksize, ksize, ksize));

            OverflowCheckedTensor y_tensor = new(Shape.Map3D(outchannels, outwidth, outheight, outdepth));

            Convolution ope = new(inwidth, inheight, indepth, inchannels, outchannels, ksize, ksize, ksize);

            ope.Execute(x_tensor, w_tensor, y_tensor);

            Cuda.Profiler.Initialize("../../../../profiler.nvsetting", "../../nvprofiles/convolution_3d_cudnn.nvvp");
            Cuda.Profiler.Start();

            ope.Execute(x_tensor, w_tensor, y_tensor);

            Cuda.Profiler.Stop();
        }

        public static Map3D Reference(Map3D x, Filter3D w, int kwidth, int kheight, int kdepth) {
            int inchannels = x.Channels, outchannels = w.OutChannels, batch = x.Batch;
            int inw = x.Width, inh = x.Height, ind = x.Depth;
            int outw = inw - kwidth + 1, outh = inh - kheight + 1, outd = ind - kdepth + 1;

            Map3D y = new(outchannels, outw, outh, outd, batch);

            for (int kx, ky, kz = 0; kz < kdepth; kz++) {
                for (ky = 0; ky < kheight; ky++) {
                    for (kx = 0; kx < kwidth; kx++) {
                        for (int th = 0; th < batch; th++) {
                            for (int ox, oy, oz = 0; oz < outd; oz++) {
                                for (oy = 0; oy < outh; oy++) {
                                    for (ox = 0; ox < outw; ox++) {
                                        for (int outch = 0; outch < outchannels; outch++) {
                                            double sum = y[outch, ox, oy, oz, th];

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
            int inchannels = 2, outchannels = 3, kwidth = 3, kheight = 5, kdepth = 7, inwidth = 13, inheight = 12, indepth = 11, batch = 2;
            int outwidth = inwidth - kwidth + 1, outheight = inheight - kheight + 1, outdepth = indepth - kdepth + 1;

            float[] xval = (new float[batch * inwidth * inheight * indepth * inchannels]).Select((_, idx) => idx * 1e-3f).ToArray();
            float[] wval = (new float[kwidth * kheight * kdepth * outchannels * inchannels]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

            Map3D x = new(inchannels, inwidth, inheight, indepth, batch, xval);
            Filter3D w = new(inchannels, outchannels, kwidth, kheight, kdepth, wval);

            Map3D y = Reference(x, w, kwidth, kheight, kdepth);

            float[] y_expect = {
                4.204809000e+01f,  4.163208000e+01f,  4.121607000e+01f,  4.218102000e+01f,  4.176417000e+01f,  4.134732000e+01f,  4.231395000e+01f,  4.189626000e+01f,
                4.147857000e+01f,  4.244688000e+01f,  4.202835000e+01f,  4.160982000e+01f,  4.257981000e+01f,  4.216044000e+01f,  4.174107000e+01f,  4.271274000e+01f,
                4.229253000e+01f,  4.187232000e+01f,  4.284567000e+01f,  4.242462000e+01f,  4.200357000e+01f,  4.297860000e+01f,  4.255671000e+01f,  4.213482000e+01f,
                4.311153000e+01f,  4.268880000e+01f,  4.226607000e+01f,  4.324446000e+01f,  4.282089000e+01f,  4.239732000e+01f,  4.337739000e+01f,  4.295298000e+01f,
                4.252857000e+01f,  4.377618000e+01f,  4.334925000e+01f,  4.292232000e+01f,  4.390911000e+01f,  4.348134000e+01f,  4.305357000e+01f,  4.404204000e+01f,
                4.361343000e+01f,  4.318482000e+01f,  4.417497000e+01f,  4.374552000e+01f,  4.331607000e+01f,  4.430790000e+01f,  4.387761000e+01f,  4.344732000e+01f,
                4.444083000e+01f,  4.400970000e+01f,  4.357857000e+01f,  4.457376000e+01f,  4.414179000e+01f,  4.370982000e+01f,  4.470669000e+01f,  4.427388000e+01f,
                4.384107000e+01f,  4.483962000e+01f,  4.440597000e+01f,  4.397232000e+01f,  4.497255000e+01f,  4.453806000e+01f,  4.410357000e+01f,  4.510548000e+01f,
                4.467015000e+01f,  4.423482000e+01f,  4.550427000e+01f,  4.506642000e+01f,  4.462857000e+01f,  4.563720000e+01f,  4.519851000e+01f,  4.475982000e+01f,
                4.577013000e+01f,  4.533060000e+01f,  4.489107000e+01f,  4.590306000e+01f,  4.546269000e+01f,  4.502232000e+01f,  4.603599000e+01f,  4.559478000e+01f,
                4.515357000e+01f,  4.616892000e+01f,  4.572687000e+01f,  4.528482000e+01f,  4.630185000e+01f,  4.585896000e+01f,  4.541607000e+01f,  4.643478000e+01f,
                4.599105000e+01f,  4.554732000e+01f,  4.656771000e+01f,  4.612314000e+01f,  4.567857000e+01f,  4.670064000e+01f,  4.625523000e+01f,  4.580982000e+01f,
                4.683357000e+01f,  4.638732000e+01f,  4.594107000e+01f,  4.723236000e+01f,  4.678359000e+01f,  4.633482000e+01f,  4.736529000e+01f,  4.691568000e+01f,
                4.646607000e+01f,  4.749822000e+01f,  4.704777000e+01f,  4.659732000e+01f,  4.763115000e+01f,  4.717986000e+01f,  4.672857000e+01f,  4.776408000e+01f,
                4.731195000e+01f,  4.685982000e+01f,  4.789701000e+01f,  4.744404000e+01f,  4.699107000e+01f,  4.802994000e+01f,  4.757613000e+01f,  4.712232000e+01f,
                4.816287000e+01f,  4.770822000e+01f,  4.725357000e+01f,  4.829580000e+01f,  4.784031000e+01f,  4.738482000e+01f,  4.842873000e+01f,  4.797240000e+01f,
                4.751607000e+01f,  4.856166000e+01f,  4.810449000e+01f,  4.764732000e+01f,  4.896045000e+01f,  4.850076000e+01f,  4.804107000e+01f,  4.909338000e+01f,
                4.863285000e+01f,  4.817232000e+01f,  4.922631000e+01f,  4.876494000e+01f,  4.830357000e+01f,  4.935924000e+01f,  4.889703000e+01f,  4.843482000e+01f,
                4.949217000e+01f,  4.902912000e+01f,  4.856607000e+01f,  4.962510000e+01f,  4.916121000e+01f,  4.869732000e+01f,  4.975803000e+01f,  4.929330000e+01f,
                4.882857000e+01f,  4.989096000e+01f,  4.942539000e+01f,  4.895982000e+01f,  5.002389000e+01f,  4.955748000e+01f,  4.909107000e+01f,  5.015682000e+01f,
                4.968957000e+01f,  4.922232000e+01f,  5.028975000e+01f,  4.982166000e+01f,  4.935357000e+01f,  5.068854000e+01f,  5.021793000e+01f,  4.974732000e+01f,
                5.082147000e+01f,  5.035002000e+01f,  4.987857000e+01f,  5.095440000e+01f,  5.048211000e+01f,  5.000982000e+01f,  5.108733000e+01f,  5.061420000e+01f,
                5.014107000e+01f,  5.122026000e+01f,  5.074629000e+01f,  5.027232000e+01f,  5.135319000e+01f,  5.087838000e+01f,  5.040357000e+01f,  5.148612000e+01f,
                5.101047000e+01f,  5.053482000e+01f,  5.161905000e+01f,  5.114256000e+01f,  5.066607000e+01f,  5.175198000e+01f,  5.127465000e+01f,  5.079732000e+01f,
                5.188491000e+01f,  5.140674000e+01f,  5.092857000e+01f,  5.201784000e+01f,  5.153883000e+01f,  5.105982000e+01f,  5.241663000e+01f,  5.193510000e+01f,
                5.145357000e+01f,  5.254956000e+01f,  5.206719000e+01f,  5.158482000e+01f,  5.268249000e+01f,  5.219928000e+01f,  5.171607000e+01f,  5.281542000e+01f,
                5.233137000e+01f,  5.184732000e+01f,  5.294835000e+01f,  5.246346000e+01f,  5.197857000e+01f,  5.308128000e+01f,  5.259555000e+01f,  5.210982000e+01f,
                5.321421000e+01f,  5.272764000e+01f,  5.224107000e+01f,  5.334714000e+01f,  5.285973000e+01f,  5.237232000e+01f,  5.348007000e+01f,  5.299182000e+01f,
                5.250357000e+01f,  5.361300000e+01f,  5.312391000e+01f,  5.263482000e+01f,  5.374593000e+01f,  5.325600000e+01f,  5.276607000e+01f,  5.414472000e+01f,
                5.365227000e+01f,  5.315982000e+01f,  5.427765000e+01f,  5.378436000e+01f,  5.329107000e+01f,  5.441058000e+01f,  5.391645000e+01f,  5.342232000e+01f,
                5.454351000e+01f,  5.404854000e+01f,  5.355357000e+01f,  5.467644000e+01f,  5.418063000e+01f,  5.368482000e+01f,  5.480937000e+01f,  5.431272000e+01f,
                5.381607000e+01f,  5.494230000e+01f,  5.444481000e+01f,  5.394732000e+01f,  5.507523000e+01f,  5.457690000e+01f,  5.407857000e+01f,  5.520816000e+01f,
                5.470899000e+01f,  5.420982000e+01f,  5.534109000e+01f,  5.484108000e+01f,  5.434107000e+01f,  5.547402000e+01f,  5.497317000e+01f,  5.447232000e+01f,
                6.278517000e+01f,  6.223812000e+01f,  6.169107000e+01f,  6.291810000e+01f,  6.237021000e+01f,  6.182232000e+01f,  6.305103000e+01f,  6.250230000e+01f,
                6.195357000e+01f,  6.318396000e+01f,  6.263439000e+01f,  6.208482000e+01f,  6.331689000e+01f,  6.276648000e+01f,  6.221607000e+01f,  6.344982000e+01f,
                6.289857000e+01f,  6.234732000e+01f,  6.358275000e+01f,  6.303066000e+01f,  6.247857000e+01f,  6.371568000e+01f,  6.316275000e+01f,  6.260982000e+01f,
                6.384861000e+01f,  6.329484000e+01f,  6.274107000e+01f,  6.398154000e+01f,  6.342693000e+01f,  6.287232000e+01f,  6.411447000e+01f,  6.355902000e+01f,
                6.300357000e+01f,  6.451326000e+01f,  6.395529000e+01f,  6.339732000e+01f,  6.464619000e+01f,  6.408738000e+01f,  6.352857000e+01f,  6.477912000e+01f,
                6.421947000e+01f,  6.365982000e+01f,  6.491205000e+01f,  6.435156000e+01f,  6.379107000e+01f,  6.504498000e+01f,  6.448365000e+01f,  6.392232000e+01f,
                6.517791000e+01f,  6.461574000e+01f,  6.405357000e+01f,  6.531084000e+01f,  6.474783000e+01f,  6.418482000e+01f,  6.544377000e+01f,  6.487992000e+01f,
                6.431607000e+01f,  6.557670000e+01f,  6.501201000e+01f,  6.444732000e+01f,  6.570963000e+01f,  6.514410000e+01f,  6.457857000e+01f,  6.584256000e+01f,
                6.527619000e+01f,  6.470982000e+01f,  6.624135000e+01f,  6.567246000e+01f,  6.510357000e+01f,  6.637428000e+01f,  6.580455000e+01f,  6.523482000e+01f,
                6.650721000e+01f,  6.593664000e+01f,  6.536607000e+01f,  6.664014000e+01f,  6.606873000e+01f,  6.549732000e+01f,  6.677307000e+01f,  6.620082000e+01f,
                6.562857000e+01f,  6.690600000e+01f,  6.633291000e+01f,  6.575982000e+01f,  6.703893000e+01f,  6.646500000e+01f,  6.589107000e+01f,  6.717186000e+01f,
                6.659709000e+01f,  6.602232000e+01f,  6.730479000e+01f,  6.672918000e+01f,  6.615357000e+01f,  6.743772000e+01f,  6.686127000e+01f,  6.628482000e+01f,
                6.757065000e+01f,  6.699336000e+01f,  6.641607000e+01f,  6.796944000e+01f,  6.738963000e+01f,  6.680982000e+01f,  6.810237000e+01f,  6.752172000e+01f,
                6.694107000e+01f,  6.823530000e+01f,  6.765381000e+01f,  6.707232000e+01f,  6.836823000e+01f,  6.778590000e+01f,  6.720357000e+01f,  6.850116000e+01f,
                6.791799000e+01f,  6.733482000e+01f,  6.863409000e+01f,  6.805008000e+01f,  6.746607000e+01f,  6.876702000e+01f,  6.818217000e+01f,  6.759732000e+01f,
                6.889995000e+01f,  6.831426000e+01f,  6.772857000e+01f,  6.903288000e+01f,  6.844635000e+01f,  6.785982000e+01f,  6.916581000e+01f,  6.857844000e+01f,
                6.799107000e+01f,  6.929874000e+01f,  6.871053000e+01f,  6.812232000e+01f,  6.969753000e+01f,  6.910680000e+01f,  6.851607000e+01f,  6.983046000e+01f,
                6.923889000e+01f,  6.864732000e+01f,  6.996339000e+01f,  6.937098000e+01f,  6.877857000e+01f,  7.009632000e+01f,  6.950307000e+01f,  6.890982000e+01f,
                7.022925000e+01f,  6.963516000e+01f,  6.904107000e+01f,  7.036218000e+01f,  6.976725000e+01f,  6.917232000e+01f,  7.049511000e+01f,  6.989934000e+01f,
                6.930357000e+01f,  7.062804000e+01f,  7.003143000e+01f,  6.943482000e+01f,  7.076097000e+01f,  7.016352000e+01f,  6.956607000e+01f,  7.089390000e+01f,
                7.029561000e+01f,  6.969732000e+01f,  7.102683000e+01f,  7.042770000e+01f,  6.982857000e+01f,  7.142562000e+01f,  7.082397000e+01f,  7.022232000e+01f,
                7.155855000e+01f,  7.095606000e+01f,  7.035357000e+01f,  7.169148000e+01f,  7.108815000e+01f,  7.048482000e+01f,  7.182441000e+01f,  7.122024000e+01f,
                7.061607000e+01f,  7.195734000e+01f,  7.135233000e+01f,  7.074732000e+01f,  7.209027000e+01f,  7.148442000e+01f,  7.087857000e+01f,  7.222320000e+01f,
                7.161651000e+01f,  7.100982000e+01f,  7.235613000e+01f,  7.174860000e+01f,  7.114107000e+01f,  7.248906000e+01f,  7.188069000e+01f,  7.127232000e+01f,
                7.262199000e+01f,  7.201278000e+01f,  7.140357000e+01f,  7.275492000e+01f,  7.214487000e+01f,  7.153482000e+01f,  7.315371000e+01f,  7.254114000e+01f,
                7.192857000e+01f,  7.328664000e+01f,  7.267323000e+01f,  7.205982000e+01f,  7.341957000e+01f,  7.280532000e+01f,  7.219107000e+01f,  7.355250000e+01f,
                7.293741000e+01f,  7.232232000e+01f,  7.368543000e+01f,  7.306950000e+01f,  7.245357000e+01f,  7.381836000e+01f,  7.320159000e+01f,  7.258482000e+01f,
                7.395129000e+01f,  7.333368000e+01f,  7.271607000e+01f,  7.408422000e+01f,  7.346577000e+01f,  7.284732000e+01f,  7.421715000e+01f,  7.359786000e+01f,
                7.297857000e+01f,  7.435008000e+01f,  7.372995000e+01f,  7.310982000e+01f,  7.448301000e+01f,  7.386204000e+01f,  7.324107000e+01f,  7.488180000e+01f,
                7.425831000e+01f,  7.363482000e+01f,  7.501473000e+01f,  7.439040000e+01f,  7.376607000e+01f,  7.514766000e+01f,  7.452249000e+01f,  7.389732000e+01f,
                7.528059000e+01f,  7.465458000e+01f,  7.402857000e+01f,  7.541352000e+01f,  7.478667000e+01f,  7.415982000e+01f,  7.554645000e+01f,  7.491876000e+01f,
                7.429107000e+01f,  7.567938000e+01f,  7.505085000e+01f,  7.442232000e+01f,  7.581231000e+01f,  7.518294000e+01f,  7.455357000e+01f,  7.594524000e+01f,
                7.531503000e+01f,  7.468482000e+01f,  7.607817000e+01f,  7.544712000e+01f,  7.481607000e+01f,  7.621110000e+01f,  7.557921000e+01f,  7.494732000e+01f,
                8.352225000e+01f,  8.284416000e+01f,  8.216607000e+01f,  8.365518000e+01f,  8.297625000e+01f,  8.229732000e+01f,  8.378811000e+01f,  8.310834000e+01f,
                8.242857000e+01f,  8.392104000e+01f,  8.324043000e+01f,  8.255982000e+01f,  8.405397000e+01f,  8.337252000e+01f,  8.269107000e+01f,  8.418690000e+01f,
                8.350461000e+01f,  8.282232000e+01f,  8.431983000e+01f,  8.363670000e+01f,  8.295357000e+01f,  8.445276000e+01f,  8.376879000e+01f,  8.308482000e+01f,
                8.458569000e+01f,  8.390088000e+01f,  8.321607000e+01f,  8.471862000e+01f,  8.403297000e+01f,  8.334732000e+01f,  8.485155000e+01f,  8.416506000e+01f,
                8.347857000e+01f,  8.525034000e+01f,  8.456133000e+01f,  8.387232000e+01f,  8.538327000e+01f,  8.469342000e+01f,  8.400357000e+01f,  8.551620000e+01f,
                8.482551000e+01f,  8.413482000e+01f,  8.564913000e+01f,  8.495760000e+01f,  8.426607000e+01f,  8.578206000e+01f,  8.508969000e+01f,  8.439732000e+01f,
                8.591499000e+01f,  8.522178000e+01f,  8.452857000e+01f,  8.604792000e+01f,  8.535387000e+01f,  8.465982000e+01f,  8.618085000e+01f,  8.548596000e+01f,
                8.479107000e+01f,  8.631378000e+01f,  8.561805000e+01f,  8.492232000e+01f,  8.644671000e+01f,  8.575014000e+01f,  8.505357000e+01f,  8.657964000e+01f,
                8.588223000e+01f,  8.518482000e+01f,  8.697843000e+01f,  8.627850000e+01f,  8.557857000e+01f,  8.711136000e+01f,  8.641059000e+01f,  8.570982000e+01f,
                8.724429000e+01f,  8.654268000e+01f,  8.584107000e+01f,  8.737722000e+01f,  8.667477000e+01f,  8.597232000e+01f,  8.751015000e+01f,  8.680686000e+01f,
                8.610357000e+01f,  8.764308000e+01f,  8.693895000e+01f,  8.623482000e+01f,  8.777601000e+01f,  8.707104000e+01f,  8.636607000e+01f,  8.790894000e+01f,
                8.720313000e+01f,  8.649732000e+01f,  8.804187000e+01f,  8.733522000e+01f,  8.662857000e+01f,  8.817480000e+01f,  8.746731000e+01f,  8.675982000e+01f,
                8.830773000e+01f,  8.759940000e+01f,  8.689107000e+01f,  8.870652000e+01f,  8.799567000e+01f,  8.728482000e+01f,  8.883945000e+01f,  8.812776000e+01f,
                8.741607000e+01f,  8.897238000e+01f,  8.825985000e+01f,  8.754732000e+01f,  8.910531000e+01f,  8.839194000e+01f,  8.767857000e+01f,  8.923824000e+01f,
                8.852403000e+01f,  8.780982000e+01f,  8.937117000e+01f,  8.865612000e+01f,  8.794107000e+01f,  8.950410000e+01f,  8.878821000e+01f,  8.807232000e+01f,
                8.963703000e+01f,  8.892030000e+01f,  8.820357000e+01f,  8.976996000e+01f,  8.905239000e+01f,  8.833482000e+01f,  8.990289000e+01f,  8.918448000e+01f,
                8.846607000e+01f,  9.003582000e+01f,  8.931657000e+01f,  8.859732000e+01f,  9.043461000e+01f,  8.971284000e+01f,  8.899107000e+01f,  9.056754000e+01f,
                8.984493000e+01f,  8.912232000e+01f,  9.070047000e+01f,  8.997702000e+01f,  8.925357000e+01f,  9.083340000e+01f,  9.010911000e+01f,  8.938482000e+01f,
                9.096633000e+01f,  9.024120000e+01f,  8.951607000e+01f,  9.109926000e+01f,  9.037329000e+01f,  8.964732000e+01f,  9.123219000e+01f,  9.050538000e+01f,
                8.977857000e+01f,  9.136512000e+01f,  9.063747000e+01f,  8.990982000e+01f,  9.149805000e+01f,  9.076956000e+01f,  9.004107000e+01f,  9.163098000e+01f,
                9.090165000e+01f,  9.017232000e+01f,  9.176391000e+01f,  9.103374000e+01f,  9.030357000e+01f,  9.216270000e+01f,  9.143001000e+01f,  9.069732000e+01f,
                9.229563000e+01f,  9.156210000e+01f,  9.082857000e+01f,  9.242856000e+01f,  9.169419000e+01f,  9.095982000e+01f,  9.256149000e+01f,  9.182628000e+01f,
                9.109107000e+01f,  9.269442000e+01f,  9.195837000e+01f,  9.122232000e+01f,  9.282735000e+01f,  9.209046000e+01f,  9.135357000e+01f,  9.296028000e+01f,
                9.222255000e+01f,  9.148482000e+01f,  9.309321000e+01f,  9.235464000e+01f,  9.161607000e+01f,  9.322614000e+01f,  9.248673000e+01f,  9.174732000e+01f,
                9.335907000e+01f,  9.261882000e+01f,  9.187857000e+01f,  9.349200000e+01f,  9.275091000e+01f,  9.200982000e+01f,  9.389079000e+01f,  9.314718000e+01f,
                9.240357000e+01f,  9.402372000e+01f,  9.327927000e+01f,  9.253482000e+01f,  9.415665000e+01f,  9.341136000e+01f,  9.266607000e+01f,  9.428958000e+01f,
                9.354345000e+01f,  9.279732000e+01f,  9.442251000e+01f,  9.367554000e+01f,  9.292857000e+01f,  9.455544000e+01f,  9.380763000e+01f,  9.305982000e+01f,
                9.468837000e+01f,  9.393972000e+01f,  9.319107000e+01f,  9.482130000e+01f,  9.407181000e+01f,  9.332232000e+01f,  9.495423000e+01f,  9.420390000e+01f,
                9.345357000e+01f,  9.508716000e+01f,  9.433599000e+01f,  9.358482000e+01f,  9.522009000e+01f,  9.446808000e+01f,  9.371607000e+01f,  9.561888000e+01f,
                9.486435000e+01f,  9.410982000e+01f,  9.575181000e+01f,  9.499644000e+01f,  9.424107000e+01f,  9.588474000e+01f,  9.512853000e+01f,  9.437232000e+01f,
                9.601767000e+01f,  9.526062000e+01f,  9.450357000e+01f,  9.615060000e+01f,  9.539271000e+01f,  9.463482000e+01f,  9.628353000e+01f,  9.552480000e+01f,
                9.476607000e+01f,  9.641646000e+01f,  9.565689000e+01f,  9.489732000e+01f,  9.654939000e+01f,  9.578898000e+01f,  9.502857000e+01f,  9.668232000e+01f,
                9.592107000e+01f,  9.515982000e+01f,  9.681525000e+01f,  9.605316000e+01f,  9.529107000e+01f,  9.694818000e+01f,  9.618525000e+01f,  9.542232000e+01f,
                1.042593300e+02f,  1.034502000e+02f,  1.026410700e+02f,  1.043922600e+02f,  1.035822900e+02f,  1.027723200e+02f,  1.045251900e+02f,  1.037143800e+02f,
                1.029035700e+02f,  1.046581200e+02f,  1.038464700e+02f,  1.030348200e+02f,  1.047910500e+02f,  1.039785600e+02f,  1.031660700e+02f,  1.049239800e+02f,
                1.041106500e+02f,  1.032973200e+02f,  1.050569100e+02f,  1.042427400e+02f,  1.034285700e+02f,  1.051898400e+02f,  1.043748300e+02f,  1.035598200e+02f,
                1.053227700e+02f,  1.045069200e+02f,  1.036910700e+02f,  1.054557000e+02f,  1.046390100e+02f,  1.038223200e+02f,  1.055886300e+02f,  1.047711000e+02f,
                1.039535700e+02f,  1.059874200e+02f,  1.051673700e+02f,  1.043473200e+02f,  1.061203500e+02f,  1.052994600e+02f,  1.044785700e+02f,  1.062532800e+02f,
                1.054315500e+02f,  1.046098200e+02f,  1.063862100e+02f,  1.055636400e+02f,  1.047410700e+02f,  1.065191400e+02f,  1.056957300e+02f,  1.048723200e+02f,
                1.066520700e+02f,  1.058278200e+02f,  1.050035700e+02f,  1.067850000e+02f,  1.059599100e+02f,  1.051348200e+02f,  1.069179300e+02f,  1.060920000e+02f,
                1.052660700e+02f,  1.070508600e+02f,  1.062240900e+02f,  1.053973200e+02f,  1.071837900e+02f,  1.063561800e+02f,  1.055285700e+02f,  1.073167200e+02f,
                1.064882700e+02f,  1.056598200e+02f,  1.077155100e+02f,  1.068845400e+02f,  1.060535700e+02f,  1.078484400e+02f,  1.070166300e+02f,  1.061848200e+02f,
                1.079813700e+02f,  1.071487200e+02f,  1.063160700e+02f,  1.081143000e+02f,  1.072808100e+02f,  1.064473200e+02f,  1.082472300e+02f,  1.074129000e+02f,
                1.065785700e+02f,  1.083801600e+02f,  1.075449900e+02f,  1.067098200e+02f,  1.085130900e+02f,  1.076770800e+02f,  1.068410700e+02f,  1.086460200e+02f,
                1.078091700e+02f,  1.069723200e+02f,  1.087789500e+02f,  1.079412600e+02f,  1.071035700e+02f,  1.089118800e+02f,  1.080733500e+02f,  1.072348200e+02f,
                1.090448100e+02f,  1.082054400e+02f,  1.073660700e+02f,  1.094436000e+02f,  1.086017100e+02f,  1.077598200e+02f,  1.095765300e+02f,  1.087338000e+02f,
                1.078910700e+02f,  1.097094600e+02f,  1.088658900e+02f,  1.080223200e+02f,  1.098423900e+02f,  1.089979800e+02f,  1.081535700e+02f,  1.099753200e+02f,
                1.091300700e+02f,  1.082848200e+02f,  1.101082500e+02f,  1.092621600e+02f,  1.084160700e+02f,  1.102411800e+02f,  1.093942500e+02f,  1.085473200e+02f,
                1.103741100e+02f,  1.095263400e+02f,  1.086785700e+02f,  1.105070400e+02f,  1.096584300e+02f,  1.088098200e+02f,  1.106399700e+02f,  1.097905200e+02f,
                1.089410700e+02f,  1.107729000e+02f,  1.099226100e+02f,  1.090723200e+02f,  1.111716900e+02f,  1.103188800e+02f,  1.094660700e+02f,  1.113046200e+02f,
                1.104509700e+02f,  1.095973200e+02f,  1.114375500e+02f,  1.105830600e+02f,  1.097285700e+02f,  1.115704800e+02f,  1.107151500e+02f,  1.098598200e+02f,
                1.117034100e+02f,  1.108472400e+02f,  1.099910700e+02f,  1.118363400e+02f,  1.109793300e+02f,  1.101223200e+02f,  1.119692700e+02f,  1.111114200e+02f,
                1.102535700e+02f,  1.121022000e+02f,  1.112435100e+02f,  1.103848200e+02f,  1.122351300e+02f,  1.113756000e+02f,  1.105160700e+02f,  1.123680600e+02f,
                1.115076900e+02f,  1.106473200e+02f,  1.125009900e+02f,  1.116397800e+02f,  1.107785700e+02f,  1.128997800e+02f,  1.120360500e+02f,  1.111723200e+02f,
                1.130327100e+02f,  1.121681400e+02f,  1.113035700e+02f,  1.131656400e+02f,  1.123002300e+02f,  1.114348200e+02f,  1.132985700e+02f,  1.124323200e+02f,
                1.115660700e+02f,  1.134315000e+02f,  1.125644100e+02f,  1.116973200e+02f,  1.135644300e+02f,  1.126965000e+02f,  1.118285700e+02f,  1.136973600e+02f,
                1.128285900e+02f,  1.119598200e+02f,  1.138302900e+02f,  1.129606800e+02f,  1.120910700e+02f,  1.139632200e+02f,  1.130927700e+02f,  1.122223200e+02f,
                1.140961500e+02f,  1.132248600e+02f,  1.123535700e+02f,  1.142290800e+02f,  1.133569500e+02f,  1.124848200e+02f,  1.146278700e+02f,  1.137532200e+02f,
                1.128785700e+02f,  1.147608000e+02f,  1.138853100e+02f,  1.130098200e+02f,  1.148937300e+02f,  1.140174000e+02f,  1.131410700e+02f,  1.150266600e+02f,
                1.141494900e+02f,  1.132723200e+02f,  1.151595900e+02f,  1.142815800e+02f,  1.134035700e+02f,  1.152925200e+02f,  1.144136700e+02f,  1.135348200e+02f,
                1.154254500e+02f,  1.145457600e+02f,  1.136660700e+02f,  1.155583800e+02f,  1.146778500e+02f,  1.137973200e+02f,  1.156913100e+02f,  1.148099400e+02f,
                1.139285700e+02f,  1.158242400e+02f,  1.149420300e+02f,  1.140598200e+02f,  1.159571700e+02f,  1.150741200e+02f,  1.141910700e+02f,  1.163559600e+02f,
                1.154703900e+02f,  1.145848200e+02f,  1.164888900e+02f,  1.156024800e+02f,  1.147160700e+02f,  1.166218200e+02f,  1.157345700e+02f,  1.148473200e+02f,
                1.167547500e+02f,  1.158666600e+02f,  1.149785700e+02f,  1.168876800e+02f,  1.159987500e+02f,  1.151098200e+02f,  1.170206100e+02f,  1.161308400e+02f,
                1.152410700e+02f,  1.171535400e+02f,  1.162629300e+02f,  1.153723200e+02f,  1.172864700e+02f,  1.163950200e+02f,  1.155035700e+02f,  1.174194000e+02f,
                1.165271100e+02f,  1.156348200e+02f,  1.175523300e+02f,  1.166592000e+02f,  1.157660700e+02f,  1.176852600e+02f,  1.167912900e+02f,  1.158973200e+02f,
                1.249964100e+02f,  1.240562400e+02f,  1.231160700e+02f,  1.251293400e+02f,  1.241883300e+02f,  1.232473200e+02f,  1.252622700e+02f,  1.243204200e+02f,
                1.233785700e+02f,  1.253952000e+02f,  1.244525100e+02f,  1.235098200e+02f,  1.255281300e+02f,  1.245846000e+02f,  1.236410700e+02f,  1.256610600e+02f,
                1.247166900e+02f,  1.237723200e+02f,  1.257939900e+02f,  1.248487800e+02f,  1.239035700e+02f,  1.259269200e+02f,  1.249808700e+02f,  1.240348200e+02f,
                1.260598500e+02f,  1.251129600e+02f,  1.241660700e+02f,  1.261927800e+02f,  1.252450500e+02f,  1.242973200e+02f,  1.263257100e+02f,  1.253771400e+02f,
                1.244285700e+02f,  1.267245000e+02f,  1.257734100e+02f,  1.248223200e+02f,  1.268574300e+02f,  1.259055000e+02f,  1.249535700e+02f,  1.269903600e+02f,
                1.260375900e+02f,  1.250848200e+02f,  1.271232900e+02f,  1.261696800e+02f,  1.252160700e+02f,  1.272562200e+02f,  1.263017700e+02f,  1.253473200e+02f,
                1.273891500e+02f,  1.264338600e+02f,  1.254785700e+02f,  1.275220800e+02f,  1.265659500e+02f,  1.256098200e+02f,  1.276550100e+02f,  1.266980400e+02f,
                1.257410700e+02f,  1.277879400e+02f,  1.268301300e+02f,  1.258723200e+02f,  1.279208700e+02f,  1.269622200e+02f,  1.260035700e+02f,  1.280538000e+02f,
                1.270943100e+02f,  1.261348200e+02f,  1.284525900e+02f,  1.274905800e+02f,  1.265285700e+02f,  1.285855200e+02f,  1.276226700e+02f,  1.266598200e+02f,
                1.287184500e+02f,  1.277547600e+02f,  1.267910700e+02f,  1.288513800e+02f,  1.278868500e+02f,  1.269223200e+02f,  1.289843100e+02f,  1.280189400e+02f,
                1.270535700e+02f,  1.291172400e+02f,  1.281510300e+02f,  1.271848200e+02f,  1.292501700e+02f,  1.282831200e+02f,  1.273160700e+02f,  1.293831000e+02f,
                1.284152100e+02f,  1.274473200e+02f,  1.295160300e+02f,  1.285473000e+02f,  1.275785700e+02f,  1.296489600e+02f,  1.286793900e+02f,  1.277098200e+02f,
                1.297818900e+02f,  1.288114800e+02f,  1.278410700e+02f,  1.301806800e+02f,  1.292077500e+02f,  1.282348200e+02f,  1.303136100e+02f,  1.293398400e+02f,
                1.283660700e+02f,  1.304465400e+02f,  1.294719300e+02f,  1.284973200e+02f,  1.305794700e+02f,  1.296040200e+02f,  1.286285700e+02f,  1.307124000e+02f,
                1.297361100e+02f,  1.287598200e+02f,  1.308453300e+02f,  1.298682000e+02f,  1.288910700e+02f,  1.309782600e+02f,  1.300002900e+02f,  1.290223200e+02f,
                1.311111900e+02f,  1.301323800e+02f,  1.291535700e+02f,  1.312441200e+02f,  1.302644700e+02f,  1.292848200e+02f,  1.313770500e+02f,  1.303965600e+02f,
                1.294160700e+02f,  1.315099800e+02f,  1.305286500e+02f,  1.295473200e+02f,  1.319087700e+02f,  1.309249200e+02f,  1.299410700e+02f,  1.320417000e+02f,
                1.310570100e+02f,  1.300723200e+02f,  1.321746300e+02f,  1.311891000e+02f,  1.302035700e+02f,  1.323075600e+02f,  1.313211900e+02f,  1.303348200e+02f,
                1.324404900e+02f,  1.314532800e+02f,  1.304660700e+02f,  1.325734200e+02f,  1.315853700e+02f,  1.305973200e+02f,  1.327063500e+02f,  1.317174600e+02f,
                1.307285700e+02f,  1.328392800e+02f,  1.318495500e+02f,  1.308598200e+02f,  1.329722100e+02f,  1.319816400e+02f,  1.309910700e+02f,  1.331051400e+02f,
                1.321137300e+02f,  1.311223200e+02f,  1.332380700e+02f,  1.322458200e+02f,  1.312535700e+02f,  1.336368600e+02f,  1.326420900e+02f,  1.316473200e+02f,
                1.337697900e+02f,  1.327741800e+02f,  1.317785700e+02f,  1.339027200e+02f,  1.329062700e+02f,  1.319098200e+02f,  1.340356500e+02f,  1.330383600e+02f,
                1.320410700e+02f,  1.341685800e+02f,  1.331704500e+02f,  1.321723200e+02f,  1.343015100e+02f,  1.333025400e+02f,  1.323035700e+02f,  1.344344400e+02f,
                1.334346300e+02f,  1.324348200e+02f,  1.345673700e+02f,  1.335667200e+02f,  1.325660700e+02f,  1.347003000e+02f,  1.336988100e+02f,  1.326973200e+02f,
                1.348332300e+02f,  1.338309000e+02f,  1.328285700e+02f,  1.349661600e+02f,  1.339629900e+02f,  1.329598200e+02f,  1.353649500e+02f,  1.343592600e+02f,
                1.333535700e+02f,  1.354978800e+02f,  1.344913500e+02f,  1.334848200e+02f,  1.356308100e+02f,  1.346234400e+02f,  1.336160700e+02f,  1.357637400e+02f,
                1.347555300e+02f,  1.337473200e+02f,  1.358966700e+02f,  1.348876200e+02f,  1.338785700e+02f,  1.360296000e+02f,  1.350197100e+02f,  1.340098200e+02f,
                1.361625300e+02f,  1.351518000e+02f,  1.341410700e+02f,  1.362954600e+02f,  1.352838900e+02f,  1.342723200e+02f,  1.364283900e+02f,  1.354159800e+02f,
                1.344035700e+02f,  1.365613200e+02f,  1.355480700e+02f,  1.345348200e+02f,  1.366942500e+02f,  1.356801600e+02f,  1.346660700e+02f,  1.370930400e+02f,
                1.360764300e+02f,  1.350598200e+02f,  1.372259700e+02f,  1.362085200e+02f,  1.351910700e+02f,  1.373589000e+02f,  1.363406100e+02f,  1.353223200e+02f,
                1.374918300e+02f,  1.364727000e+02f,  1.354535700e+02f,  1.376247600e+02f,  1.366047900e+02f,  1.355848200e+02f,  1.377576900e+02f,  1.367368800e+02f,
                1.357160700e+02f,  1.378906200e+02f,  1.368689700e+02f,  1.358473200e+02f,  1.380235500e+02f,  1.370010600e+02f,  1.359785700e+02f,  1.381564800e+02f,
                1.371331500e+02f,  1.361098200e+02f,  1.382894100e+02f,  1.372652400e+02f,  1.362410700e+02f,  1.384223400e+02f,  1.373973300e+02f,  1.363723200e+02f,
                2.701559700e+02f,  2.682985200e+02f,  2.664410700e+02f,  2.702889000e+02f,  2.684306100e+02f,  2.665723200e+02f,  2.704218300e+02f,  2.685627000e+02f,
                2.667035700e+02f,  2.705547600e+02f,  2.686947900e+02f,  2.668348200e+02f,  2.706876900e+02f,  2.688268800e+02f,  2.669660700e+02f,  2.708206200e+02f,
                2.689589700e+02f,  2.670973200e+02f,  2.709535500e+02f,  2.690910600e+02f,  2.672285700e+02f,  2.710864800e+02f,  2.692231500e+02f,  2.673598200e+02f,
                2.712194100e+02f,  2.693552400e+02f,  2.674910700e+02f,  2.713523400e+02f,  2.694873300e+02f,  2.676223200e+02f,  2.714852700e+02f,  2.696194200e+02f,
                2.677535700e+02f,  2.718840600e+02f,  2.700156900e+02f,  2.681473200e+02f,  2.720169900e+02f,  2.701477800e+02f,  2.682785700e+02f,  2.721499200e+02f,
                2.702798700e+02f,  2.684098200e+02f,  2.722828500e+02f,  2.704119600e+02f,  2.685410700e+02f,  2.724157800e+02f,  2.705440500e+02f,  2.686723200e+02f,
                2.725487100e+02f,  2.706761400e+02f,  2.688035700e+02f,  2.726816400e+02f,  2.708082300e+02f,  2.689348200e+02f,  2.728145700e+02f,  2.709403200e+02f,
                2.690660700e+02f,  2.729475000e+02f,  2.710724100e+02f,  2.691973200e+02f,  2.730804300e+02f,  2.712045000e+02f,  2.693285700e+02f,  2.732133600e+02f,
                2.713365900e+02f,  2.694598200e+02f,  2.736121500e+02f,  2.717328600e+02f,  2.698535700e+02f,  2.737450800e+02f,  2.718649500e+02f,  2.699848200e+02f,
                2.738780100e+02f,  2.719970400e+02f,  2.701160700e+02f,  2.740109400e+02f,  2.721291300e+02f,  2.702473200e+02f,  2.741438700e+02f,  2.722612200e+02f,
                2.703785700e+02f,  2.742768000e+02f,  2.723933100e+02f,  2.705098200e+02f,  2.744097300e+02f,  2.725254000e+02f,  2.706410700e+02f,  2.745426600e+02f,
                2.726574900e+02f,  2.707723200e+02f,  2.746755900e+02f,  2.727895800e+02f,  2.709035700e+02f,  2.748085200e+02f,  2.729216700e+02f,  2.710348200e+02f,
                2.749414500e+02f,  2.730537600e+02f,  2.711660700e+02f,  2.753402400e+02f,  2.734500300e+02f,  2.715598200e+02f,  2.754731700e+02f,  2.735821200e+02f,
                2.716910700e+02f,  2.756061000e+02f,  2.737142100e+02f,  2.718223200e+02f,  2.757390300e+02f,  2.738463000e+02f,  2.719535700e+02f,  2.758719600e+02f,
                2.739783900e+02f,  2.720848200e+02f,  2.760048900e+02f,  2.741104800e+02f,  2.722160700e+02f,  2.761378200e+02f,  2.742425700e+02f,  2.723473200e+02f,
                2.762707500e+02f,  2.743746600e+02f,  2.724785700e+02f,  2.764036800e+02f,  2.745067500e+02f,  2.726098200e+02f,  2.765366100e+02f,  2.746388400e+02f,
                2.727410700e+02f,  2.766695400e+02f,  2.747709300e+02f,  2.728723200e+02f,  2.770683300e+02f,  2.751672000e+02f,  2.732660700e+02f,  2.772012600e+02f,
                2.752992900e+02f,  2.733973200e+02f,  2.773341900e+02f,  2.754313800e+02f,  2.735285700e+02f,  2.774671200e+02f,  2.755634700e+02f,  2.736598200e+02f,
                2.776000500e+02f,  2.756955600e+02f,  2.737910700e+02f,  2.777329800e+02f,  2.758276500e+02f,  2.739223200e+02f,  2.778659100e+02f,  2.759597400e+02f,
                2.740535700e+02f,  2.779988400e+02f,  2.760918300e+02f,  2.741848200e+02f,  2.781317700e+02f,  2.762239200e+02f,  2.743160700e+02f,  2.782647000e+02f,
                2.763560100e+02f,  2.744473200e+02f,  2.783976300e+02f,  2.764881000e+02f,  2.745785700e+02f,  2.787964200e+02f,  2.768843700e+02f,  2.749723200e+02f,
                2.789293500e+02f,  2.770164600e+02f,  2.751035700e+02f,  2.790622800e+02f,  2.771485500e+02f,  2.752348200e+02f,  2.791952100e+02f,  2.772806400e+02f,
                2.753660700e+02f,  2.793281400e+02f,  2.774127300e+02f,  2.754973200e+02f,  2.794610700e+02f,  2.775448200e+02f,  2.756285700e+02f,  2.795940000e+02f,
                2.776769100e+02f,  2.757598200e+02f,  2.797269300e+02f,  2.778090000e+02f,  2.758910700e+02f,  2.798598600e+02f,  2.779410900e+02f,  2.760223200e+02f,
                2.799927900e+02f,  2.780731800e+02f,  2.761535700e+02f,  2.801257200e+02f,  2.782052700e+02f,  2.762848200e+02f,  2.805245100e+02f,  2.786015400e+02f,
                2.766785700e+02f,  2.806574400e+02f,  2.787336300e+02f,  2.768098200e+02f,  2.807903700e+02f,  2.788657200e+02f,  2.769410700e+02f,  2.809233000e+02f,
                2.789978100e+02f,  2.770723200e+02f,  2.810562300e+02f,  2.791299000e+02f,  2.772035700e+02f,  2.811891600e+02f,  2.792619900e+02f,  2.773348200e+02f,
                2.813220900e+02f,  2.793940800e+02f,  2.774660700e+02f,  2.814550200e+02f,  2.795261700e+02f,  2.775973200e+02f,  2.815879500e+02f,  2.796582600e+02f,
                2.777285700e+02f,  2.817208800e+02f,  2.797903500e+02f,  2.778598200e+02f,  2.818538100e+02f,  2.799224400e+02f,  2.779910700e+02f,  2.822526000e+02f,
                2.803187100e+02f,  2.783848200e+02f,  2.823855300e+02f,  2.804508000e+02f,  2.785160700e+02f,  2.825184600e+02f,  2.805828900e+02f,  2.786473200e+02f,
                2.826513900e+02f,  2.807149800e+02f,  2.787785700e+02f,  2.827843200e+02f,  2.808470700e+02f,  2.789098200e+02f,  2.829172500e+02f,  2.809791600e+02f,
                2.790410700e+02f,  2.830501800e+02f,  2.811112500e+02f,  2.791723200e+02f,  2.831831100e+02f,  2.812433400e+02f,  2.793035700e+02f,  2.833160400e+02f,
                2.813754300e+02f,  2.794348200e+02f,  2.834489700e+02f,  2.815075200e+02f,  2.795660700e+02f,  2.835819000e+02f,  2.816396100e+02f,  2.796973200e+02f,
                2.908930500e+02f,  2.889045600e+02f,  2.869160700e+02f,  2.910259800e+02f,  2.890366500e+02f,  2.870473200e+02f,  2.911589100e+02f,  2.891687400e+02f,
                2.871785700e+02f,  2.912918400e+02f,  2.893008300e+02f,  2.873098200e+02f,  2.914247700e+02f,  2.894329200e+02f,  2.874410700e+02f,  2.915577000e+02f,
                2.895650100e+02f,  2.875723200e+02f,  2.916906300e+02f,  2.896971000e+02f,  2.877035700e+02f,  2.918235600e+02f,  2.898291900e+02f,  2.878348200e+02f,
                2.919564900e+02f,  2.899612800e+02f,  2.879660700e+02f,  2.920894200e+02f,  2.900933700e+02f,  2.880973200e+02f,  2.922223500e+02f,  2.902254600e+02f,
                2.882285700e+02f,  2.926211400e+02f,  2.906217300e+02f,  2.886223200e+02f,  2.927540700e+02f,  2.907538200e+02f,  2.887535700e+02f,  2.928870000e+02f,
                2.908859100e+02f,  2.888848200e+02f,  2.930199300e+02f,  2.910180000e+02f,  2.890160700e+02f,  2.931528600e+02f,  2.911500900e+02f,  2.891473200e+02f,
                2.932857900e+02f,  2.912821800e+02f,  2.892785700e+02f,  2.934187200e+02f,  2.914142700e+02f,  2.894098200e+02f,  2.935516500e+02f,  2.915463600e+02f,
                2.895410700e+02f,  2.936845800e+02f,  2.916784500e+02f,  2.896723200e+02f,  2.938175100e+02f,  2.918105400e+02f,  2.898035700e+02f,  2.939504400e+02f,
                2.919426300e+02f,  2.899348200e+02f,  2.943492300e+02f,  2.923389000e+02f,  2.903285700e+02f,  2.944821600e+02f,  2.924709900e+02f,  2.904598200e+02f,
                2.946150900e+02f,  2.926030800e+02f,  2.905910700e+02f,  2.947480200e+02f,  2.927351700e+02f,  2.907223200e+02f,  2.948809500e+02f,  2.928672600e+02f,
                2.908535700e+02f,  2.950138800e+02f,  2.929993500e+02f,  2.909848200e+02f,  2.951468100e+02f,  2.931314400e+02f,  2.911160700e+02f,  2.952797400e+02f,
                2.932635300e+02f,  2.912473200e+02f,  2.954126700e+02f,  2.933956200e+02f,  2.913785700e+02f,  2.955456000e+02f,  2.935277100e+02f,  2.915098200e+02f,
                2.956785300e+02f,  2.936598000e+02f,  2.916410700e+02f,  2.960773200e+02f,  2.940560700e+02f,  2.920348200e+02f,  2.962102500e+02f,  2.941881600e+02f,
                2.921660700e+02f,  2.963431800e+02f,  2.943202500e+02f,  2.922973200e+02f,  2.964761100e+02f,  2.944523400e+02f,  2.924285700e+02f,  2.966090400e+02f,
                2.945844300e+02f,  2.925598200e+02f,  2.967419700e+02f,  2.947165200e+02f,  2.926910700e+02f,  2.968749000e+02f,  2.948486100e+02f,  2.928223200e+02f,
                2.970078300e+02f,  2.949807000e+02f,  2.929535700e+02f,  2.971407600e+02f,  2.951127900e+02f,  2.930848200e+02f,  2.972736900e+02f,  2.952448800e+02f,
                2.932160700e+02f,  2.974066200e+02f,  2.953769700e+02f,  2.933473200e+02f,  2.978054100e+02f,  2.957732400e+02f,  2.937410700e+02f,  2.979383400e+02f,
                2.959053300e+02f,  2.938723200e+02f,  2.980712700e+02f,  2.960374200e+02f,  2.940035700e+02f,  2.982042000e+02f,  2.961695100e+02f,  2.941348200e+02f,
                2.983371300e+02f,  2.963016000e+02f,  2.942660700e+02f,  2.984700600e+02f,  2.964336900e+02f,  2.943973200e+02f,  2.986029900e+02f,  2.965657800e+02f,
                2.945285700e+02f,  2.987359200e+02f,  2.966978700e+02f,  2.946598200e+02f,  2.988688500e+02f,  2.968299600e+02f,  2.947910700e+02f,  2.990017800e+02f,
                2.969620500e+02f,  2.949223200e+02f,  2.991347100e+02f,  2.970941400e+02f,  2.950535700e+02f,  2.995335000e+02f,  2.974904100e+02f,  2.954473200e+02f,
                2.996664300e+02f,  2.976225000e+02f,  2.955785700e+02f,  2.997993600e+02f,  2.977545900e+02f,  2.957098200e+02f,  2.999322900e+02f,  2.978866800e+02f,
                2.958410700e+02f,  3.000652200e+02f,  2.980187700e+02f,  2.959723200e+02f,  3.001981500e+02f,  2.981508600e+02f,  2.961035700e+02f,  3.003310800e+02f,
                2.982829500e+02f,  2.962348200e+02f,  3.004640100e+02f,  2.984150400e+02f,  2.963660700e+02f,  3.005969400e+02f,  2.985471300e+02f,  2.964973200e+02f,
                3.007298700e+02f,  2.986792200e+02f,  2.966285700e+02f,  3.008628000e+02f,  2.988113100e+02f,  2.967598200e+02f,  3.012615900e+02f,  2.992075800e+02f,
                2.971535700e+02f,  3.013945200e+02f,  2.993396700e+02f,  2.972848200e+02f,  3.015274500e+02f,  2.994717600e+02f,  2.974160700e+02f,  3.016603800e+02f,
                2.996038500e+02f,  2.975473200e+02f,  3.017933100e+02f,  2.997359400e+02f,  2.976785700e+02f,  3.019262400e+02f,  2.998680300e+02f,  2.978098200e+02f,
                3.020591700e+02f,  3.000001200e+02f,  2.979410700e+02f,  3.021921000e+02f,  3.001322100e+02f,  2.980723200e+02f,  3.023250300e+02f,  3.002643000e+02f,
                2.982035700e+02f,  3.024579600e+02f,  3.003963900e+02f,  2.983348200e+02f,  3.025908900e+02f,  3.005284800e+02f,  2.984660700e+02f,  3.029896800e+02f,
                3.009247500e+02f,  2.988598200e+02f,  3.031226100e+02f,  3.010568400e+02f,  2.989910700e+02f,  3.032555400e+02f,  3.011889300e+02f,  2.991223200e+02f,
                3.033884700e+02f,  3.013210200e+02f,  2.992535700e+02f,  3.035214000e+02f,  3.014531100e+02f,  2.993848200e+02f,  3.036543300e+02f,  3.015852000e+02f,
                2.995160700e+02f,  3.037872600e+02f,  3.017172900e+02f,  2.996473200e+02f,  3.039201900e+02f,  3.018493800e+02f,  2.997785700e+02f,  3.040531200e+02f,
                3.019814700e+02f,  2.999098200e+02f,  3.041860500e+02f,  3.021135600e+02f,  3.000410700e+02f,  3.043189800e+02f,  3.022456500e+02f,  3.001723200e+02f,
                3.116301300e+02f,  3.095106000e+02f,  3.073910700e+02f,  3.117630600e+02f,  3.096426900e+02f,  3.075223200e+02f,  3.118959900e+02f,  3.097747800e+02f,
                3.076535700e+02f,  3.120289200e+02f,  3.099068700e+02f,  3.077848200e+02f,  3.121618500e+02f,  3.100389600e+02f,  3.079160700e+02f,  3.122947800e+02f,
                3.101710500e+02f,  3.080473200e+02f,  3.124277100e+02f,  3.103031400e+02f,  3.081785700e+02f,  3.125606400e+02f,  3.104352300e+02f,  3.083098200e+02f,
                3.126935700e+02f,  3.105673200e+02f,  3.084410700e+02f,  3.128265000e+02f,  3.106994100e+02f,  3.085723200e+02f,  3.129594300e+02f,  3.108315000e+02f,
                3.087035700e+02f,  3.133582200e+02f,  3.112277700e+02f,  3.090973200e+02f,  3.134911500e+02f,  3.113598600e+02f,  3.092285700e+02f,  3.136240800e+02f,
                3.114919500e+02f,  3.093598200e+02f,  3.137570100e+02f,  3.116240400e+02f,  3.094910700e+02f,  3.138899400e+02f,  3.117561300e+02f,  3.096223200e+02f,
                3.140228700e+02f,  3.118882200e+02f,  3.097535700e+02f,  3.141558000e+02f,  3.120203100e+02f,  3.098848200e+02f,  3.142887300e+02f,  3.121524000e+02f,
                3.100160700e+02f,  3.144216600e+02f,  3.122844900e+02f,  3.101473200e+02f,  3.145545900e+02f,  3.124165800e+02f,  3.102785700e+02f,  3.146875200e+02f,
                3.125486700e+02f,  3.104098200e+02f,  3.150863100e+02f,  3.129449400e+02f,  3.108035700e+02f,  3.152192400e+02f,  3.130770300e+02f,  3.109348200e+02f,
                3.153521700e+02f,  3.132091200e+02f,  3.110660700e+02f,  3.154851000e+02f,  3.133412100e+02f,  3.111973200e+02f,  3.156180300e+02f,  3.134733000e+02f,
                3.113285700e+02f,  3.157509600e+02f,  3.136053900e+02f,  3.114598200e+02f,  3.158838900e+02f,  3.137374800e+02f,  3.115910700e+02f,  3.160168200e+02f,
                3.138695700e+02f,  3.117223200e+02f,  3.161497500e+02f,  3.140016600e+02f,  3.118535700e+02f,  3.162826800e+02f,  3.141337500e+02f,  3.119848200e+02f,
                3.164156100e+02f,  3.142658400e+02f,  3.121160700e+02f,  3.168144000e+02f,  3.146621100e+02f,  3.125098200e+02f,  3.169473300e+02f,  3.147942000e+02f,
                3.126410700e+02f,  3.170802600e+02f,  3.149262900e+02f,  3.127723200e+02f,  3.172131900e+02f,  3.150583800e+02f,  3.129035700e+02f,  3.173461200e+02f,
                3.151904700e+02f,  3.130348200e+02f,  3.174790500e+02f,  3.153225600e+02f,  3.131660700e+02f,  3.176119800e+02f,  3.154546500e+02f,  3.132973200e+02f,
                3.177449100e+02f,  3.155867400e+02f,  3.134285700e+02f,  3.178778400e+02f,  3.157188300e+02f,  3.135598200e+02f,  3.180107700e+02f,  3.158509200e+02f,
                3.136910700e+02f,  3.181437000e+02f,  3.159830100e+02f,  3.138223200e+02f,  3.185424900e+02f,  3.163792800e+02f,  3.142160700e+02f,  3.186754200e+02f,
                3.165113700e+02f,  3.143473200e+02f,  3.188083500e+02f,  3.166434600e+02f,  3.144785700e+02f,  3.189412800e+02f,  3.167755500e+02f,  3.146098200e+02f,
                3.190742100e+02f,  3.169076400e+02f,  3.147410700e+02f,  3.192071400e+02f,  3.170397300e+02f,  3.148723200e+02f,  3.193400700e+02f,  3.171718200e+02f,
                3.150035700e+02f,  3.194730000e+02f,  3.173039100e+02f,  3.151348200e+02f,  3.196059300e+02f,  3.174360000e+02f,  3.152660700e+02f,  3.197388600e+02f,
                3.175680900e+02f,  3.153973200e+02f,  3.198717900e+02f,  3.177001800e+02f,  3.155285700e+02f,  3.202705800e+02f,  3.180964500e+02f,  3.159223200e+02f,
                3.204035100e+02f,  3.182285400e+02f,  3.160535700e+02f,  3.205364400e+02f,  3.183606300e+02f,  3.161848200e+02f,  3.206693700e+02f,  3.184927200e+02f,
                3.163160700e+02f,  3.208023000e+02f,  3.186248100e+02f,  3.164473200e+02f,  3.209352300e+02f,  3.187569000e+02f,  3.165785700e+02f,  3.210681600e+02f,
                3.188889900e+02f,  3.167098200e+02f,  3.212010900e+02f,  3.190210800e+02f,  3.168410700e+02f,  3.213340200e+02f,  3.191531700e+02f,  3.169723200e+02f,
                3.214669500e+02f,  3.192852600e+02f,  3.171035700e+02f,  3.215998800e+02f,  3.194173500e+02f,  3.172348200e+02f,  3.219986700e+02f,  3.198136200e+02f,
                3.176285700e+02f,  3.221316000e+02f,  3.199457100e+02f,  3.177598200e+02f,  3.222645300e+02f,  3.200778000e+02f,  3.178910700e+02f,  3.223974600e+02f,
                3.202098900e+02f,  3.180223200e+02f,  3.225303900e+02f,  3.203419800e+02f,  3.181535700e+02f,  3.226633200e+02f,  3.204740700e+02f,  3.182848200e+02f,
                3.227962500e+02f,  3.206061600e+02f,  3.184160700e+02f,  3.229291800e+02f,  3.207382500e+02f,  3.185473200e+02f,  3.230621100e+02f,  3.208703400e+02f,
                3.186785700e+02f,  3.231950400e+02f,  3.210024300e+02f,  3.188098200e+02f,  3.233279700e+02f,  3.211345200e+02f,  3.189410700e+02f,  3.237267600e+02f,
                3.215307900e+02f,  3.193348200e+02f,  3.238596900e+02f,  3.216628800e+02f,  3.194660700e+02f,  3.239926200e+02f,  3.217949700e+02f,  3.195973200e+02f,
                3.241255500e+02f,  3.219270600e+02f,  3.197285700e+02f,  3.242584800e+02f,  3.220591500e+02f,  3.198598200e+02f,  3.243914100e+02f,  3.221912400e+02f,
                3.199910700e+02f,  3.245243400e+02f,  3.223233300e+02f,  3.201223200e+02f,  3.246572700e+02f,  3.224554200e+02f,  3.202535700e+02f,  3.247902000e+02f,
                3.225875100e+02f,  3.203848200e+02f,  3.249231300e+02f,  3.227196000e+02f,  3.205160700e+02f,  3.250560600e+02f,  3.228516900e+02f,  3.206473200e+02f,
                3.323672100e+02f,  3.301166400e+02f,  3.278660700e+02f,  3.325001400e+02f,  3.302487300e+02f,  3.279973200e+02f,  3.326330700e+02f,  3.303808200e+02f,
                3.281285700e+02f,  3.327660000e+02f,  3.305129100e+02f,  3.282598200e+02f,  3.328989300e+02f,  3.306450000e+02f,  3.283910700e+02f,  3.330318600e+02f,
                3.307770900e+02f,  3.285223200e+02f,  3.331647900e+02f,  3.309091800e+02f,  3.286535700e+02f,  3.332977200e+02f,  3.310412700e+02f,  3.287848200e+02f,
                3.334306500e+02f,  3.311733600e+02f,  3.289160700e+02f,  3.335635800e+02f,  3.313054500e+02f,  3.290473200e+02f,  3.336965100e+02f,  3.314375400e+02f,
                3.291785700e+02f,  3.340953000e+02f,  3.318338100e+02f,  3.295723200e+02f,  3.342282300e+02f,  3.319659000e+02f,  3.297035700e+02f,  3.343611600e+02f,
                3.320979900e+02f,  3.298348200e+02f,  3.344940900e+02f,  3.322300800e+02f,  3.299660700e+02f,  3.346270200e+02f,  3.323621700e+02f,  3.300973200e+02f,
                3.347599500e+02f,  3.324942600e+02f,  3.302285700e+02f,  3.348928800e+02f,  3.326263500e+02f,  3.303598200e+02f,  3.350258100e+02f,  3.327584400e+02f,
                3.304910700e+02f,  3.351587400e+02f,  3.328905300e+02f,  3.306223200e+02f,  3.352916700e+02f,  3.330226200e+02f,  3.307535700e+02f,  3.354246000e+02f,
                3.331547100e+02f,  3.308848200e+02f,  3.358233900e+02f,  3.335509800e+02f,  3.312785700e+02f,  3.359563200e+02f,  3.336830700e+02f,  3.314098200e+02f,
                3.360892500e+02f,  3.338151600e+02f,  3.315410700e+02f,  3.362221800e+02f,  3.339472500e+02f,  3.316723200e+02f,  3.363551100e+02f,  3.340793400e+02f,
                3.318035700e+02f,  3.364880400e+02f,  3.342114300e+02f,  3.319348200e+02f,  3.366209700e+02f,  3.343435200e+02f,  3.320660700e+02f,  3.367539000e+02f,
                3.344756100e+02f,  3.321973200e+02f,  3.368868300e+02f,  3.346077000e+02f,  3.323285700e+02f,  3.370197600e+02f,  3.347397900e+02f,  3.324598200e+02f,
                3.371526900e+02f,  3.348718800e+02f,  3.325910700e+02f,  3.375514800e+02f,  3.352681500e+02f,  3.329848200e+02f,  3.376844100e+02f,  3.354002400e+02f,
                3.331160700e+02f,  3.378173400e+02f,  3.355323300e+02f,  3.332473200e+02f,  3.379502700e+02f,  3.356644200e+02f,  3.333785700e+02f,  3.380832000e+02f,
                3.357965100e+02f,  3.335098200e+02f,  3.382161300e+02f,  3.359286000e+02f,  3.336410700e+02f,  3.383490600e+02f,  3.360606900e+02f,  3.337723200e+02f,
                3.384819900e+02f,  3.361927800e+02f,  3.339035700e+02f,  3.386149200e+02f,  3.363248700e+02f,  3.340348200e+02f,  3.387478500e+02f,  3.364569600e+02f,
                3.341660700e+02f,  3.388807800e+02f,  3.365890500e+02f,  3.342973200e+02f,  3.392795700e+02f,  3.369853200e+02f,  3.346910700e+02f,  3.394125000e+02f,
                3.371174100e+02f,  3.348223200e+02f,  3.395454300e+02f,  3.372495000e+02f,  3.349535700e+02f,  3.396783600e+02f,  3.373815900e+02f,  3.350848200e+02f,
                3.398112900e+02f,  3.375136800e+02f,  3.352160700e+02f,  3.399442200e+02f,  3.376457700e+02f,  3.353473200e+02f,  3.400771500e+02f,  3.377778600e+02f,
                3.354785700e+02f,  3.402100800e+02f,  3.379099500e+02f,  3.356098200e+02f,  3.403430100e+02f,  3.380420400e+02f,  3.357410700e+02f,  3.404759400e+02f,
                3.381741300e+02f,  3.358723200e+02f,  3.406088700e+02f,  3.383062200e+02f,  3.360035700e+02f,  3.410076600e+02f,  3.387024900e+02f,  3.363973200e+02f,
                3.411405900e+02f,  3.388345800e+02f,  3.365285700e+02f,  3.412735200e+02f,  3.389666700e+02f,  3.366598200e+02f,  3.414064500e+02f,  3.390987600e+02f,
                3.367910700e+02f,  3.415393800e+02f,  3.392308500e+02f,  3.369223200e+02f,  3.416723100e+02f,  3.393629400e+02f,  3.370535700e+02f,  3.418052400e+02f,
                3.394950300e+02f,  3.371848200e+02f,  3.419381700e+02f,  3.396271200e+02f,  3.373160700e+02f,  3.420711000e+02f,  3.397592100e+02f,  3.374473200e+02f,
                3.422040300e+02f,  3.398913000e+02f,  3.375785700e+02f,  3.423369600e+02f,  3.400233900e+02f,  3.377098200e+02f,  3.427357500e+02f,  3.404196600e+02f,
                3.381035700e+02f,  3.428686800e+02f,  3.405517500e+02f,  3.382348200e+02f,  3.430016100e+02f,  3.406838400e+02f,  3.383660700e+02f,  3.431345400e+02f,
                3.408159300e+02f,  3.384973200e+02f,  3.432674700e+02f,  3.409480200e+02f,  3.386285700e+02f,  3.434004000e+02f,  3.410801100e+02f,  3.387598200e+02f,
                3.435333300e+02f,  3.412122000e+02f,  3.388910700e+02f,  3.436662600e+02f,  3.413442900e+02f,  3.390223200e+02f,  3.437991900e+02f,  3.414763800e+02f,
                3.391535700e+02f,  3.439321200e+02f,  3.416084700e+02f,  3.392848200e+02f,  3.440650500e+02f,  3.417405600e+02f,  3.394160700e+02f,  3.444638400e+02f,
                3.421368300e+02f,  3.398098200e+02f,  3.445967700e+02f,  3.422689200e+02f,  3.399410700e+02f,  3.447297000e+02f,  3.424010100e+02f,  3.400723200e+02f,
                3.448626300e+02f,  3.425331000e+02f,  3.402035700e+02f,  3.449955600e+02f,  3.426651900e+02f,  3.403348200e+02f,  3.451284900e+02f,  3.427972800e+02f,
                3.404660700e+02f,  3.452614200e+02f,  3.429293700e+02f,  3.405973200e+02f,  3.453943500e+02f,  3.430614600e+02f,  3.407285700e+02f,  3.455272800e+02f,
                3.431935500e+02f,  3.408598200e+02f,  3.456602100e+02f,  3.433256400e+02f,  3.409910700e+02f,  3.457931400e+02f,  3.434577300e+02f,  3.411223200e+02f,
                3.531042900e+02f,  3.507226800e+02f,  3.483410700e+02f,  3.532372200e+02f,  3.508547700e+02f,  3.484723200e+02f,  3.533701500e+02f,  3.509868600e+02f,
                3.486035700e+02f,  3.535030800e+02f,  3.511189500e+02f,  3.487348200e+02f,  3.536360100e+02f,  3.512510400e+02f,  3.488660700e+02f,  3.537689400e+02f,
                3.513831300e+02f,  3.489973200e+02f,  3.539018700e+02f,  3.515152200e+02f,  3.491285700e+02f,  3.540348000e+02f,  3.516473100e+02f,  3.492598200e+02f,
                3.541677300e+02f,  3.517794000e+02f,  3.493910700e+02f,  3.543006600e+02f,  3.519114900e+02f,  3.495223200e+02f,  3.544335900e+02f,  3.520435800e+02f,
                3.496535700e+02f,  3.548323800e+02f,  3.524398500e+02f,  3.500473200e+02f,  3.549653100e+02f,  3.525719400e+02f,  3.501785700e+02f,  3.550982400e+02f,
                3.527040300e+02f,  3.503098200e+02f,  3.552311700e+02f,  3.528361200e+02f,  3.504410700e+02f,  3.553641000e+02f,  3.529682100e+02f,  3.505723200e+02f,
                3.554970300e+02f,  3.531003000e+02f,  3.507035700e+02f,  3.556299600e+02f,  3.532323900e+02f,  3.508348200e+02f,  3.557628900e+02f,  3.533644800e+02f,
                3.509660700e+02f,  3.558958200e+02f,  3.534965700e+02f,  3.510973200e+02f,  3.560287500e+02f,  3.536286600e+02f,  3.512285700e+02f,  3.561616800e+02f,
                3.537607500e+02f,  3.513598200e+02f,  3.565604700e+02f,  3.541570200e+02f,  3.517535700e+02f,  3.566934000e+02f,  3.542891100e+02f,  3.518848200e+02f,
                3.568263300e+02f,  3.544212000e+02f,  3.520160700e+02f,  3.569592600e+02f,  3.545532900e+02f,  3.521473200e+02f,  3.570921900e+02f,  3.546853800e+02f,
                3.522785700e+02f,  3.572251200e+02f,  3.548174700e+02f,  3.524098200e+02f,  3.573580500e+02f,  3.549495600e+02f,  3.525410700e+02f,  3.574909800e+02f,
                3.550816500e+02f,  3.526723200e+02f,  3.576239100e+02f,  3.552137400e+02f,  3.528035700e+02f,  3.577568400e+02f,  3.553458300e+02f,  3.529348200e+02f,
                3.578897700e+02f,  3.554779200e+02f,  3.530660700e+02f,  3.582885600e+02f,  3.558741900e+02f,  3.534598200e+02f,  3.584214900e+02f,  3.560062800e+02f,
                3.535910700e+02f,  3.585544200e+02f,  3.561383700e+02f,  3.537223200e+02f,  3.586873500e+02f,  3.562704600e+02f,  3.538535700e+02f,  3.588202800e+02f,
                3.564025500e+02f,  3.539848200e+02f,  3.589532100e+02f,  3.565346400e+02f,  3.541160700e+02f,  3.590861400e+02f,  3.566667300e+02f,  3.542473200e+02f,
                3.592190700e+02f,  3.567988200e+02f,  3.543785700e+02f,  3.593520000e+02f,  3.569309100e+02f,  3.545098200e+02f,  3.594849300e+02f,  3.570630000e+02f,
                3.546410700e+02f,  3.596178600e+02f,  3.571950900e+02f,  3.547723200e+02f,  3.600166500e+02f,  3.575913600e+02f,  3.551660700e+02f,  3.601495800e+02f,
                3.577234500e+02f,  3.552973200e+02f,  3.602825100e+02f,  3.578555400e+02f,  3.554285700e+02f,  3.604154400e+02f,  3.579876300e+02f,  3.555598200e+02f,
                3.605483700e+02f,  3.581197200e+02f,  3.556910700e+02f,  3.606813000e+02f,  3.582518100e+02f,  3.558223200e+02f,  3.608142300e+02f,  3.583839000e+02f,
                3.559535700e+02f,  3.609471600e+02f,  3.585159900e+02f,  3.560848200e+02f,  3.610800900e+02f,  3.586480800e+02f,  3.562160700e+02f,  3.612130200e+02f,
                3.587801700e+02f,  3.563473200e+02f,  3.613459500e+02f,  3.589122600e+02f,  3.564785700e+02f,  3.617447400e+02f,  3.593085300e+02f,  3.568723200e+02f,
                3.618776700e+02f,  3.594406200e+02f,  3.570035700e+02f,  3.620106000e+02f,  3.595727100e+02f,  3.571348200e+02f,  3.621435300e+02f,  3.597048000e+02f,
                3.572660700e+02f,  3.622764600e+02f,  3.598368900e+02f,  3.573973200e+02f,  3.624093900e+02f,  3.599689800e+02f,  3.575285700e+02f,  3.625423200e+02f,
                3.601010700e+02f,  3.576598200e+02f,  3.626752500e+02f,  3.602331600e+02f,  3.577910700e+02f,  3.628081800e+02f,  3.603652500e+02f,  3.579223200e+02f,
                3.629411100e+02f,  3.604973400e+02f,  3.580535700e+02f,  3.630740400e+02f,  3.606294300e+02f,  3.581848200e+02f,  3.634728300e+02f,  3.610257000e+02f,
                3.585785700e+02f,  3.636057600e+02f,  3.611577900e+02f,  3.587098200e+02f,  3.637386900e+02f,  3.612898800e+02f,  3.588410700e+02f,  3.638716200e+02f,
                3.614219700e+02f,  3.589723200e+02f,  3.640045500e+02f,  3.615540600e+02f,  3.591035700e+02f,  3.641374800e+02f,  3.616861500e+02f,  3.592348200e+02f,
                3.642704100e+02f,  3.618182400e+02f,  3.593660700e+02f,  3.644033400e+02f,  3.619503300e+02f,  3.594973200e+02f,  3.645362700e+02f,  3.620824200e+02f,
                3.596285700e+02f,  3.646692000e+02f,  3.622145100e+02f,  3.597598200e+02f,  3.648021300e+02f,  3.623466000e+02f,  3.598910700e+02f,  3.652009200e+02f,
                3.627428700e+02f,  3.602848200e+02f,  3.653338500e+02f,  3.628749600e+02f,  3.604160700e+02f,  3.654667800e+02f,  3.630070500e+02f,  3.605473200e+02f,
                3.655997100e+02f,  3.631391400e+02f,  3.606785700e+02f,  3.657326400e+02f,  3.632712300e+02f,  3.608098200e+02f,  3.658655700e+02f,  3.634033200e+02f,
                3.609410700e+02f,  3.659985000e+02f,  3.635354100e+02f,  3.610723200e+02f,  3.661314300e+02f,  3.636675000e+02f,  3.612035700e+02f,  3.662643600e+02f,
                3.637995900e+02f,  3.613348200e+02f,  3.663972900e+02f,  3.639316800e+02f,  3.614660700e+02f,  3.665302200e+02f,  3.640637700e+02f,  3.615973200e+02f,
            };

            float[] y_actual = y.ToArray();

            AssertError.Tolerance(y_expect, y_actual, 1e-7f, 1e-5f, $"mismatch value {inchannels},{outchannels},{kwidth},{kheight},{kdepth},{inwidth},{inheight},{batch}");
        }
    }
}
