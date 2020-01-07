using System;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using TensorShader.Operators.Connection2D;
using TensorShaderCudaBackend.API;

namespace TensorShaderTest.Operators.Connection2D {
    [TestClass]
    public class PointwiseKernelProductTest {
        [TestMethod]
        public void ExecuteTest() {
            float max_err = 0;

            foreach (int batch in new int[] { 1, 2 }) {
                foreach (int inchannels in new int[] { 1, 2, 3, 4, 5, 10, 15, 20 }) {
                    foreach (int outchannels in new int[] { 7, 13 }) {
                        foreach (int height in new int[] { 8, 9, 19, 23 }) {
                            foreach (int width in new int[] { 8, 9, 13, 17 }) {
                                float[] xval = (new float[width * height * inchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
                                float[] gyval = (new float[width * height * outchannels * batch]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

                                Map2D x = new Map2D(inchannels, width, height, batch, xval);
                                Map2D gy = new Map2D(outchannels, width, height, batch, gyval);

                                Filter2D gw = Reference(x, gy);

                                OverflowCheckedTensor x_tensor = new OverflowCheckedTensor(Shape.Map2D(inchannels, width, height, batch), xval);
                                OverflowCheckedTensor gy_tensor = new OverflowCheckedTensor(Shape.Map2D(outchannels, width, height, batch), gyval);

                                OverflowCheckedTensor gw_tensor = new OverflowCheckedTensor(Shape.Kernel0D(inchannels, outchannels));

                                PointwiseKernelProduct ope = new PointwiseKernelProduct(width, height, inchannels, outchannels, batch);

                                ope.Execute(x_tensor, gy_tensor, gw_tensor);

                                float[] gw_expect = gw.ToArray();
                                float[] gw_actual = gw_tensor.State;

                                CollectionAssert.AreEqual(xval, x_tensor.State);
                                CollectionAssert.AreEqual(gyval, gy_tensor.State);

                                AssertError.Tolerance(gw_expect, gw_actual, 1e-7f, 1e-5f, ref max_err, $"mismatch value {inchannels},{outchannels},{width},{height},{batch}");

                                Console.WriteLine($"pass: {inchannels},{outchannels},{width},{height},{batch}");

                            }
                        }
                    }
                }
            }

            Console.WriteLine($"maxerr:{max_err}");
        }

        [TestMethod]
        public void LargeMapTest() {
            float max_err = 0;

            Random random = new Random(1234);

            int batch = 3;
            int inchannels = 49, outchannels = 50;
            int width = 128, height = 196;

            float[] xval = (new float[width * height * inchannels * batch]).Select((_, idx) => (float)random.NextDouble() * 1e-2f).ToArray();
            float[] gyval = (new float[width * height * outchannels * batch]).Select((_, idx) => (float)random.NextDouble() * 1e-2f).ToArray();

            Map2D x = new Map2D(inchannels, width, height, batch, xval);
            Map2D gy = new Map2D(outchannels, width, height, batch, gyval);

            Filter2D gw = Reference(x, gy);

            OverflowCheckedTensor x_tensor = new OverflowCheckedTensor(Shape.Map2D(inchannels, width, height, batch), xval);
            OverflowCheckedTensor gy_tensor = new OverflowCheckedTensor(Shape.Map2D(outchannels, width, height, batch), gyval);

            OverflowCheckedTensor gw_tensor = new OverflowCheckedTensor(Shape.Kernel0D(inchannels, outchannels));

            PointwiseKernelProduct ope = new PointwiseKernelProduct(width, height, inchannels, outchannels, batch);

            ope.Execute(x_tensor, gy_tensor, gw_tensor);

            float[] gw_expect = gw.ToArray();
            float[] gw_actual = gw_tensor.State;

            CollectionAssert.AreEqual(xval, x_tensor.State);
            CollectionAssert.AreEqual(gyval, gy_tensor.State);

            AssertError.Tolerance(gw_expect, gw_actual, 1e-7f, 1e-5f, ref max_err, $"mismatch value {inchannels},{outchannels},{width},{height},{batch}");

            Console.WriteLine($"pass: {inchannels},{outchannels},{width},{height},{batch}");

            Console.WriteLine($"maxerr:{max_err}");
        }

        [TestMethod]
        public void SpeedTest() {
            int inwidth = 512, inheight = 512, inchannels = 32, outchannels = 63;

            OverflowCheckedTensor x_tensor = new OverflowCheckedTensor(Shape.Map2D(inchannels, inwidth, inheight));
            OverflowCheckedTensor gy_tensor = new OverflowCheckedTensor(Shape.Map2D(outchannels, inwidth, inheight));

            OverflowCheckedTensor gw_tensor = new OverflowCheckedTensor(Shape.Kernel0D(inchannels, outchannels));

            PointwiseKernelProduct ope = new PointwiseKernelProduct(inwidth, inheight, inchannels, outchannels);

            Cuda.Profiler.Initialize("../../../profiler.nvsetting", "../../nvprofiles/ptwise_convolution_2d.nvvp");
            Cuda.Profiler.Start();

            ope.Execute(x_tensor, gy_tensor, gw_tensor);

            Cuda.Profiler.Stop();
        }

        public static Filter2D Reference(Map2D x, Map2D gy) {
            int inchannels = x.Channels, outchannels = gy.Channels, batch = x.Batch;
            int inw = x.Width, inh = x.Height;

            Filter2D w = new Filter2D(inchannels, outchannels, 1, 1);

            for (int th = 0; th < batch; th++) {
                for (int inch, outch = 0; outch < outchannels; outch++) {
                    for (inch = 0; inch < inchannels; inch++) {
                        double sum = 0;

                        for (int ix, iy = 0; iy < inh; iy++) {
                            for (ix = 0; ix < inw; ix++) {
                                sum += x[inch, ix, iy, th] * gy[outch, ix, iy, th];
                            }
                        }

                        w[inch, outch, 0, 0] += sum;
                    }
                }
            }

            return w;
        }

        [TestMethod]
        public void ReferenceTest() {
            int inchannels = 7, outchannels = 11, inwidth = 13, inheight = 17;

            float[] xval = (new float[inwidth * inheight * inchannels]).Select((_, idx) => idx * 1e-3f).ToArray();
            float[] gyval = (new float[inwidth * inheight * outchannels]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

            Map2D x = new Map2D(inchannels, inwidth, inheight, 1, xval);
            Map2D gy = new Map2D(outchannels, inwidth, inheight, 1, gyval);

            Filter2D gw = Reference(x, gy);

            float[] gw_expect = {
                1.383482361e+02f,  1.386178131e+02f,  1.388874359e+02f,  1.391571198e+02f,  1.394267426e+02f,  1.396963806e+02f,  1.399660034e+02f,
                1.381780396e+02f,  1.384474487e+02f,  1.387168427e+02f,  1.389862366e+02f,  1.392556610e+02f,  1.395250397e+02f,  1.397944641e+02f,
                1.380079041e+02f,  1.382770691e+02f,  1.385462189e+02f,  1.388154602e+02f,  1.390845795e+02f,  1.393537750e+02f,  1.396229858e+02f,
                1.378377228e+02f,  1.381066284e+02f,  1.383756409e+02f,  1.386446075e+02f,  1.389135284e+02f,  1.391824799e+02f,  1.394515228e+02f,
                1.376675415e+02f,  1.379362793e+02f,  1.382050018e+02f,  1.384737854e+02f,  1.387424622e+02f,  1.390112457e+02f,  1.392799835e+02f,
                1.374974060e+02f,  1.377659149e+02f,  1.380344086e+02f,  1.383029480e+02f,  1.385714569e+02f,  1.388399658e+02f,  1.391084747e+02f,
                1.373271637e+02f,  1.375955505e+02f,  1.378637238e+02f,  1.381321106e+02f,  1.384004211e+02f,  1.386686859e+02f,  1.389369659e+02f,
                1.371570129e+02f,  1.374250946e+02f,  1.376931763e+02f,  1.379611816e+02f,  1.382293243e+02f,  1.384973450e+02f,  1.387654724e+02f,
                1.369868317e+02f,  1.372547607e+02f,  1.375225983e+02f,  1.377903900e+02f,  1.380581970e+02f,  1.383261108e+02f,  1.385939789e+02f,
                1.368167267e+02f,  1.370843048e+02f,  1.373518982e+02f,  1.376195526e+02f,  1.378872375e+02f,  1.381548462e+02f,  1.384224548e+02f,
                1.366464996e+02f,  1.369139099e+02f,  1.371813660e+02f,  1.374488068e+02f,  1.377162170e+02f,  1.379835205e+02f,  1.382509308e+02f,
            };

            float[] gw_actual = gw.ToArray();

            AssertError.Tolerance(gw_expect, gw_actual, 1e-7f, 1e-5f, $"mismatch value {inchannels},{outchannels},{inwidth},{inheight}");
        }
    }
}
