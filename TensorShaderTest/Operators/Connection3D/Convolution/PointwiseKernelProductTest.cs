using System;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using TensorShader.Operators.Connection3D;
using TensorShaderCudaBackend.API;

namespace TensorShaderTest.Operators.Connection3D {
    [TestClass]
    public class PointwiseKernelProductTest {
        [TestMethod]
        public void ExecuteTest() {
            float max_err = 0;

            foreach (int batch in new int[] { 1, 2 }) {
                foreach (int inchannels in new int[] { 1, 2, 3, 4, 5, 10, 15, 20 }) {
                    foreach (int outchannels in new int[] { 7, 13 }) {
                        foreach ((int width, int height, int depth) in new (int, int, int)[] { (13, 13, 13), (17, 17, 17), (19, 19, 19), (17, 19, 13), (13, 17, 19), (19, 13, 17) }) {
                            float[] xval = (new float[width * height * depth * inchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
                            float[] gyval = (new float[width * height * depth * outchannels * batch]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

                            Map3D x = new Map3D(inchannels, width, height, depth, batch, xval);
                            Map3D gy = new Map3D(outchannels, width, height, depth, batch, gyval);

                            Filter3D gw = Reference(x, gy);

                            OverflowCheckedTensor x_tensor = new OverflowCheckedTensor(Shape.Map3D(inchannels, width, height, depth, batch), xval);
                            OverflowCheckedTensor gy_tensor = new OverflowCheckedTensor(Shape.Map3D(outchannels, width, height, depth, batch), gyval);

                            OverflowCheckedTensor gw_tensor = new OverflowCheckedTensor(Shape.Kernel0D(inchannels, outchannels));

                            PointwiseKernelProduct ope = new PointwiseKernelProduct(width, height, depth, inchannels, outchannels, batch);

                            ope.Execute(x_tensor, gy_tensor, gw_tensor);

                            float[] gw_expect = gw.ToArray();
                            float[] gw_actual = gw_tensor.State.Value;

                            CollectionAssert.AreEqual(xval, x_tensor.State.Value);
                            CollectionAssert.AreEqual(gyval, gy_tensor.State.Value);

                            AssertError.Tolerance(gw_expect, gw_actual, 1e-7f, 1e-5f, ref max_err, $"mismatch value {inchannels},{outchannels},{width},{height},{depth},{batch}");

                            Console.WriteLine($"pass: {inchannels},{outchannels},{width},{height},{depth},{batch}");
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
            int width = 128, height = 196, depth = 4;

            float[] xval = (new float[width * height * depth * inchannels * batch]).Select((_, idx) => (float)random.NextDouble() * 1e-2f).ToArray();
            float[] gyval = (new float[width * height * depth * outchannels * batch]).Select((_, idx) => (float)random.NextDouble() * 1e-2f).ToArray();

            Map3D x = new Map3D(inchannels, width, height, depth, batch, xval);
            Map3D gy = new Map3D(outchannels, width, height, depth, batch, gyval);

            Filter3D gw = Reference(x, gy);

            OverflowCheckedTensor x_tensor = new OverflowCheckedTensor(Shape.Map3D(inchannels, width, height, depth, batch), xval);
            OverflowCheckedTensor gy_tensor = new OverflowCheckedTensor(Shape.Map3D(outchannels, width, height, depth, batch), gyval);

            OverflowCheckedTensor gw_tensor = new OverflowCheckedTensor(Shape.Kernel0D(inchannels, outchannels));

            PointwiseKernelProduct ope = new PointwiseKernelProduct(width, height, depth, inchannels, outchannels, batch);

            ope.Execute(x_tensor, gy_tensor, gw_tensor);

            float[] gw_expect = gw.ToArray();
            float[] gw_actual = gw_tensor.State.Value;

            CollectionAssert.AreEqual(xval, x_tensor.State.Value);
            CollectionAssert.AreEqual(gyval, gy_tensor.State.Value);

            AssertError.Tolerance(gw_expect, gw_actual, 1e-7f, 1e-5f, ref max_err, $"mismatch value {inchannels},{outchannels},{width},{height},{depth},{batch}");

            Console.WriteLine($"pass: {inchannels},{outchannels},{width},{height},{depth},{batch}");

            Console.WriteLine($"maxerr:{max_err}");
        }

        [TestMethod]
        public void SpeedTest() {
            int width = 64, height = 64, depth = 64, inchannels = 32, outchannels = 32;

            OverflowCheckedTensor x_tensor = new OverflowCheckedTensor(Shape.Map3D(inchannels, width, height, depth));
            OverflowCheckedTensor gy_tensor = new OverflowCheckedTensor(Shape.Map3D(outchannels, width, height, depth));

            OverflowCheckedTensor gw_tensor = new OverflowCheckedTensor(Shape.Kernel0D(inchannels, outchannels));

            PointwiseKernelProduct ope = new PointwiseKernelProduct(width, height, depth, inchannels, outchannels);

            Cuda.Profiler.Initialize("../../../../profiler.nvsetting", "../../nvprofiles/ptwise_kernelproduct_3d.nvvp");
            Cuda.Profiler.Start();

            ope.Execute(x_tensor, gy_tensor, gw_tensor);

            Cuda.Profiler.Stop();
        }

        public static Filter3D Reference(Map3D x, Map3D gy) {
            int inchannels = x.Channels, outchannels = gy.Channels, batch = x.Batch;
            int inw = x.Width, inh = x.Height, ind = x.Depth;

            Filter3D w = new Filter3D(inchannels, outchannels, 1, 1, 1);

            for (int th = 0; th < batch; th++) {
                for (int ix, iy, iz = 0; iz < ind; iz++) {
                    for (iy = 0; iy < inh; iy++) {
                        for (ix = 0; ix < inw; ix++) {
                            for (int inch, outch = 0; outch < outchannels; outch++) {
                                for (inch = 0; inch < inchannels; inch++) {
                                    w[inch, outch, 0, 0, 0] += x[inch, ix, iy, iz, th] * gy[outch, ix, iy, iz, th];
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
            int inchannels = 2, outchannels = 3, width = 7, height = 6, depth = 5;

            float[] xval = (new float[width * height * depth * inchannels]).Select((_, idx) => idx * 1e-3f).ToArray();
            float[] gyval = (new float[width * height * depth * outchannels]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

            Map3D x = new Map3D(inchannels, width, height, depth, 1, xval);
            Map3D gy = new Map3D(outchannels, width, height, depth, 1, gyval);

            Filter3D gw = Reference(x, gy);

            float[] gw_expect = {
                9.216900e+00f,  9.283155e+00f,  9.173010e+00f,  9.239055e+00f,  9.129120e+00f,  9.194955e+00f,
            };

            float[] gw_actual = gw.ToArray();

            AssertError.Tolerance(gw_expect, gw_actual, 1e-7f, 1e-5f, $"mismatch value {inchannels},{outchannels},{width},{height},{depth}");
        }
    }
}
