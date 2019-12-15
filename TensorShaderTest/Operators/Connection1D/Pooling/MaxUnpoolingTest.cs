using System;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using TensorShader.Operators.Connection1D;
using TensorShaderCudaBackend.API;

namespace TensorShaderTest.Operators.Connection1D {
    [TestClass]
    public class MaxUnpoolingTest {
        [TestMethod]
        public void ExecuteTest() {
            float max_err = 0;

            Random rd = new Random(1234);

            foreach (int batch in new int[] { 1, 2, 3 }) {
                foreach (int channels in new int[] { 1, 2, 3, 5 }) {
                    foreach (int stride in new int[] { 2, 3, 4 }) {
                        foreach (int inwidth in new int[] { stride, 5, 7, 8, 11 }) {
                            int outwidth = inwidth / stride;

                            float[] xval = (new float[inwidth * channels * batch]).Select((_) => (float)rd.NextDouble()).ToArray();
                            float[] gyval = (new float[outwidth * channels * batch]).Select((_) => (float)rd.NextDouble()).ToArray();

                            Map1D x = new Map1D(channels, inwidth, batch, xval);
                            Map1D gy = new Map1D(channels, outwidth, batch, gyval);

                            Map1D gx = Reference(x, gy, stride);

                            OverflowCheckedTensor x_tensor = new OverflowCheckedTensor(Shape.Map1D(channels, inwidth, batch), xval);
                            OverflowCheckedTensor y_tensor = new OverflowCheckedTensor(Shape.Map1D(channels, outwidth, batch));
                            OverflowCheckedTensor gy_tensor = new OverflowCheckedTensor(Shape.Map1D(channels, outwidth, batch), gyval);
                            OverflowCheckedTensor gx_tensor = new OverflowCheckedTensor(Shape.Map1D(channels, inwidth, batch));

                            MaxPooling ope_pool = new MaxPooling(inwidth, channels, stride, batch);
                            ope_pool.Execute(x_tensor, y_tensor);

                            MaxUnpooling ope_unpool = new MaxUnpooling(inwidth, channels, stride, batch);
                            ope_unpool.Execute(gy_tensor, x_tensor, y_tensor, gx_tensor);

                            float[] gx_expect = gx.ToArray();
                            float[] gx_actual = gx_tensor.State;

                            int gx_expect_nonzero = gx_expect.Count((v) => v != 0);
                            int gx_actual_nonzero = gx_expect.Count((v) => v != 0);

                            CollectionAssert.AreEqual(xval, x_tensor.State);
                            CollectionAssert.AreEqual(gyval, gy_tensor.State);

                            Assert.AreEqual(y_tensor.Length, gx_expect_nonzero);
                            Assert.AreEqual(y_tensor.Length, gx_actual_nonzero);

                            AssertError.Tolerance(gx_expect, gx_actual, 1e-7f, 1e-5f, $"mismatch value {channels},{stride},{inwidth},{batch}");

                            Console.WriteLine($"pass: {channels},{stride},{inwidth},{batch}");

                        }
                    }
                }
            }

            Console.WriteLine($"maxerr:{max_err}");
        }

        public static Map1D Reference(Map1D x, Map1D gy, int stride) {
            int channels = x.Channels, batch = x.Batch;
            int inw = x.Width, outw = inw / stride;

            Map1D y = new Map1D(channels, outw, batch);
            Map1D gx = new Map1D(channels, inw, batch);

            for (int th = 0; th < batch; th++) {
                for (int ox = 0; ox < outw; ox++) {
                    for (int ch = 0; ch < channels; ch++) {
                        double max = 0;
                        for (int kx = 0; kx < stride; kx++) {
                            max = Math.Max(max, x[ch, ox * stride + kx, th]);
                        }

                        y[ch, ox, th] = max;
                    }
                }

            }

            for (int th = 0; th < batch; th++) {
                for (int ix = 0; ix < inw; ix++) {
                    int ox = ix / stride;

                    if (ox < outw) {
                        for (int ch = 0; ch < channels; ch++) {
                            gx[ch, ix, th] = (y[ch, ox, th] <= x[ch, ix, th]) ? gy[ch, ox, th] : 0;
                        }
                    }
                    else {
                        for (int ch = 0; ch < channels; ch++) {
                            gx[ch, ix, th] = 0;
                        }
                    }
                }

            }

            return gx;
        }

        [TestMethod]
        public void SpeedTest() {
            int inwidth = 2048, channels = 1024, stride = 2, batch = 4;
            int outwidth = inwidth / stride;

            OverflowCheckedTensor x_tensor = new OverflowCheckedTensor(Shape.Map1D(channels, inwidth, batch));
            OverflowCheckedTensor y_tensor = new OverflowCheckedTensor(Shape.Map1D(channels, outwidth, batch));
            OverflowCheckedTensor gy_tensor = new OverflowCheckedTensor(Shape.Map1D(channels, outwidth, batch));
            OverflowCheckedTensor gx_tensor = new OverflowCheckedTensor(Shape.Map1D(channels, inwidth, batch));

            MaxUnpooling ope = new MaxUnpooling(inwidth, channels, stride, batch);

            Cuda.Profiler.Initialize("../../../profiler.nvsetting", "../../nvprofiles/maxunpool1d.nvvp");
            Cuda.Profiler.Start();

            ope.Execute(gy_tensor, x_tensor, y_tensor, gx_tensor);

            Cuda.Profiler.Stop();
        }
    }
}
