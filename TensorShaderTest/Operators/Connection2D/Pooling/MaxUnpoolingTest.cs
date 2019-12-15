using System;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using TensorShader.Operators.Connection2D;
using TensorShaderCudaBackend.API;

namespace TensorShaderTest.Operators.Connection2D {
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
                            foreach (int inheight in new int[] { stride, 5, 7, 8, 11 }) {
                                int outwidth = inwidth / stride, outheight = inheight / stride;

                                float[] xval = (new float[inwidth * inheight * channels * batch]).Select((_) => (float)rd.NextDouble()).ToArray();
                                float[] gyval = (new float[outwidth * outheight * channels * batch]).Select((_) => (float)rd.NextDouble()).ToArray();

                                Map2D x = new Map2D(channels, inwidth, inheight, batch, xval);
                                Map2D gy = new Map2D(channels, outwidth, outheight, batch, gyval);

                                Map2D gx = Reference(x, gy, stride);

                                OverflowCheckedTensor x_tensor = new OverflowCheckedTensor(Shape.Map2D(channels, inwidth, inheight, batch), xval);
                                OverflowCheckedTensor y_tensor = new OverflowCheckedTensor(Shape.Map2D(channels, outwidth, outheight, batch));
                                OverflowCheckedTensor gy_tensor = new OverflowCheckedTensor(Shape.Map2D(channels, outwidth, outheight, batch), gyval);
                                OverflowCheckedTensor gx_tensor = new OverflowCheckedTensor(Shape.Map2D(channels, inwidth, inheight, batch));

                                MaxPooling ope_pool = new MaxPooling(inwidth, inheight, channels, stride, batch);
                                ope_pool.Execute(x_tensor, y_tensor);

                                MaxUnpooling ope_unpool = new MaxUnpooling(inwidth, inheight, channels, stride, batch);
                                ope_unpool.Execute(gy_tensor, x_tensor, y_tensor, gx_tensor);

                                float[] gx_expect = gx.ToArray();
                                float[] gx_actual = gx_tensor.State;

                                int gx_expect_nonzero = gx_expect.Count((v) => v != 0);
                                int gx_actual_nonzero = gx_expect.Count((v) => v != 0);

                                CollectionAssert.AreEqual(xval, x_tensor.State);
                                CollectionAssert.AreEqual(gyval, gy_tensor.State);

                                Assert.AreEqual(y_tensor.Length, gx_expect_nonzero);
                                Assert.AreEqual(y_tensor.Length, gx_actual_nonzero);

                                AssertError.Tolerance(gx_expect, gx_actual, 1e-7f, 1e-5f, ref max_err, $"mismatch value {channels},{stride},{inwidth},{inheight},{batch}");

                                Console.WriteLine($"pass: {channels},{stride},{inwidth},{inheight},{batch}");

                            }
                        }
                    }
                }
            }

            Console.WriteLine($"maxerr:{max_err}");
        }

        public static Map2D Reference(Map2D x, Map2D gy, int stride) {
            int channels = x.Channels, batch = x.Batch;
            int inw = x.Width, inh = x.Height, outw = inw / stride, outh = inh / stride;

            Map2D y = new Map2D(channels, outw, outh, batch);
            Map2D gx = new Map2D(channels, inw, inh, batch);

            for (int th = 0; th < batch; th++) {
                for (int ox, oy = 0; oy < outh; oy++) {
                    for (ox = 0; ox < outw; ox++) {
                        for (int ch = 0; ch < channels; ch++) {
                            double max = 0;
                            for (int kx, ky = 0; ky < stride; ky++) {
                                for (kx = 0; kx < stride; kx++) {
                                    max = Math.Max(max, x[ch, ox * stride + kx, oy * stride + ky, th]);
                                }
                            }

                            y[ch, ox, oy, th] = max;
                        }
                    }
                }

            }

            for (int th = 0; th < batch; th++) {
                for (int ix, iy = 0; iy < inh; iy++) {
                    for (ix = 0; ix < inw; ix++) {
                        int ox = ix / stride, oy = iy / stride;

                        if (ox < outw && oy < outh) {
                            for (int ch = 0; ch < channels; ch++) {
                                gx[ch, ix, iy, th] = (y[ch, ox, oy, th] <= x[ch, ix, iy, th]) ? gy[ch, ox, oy, th] : 0;
                            }
                        }
                        else {
                            for (int ch = 0; ch < channels; ch++) {
                                gx[ch, ix, iy, th] = 0;
                            }
                        }

                    }
                }

            }

            return gx;
        }

        [TestMethod]
        public void SpeedTest() {
            int inwidth = 512, inheight = 512, channels = 32, stride = 2, batch = 4;
            int outwidth = inwidth / stride, outheight = inheight / stride;

            OverflowCheckedTensor x_tensor = new OverflowCheckedTensor(Shape.Map2D(channels, inwidth, inheight, batch));
            OverflowCheckedTensor y_tensor = new OverflowCheckedTensor(Shape.Map2D(channels, outwidth, outheight, batch));
            OverflowCheckedTensor gy_tensor = new OverflowCheckedTensor(Shape.Map2D(channels, outwidth, outheight, batch));
            OverflowCheckedTensor gx_tensor = new OverflowCheckedTensor(Shape.Map2D(channels, inwidth, inheight, batch));

            MaxUnpooling ope = new MaxUnpooling(inwidth, inheight, channels, stride, batch);

            Cuda.Profiler.Initialize("../../../profiler.nvsetting", "../../nvprofiles/maxunpool2d.nvvp");
            Cuda.Profiler.Start();

            ope.Execute(gy_tensor, x_tensor, y_tensor, gx_tensor);

            Cuda.Profiler.Stop();
        }
    }
}
