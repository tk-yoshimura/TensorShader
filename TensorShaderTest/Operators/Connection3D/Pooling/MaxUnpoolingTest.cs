using System;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using TensorShader.Operators.Connection3D;
using TensorShaderCudaBackend.API;

namespace TensorShaderTest.Operators.Connection3D {
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
                                foreach (int indepth in new int[] { stride, 5, 7, 8, 11 }) {
                                    int outwidth = inwidth / stride, outheight = inheight / stride, outdepth = indepth / stride;

                                    float[] xval = (new float[inwidth * inheight * indepth * channels * batch]).Select((_) => (float)rd.NextDouble()).ToArray();
                                    float[] gyval = (new float[outwidth * outheight * outdepth * channels * batch]).Select((_) => (float)rd.NextDouble()).ToArray();

                                    Map3D x = new Map3D(channels, inwidth, inheight, indepth, batch, xval);
                                    Map3D gy = new Map3D(channels, outwidth, outheight, outdepth, batch, gyval);

                                    Map3D gx = Reference(x, gy, stride);

                                    OverflowCheckedTensor x_tensor = new OverflowCheckedTensor(Shape.Map3D(channels, inwidth, inheight, indepth, batch), xval);
                                    OverflowCheckedTensor y_tensor = new OverflowCheckedTensor(Shape.Map3D(channels, outwidth, outheight, outdepth, batch));
                                    OverflowCheckedTensor gy_tensor = new OverflowCheckedTensor(Shape.Map3D(channels, outwidth, outheight, outdepth, batch), gyval);
                                    OverflowCheckedTensor gx_tensor = new OverflowCheckedTensor(Shape.Map3D(channels, inwidth, inheight, indepth, batch));

                                    MaxPooling ope_pool = new MaxPooling(inwidth, inheight, indepth, channels, stride, batch);
                                    ope_pool.Execute(x_tensor, y_tensor);

                                    MaxUnpooling ope_unpool = new MaxUnpooling(inwidth, inheight, indepth, channels, stride, batch);
                                    ope_unpool.Execute(gy_tensor, x_tensor, y_tensor, gx_tensor);

                                    float[] gx_expect = gx.ToArray();
                                    float[] gx_actual = gx_tensor.State;

                                    int gx_expect_nonzero = gx_expect.Count((v) => v != 0);
                                    int gx_actual_nonzero = gx_expect.Count((v) => v != 0);

                                    CollectionAssert.AreEqual(xval, x_tensor.State);
                                    CollectionAssert.AreEqual(gyval, gy_tensor.State);

                                    Assert.AreEqual(y_tensor.Length, gx_expect_nonzero);
                                    Assert.AreEqual(y_tensor.Length, gx_actual_nonzero);

                                    AssertError.Tolerance(gx_expect, gx_actual, 1e-7f, 1e-5f, ref max_err, $"mismatch value {channels},{stride},{inwidth},{inheight},{indepth},{batch}");

                                    Console.WriteLine($"pass: {channels},{stride},{inwidth},{inheight},{batch}");

                                }
                            }
                        }
                    }
                }
            }

            Console.WriteLine($"maxerr:{max_err}");
        }

        public static Map3D Reference(Map3D x, Map3D gy, int stride) {
            int channels = x.Channels, batch = x.Batch;
            int inw = x.Width, inh = x.Height, ind = x.Depth, outw = inw / stride, outh = inh / stride, outd = ind / stride;

            Map3D y = new Map3D(channels, outw, outh, outd, batch);
            Map3D gx = new Map3D(channels, inw, inh, ind, batch);

            for (int th = 0; th < batch; th++) {
                for (int ox, oy, oz = 0; oz < outd; oz++) {
                    for (oy = 0; oy < outh; oy++) {
                        for (ox = 0; ox < outw; ox++) {
                            for (int ch = 0; ch < channels; ch++) {
                                double max = 0;
                                for (int kx, ky, kz = 0; kz < stride; kz++) {
                                    for (ky = 0; ky < stride; ky++) {
                                        for (kx = 0; kx < stride; kx++) {
                                            max = Math.Max(max, x[ch, ox * stride + kx, oy * stride + ky, oz * stride + kz, th]);
                                        }
                                    }
                                }

                                y[ch, ox, oy, oz, th] = max;
                            }
                        }
                    }
                }

            }

            for (int th = 0; th < batch; th++) {
                for (int ix, iy, iz = 0; iz < ind; iz++) {
                    for (iy = 0; iy < inh; iy++) {
                        for (ix = 0; ix < inw; ix++) {
                            int ox = ix / stride, oy = iy / stride, oz = iz / stride;

                            if (ox < outw && oy < outh && oz < outd) {
                                for (int ch = 0; ch < channels; ch++) {
                                    gx[ch, ix, iy, iz, th] = (y[ch, ox, oy, oz, th] <= x[ch, ix, iy, iz, th]) ? gy[ch, ox, oy, oz, th] : 0;
                                }
                            }
                            else {
                                for (int ch = 0; ch < channels; ch++) {
                                    gx[ch, ix, iy, iz, th] = 0;
                                }
                            }

                        }
                    }
                }

            }

            return gx;
        }

        [TestMethod]
        public void SpeedTest() {
            int inwidth = 64, inheight = 64, indepth = 64, channels = 32, stride = 2, batch = 4;
            int outwidth = inwidth / stride, outheight = inheight / stride, outdepth = inheight / stride;

            OverflowCheckedTensor x_tensor = new OverflowCheckedTensor(Shape.Map3D(channels, inwidth, inheight, indepth, batch));
            OverflowCheckedTensor y_tensor = new OverflowCheckedTensor(Shape.Map3D(channels, outwidth, outheight, outdepth, batch));
            OverflowCheckedTensor gy_tensor = new OverflowCheckedTensor(Shape.Map3D(channels, outwidth, outheight, outdepth, batch));
            OverflowCheckedTensor gx_tensor = new OverflowCheckedTensor(Shape.Map3D(channels, inwidth, inheight, indepth, batch));

            MaxUnpooling ope = new MaxUnpooling(inwidth, inheight, indepth, channels, stride, batch);

            Cuda.Profiler.Initialize("../../../profiler.nvsetting", "../../nvprofiles/maxunpool_3d.nvvp");
            Cuda.Profiler.Start();

            ope.Execute(gy_tensor, x_tensor, y_tensor, gx_tensor);

            Cuda.Profiler.Stop();
        }
    }
}
