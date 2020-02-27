using System;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using TensorShader.Operators.Connection3D;
using TensorShaderCudaBackend.API;

namespace TensorShaderTest.Operators.Connection3D {
    [TestClass]
    public class AveragePoolingTest {
        [TestMethod]
        public void ExecuteTest() {
            float max_err = 0;

            foreach (int batch in new int[] { 1, 2, 3 }) {
                foreach (int channels in new int[] { 1, 2, 3, 5 }) {
                    foreach (int stride in new int[] { 2, 3, 4 }) {
                        foreach (int indepth in new int[] { stride, 5, 7, 8, 11 }) {
                            foreach (int inheight in new int[] { stride, 5, 7, 8, 11 }) {
                                foreach (int inwidth in new int[] { stride, 5, 7, 8, 11 }) {
                                    int outwidth = inwidth / stride, outheight = inheight / stride, outdepth = indepth / stride;

                                    float[] xval = (new float[inwidth * inheight * indepth * channels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();

                                    Map3D x = new Map3D(channels, inwidth, inheight, indepth, batch, xval);

                                    Map3D y = Reference(x, stride);

                                    OverflowCheckedTensor x_tensor = new OverflowCheckedTensor(Shape.Map3D(channels, inwidth, inheight, indepth, batch), xval);
                                    OverflowCheckedTensor y_tensor = new OverflowCheckedTensor(Shape.Map3D(channels, outwidth, outheight, outdepth, batch));

                                    AveragePooling ope = new AveragePooling(inwidth, inheight, indepth, channels, stride, batch);

                                    ope.Execute(x_tensor, y_tensor);

                                    float[] y_expect = y.ToArray();
                                    float[] y_actual = y_tensor.State;

                                    CollectionAssert.AreEqual(xval, x_tensor.State);

                                    AssertError.Tolerance(y_expect, y_actual, 1e-7f, 1e-5f, ref max_err, $"mismatch value {channels},{stride},{inwidth},{inheight},{indepth},{batch}");

                                    Console.WriteLine($"pass: {channels},{stride},{inwidth},{inheight},{indepth},{batch}");

                                }
                            }
                        }
                    }
                }
            }

            Console.WriteLine($"maxerr:{max_err}");
        }

        public static Map3D Reference(Map3D x, int stride) {
            int channels = x.Channels, batch = x.Batch;
            int inw = x.Width, inh = x.Height, ind = x.Depth;
            int outw = inw / stride, outh = inh / stride, outd = ind / stride;

            Map3D y = new Map3D(channels, outw, outh, outd, batch);

            for (int th = 0; th < batch; th++) {
                for (int ox, oy, oz = 0; oz < outd; oz++) {
                    for (oy = 0; oy < outh; oy++) {
                        for (ox = 0; ox < outw; ox++) {
                            for (int ch = 0; ch < channels; ch++) {
                                double sum = 0;

                                for (int kx, ky, kz = 0; kz < stride; kz++) {
                                    for (ky = 0; ky < stride; ky++) {
                                        for (kx = 0; kx < stride; kx++) {
                                            sum += x[ch, ox * stride + kx, oy * stride + ky, oz * stride + kz, th];
                                        }
                                    }
                                }

                                y[ch, ox, oy, oz, th] = sum / (stride * stride * stride);
                            }
                        }
                    }
                }
            }

            return y;
        }

        [TestMethod]
        public void SpeedTest() {
            int inwidth = 64, inheight = 64, indepth = 64, channels = 32, stride = 2, batch = 4;
            int outwidth = inwidth / stride, outheight = inheight / stride, outdepth = indepth / stride;

            OverflowCheckedTensor x_tensor = new OverflowCheckedTensor(Shape.Map3D(channels, inwidth, inheight, indepth, batch));
            OverflowCheckedTensor y_tensor = new OverflowCheckedTensor(Shape.Map3D(channels, outwidth, outheight, outdepth, batch));

            AveragePooling ope = new AveragePooling(inwidth, inheight, indepth, channels, stride, batch);

            Cuda.Profiler.Initialize("../../../../profiler.nvsetting", "../../nvprofiles/averagepool_3d.nvvp");
            Cuda.Profiler.Start();

            ope.Execute(x_tensor, y_tensor);

            Cuda.Profiler.Stop();
        }
    }
}
