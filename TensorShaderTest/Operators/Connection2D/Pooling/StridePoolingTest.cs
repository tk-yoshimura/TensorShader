using System;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using TensorShader.Operators.Connection2D;
using TensorShaderCudaBackend.API;

namespace TensorShaderTest.Operators.Connection2D {
    [TestClass]
    public class StridePoolingTest {
        [TestMethod]
        public void ExecuteTest() {
            float max_err = 0;

            foreach (int batch in new int[] { 1, 2, 3 }) {
                foreach (int channels in new int[] { 1, 2, 3, 5 }) {
                    foreach (int stride in new int[] { 2, 3, 4 }) {
                        foreach (int inheight in new int[] { stride, 5, 7, 8, 11 }) {
                            foreach (int inwidth in new int[] { stride, 5, 7, 8, 11 }) {
                                int outwidth = inwidth / stride, outheight = inheight / stride;

                                float[] xval = (new float[inwidth * inheight * channels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();

                                Map2D x = new Map2D(channels, inwidth, inheight, batch, xval);

                                Map2D y = Reference(x, stride);

                                OverflowCheckedTensor x_tensor = new OverflowCheckedTensor(Shape.Map2D(channels, inwidth, inheight, batch), xval);
                                OverflowCheckedTensor y_tensor = new OverflowCheckedTensor(Shape.Map2D(channels, outwidth, outheight, batch));

                                StridePooling ope = new StridePooling(inwidth, inheight, channels, stride, batch);

                                ope.Execute(x_tensor, y_tensor);

                                float[] y_expect = y.ToArray();
                                float[] y_actual = y_tensor.State;

                                CollectionAssert.AreEqual(xval, x_tensor.State);

                                AssertError.Tolerance(y_expect, y_actual, 1e-7f, 1e-5f, ref max_err, $"mismatch value {channels},{stride},{inwidth},{inheight},{batch}");

                                Console.WriteLine($"pass: {channels},{stride},{inwidth},{inheight},{batch}");

                            }
                        }
                    }
                }
            }

            Console.WriteLine($"maxerr:{max_err}");
        }

        public static Map2D Reference(Map2D x, int stride) {
            int channels = x.Channels, batch = x.Batch;
            int inw = x.Width, inh = x.Height, outw = inw / stride, outh = inh / stride;

            Map2D y = new Map2D(channels, outw, outh, batch);

            for (int th = 0; th < batch; th++) {
                for (int ox, oy = 0; oy < outh; oy++) {
                    int iy = oy * stride;

                    for (ox = 0; ox < outw; ox++) {
                        int ix = ox * stride;

                        for (int ch = 0; ch < channels; ch++) {
                            y[ch, ox, oy, th] = x[ch, ix, iy, th];
                        }
                    }
                }
            }

            return y;
        }

        [TestMethod]
        public void SpeedTest() {
            int inwidth = 512, inheight = 512, channels = 32, stride = 2, batch = 4;
            int outwidth = inwidth / stride, outheight = inheight / stride;

            OverflowCheckedTensor x_tensor = new OverflowCheckedTensor(Shape.Map2D(channels, inwidth, inheight, batch));
            OverflowCheckedTensor y_tensor = new OverflowCheckedTensor(Shape.Map2D(channels, outwidth, outheight, batch));

            StridePooling ope = new StridePooling(inwidth, inheight, channels, stride, batch);

            Cuda.Profiler.Initialize("../../../../profiler.nvsetting", "../../nvprofiles/stridepool_2d.nvvp");
            Cuda.Profiler.Start();

            ope.Execute(x_tensor, y_tensor);

            Cuda.Profiler.Stop();
        }
    }
}
