using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;
using TensorShader;
using TensorShader.Operators.Connection2D;
using TensorShaderCudaBackend.API;

namespace TensorShaderTest.Operators.Connection2D {
    [TestClass]
    public class AverageUnpoolingTest {
        [TestMethod]
        public void ExecuteTest() {
            float max_err = 0;

            foreach (int batch in new int[] { 1, 2, 3 }) {
                foreach (int channels in new int[] { 1, 2, 3, 5 }) {
                    foreach (int stride in new int[] { 2, 3, 4 }) {
                        foreach (int inheight in new int[] { stride, 5, 7, 8, 11 }) {
                            foreach (int inwidth in new int[] { stride, 5, 7, 8, 11 }) {
                                int outwidth = inwidth / stride, outheight = inheight / stride;

                                float[] yval = (new float[outwidth * outheight * channels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();

                                Map2D y = new(channels, outwidth, outheight, batch, yval);

                                Map2D x = Reference(y, inwidth, inheight, stride);

                                OverflowCheckedTensor x_tensor = new(Shape.Map2D(channels, inwidth, inheight, batch));
                                OverflowCheckedTensor y_tensor = new(Shape.Map2D(channels, outwidth, outheight, batch), yval);

                                AverageUnpooling ope = new(inwidth, inheight, channels, stride, batch);

                                ope.Execute(y_tensor, x_tensor);

                                float[] x_expect = x.ToArray();
                                float[] x_actual = x_tensor.State.Value;

                                CollectionAssert.AreEqual(yval, y_tensor.State.Value);

                                AssertError.Tolerance(x_expect, x_actual, 1e-7f, 1e-5f, ref max_err, $"mismatch value {channels},{stride},{inwidth},{inheight},{batch}");

                                Console.WriteLine($"pass: {channels},{stride},{inwidth},{inheight},{batch}");

                            }
                        }
                    }
                }
            }

            Console.WriteLine($"maxerr:{max_err}");
        }

        public static Map2D Reference(Map2D y, int outw, int outh, int stride) {
            int channels = y.Channels, batch = y.Batch;
            int inw = outw / stride, inh = outh / stride;

            Map2D x = new(channels, outw, outh, batch);

            for (int th = 0; th < batch; th++) {
                for (int ox, oy = 0; oy < outh; oy++) {
                    for (ox = 0; ox < outw; ox++) {
                        int ix = ox / stride, iy = oy / stride;

                        if (ix < inw && iy < inh) {
                            for (int ch = 0; ch < channels; ch++) {
                                x[ch, ox, oy, th] = y[ch, ix, iy, th] / (stride * stride);
                            }
                        }
                        else {
                            for (int ch = 0; ch < channels; ch++) {
                                x[ch, ox, oy, th] = 0;
                            }
                        }
                    }
                }
            }

            return x;
        }

        [TestMethod]
        public void SpeedTest() {
            int outwidth = 512, outheight = 512, channels = 32, stride = 2, batch = 4;
            int inwidth = outwidth / stride, inheight = outheight / stride;

            OverflowCheckedTensor x_tensor = new(Shape.Map2D(channels, inwidth, inheight, batch));
            OverflowCheckedTensor y_tensor = new(Shape.Map2D(channels, outwidth, outheight, batch));

            AverageUnpooling ope = new(outwidth, outheight, channels, stride, batch);

            Cuda.Profiler.Initialize("../../../../profiler.nvsetting", "../../nvprofiles/averageunpool_2d.nvvp");
            Cuda.Profiler.Start();

            ope.Execute(x_tensor, y_tensor);

            Cuda.Profiler.Stop();
        }
    }
}
