using System;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using TensorShader.Operators.Connection2D;
using TensorShaderCudaBackend.API;

namespace TensorShaderTest.Operators.Connection2D {
    [TestClass]
    public class StrideUnpoolingTest {
        [TestMethod]
        public void ExecuteTest() {
            float max_err = 0;

            Random random = new Random();

            foreach (int batch in new int[] { 1, 2, 3 }) {
                foreach (int channels in new int[] { 1, 2, 3, 5 }) {
                    foreach (int stride in new int[] { 2, 3, 4 }) {
                        foreach (int inheight in new int[] { stride, 5, 7, 8, 11 }) {
                            foreach (int inwidth in new int[] { stride, 5, 7, 8, 11 }) {
                                int outwidth = inwidth / stride, outheight = inheight / stride;

                                float[] yval = (new float[outwidth * outheight * channels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();

                                Map2D y = new Map2D(channels, outwidth, outheight, batch, yval);

                                Map2D x = Reference(y, inwidth, inheight, stride);

                                OverflowCheckedTensor x_tensor = new OverflowCheckedTensor(Shape.Map2D(channels, inwidth, inheight, batch));
                                OverflowCheckedTensor y_tensor = new OverflowCheckedTensor(Shape.Map2D(channels, outwidth, outheight, batch), yval);

                                TensorShaderCudaBackend.Randomize.Uniform((uint)x_tensor.Length, x_tensor.Buffer, random);

                                StrideUnpooling ope = new StrideUnpooling(inwidth, inheight, channels, stride, batch);

                                ope.Execute(y_tensor, x_tensor);

                                float[] x_expect = x.ToArray();
                                float[] x_actual = x_tensor.State;

                                CollectionAssert.AreEqual(yval, y_tensor.State);

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

            Map2D x = new Map2D(channels, outw, outh, batch);

            for (int th = 0; th < batch; th++) {
                for (int ix, iy = 0; iy < inh; iy++) {
                    for (ix = 0; ix < inw; ix++) {
                        int ox = ix * stride, oy = iy * stride;

                        for (int ch = 0; ch < channels; ch++) {
                            x[ch, ox, oy, th] = y[ch, ix, iy, th];
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

            OverflowCheckedTensor x_tensor = new OverflowCheckedTensor(Shape.Map2D(channels, inwidth, inheight, batch));
            OverflowCheckedTensor y_tensor = new OverflowCheckedTensor(Shape.Map2D(channels, outwidth, outheight, batch));

            StrideUnpooling ope = new StrideUnpooling(outwidth, outheight, channels, stride, batch);

            Cuda.Profiler.Initialize("../../../../profiler.nvsetting", "../../nvprofiles/strideunpool_2d.nvvp");
            Cuda.Profiler.Start();

            ope.Execute(x_tensor, y_tensor);

            Cuda.Profiler.Stop();
        }
    }
}
