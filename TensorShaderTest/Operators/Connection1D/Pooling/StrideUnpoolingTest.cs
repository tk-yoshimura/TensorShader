using System;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using TensorShader.Operators.Connection1D;
using TensorShaderCudaBackend.API;

namespace TensorShaderTest.Operators.Connection1D {
    [TestClass]
    public class StrideUnpoolingTest {
        [TestMethod]
        public void ExecuteTest() {
            float max_err = 0;

            Random random = new Random();

            foreach (int batch in new int[] { 1, 2, 3 }) {
                foreach (int channels in new int[] { 1, 2, 3, 5 }) {
                    foreach (int stride in new int[] { 2, 3, 4 }) {
                        foreach (int inwidth in new int[] { stride, 5, 7, 8, 11 }) {
                            int outwidth = inwidth / stride;

                            float[] yval = (new float[outwidth * channels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();

                            Map1D y = new Map1D(channels, outwidth, batch, yval);

                            Map1D x = Reference(y, inwidth, stride);

                            OverflowCheckedTensor x_tensor = new OverflowCheckedTensor(Shape.Map1D(channels, inwidth, batch));
                            OverflowCheckedTensor y_tensor = new OverflowCheckedTensor(Shape.Map1D(channels, outwidth, batch), yval);

                            TensorShaderCudaBackend.Randomize.Uniform((uint)x_tensor.Length, x_tensor.Buffer, random);

                            StrideUnpooling ope = new StrideUnpooling(inwidth, channels, stride, batch);

                            ope.Execute(y_tensor, x_tensor);

                            float[] x_expect = x.ToArray();
                            float[] x_actual = x_tensor.State;

                            CollectionAssert.AreEqual(yval, y_tensor.State);

                            AssertError.Tolerance(x_expect, x_actual, 1e-7f, 1e-5f, ref max_err, $"mismatch value {channels},{stride},{inwidth},{batch}");

                            Console.WriteLine($"pass: {channels},{stride},{inwidth},{batch}");

                        }
                    }
                }
            }

            Console.WriteLine($"maxerr:{max_err}");
        }

        public static Map1D Reference(Map1D y, int outw, int stride) {
            int channels = y.Channels, batch = y.Batch;
            int inw = outw / stride;

            Map1D x = new Map1D(channels, outw, batch);

            for (int th = 0; th < batch; th++) {
                for (int ix = 0; ix < inw; ix++) {
                    int ox = ix * stride;

                    for (int ch = 0; ch < channels; ch++) {
                        x[ch, ox, th] = y[ch, ix, th];
                    }
                }
            }

            return x;
        }

        [TestMethod]
        public void SpeedTest() {
            int outwidth = 2048, channels = 1024, stride = 2, batch = 4;
            int inwidth = outwidth / stride;

            OverflowCheckedTensor x_tensor = new OverflowCheckedTensor(Shape.Map1D(channels, inwidth, batch));
            OverflowCheckedTensor y_tensor = new OverflowCheckedTensor(Shape.Map1D(channels, outwidth, batch));

            StrideUnpooling ope = new StrideUnpooling(outwidth, channels, stride, batch);

            Cuda.Profiler.Initialize("../../../../profiler.nvsetting", "../../nvprofiles/strideunpool_1d.nvvp");
            Cuda.Profiler.Start();

            ope.Execute(x_tensor, y_tensor);

            Cuda.Profiler.Stop();
        }
    }
}
