using System;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using TensorShader.Operators.Connection1D;
using TensorShaderCudaBackend.API;

namespace TensorShaderTest.Operators.Connection1D {
    [TestClass]
    public class ZeroPaddingTest {
        [TestMethod]
        public void ExecuteTest() {
            float max_err = 0;

            Random random = new Random();

            foreach (int batch in new int[] { 1, 2 }) {
                foreach (int channels in new int[] { 1, 2, 3, 4, 5, 6, 7, 8 }) {
                    foreach (int leftpad in new int[] { 0, 1, 2 }) {
                        foreach (int rightpad in new int[] { 0, 1, 2 }) {
                            foreach (int inwidth in new int[] { 5, 7, 11 }) {
                                int outwidth = inwidth + leftpad + rightpad;

                                float[] xval = (new float[inwidth * channels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();

                                Map1D x = new Map1D(channels, inwidth, batch, xval);

                                Map1D y = Reference(x, leftpad, rightpad);

                                OverflowCheckedTensor x_tensor = new OverflowCheckedTensor(Shape.Map1D(channels, inwidth, batch), xval);
                                OverflowCheckedTensor y_tensor = new OverflowCheckedTensor(Shape.Map1D(channels, outwidth, batch));

                                TensorShaderCudaBackend.Randomize.Uniform((uint)y_tensor.Length, y_tensor.Buffer, random);

                                ZeroPadding ope = new ZeroPadding(inwidth, channels, leftpad, rightpad, batch);

                                ope.Execute(x_tensor, y_tensor);

                                float[] y_expect = y.ToArray();
                                float[] y_actual = y_tensor.State;

                                CollectionAssert.AreEqual(xval, x_tensor.State);

                                AssertError.Tolerance(y_expect, y_actual, 1e-7f, 1e-5f, ref max_err, $"mismatch value {channels},{leftpad},{rightpad},{inwidth},{batch}");

                                Console.WriteLine($"pass: {channels},{leftpad},{rightpad},{inwidth},{batch}");

                            }
                        }
                    }
                }
            }

            Console.WriteLine($"maxerr:{max_err}");
        }

        public static Map1D Reference(Map1D x, int leftpad, int rightpad) {
            int channels = x.Channels, batch = x.Batch;
            int inw = x.Width, outw = inw + leftpad + rightpad;

            Map1D y = new Map1D(channels, outw, batch);

            for (int th = 0; th < batch; th++) {
                for (int ix = 0; ix < inw; ix++) {
                    for (int ch = 0; ch < channels; ch++) {
                        y[ch, ix + leftpad, th] = x[ch, ix, th];
                    }
                }
            }

            return y;
        }

        [TestMethod]
        public void SpeedTest() {
            int inwidth = 2048, channels = 1024, leftpad = 2, rightpad = 3, batch = 4;
            int outwidth = inwidth + leftpad + rightpad;

            OverflowCheckedTensor x_tensor = new OverflowCheckedTensor(Shape.Map1D(channels, inwidth, batch));
            OverflowCheckedTensor y_tensor = new OverflowCheckedTensor(Shape.Map1D(channels, outwidth, batch));

            ZeroPadding ope = new ZeroPadding(inwidth, channels, leftpad, rightpad, batch);

            Cuda.Profiler.Initialize("../../../../profiler.nvsetting", "../../nvprofiles/zeropadding_1d.nvvp");
            Cuda.Profiler.Start();

            ope.Execute(x_tensor, y_tensor);

            Cuda.Profiler.Stop();
        }
    }
}
