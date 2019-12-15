using System;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using TensorShader.Operators.Connection1D;
using TensorShaderCudaBackend.API;

namespace TensorShaderTest.Operators.Connection1D {
    [TestClass]
    public class TrimmingTest {
        [TestMethod]
        public void ExecuteTest() {
            float max_err = 0;

            foreach (int batch in new int[] { 1, 2 }) {
                foreach (int channels in new int[] { 3, 5 }) {
                    foreach (int lefttrim in new int[] { 0, 1, 2 }) {
                        foreach (int righttrim in new int[] { 0, 1, 2 }) {
                            foreach (int inwidth in new int[] { 5, 7, 11 }) {
                                int outwidth = inwidth - lefttrim - righttrim;

                                float[] xval = (new float[inwidth * channels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();

                                Map1D x = new Map1D(channels, inwidth, batch, xval);

                                Map1D y = Reference(x, lefttrim, righttrim);

                                OverflowCheckedTensor x_tensor = new OverflowCheckedTensor(Shape.Map1D(channels, inwidth, batch), xval);
                                OverflowCheckedTensor y_tensor = new OverflowCheckedTensor(Shape.Map1D(channels, outwidth, batch));

                                Trimming ope = new Trimming(inwidth, channels, lefttrim, righttrim, batch);

                                ope.Execute(x_tensor, y_tensor);

                                float[] y_expect = y.ToArray();
                                float[] y_actual = y_tensor.State;

                                CollectionAssert.AreEqual(xval, x_tensor.State);

                                AssertError.Tolerance(y_expect, y_actual, 1e-7f, 1e-5f, ref max_err, $"mismatch value {channels},{lefttrim},{righttrim},{inwidth},{batch}");

                                Console.WriteLine($"pass: {channels},{lefttrim},{righttrim},{inwidth},{batch}");

                            }
                        }
                    }
                }
            }

            Console.WriteLine($"maxerr:{max_err}");
        }

        public static Map1D Reference(Map1D x, int lefttrim, int righttrim) {
            int channels = x.Channels, batch = x.Batch;
            int inw = x.Width, outw = inw - lefttrim - righttrim;

            Map1D y = new Map1D(channels, outw, batch);

            for (int th = 0; th < batch; th++) {
                for (int ox = 0; ox < outw; ox++) {
                    for (int ch = 0; ch < channels; ch++) {
                        y[ch, ox, th] = x[ch, ox + lefttrim, th];
                    }
                }

            }

            return y;
        }

        [TestMethod]
        public void SpeedTest() {
            int inwidth = 512, channels = 32, lefttrim = 1, righttrim = 1, batch = 4;
            int outwidth = inwidth - lefttrim - righttrim;

            OverflowCheckedTensor x_tensor = new OverflowCheckedTensor(Shape.Map1D(channels, inwidth, batch));
            OverflowCheckedTensor y_tensor = new OverflowCheckedTensor(Shape.Map1D(channels, outwidth, batch));

            Trimming ope = new Trimming(inwidth, channels, lefttrim, righttrim, batch);

            Cuda.Profiler.Initialize("../../../profiler.nvsetting", "../../nvprofiles/trimming1d.nvvp");
            Cuda.Profiler.Start();

            ope.Execute(x_tensor, y_tensor);

            Cuda.Profiler.Stop();
        }
    }
}
