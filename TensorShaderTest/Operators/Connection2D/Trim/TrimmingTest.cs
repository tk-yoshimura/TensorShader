using System;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using TensorShader.Operators.Connection2D;
using TensorShaderCudaBackend.API;

namespace TensorShaderTest.Operators.Connection2D {
    [TestClass]
    public class TrimmingTest {
        [TestMethod]
        public void ExecuteTest() {
            float max_err = 0;

            foreach (int batch in new int[] { 1, 2 }) {
                foreach (int channels in new int[] { 3, 5 }) {
                    foreach (int lefttrim in new int[] { 0, 1, 2 }) {
                        foreach (int righttrim in new int[] { 0, 1, 2 }) {
                            foreach (int toptrim in new int[] { 0, 1, 2 }) {
                                foreach (int bottomtrim in new int[] { 0, 1, 2 }) {
                                    foreach (int inwidth in new int[] { 5, 7, 11 }) {
                                        foreach (int inheight in new int[] { 5, 7, 11 }) {
                                            int outwidth = inwidth - lefttrim - righttrim, outheight = inheight - toptrim - bottomtrim;

                                            float[] xval = (new float[inwidth * inheight * channels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();

                                            Map2D x = new Map2D(channels, inwidth, inheight, batch, xval);

                                            Map2D y = Reference(x, lefttrim, righttrim, toptrim, bottomtrim);

                                            OverflowCheckedTensor x_tensor = new OverflowCheckedTensor(Shape.Map2D(channels, inwidth, inheight, batch), xval);
                                            OverflowCheckedTensor y_tensor = new OverflowCheckedTensor(Shape.Map2D(channels, outwidth, outheight, batch));

                                            Trimming ope = new Trimming(inwidth, inheight, channels, lefttrim, righttrim, toptrim, bottomtrim, batch);

                                            ope.Execute(x_tensor, y_tensor);

                                            float[] y_expect = y.ToArray();
                                            float[] y_actual = y_tensor.State;

                                            CollectionAssert.AreEqual(xval, x_tensor.State);

                                            AssertError.Tolerance(y_expect, y_actual, 1e-7f, 1e-5f, ref max_err, $"mismatch value {channels},{lefttrim},{righttrim},{toptrim},{bottomtrim},{inwidth},{inheight},{batch}");

                                            Console.WriteLine($"pass: {channels},{lefttrim},{righttrim},{toptrim},{bottomtrim},{inwidth},{inheight},{batch}");

                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            Console.WriteLine($"maxerr:{max_err}");
        }

        public static Map2D Reference(Map2D x, int lefttrim, int righttrim, int toptrim, int bottomtrim) {
            int channels = x.Channels, batch = x.Batch;
            int inw = x.Width, inh = x.Height, outw = inw - lefttrim - righttrim, outh = inh - toptrim - bottomtrim;

            Map2D y = new Map2D(channels, outw, outh, batch);

            for (int th = 0; th < batch; th++) {
                for (int ox, oy = 0; oy < outh; oy++) {
                    for (ox = 0; ox < outw; ox++) {
                        for (int ch = 0; ch < channels; ch++) {
                            y[ch, ox, oy, th] = x[ch, ox + lefttrim, oy + toptrim, th];
                        }
                    }
                }

            }

            return y;
        }

        [TestMethod]
        public void SpeedTest() {
            int inwidth = 512, inheight = 512, channels = 32, lefttrim = 1, righttrim = 1, toptrim = 1, bottomtrim = 1, batch = 4;
            int outwidth = inwidth - lefttrim - righttrim, outheight = inheight - toptrim - bottomtrim;

            OverflowCheckedTensor x_tensor = new OverflowCheckedTensor(Shape.Map2D(channels, inwidth, inheight, batch));
            OverflowCheckedTensor y_tensor = new OverflowCheckedTensor(Shape.Map2D(channels, outwidth, outheight, batch));

            Trimming ope = new Trimming(inwidth, inheight, channels, lefttrim, righttrim, toptrim, bottomtrim, batch);

            Cuda.Profiler.Initialize("../../../profiler.nvsetting", "../../nvprofiles/trimming2d.nvvp");
            Cuda.Profiler.Start();

            ope.Execute(x_tensor, y_tensor);

            Cuda.Profiler.Stop();
        }
    }
}
