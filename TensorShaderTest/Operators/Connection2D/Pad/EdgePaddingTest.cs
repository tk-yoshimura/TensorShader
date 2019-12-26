using System;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using TensorShader.Operators.Connection2D;
using TensorShaderCudaBackend.API;

namespace TensorShaderTest.Operators.Connection2D {
    [TestClass]
    public class EdgePaddingTest {
        [TestMethod]
        public void ExecuteTest() {
            float max_err = 0;

            foreach (int batch in new int[] { 1, 2 }) {
                foreach (int channels in new int[] { 1, 2, 3, 4, 5, 6, 7, 8 }) {
                    foreach (int toppad in new int[] { 0, 1, 2 }) {
                        foreach (int bottompad in new int[] { 0, 1, 2 }) {
                            foreach (int leftpad in new int[] { 0, 1, 2 }) {
                                foreach (int rightpad in new int[] { 0, 1, 2 }) {
                                    foreach (int inwidth in new int[] { 5, 7, 11 }) {
                                        foreach (int inheight in new int[] { 5, 7, 11 }) {
                                            int outwidth = inwidth + leftpad + rightpad, outheight = inheight + toppad + bottompad;

                                            float[] xval = (new float[inwidth * inheight * channels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();

                                            Map2D x = new Map2D(channels, inwidth, inheight, batch, xval);

                                            Map2D y = Reference(x, leftpad, rightpad, toppad, bottompad);

                                            OverflowCheckedTensor x_tensor = new OverflowCheckedTensor(Shape.Map2D(channels, inwidth, inheight, batch), xval);
                                            OverflowCheckedTensor y_tensor = new OverflowCheckedTensor(Shape.Map2D(channels, outwidth, outheight, batch));

                                            EdgePadding ope = new EdgePadding(inwidth, inheight, channels, leftpad, rightpad, toppad, bottompad, batch);

                                            ope.Execute(x_tensor, y_tensor);

                                            float[] y_expect = y.ToArray();
                                            float[] y_actual = y_tensor.State;

                                            CollectionAssert.AreEqual(xval, x_tensor.State);

                                            AssertError.Tolerance(y_expect, y_actual, 1e-7f, 1e-5f, ref max_err, $"mismatch value {channels},{leftpad},{rightpad},{toppad},{bottompad},{inwidth},{inheight},{batch}");

                                            Console.WriteLine($"pass: {channels},{leftpad},{rightpad},{toppad},{bottompad},{inwidth},{inheight},{batch}");

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

        public static Map2D Reference(Map2D x, int leftpad, int rightpad, int toppad, int bottompad) {
            int channels = x.Channels, batch = x.Batch;
            int inw = x.Width, inh = x.Height, outw = inw + leftpad + rightpad, outh = inh + toppad + bottompad;

            Map2D y = new Map2D(channels, outw, outh, batch);

            for (int th = 0; th < batch; th++) {
                for (int ox, oy = 0; oy < outh; oy++) {
                    int iy = Math.Min(inh - 1, Math.Max(0, oy - toppad));

                    for (ox = 0; ox < outw; ox++) {
                        int ix = Math.Min(inw - 1, Math.Max(0, ox - leftpad));

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
            int inwidth = 512, inheight = 512, channels = 32, leftpad = 1, rightpad = 1, toppad = 1, bottompad = 1, batch = 4;
            int outwidth = inwidth + leftpad + rightpad, outheight = inheight + toppad + bottompad;

            OverflowCheckedTensor x_tensor = new OverflowCheckedTensor(Shape.Map2D(channels, inwidth, inheight, batch));
            OverflowCheckedTensor y_tensor = new OverflowCheckedTensor(Shape.Map2D(channels, outwidth, outheight, batch));

            EdgePadding ope = new EdgePadding(inwidth, inheight, channels, leftpad, rightpad, toppad, bottompad, batch);

            Cuda.Profiler.Initialize("../../../profiler.nvsetting", "../../nvprofiles/edgepadding_2d.nvvp");
            Cuda.Profiler.Start();

            ope.Execute(x_tensor, y_tensor);

            Cuda.Profiler.Stop();
        }
    }
}
