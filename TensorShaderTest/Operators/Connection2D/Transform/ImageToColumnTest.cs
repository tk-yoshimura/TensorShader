using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;
using TensorShader;
using TensorShader.Operators.Connection2D;
using TensorShaderCudaBackend.API;

namespace TensorShaderTest.Operators.Connection2D {
    [TestClass]
    public class ImageToColumnTest {
        [TestMethod]
        public void ExecuteTest() {
            float max_err = 0;

            foreach (int batch in new int[] { 1, 2 }) {
                foreach (int channels in new int[] { 1, 2, 3, 4, 5, 10, 15, 20 }) {
                    foreach (int kheight in new int[] { 1, 3, 5 }) {
                        foreach (int kwidth in new int[] { 1, 3, 5 }) {
                            foreach (int inwidth in new int[] { 8, 9, 13, 17 }) {
                                foreach (int inheight in new int[] { 8, 9, 19, 23 }) {
                                    int outwidth = inwidth - kwidth + 1, outheight = inheight - kheight + 1;

                                    float[] xval = (new float[inwidth * inheight * channels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();

                                    Map2D x = new(channels, inwidth, inheight, batch, xval);

                                    Map2D y = Reference(x, kwidth, kheight);

                                    OverflowCheckedTensor x_tensor = new(Shape.Map2D(channels, inwidth, inheight, batch), xval);

                                    OverflowCheckedTensor y_tensor = new(new Shape(ShapeType.Column, kwidth * kheight, channels, outwidth, outheight, batch));

                                    ImageToColumn ope = new(inwidth, inheight, channels, kwidth, kheight, batch);

                                    ope.Execute(x_tensor, y_tensor);

                                    float[] y_expect = y.ToArray();
                                    float[] y_actual = y_tensor.State.Value;

                                    CollectionAssert.AreEqual(xval, x_tensor.State.Value);

                                    AssertError.Tolerance(y_expect, y_actual, 1e-7f, 1e-5f, ref max_err, $"mismatch value {channels},{kwidth},{inwidth},{inheight},{batch}");

                                    Console.WriteLine($"pass: {channels},{kwidth},{kheight},{inwidth},{inheight},{batch}");
                                }
                            }
                        }
                    }
                }
            }

            Console.WriteLine($"maxerr:{max_err}");
        }

        [TestMethod]
        public void SpeedTest() {
            int inwidth = 512, inheight = 512, channels = 31, ksize = 3;
            int outwidth = inwidth - ksize + 1, outheight = inheight - ksize + 1;

            OverflowCheckedTensor x_tensor = new(Shape.Map2D(channels, inwidth, inheight));

            OverflowCheckedTensor y_tensor = new(new Shape(ShapeType.Column, ksize * ksize, channels, outwidth, outheight, 1));

            ImageToColumn ope = new(inwidth, inheight, channels, ksize, ksize);

            Cuda.Profiler.Initialize("../../../../profiler.nvsetting", "../../nvprofiles/image_to_column_2d.nvvp");
            Cuda.Profiler.Start();

            ope.Execute(x_tensor, y_tensor);

            Cuda.Profiler.Stop();
        }

        public static Map2D Reference(Map2D x, int kwidth, int kheight) {
            int channels = x.Channels, batch = x.Batch;
            int inw = x.Width, inh = x.Height, outw = inw - kwidth + 1, outh = inh - kheight + 1;

            Map2D y = new(kwidth * kheight * channels, outw, outh, batch);

            for (int k = 0, kx, ky = 0; ky < kheight; ky++) {
                for (kx = 0; kx < kwidth; k++, kx++) {
                    for (int th = 0; th < batch; th++) {
                        for (int ox, oy = 0; oy < outh; oy++) {
                            for (ox = 0; ox < outw; ox++) {
                                for (int ch = 0; ch < channels; ch++) {
                                    y[k + kwidth * kheight * ch, ox, oy, th] = x[ch, kx + ox, ky + oy, th];
                                }
                            }
                        }
                    }
                }
            }

            return y;
        }
    }
}
