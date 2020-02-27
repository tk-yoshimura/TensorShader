using System;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using TensorShader.Operators.Connection1D;
using TensorShaderCudaBackend.API;

namespace TensorShaderTest.Operators.Connection1D {
    [TestClass]
    public class ImageToColumnTest {
        [TestMethod]
        public void ExecuteTest() {
            float max_err = 0;

            foreach (int batch in new int[] { 1, 2 }) {
                foreach (int channels in new int[] { 1, 2, 3, 4, 5, 10, 15, 20 }) {
                    foreach (int kwidth in new int[] { 1, 3, 5 }) {
                        foreach (int inwidth in new int[] { 8, 9, 13, 17 }) {
                            int outwidth = inwidth - kwidth + 1;

                            float[] xval = (new float[inwidth * channels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();

                            Map1D x = new Map1D(channels, inwidth, batch, xval);

                            Map1D y = Reference(x, kwidth);

                            OverflowCheckedTensor x_tensor = new OverflowCheckedTensor(Shape.Map1D(channels, inwidth, batch), xval);

                            OverflowCheckedTensor y_tensor = new OverflowCheckedTensor(new Shape(ShapeType.Column, kwidth, channels, outwidth, batch));

                            ImageToColumn ope = new ImageToColumn(inwidth, channels, kwidth, batch);

                            ope.Execute(x_tensor, y_tensor);

                            float[] y_expect = y.ToArray();
                            float[] y_actual = y_tensor.State;

                            CollectionAssert.AreEqual(xval, x_tensor.State);

                            AssertError.Tolerance(y_expect, y_actual, 1e-7f, 1e-5f, ref max_err, $"mismatch value {channels},{kwidth},{inwidth},{batch}");

                            Console.WriteLine($"pass: {channels},{kwidth},{inwidth},{batch}");
                        }
                    }
                }
            }

            Console.WriteLine($"maxerr:{max_err}");
        }

        [TestMethod]
        public void SpeedTest() {
            int inwidth = 512, channels = 31, ksize = 3;
            int outwidth = inwidth - ksize + 1;

            OverflowCheckedTensor x_tensor = new OverflowCheckedTensor(Shape.Map1D(channels, inwidth));

            OverflowCheckedTensor y_tensor = new OverflowCheckedTensor(new Shape(ShapeType.Column, ksize, channels, outwidth, 1));

            ImageToColumn ope = new ImageToColumn(inwidth, channels, ksize);

            Cuda.Profiler.Initialize("../../../../profiler.nvsetting", "../../nvprofiles/image_to_column_1d.nvvp");
            Cuda.Profiler.Start();

            ope.Execute(x_tensor, y_tensor);

            Cuda.Profiler.Stop();
        }

        public static Map1D Reference(Map1D x, int kwidth) {
            int channels = x.Channels, batch = x.Batch;
            int inw = x.Width, outw = inw - kwidth + 1;

            Map1D y = new Map1D(kwidth * channels, outw, batch);

            for (int k = 0, kx = 0; kx < kwidth; k++, kx++) {
                for (int th = 0; th < batch; th++) {
                    for (int ox = 0; ox < outw; ox++) {
                        for (int ch = 0; ch < channels; ch++) {
                            y[k + kwidth * ch, ox, th] = x[ch, kx + ox, th];
                        }
                    }
                }
            }

            return y;
        }
    }
}
