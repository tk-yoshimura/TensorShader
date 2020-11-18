using System;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using TensorShader.Operators.Connection1D;
using TensorShaderCudaBackend.API;

namespace TensorShaderTest.Operators.Connection1D {
    [TestClass]
    public class ColumnToImageTest {
        [TestMethod]
        public void ExecuteTest() {
            float max_err = 0;

            foreach (int batch in new int[] { 1, 2 }) {
                foreach (int channels in new int[] { 1, 2, 3, 4, 5, 10, 15, 20 }) {
                    foreach (int kwidth in new int[] { 1, 3, 5 }) {
                        foreach (int inwidth in new int[] { 8, 9, 13, 17 }) {
                            int outwidth = inwidth - kwidth + 1;

                            float[] xval = (new float[kwidth * outwidth * channels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();

                            Map1D x = new Map1D(kwidth * channels, outwidth, batch, xval);

                            Map1D y = Reference(x, kwidth);

                            OverflowCheckedTensor x_tensor = new OverflowCheckedTensor(new Shape(ShapeType.Column, kwidth, channels, outwidth, batch), xval);

                            OverflowCheckedTensor y_tensor = new OverflowCheckedTensor(Shape.Map1D(channels, inwidth, batch));

                            ColumnToImage ope = new ColumnToImage(outwidth, channels, kwidth, batch);

                            ope.Execute(x_tensor, y_tensor);

                            float[] y_expect = y.ToArray();
                            float[] y_actual = y_tensor.State.Value;

                            CollectionAssert.AreEqual(xval, x_tensor.State.Value);

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

            OverflowCheckedTensor x_tensor = new OverflowCheckedTensor(new Shape(ShapeType.Column, ksize, channels, outwidth, 1));

            OverflowCheckedTensor y_tensor = new OverflowCheckedTensor(Shape.Map1D(channels, inwidth));

            ColumnToImage ope = new ColumnToImage(outwidth, channels, ksize);

            Cuda.Profiler.Initialize("../../../../profiler.nvsetting", "../../nvprofiles/column_to_image_1d.nvvp");
            Cuda.Profiler.Start();

            ope.Execute(x_tensor, y_tensor);

            Cuda.Profiler.Stop();
        }

        public static Map1D Reference(Map1D x, int kwidth) {
            int channels = x.Channels / kwidth, batch = x.Batch;
            int inw = x.Width, outw = inw + kwidth - 1;

            Map1D y = new Map1D(channels, outw, batch);

            for (int k = 0, kx = 0; kx < kwidth; k++, kx++) {
                for (int th = 0; th < batch; th++) {
                    for (int ix = 0; ix < inw; ix++) {
                        for (int ch = 0; ch < channels; ch++) {
                            y[ch, kx + ix, th] += x[k + kwidth * ch, ix, th];
                        }
                    }
                }
            }

            return y;
        }
    }
}
