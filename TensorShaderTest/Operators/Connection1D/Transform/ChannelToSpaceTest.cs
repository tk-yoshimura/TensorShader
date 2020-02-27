using System;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using TensorShader.Operators.Connection1D;
using TensorShaderCudaBackend.API;

namespace TensorShaderTest.Operators.Connection1D {
    [TestClass]
    public class ChannelToSpaceTest {
        [TestMethod]
        public void ExecuteTest() {
            float max_err = 0;

            foreach (int batch in new int[] { 1, 2 }) {
                foreach (int outchannels in new int[] { 3, 5 }) {
                    foreach (int scale in new int[] { 2, 3, 4 }) {
                        foreach (int inwidth in new int[] { 5, 7, 11 }) {
                            int outwidth = inwidth * scale, inchannels = outchannels * scale;

                            float[] xval = (new float[inwidth * inchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();

                            Map1D x = new Map1D(inchannels, inwidth, batch, xval);

                            Map1D y = Reference(x, scale);

                            OverflowCheckedTensor x_tensor = new OverflowCheckedTensor(Shape.Map1D(inchannels, inwidth, batch), xval);
                            OverflowCheckedTensor y_tensor = new OverflowCheckedTensor(Shape.Map1D(outchannels, outwidth, batch));

                            ChannelToSpace ope = new ChannelToSpace(inwidth, inchannels, scale, batch);

                            ope.Execute(x_tensor, y_tensor);

                            float[] y_expect = y.ToArray();
                            float[] y_actual = y_tensor.State;

                            CollectionAssert.AreEqual(xval, x_tensor.State);

                            AssertError.Tolerance(y_expect, y_actual, 1e-7f, 1e-5f, ref max_err, $"mismatch value {inchannels},{outchannels},{scale},{inwidth},{batch}");

                            Console.WriteLine($"pass: {inchannels},{outchannels},{scale},{inwidth},{batch}");

                        }
                    }
                }
            }

            Console.WriteLine($"maxerr:{max_err}");
        }

        [TestMethod]
        public void SpeedTest() {
            int inwidth = 512, inchannels = 32, scale = 2;
            int outwidth = inwidth * scale, outchannels = inchannels / scale;

            OverflowCheckedTensor x_tensor = new OverflowCheckedTensor(Shape.Map1D(inchannels, inwidth));
            OverflowCheckedTensor y_tensor = new OverflowCheckedTensor(Shape.Map1D(outchannels, outwidth));

            ChannelToSpace ope = new ChannelToSpace(inwidth, inchannels, scale);

            Cuda.Profiler.Initialize("../../../../profiler.nvsetting", "../../nvprofiles/channel_to_space_1d.nvvp");
            Cuda.Profiler.Start();

            ope.Execute(x_tensor, y_tensor);

            Cuda.Profiler.Stop();
        }

        public static Map1D Reference(Map1D x, int scale) {
            int inchannels = x.Channels, batch = x.Batch;
            if (inchannels % scale != 0) {
                throw new ArgumentException(nameof(scale));
            }

            int inw = x.Width, outw = inw * scale;
            int outchannels = inchannels / scale;

            Map1D y = new Map1D(outchannels, outw, batch);

            for (int th = 0; th < batch; th++) {
                for (int ix = 0; ix < inw; ix++) {
                    for (int kx = 0; kx < scale; kx++) {
                        for (int outch = 0; outch < outchannels; outch++) {
                            int inch = outch + kx * outchannels;

                            y[outch, ix * scale + kx, th] = x[inch, ix, th];

                        }
                    }
                }

            }

            return y;
        }

        [TestMethod]
        public void ReferenceTest() {
            int inchannels = 12, scale = 2, inwidth = 7;
            int outchannels = inchannels / scale, outwidth = inwidth * scale;

            float[] xval = (new float[inwidth * inchannels]).Select((_, idx) => idx * 1e-3f).ToArray();

            Map1D x = new Map1D(inchannels, inwidth, 1, xval);

            Map1D y = Reference(x, scale);

            float[] y_expect = {
                0.000f,0.001f,0.002f,0.003f,0.004f,0.005f,
                0.006f,0.007f,0.008f,0.009f,0.010f,0.011f,
                0.012f,0.013f,0.014f,0.015f,0.016f,0.017f,
                0.018f,0.019f,0.020f,0.021f,0.022f,0.023f,
                0.024f,0.025f,0.026f,0.027f,0.028f,0.029f,
                0.030f,0.031f,0.032f,0.033f,0.034f,0.035f,
                0.036f,0.037f,0.038f,0.039f,0.040f,0.041f,
                0.042f,0.043f,0.044f,0.045f,0.046f,0.047f,
                0.048f,0.049f,0.050f,0.051f,0.052f,0.053f,
                0.054f,0.055f,0.056f,0.057f,0.058f,0.059f,
                0.060f,0.061f,0.062f,0.063f,0.064f,0.065f,
                0.066f,0.067f,0.068f,0.069f,0.070f,0.071f,
                0.072f,0.073f,0.074f,0.075f,0.076f,0.077f,
                0.078f,0.079f,0.080f,0.081f,0.082f,0.083f,
            };

            float[] y_actual = y.ToArray();

            AssertError.Tolerance(y_expect, y_actual, 1e-7f, 1e-5f, $"mismatch value {inchannels},{outchannels},{scale},{inwidth}");
        }
    }
}
