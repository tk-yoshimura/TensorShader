using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;
using TensorShader;
using TensorShader.Operators.Connection1D;
using TensorShaderCudaBackend.API;

namespace TensorShaderTest.Operators.Connection1D {
    [TestClass]
    public class SpaceToChannelTest {
        [TestMethod]
        public void ExecuteTest() {
            float max_err = 0;

            foreach (int batch in new int[] { 1, 2 }) {
                foreach (int inchannels in new int[] { 3, 5 }) {
                    foreach (int scale in new int[] { 2, 3, 4 }) {
                        foreach (int outwidth in new int[] { 5, 7, 11 }) {
                            int inwidth = outwidth * scale, outchannels = inchannels * scale;

                            float[] xval = (new float[inwidth * inchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();

                            Map1D x = new(inchannels, inwidth, batch, xval);

                            Map1D y = Reference(x, scale);

                            OverflowCheckedTensor x_tensor = new(Shape.Map1D(inchannels, inwidth, batch), xval);
                            OverflowCheckedTensor y_tensor = new(Shape.Map1D(outchannels, outwidth, batch));

                            SpaceToChannel ope = new(inwidth, inchannels, scale, batch);

                            ope.Execute(x_tensor, y_tensor);

                            float[] y_expect = y.ToArray();
                            float[] y_actual = y_tensor.State.Value;

                            CollectionAssert.AreEqual(xval, x_tensor.State.Value);

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
            int outwidth = inwidth / scale, outchannels = inchannels * scale;

            OverflowCheckedTensor x_tensor = new(Shape.Map1D(inchannels, inwidth));
            OverflowCheckedTensor y_tensor = new(Shape.Map1D(outchannels, outwidth));

            SpaceToChannel ope = new(inwidth, inchannels, scale);

            Cuda.Profiler.Initialize("../../../../profiler.nvsetting", "../../nvprofiles/space_to_channel_1d.nvvp");
            Cuda.Profiler.Start();

            ope.Execute(x_tensor, y_tensor);

            Cuda.Profiler.Stop();
        }

        public static Map1D Reference(Map1D x, int scale) {
            int inw = x.Width, inchannels = x.Channels, batch = x.Batch;
            if (inw % scale != 0) {
                throw new ArgumentException(null, nameof(scale));
            }

            int outw = inw / scale;
            int outchannels = inchannels * scale;

            Map1D y = new(outchannels, outw, batch);

            for (int th = 0; th < batch; th++) {
                for (int ox = 0; ox < outw; ox++) {
                    for (int kx = 0; kx < scale; kx++) {
                        for (int inch = 0; inch < inchannels; inch++) {
                            int outch = inch + kx * inchannels;

                            y[outch, ox, th] = x[inch, ox * scale + kx, th];

                        }
                    }
                }

            }

            return y;
        }

        [TestMethod]
        public void ReferenceTest() {
            int inchannels = 12, scale = 2, inwidth = 7;

            float[] xval = (new float[inwidth * inchannels]).Select((_, idx) => idx * 1e-3f).ToArray();

            Map1D x = new(inchannels, inwidth, 1, xval);

            Map1D y = Reference(ChannelToSpaceTest.Reference(x, scale), scale);

            CollectionAssert.AreEqual(x.ToArray(), y.ToArray());
        }
    }
}
