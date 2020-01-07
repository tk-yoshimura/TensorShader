using System;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using TensorShader.Operators.Connection2D;
using TensorShaderCudaBackend.API;

namespace TensorShaderTest.Operators.Connection2D {
    [TestClass]
    public class ChannelToSpaceTest {
        [TestMethod]
        public void ExecuteTest() {
            float max_err = 0;

            foreach (int batch in new int[] { 1, 2 }) {
                foreach (int outchannels in new int[] { 3, 5 }) {
                    foreach (int scale in new int[] { 2, 3, 4 }) {
                        foreach (int inheight in new int[] { 5, 7, 11 }) {
                            foreach (int inwidth in new int[] { 5, 7, 11 }) {
                                int outwidth = inwidth * scale, outheight = inheight * scale, inchannels = outchannels * scale * scale;

                                float[] xval = (new float[inwidth * inheight * inchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();

                                Map2D x = new Map2D(inchannels, inwidth, inheight, batch, xval);

                                Map2D y = Reference(x, scale);

                                OverflowCheckedTensor x_tensor = new OverflowCheckedTensor(Shape.Map2D(inchannels, inwidth, inheight, batch), xval);
                                OverflowCheckedTensor y_tensor = new OverflowCheckedTensor(Shape.Map2D(outchannels, outwidth, outheight, batch));

                                ChannelToSpace ope = new ChannelToSpace(inwidth, inheight, inchannels, scale, batch);

                                ope.Execute(x_tensor, y_tensor);

                                float[] y_expect = y.ToArray();
                                float[] y_actual = y_tensor.State;

                                CollectionAssert.AreEqual(xval, x_tensor.State);

                                AssertError.Tolerance(y_expect, y_actual, 1e-7f, 1e-5f, ref max_err, $"mismatch value {inchannels},{scale},{inwidth},{inheight},{batch}");

                                Console.WriteLine($"pass: {inchannels},{scale},{inwidth},{inheight},{batch}");

                            }
                        }
                    }
                }
            }

            Console.WriteLine($"maxerr:{max_err}");
        }

        [TestMethod]
        public void SpeedTest() {
            int inwidth = 512, inheight = 512, inchannels = 32, scale = 2;
            int outwidth = inwidth * scale, outheight = inheight * scale, outchannels = inchannels / (scale * scale);

            OverflowCheckedTensor x_tensor = new OverflowCheckedTensor(Shape.Map2D(inchannels, inwidth, inheight));
            OverflowCheckedTensor y_tensor = new OverflowCheckedTensor(Shape.Map2D(outchannels, outwidth, outheight));

            ChannelToSpace ope = new ChannelToSpace(inwidth, inheight, inchannels, scale);

            Cuda.Profiler.Initialize("../../../profiler.nvsetting", "../../nvprofiles/channel_to_space_2d.nvvp");
            Cuda.Profiler.Start();

            ope.Execute(x_tensor, y_tensor);

            Cuda.Profiler.Stop();
        }

        public static Map2D Reference(Map2D x, int scale) {
            int inchannels = x.Channels, batch = x.Batch;
            if (inchannels % (scale * scale) != 0) {
                throw new ArgumentException(nameof(scale));
            }

            int inw = x.Width, inh = x.Height, outw = inw * scale, outh = inh * scale;
            int outchannels = inchannels / (scale * scale);

            Map2D y = new Map2D(outchannels, outw, outh, batch);

            for (int th = 0; th < batch; th++) {
                for (int ix, iy = 0; iy < inh; iy++) {
                    for (ix = 0; ix < inw; ix++) {
                        for (int kx, ky = 0; ky < scale; ky++) {
                            for (kx = 0; kx < scale; kx++) {
                                for (int outch = 0; outch < outchannels; outch++) {
                                    int inch = outch + kx * outchannels + ky * outchannels * scale;

                                    y[outch, ix * scale + kx, iy * scale + ky, th] = x[inch, ix, iy, th];

                                }
                            }
                        }

                    }
                }

            }

            return y;
        }

        [TestMethod]
        public void ReferenceTest() {
            int inchannels = 12, scale = 2, inwidth = 7, inheight = 5;
            int outchannels = inchannels / (scale * scale), outwidth = inwidth * scale, outheight = inheight * scale;

            float[] xval = (new float[inwidth * inheight * inchannels]).Select((_, idx) => idx * 1e-3f).ToArray();

            Map2D x = new Map2D(inchannels, inwidth, inheight, 1, xval);

            Map2D y = Reference(x, scale);

            float[] y_expect = {
                0.000f,0.001f,0.002f,0.003f,0.004f,0.005f,
                0.012f,0.013f,0.014f,0.015f,0.016f,0.017f,
                0.024f,0.025f,0.026f,0.027f,0.028f,0.029f,
                0.036f,0.037f,0.038f,0.039f,0.040f,0.041f,
                0.048f,0.049f,0.050f,0.051f,0.052f,0.053f,
                0.060f,0.061f,0.062f,0.063f,0.064f,0.065f,
                0.072f,0.073f,0.074f,0.075f,0.076f,0.077f,
                0.006f,0.007f,0.008f,0.009f,0.010f,0.011f,
                0.018f,0.019f,0.020f,0.021f,0.022f,0.023f,
                0.030f,0.031f,0.032f,0.033f,0.034f,0.035f,
                0.042f,0.043f,0.044f,0.045f,0.046f,0.047f,
                0.054f,0.055f,0.056f,0.057f,0.058f,0.059f,
                0.066f,0.067f,0.068f,0.069f,0.070f,0.071f,
                0.078f,0.079f,0.080f,0.081f,0.082f,0.083f,
                0.084f,0.085f,0.086f,0.087f,0.088f,0.089f,
                0.096f,0.097f,0.098f,0.099f,0.100f,0.101f,
                0.108f,0.109f,0.110f,0.111f,0.112f,0.113f,
                0.120f,0.121f,0.122f,0.123f,0.124f,0.125f,
                0.132f,0.133f,0.134f,0.135f,0.136f,0.137f,
                0.144f,0.145f,0.146f,0.147f,0.148f,0.149f,
                0.156f,0.157f,0.158f,0.159f,0.160f,0.161f,
                0.090f,0.091f,0.092f,0.093f,0.094f,0.095f,
                0.102f,0.103f,0.104f,0.105f,0.106f,0.107f,
                0.114f,0.115f,0.116f,0.117f,0.118f,0.119f,
                0.126f,0.127f,0.128f,0.129f,0.130f,0.131f,
                0.138f,0.139f,0.140f,0.141f,0.142f,0.143f,
                0.150f,0.151f,0.152f,0.153f,0.154f,0.155f,
                0.162f,0.163f,0.164f,0.165f,0.166f,0.167f,
                0.168f,0.169f,0.170f,0.171f,0.172f,0.173f,
                0.180f,0.181f,0.182f,0.183f,0.184f,0.185f,
                0.192f,0.193f,0.194f,0.195f,0.196f,0.197f,
                0.204f,0.205f,0.206f,0.207f,0.208f,0.209f,
                0.216f,0.217f,0.218f,0.219f,0.220f,0.221f,
                0.228f,0.229f,0.230f,0.231f,0.232f,0.233f,
                0.240f,0.241f,0.242f,0.243f,0.244f,0.245f,
                0.174f,0.175f,0.176f,0.177f,0.178f,0.179f,
                0.186f,0.187f,0.188f,0.189f,0.190f,0.191f,
                0.198f,0.199f,0.200f,0.201f,0.202f,0.203f,
                0.210f,0.211f,0.212f,0.213f,0.214f,0.215f,
                0.222f,0.223f,0.224f,0.225f,0.226f,0.227f,
                0.234f,0.235f,0.236f,0.237f,0.238f,0.239f,
                0.246f,0.247f,0.248f,0.249f,0.250f,0.251f,
                0.252f,0.253f,0.254f,0.255f,0.256f,0.257f,
                0.264f,0.265f,0.266f,0.267f,0.268f,0.269f,
                0.276f,0.277f,0.278f,0.279f,0.280f,0.281f,
                0.288f,0.289f,0.290f,0.291f,0.292f,0.293f,
                0.300f,0.301f,0.302f,0.303f,0.304f,0.305f,
                0.312f,0.313f,0.314f,0.315f,0.316f,0.317f,
                0.324f,0.325f,0.326f,0.327f,0.328f,0.329f,
                0.258f,0.259f,0.260f,0.261f,0.262f,0.263f,
                0.270f,0.271f,0.272f,0.273f,0.274f,0.275f,
                0.282f,0.283f,0.284f,0.285f,0.286f,0.287f,
                0.294f,0.295f,0.296f,0.297f,0.298f,0.299f,
                0.306f,0.307f,0.308f,0.309f,0.310f,0.311f,
                0.318f,0.319f,0.320f,0.321f,0.322f,0.323f,
                0.330f,0.331f,0.332f,0.333f,0.334f,0.335f,
                0.336f,0.337f,0.338f,0.339f,0.340f,0.341f,
                0.348f,0.349f,0.350f,0.351f,0.352f,0.353f,
                0.360f,0.361f,0.362f,0.363f,0.364f,0.365f,
                0.372f,0.373f,0.374f,0.375f,0.376f,0.377f,
                0.384f,0.385f,0.386f,0.387f,0.388f,0.389f,
                0.396f,0.397f,0.398f,0.399f,0.400f,0.401f,
                0.408f,0.409f,0.410f,0.411f,0.412f,0.413f,
                0.342f,0.343f,0.344f,0.345f,0.346f,0.347f,
                0.354f,0.355f,0.356f,0.357f,0.358f,0.359f,
                0.366f,0.367f,0.368f,0.369f,0.370f,0.371f,
                0.378f,0.379f,0.380f,0.381f,0.382f,0.383f,
                0.390f,0.391f,0.392f,0.393f,0.394f,0.395f,
                0.402f,0.403f,0.404f,0.405f,0.406f,0.407f,
                0.414f,0.415f,0.416f,0.417f,0.418f,0.419f,
            };

            float[] y_actual = y.ToArray();

            AssertError.Tolerance(y_expect, y_actual, 1e-7f, 1e-5f, $"mismatch value {inchannels},{outchannels},{scale},{inwidth},{inheight}");
        }
    }
}
