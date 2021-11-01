using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;
using TensorShader;
using TensorShader.Operators.Connection3D;
using TensorShaderCudaBackend.API;

namespace TensorShaderTest.Operators.Connection3D {
    [TestClass]
    public class ChannelToSpaceTest {
        [TestMethod]
        public void ExecuteTest() {
            float max_err = 0;

            foreach (int batch in new int[] { 1, 2 }) {
                foreach (int outchannels in new int[] { 3, 5 }) {
                    foreach (int scale in new int[] { 2, 3, 4 }) {
                        foreach (int indepth in new int[] { 5, 7, 11 }) {
                            foreach (int inheight in new int[] { 5, 7, 11 }) {
                                foreach (int inwidth in new int[] { 5, 7, 11 }) {
                                    int outwidth = inwidth * scale, outheight = inheight * scale, outdepth = indepth * scale, inchannels = outchannels * scale * scale * scale;

                                    float[] xval = (new float[inwidth * inheight * inchannels * indepth * batch]).Select((_, idx) => idx * 1e-3f).ToArray();

                                    Map3D x = new(inchannels, inwidth, inheight, indepth, batch, xval);

                                    Map3D y = Reference(x, scale);

                                    OverflowCheckedTensor x_tensor = new(Shape.Map3D(inchannels, inwidth, inheight, indepth, batch), xval);
                                    OverflowCheckedTensor y_tensor = new(Shape.Map3D(outchannels, outwidth, outheight, outdepth, batch));

                                    ChannelToSpace ope = new(inwidth, inheight, indepth, inchannels, scale, batch);

                                    ope.Execute(x_tensor, y_tensor);

                                    float[] y_expect = y.ToArray();
                                    float[] y_actual = y_tensor.State.Value;

                                    CollectionAssert.AreEqual(xval, x_tensor.State.Value);

                                    AssertError.Tolerance(y_expect, y_actual, 1e-7f, 1e-5f, ref max_err, $"mismatch value {inchannels},{outchannels},{scale},{inwidth},{inheight},{indepth},{batch}");

                                    Console.WriteLine($"pass: {inchannels},{outchannels},{scale},{inwidth},{inheight},{indepth},{batch}");

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
            int inwidth = 64, inheight = 64, indepth = 64, inchannels = 32, scale = 2;
            int outwidth = inwidth * scale, outheight = inheight * scale, outdepth = indepth * scale, outchannels = inchannels / (scale * scale * scale);

            OverflowCheckedTensor x_tensor = new(Shape.Map3D(inchannels, inwidth, inheight, indepth));
            OverflowCheckedTensor y_tensor = new(Shape.Map3D(outchannels, outwidth, outheight, outdepth));

            ChannelToSpace ope = new(inwidth, inheight, indepth, inchannels, scale);

            Cuda.Profiler.Initialize("../../../../profiler.nvsetting", "../../nvprofiles/channel_to_space_3d.nvvp");
            Cuda.Profiler.Start();

            ope.Execute(x_tensor, y_tensor);

            Cuda.Profiler.Stop();
        }

        public static Map3D Reference(Map3D x, int scale) {
            int inchannels = x.Channels, batch = x.Batch;
            if (inchannels % (scale * scale * scale) != 0) {
                throw new ArgumentException(null, nameof(scale));
            }

            int inw = x.Width, inh = x.Height, ind = x.Depth, outw = inw * scale, outh = inh * scale, outd = ind * scale;
            int outchannels = inchannels / (scale * scale * scale);

            Map3D y = new(outchannels, outw, outh, outd, batch);

            for (int th = 0; th < batch; th++) {
                for (int ix, iy, iz = 0; iz < ind; iz++) {
                    for (iy = 0; iy < inh; iy++) {
                        for (ix = 0; ix < inw; ix++) {
                            for (int kx, ky, kz = 0; kz < scale; kz++) {
                                for (ky = 0; ky < scale; ky++) {
                                    for (kx = 0; kx < scale; kx++) {
                                        for (int outch = 0; outch < outchannels; outch++) {
                                            int inch = outch + (kx + ky * scale + kz * scale * scale) * outchannels;

                                            y[outch, ix * scale + kx, iy * scale + ky, iz * scale + kz, th] = x[inch, ix, iy, iz, th];

                                        }
                                    }
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
            int inchannels = 32, scale = 2, inwidth = 7, inheight = 5, indepth = 3;
            int outchannels = inchannels / (scale * scale * scale), outwidth = inwidth * scale, outheight = inheight * scale, outdepth = indepth * scale;

            float[] xval = (new float[inwidth * inheight * indepth * inchannels]).Select((_, idx) => idx * 1e-3f).ToArray();

            Map3D x = new(inchannels, inwidth, inheight, indepth, 1, xval);

            Map3D y = Reference(x, scale);

            float[] y_expect = {
                0.000f, 0.001f, 0.002f, 0.003f, 0.004f, 0.005f, 0.006f, 0.007f, 0.032f, 0.033f, 0.034f, 0.035f, 0.036f, 0.037f, 0.038f, 0.039f, 0.064f, 0.065f, 0.066f, 0.067f, 0.068f, 0.069f, 0.070f, 0.071f, 0.096f,
                0.097f, 0.098f, 0.099f, 0.100f, 0.101f, 0.102f, 0.103f, 0.128f, 0.129f, 0.130f, 0.131f, 0.132f, 0.133f, 0.134f, 0.135f, 0.160f, 0.161f, 0.162f, 0.163f, 0.164f, 0.165f, 0.166f, 0.167f, 0.192f, 0.193f,
                0.194f, 0.195f, 0.196f, 0.197f, 0.198f, 0.199f, 0.008f, 0.009f, 0.010f, 0.011f, 0.012f, 0.013f, 0.014f, 0.015f, 0.040f, 0.041f, 0.042f, 0.043f, 0.044f, 0.045f, 0.046f, 0.047f, 0.072f, 0.073f, 0.074f,
                0.075f, 0.076f, 0.077f, 0.078f, 0.079f, 0.104f, 0.105f, 0.106f, 0.107f, 0.108f, 0.109f, 0.110f, 0.111f, 0.136f, 0.137f, 0.138f, 0.139f, 0.140f, 0.141f, 0.142f, 0.143f, 0.168f, 0.169f, 0.170f, 0.171f,
                0.172f, 0.173f, 0.174f, 0.175f, 0.200f, 0.201f, 0.202f, 0.203f, 0.204f, 0.205f, 0.206f, 0.207f, 0.224f, 0.225f, 0.226f, 0.227f, 0.228f, 0.229f, 0.230f, 0.231f, 0.256f, 0.257f, 0.258f, 0.259f, 0.260f,
                0.261f, 0.262f, 0.263f, 0.288f, 0.289f, 0.290f, 0.291f, 0.292f, 0.293f, 0.294f, 0.295f, 0.320f, 0.321f, 0.322f, 0.323f, 0.324f, 0.325f, 0.326f, 0.327f, 0.352f, 0.353f, 0.354f, 0.355f, 0.356f, 0.357f,
                0.358f, 0.359f, 0.384f, 0.385f, 0.386f, 0.387f, 0.388f, 0.389f, 0.390f, 0.391f, 0.416f, 0.417f, 0.418f, 0.419f, 0.420f, 0.421f, 0.422f, 0.423f, 0.232f, 0.233f, 0.234f, 0.235f, 0.236f, 0.237f, 0.238f,
                0.239f, 0.264f, 0.265f, 0.266f, 0.267f, 0.268f, 0.269f, 0.270f, 0.271f, 0.296f, 0.297f, 0.298f, 0.299f, 0.300f, 0.301f, 0.302f, 0.303f, 0.328f, 0.329f, 0.330f, 0.331f, 0.332f, 0.333f, 0.334f, 0.335f,
                0.360f, 0.361f, 0.362f, 0.363f, 0.364f, 0.365f, 0.366f, 0.367f, 0.392f, 0.393f, 0.394f, 0.395f, 0.396f, 0.397f, 0.398f, 0.399f, 0.424f, 0.425f, 0.426f, 0.427f, 0.428f, 0.429f, 0.430f, 0.431f, 0.448f,
                0.449f, 0.450f, 0.451f, 0.452f, 0.453f, 0.454f, 0.455f, 0.480f, 0.481f, 0.482f, 0.483f, 0.484f, 0.485f, 0.486f, 0.487f, 0.512f, 0.513f, 0.514f, 0.515f, 0.516f, 0.517f, 0.518f, 0.519f, 0.544f, 0.545f,
                0.546f, 0.547f, 0.548f, 0.549f, 0.550f, 0.551f, 0.576f, 0.577f, 0.578f, 0.579f, 0.580f, 0.581f, 0.582f, 0.583f, 0.608f, 0.609f, 0.610f, 0.611f, 0.612f, 0.613f, 0.614f, 0.615f, 0.640f, 0.641f, 0.642f,
                0.643f, 0.644f, 0.645f, 0.646f, 0.647f, 0.456f, 0.457f, 0.458f, 0.459f, 0.460f, 0.461f, 0.462f, 0.463f, 0.488f, 0.489f, 0.490f, 0.491f, 0.492f, 0.493f, 0.494f, 0.495f, 0.520f, 0.521f, 0.522f, 0.523f,
                0.524f, 0.525f, 0.526f, 0.527f, 0.552f, 0.553f, 0.554f, 0.555f, 0.556f, 0.557f, 0.558f, 0.559f, 0.584f, 0.585f, 0.586f, 0.587f, 0.588f, 0.589f, 0.590f, 0.591f, 0.616f, 0.617f, 0.618f, 0.619f, 0.620f,
                0.621f, 0.622f, 0.623f, 0.648f, 0.649f, 0.650f, 0.651f, 0.652f, 0.653f, 0.654f, 0.655f, 0.672f, 0.673f, 0.674f, 0.675f, 0.676f, 0.677f, 0.678f, 0.679f, 0.704f, 0.705f, 0.706f, 0.707f, 0.708f, 0.709f,
                0.710f, 0.711f, 0.736f, 0.737f, 0.738f, 0.739f, 0.740f, 0.741f, 0.742f, 0.743f, 0.768f, 0.769f, 0.770f, 0.771f, 0.772f, 0.773f, 0.774f, 0.775f, 0.800f, 0.801f, 0.802f, 0.803f, 0.804f, 0.805f, 0.806f,
                0.807f, 0.832f, 0.833f, 0.834f, 0.835f, 0.836f, 0.837f, 0.838f, 0.839f, 0.864f, 0.865f, 0.866f, 0.867f, 0.868f, 0.869f, 0.870f, 0.871f, 0.680f, 0.681f, 0.682f, 0.683f, 0.684f, 0.685f, 0.686f, 0.687f,
                0.712f, 0.713f, 0.714f, 0.715f, 0.716f, 0.717f, 0.718f, 0.719f, 0.744f, 0.745f, 0.746f, 0.747f, 0.748f, 0.749f, 0.750f, 0.751f, 0.776f, 0.777f, 0.778f, 0.779f, 0.780f, 0.781f, 0.782f, 0.783f, 0.808f,
                0.809f, 0.810f, 0.811f, 0.812f, 0.813f, 0.814f, 0.815f, 0.840f, 0.841f, 0.842f, 0.843f, 0.844f, 0.845f, 0.846f, 0.847f, 0.872f, 0.873f, 0.874f, 0.875f, 0.876f, 0.877f, 0.878f, 0.879f, 0.896f, 0.897f,
                0.898f, 0.899f, 0.900f, 0.901f, 0.902f, 0.903f, 0.928f, 0.929f, 0.930f, 0.931f, 0.932f, 0.933f, 0.934f, 0.935f, 0.960f, 0.961f, 0.962f, 0.963f, 0.964f, 0.965f, 0.966f, 0.967f, 0.992f, 0.993f, 0.994f,
                0.995f, 0.996f, 0.997f, 0.998f, 0.999f, 1.024f, 1.025f, 1.026f, 1.027f, 1.028f, 1.029f, 1.030f, 1.031f, 1.056f, 1.057f, 1.058f, 1.059f, 1.060f, 1.061f, 1.062f, 1.063f, 1.088f, 1.089f, 1.090f, 1.091f,
                1.092f, 1.093f, 1.094f, 1.095f, 0.904f, 0.905f, 0.906f, 0.907f, 0.908f, 0.909f, 0.910f, 0.911f, 0.936f, 0.937f, 0.938f, 0.939f, 0.940f, 0.941f, 0.942f, 0.943f, 0.968f, 0.969f, 0.970f, 0.971f, 0.972f,
                0.973f, 0.974f, 0.975f, 1.000f, 1.001f, 1.002f, 1.003f, 1.004f, 1.005f, 1.006f, 1.007f, 1.032f, 1.033f, 1.034f, 1.035f, 1.036f, 1.037f, 1.038f, 1.039f, 1.064f, 1.065f, 1.066f, 1.067f, 1.068f, 1.069f,
                1.070f, 1.071f, 1.096f, 1.097f, 1.098f, 1.099f, 1.100f, 1.101f, 1.102f, 1.103f, 0.016f, 0.017f, 0.018f, 0.019f, 0.020f, 0.021f, 0.022f, 0.023f, 0.048f, 0.049f, 0.050f, 0.051f, 0.052f, 0.053f, 0.054f,
                0.055f, 0.080f, 0.081f, 0.082f, 0.083f, 0.084f, 0.085f, 0.086f, 0.087f, 0.112f, 0.113f, 0.114f, 0.115f, 0.116f, 0.117f, 0.118f, 0.119f, 0.144f, 0.145f, 0.146f, 0.147f, 0.148f, 0.149f, 0.150f, 0.151f,
                0.176f, 0.177f, 0.178f, 0.179f, 0.180f, 0.181f, 0.182f, 0.183f, 0.208f, 0.209f, 0.210f, 0.211f, 0.212f, 0.213f, 0.214f, 0.215f, 0.024f, 0.025f, 0.026f, 0.027f, 0.028f, 0.029f, 0.030f, 0.031f, 0.056f,
                0.057f, 0.058f, 0.059f, 0.060f, 0.061f, 0.062f, 0.063f, 0.088f, 0.089f, 0.090f, 0.091f, 0.092f, 0.093f, 0.094f, 0.095f, 0.120f, 0.121f, 0.122f, 0.123f, 0.124f, 0.125f, 0.126f, 0.127f, 0.152f, 0.153f,
                0.154f, 0.155f, 0.156f, 0.157f, 0.158f, 0.159f, 0.184f, 0.185f, 0.186f, 0.187f, 0.188f, 0.189f, 0.190f, 0.191f, 0.216f, 0.217f, 0.218f, 0.219f, 0.220f, 0.221f, 0.222f, 0.223f, 0.240f, 0.241f, 0.242f,
                0.243f, 0.244f, 0.245f, 0.246f, 0.247f, 0.272f, 0.273f, 0.274f, 0.275f, 0.276f, 0.277f, 0.278f, 0.279f, 0.304f, 0.305f, 0.306f, 0.307f, 0.308f, 0.309f, 0.310f, 0.311f, 0.336f, 0.337f, 0.338f, 0.339f,
                0.340f, 0.341f, 0.342f, 0.343f, 0.368f, 0.369f, 0.370f, 0.371f, 0.372f, 0.373f, 0.374f, 0.375f, 0.400f, 0.401f, 0.402f, 0.403f, 0.404f, 0.405f, 0.406f, 0.407f, 0.432f, 0.433f, 0.434f, 0.435f, 0.436f,
                0.437f, 0.438f, 0.439f, 0.248f, 0.249f, 0.250f, 0.251f, 0.252f, 0.253f, 0.254f, 0.255f, 0.280f, 0.281f, 0.282f, 0.283f, 0.284f, 0.285f, 0.286f, 0.287f, 0.312f, 0.313f, 0.314f, 0.315f, 0.316f, 0.317f,
                0.318f, 0.319f, 0.344f, 0.345f, 0.346f, 0.347f, 0.348f, 0.349f, 0.350f, 0.351f, 0.376f, 0.377f, 0.378f, 0.379f, 0.380f, 0.381f, 0.382f, 0.383f, 0.408f, 0.409f, 0.410f, 0.411f, 0.412f, 0.413f, 0.414f,
                0.415f, 0.440f, 0.441f, 0.442f, 0.443f, 0.444f, 0.445f, 0.446f, 0.447f, 0.464f, 0.465f, 0.466f, 0.467f, 0.468f, 0.469f, 0.470f, 0.471f, 0.496f, 0.497f, 0.498f, 0.499f, 0.500f, 0.501f, 0.502f, 0.503f,
                0.528f, 0.529f, 0.530f, 0.531f, 0.532f, 0.533f, 0.534f, 0.535f, 0.560f, 0.561f, 0.562f, 0.563f, 0.564f, 0.565f, 0.566f, 0.567f, 0.592f, 0.593f, 0.594f, 0.595f, 0.596f, 0.597f, 0.598f, 0.599f, 0.624f,
                0.625f, 0.626f, 0.627f, 0.628f, 0.629f, 0.630f, 0.631f, 0.656f, 0.657f, 0.658f, 0.659f, 0.660f, 0.661f, 0.662f, 0.663f, 0.472f, 0.473f, 0.474f, 0.475f, 0.476f, 0.477f, 0.478f, 0.479f, 0.504f, 0.505f,
                0.506f, 0.507f, 0.508f, 0.509f, 0.510f, 0.511f, 0.536f, 0.537f, 0.538f, 0.539f, 0.540f, 0.541f, 0.542f, 0.543f, 0.568f, 0.569f, 0.570f, 0.571f, 0.572f, 0.573f, 0.574f, 0.575f, 0.600f, 0.601f, 0.602f,
                0.603f, 0.604f, 0.605f, 0.606f, 0.607f, 0.632f, 0.633f, 0.634f, 0.635f, 0.636f, 0.637f, 0.638f, 0.639f, 0.664f, 0.665f, 0.666f, 0.667f, 0.668f, 0.669f, 0.670f, 0.671f, 0.688f, 0.689f, 0.690f, 0.691f,
                0.692f, 0.693f, 0.694f, 0.695f, 0.720f, 0.721f, 0.722f, 0.723f, 0.724f, 0.725f, 0.726f, 0.727f, 0.752f, 0.753f, 0.754f, 0.755f, 0.756f, 0.757f, 0.758f, 0.759f, 0.784f, 0.785f, 0.786f, 0.787f, 0.788f,
                0.789f, 0.790f, 0.791f, 0.816f, 0.817f, 0.818f, 0.819f, 0.820f, 0.821f, 0.822f, 0.823f, 0.848f, 0.849f, 0.850f, 0.851f, 0.852f, 0.853f, 0.854f, 0.855f, 0.880f, 0.881f, 0.882f, 0.883f, 0.884f, 0.885f,
                0.886f, 0.887f, 0.696f, 0.697f, 0.698f, 0.699f, 0.700f, 0.701f, 0.702f, 0.703f, 0.728f, 0.729f, 0.730f, 0.731f, 0.732f, 0.733f, 0.734f, 0.735f, 0.760f, 0.761f, 0.762f, 0.763f, 0.764f, 0.765f, 0.766f,
                0.767f, 0.792f, 0.793f, 0.794f, 0.795f, 0.796f, 0.797f, 0.798f, 0.799f, 0.824f, 0.825f, 0.826f, 0.827f, 0.828f, 0.829f, 0.830f, 0.831f, 0.856f, 0.857f, 0.858f, 0.859f, 0.860f, 0.861f, 0.862f, 0.863f,
                0.888f, 0.889f, 0.890f, 0.891f, 0.892f, 0.893f, 0.894f, 0.895f, 0.912f, 0.913f, 0.914f, 0.915f, 0.916f, 0.917f, 0.918f, 0.919f, 0.944f, 0.945f, 0.946f, 0.947f, 0.948f, 0.949f, 0.950f, 0.951f, 0.976f,
                0.977f, 0.978f, 0.979f, 0.980f, 0.981f, 0.982f, 0.983f, 1.008f, 1.009f, 1.010f, 1.011f, 1.012f, 1.013f, 1.014f, 1.015f, 1.040f, 1.041f, 1.042f, 1.043f, 1.044f, 1.045f, 1.046f, 1.047f, 1.072f, 1.073f,
                1.074f, 1.075f, 1.076f, 1.077f, 1.078f, 1.079f, 1.104f, 1.105f, 1.106f, 1.107f, 1.108f, 1.109f, 1.110f, 1.111f, 0.920f, 0.921f, 0.922f, 0.923f, 0.924f, 0.925f, 0.926f, 0.927f, 0.952f, 0.953f, 0.954f,
                0.955f, 0.956f, 0.957f, 0.958f, 0.959f, 0.984f, 0.985f, 0.986f, 0.987f, 0.988f, 0.989f, 0.990f, 0.991f, 1.016f, 1.017f, 1.018f, 1.019f, 1.020f, 1.021f, 1.022f, 1.023f, 1.048f, 1.049f, 1.050f, 1.051f,
                1.052f, 1.053f, 1.054f, 1.055f, 1.080f, 1.081f, 1.082f, 1.083f, 1.084f, 1.085f, 1.086f, 1.087f, 1.112f, 1.113f, 1.114f, 1.115f, 1.116f, 1.117f, 1.118f, 1.119f, 1.120f, 1.121f, 1.122f, 1.123f, 1.124f,
                1.125f, 1.126f, 1.127f, 1.152f, 1.153f, 1.154f, 1.155f, 1.156f, 1.157f, 1.158f, 1.159f, 1.184f, 1.185f, 1.186f, 1.187f, 1.188f, 1.189f, 1.190f, 1.191f, 1.216f, 1.217f, 1.218f, 1.219f, 1.220f, 1.221f,
                1.222f, 1.223f, 1.248f, 1.249f, 1.250f, 1.251f, 1.252f, 1.253f, 1.254f, 1.255f, 1.280f, 1.281f, 1.282f, 1.283f, 1.284f, 1.285f, 1.286f, 1.287f, 1.312f, 1.313f, 1.314f, 1.315f, 1.316f, 1.317f, 1.318f,
                1.319f, 1.128f, 1.129f, 1.130f, 1.131f, 1.132f, 1.133f, 1.134f, 1.135f, 1.160f, 1.161f, 1.162f, 1.163f, 1.164f, 1.165f, 1.166f, 1.167f, 1.192f, 1.193f, 1.194f, 1.195f, 1.196f, 1.197f, 1.198f, 1.199f,
                1.224f, 1.225f, 1.226f, 1.227f, 1.228f, 1.229f, 1.230f, 1.231f, 1.256f, 1.257f, 1.258f, 1.259f, 1.260f, 1.261f, 1.262f, 1.263f, 1.288f, 1.289f, 1.290f, 1.291f, 1.292f, 1.293f, 1.294f, 1.295f, 1.320f,
                1.321f, 1.322f, 1.323f, 1.324f, 1.325f, 1.326f, 1.327f, 1.344f, 1.345f, 1.346f, 1.347f, 1.348f, 1.349f, 1.350f, 1.351f, 1.376f, 1.377f, 1.378f, 1.379f, 1.380f, 1.381f, 1.382f, 1.383f, 1.408f, 1.409f,
                1.410f, 1.411f, 1.412f, 1.413f, 1.414f, 1.415f, 1.440f, 1.441f, 1.442f, 1.443f, 1.444f, 1.445f, 1.446f, 1.447f, 1.472f, 1.473f, 1.474f, 1.475f, 1.476f, 1.477f, 1.478f, 1.479f, 1.504f, 1.505f, 1.506f,
                1.507f, 1.508f, 1.509f, 1.510f, 1.511f, 1.536f, 1.537f, 1.538f, 1.539f, 1.540f, 1.541f, 1.542f, 1.543f, 1.352f, 1.353f, 1.354f, 1.355f, 1.356f, 1.357f, 1.358f, 1.359f, 1.384f, 1.385f, 1.386f, 1.387f,
                1.388f, 1.389f, 1.390f, 1.391f, 1.416f, 1.417f, 1.418f, 1.419f, 1.420f, 1.421f, 1.422f, 1.423f, 1.448f, 1.449f, 1.450f, 1.451f, 1.452f, 1.453f, 1.454f, 1.455f, 1.480f, 1.481f, 1.482f, 1.483f, 1.484f,
                1.485f, 1.486f, 1.487f, 1.512f, 1.513f, 1.514f, 1.515f, 1.516f, 1.517f, 1.518f, 1.519f, 1.544f, 1.545f, 1.546f, 1.547f, 1.548f, 1.549f, 1.550f, 1.551f, 1.568f, 1.569f, 1.570f, 1.571f, 1.572f, 1.573f,
                1.574f, 1.575f, 1.600f, 1.601f, 1.602f, 1.603f, 1.604f, 1.605f, 1.606f, 1.607f, 1.632f, 1.633f, 1.634f, 1.635f, 1.636f, 1.637f, 1.638f, 1.639f, 1.664f, 1.665f, 1.666f, 1.667f, 1.668f, 1.669f, 1.670f,
                1.671f, 1.696f, 1.697f, 1.698f, 1.699f, 1.700f, 1.701f, 1.702f, 1.703f, 1.728f, 1.729f, 1.730f, 1.731f, 1.732f, 1.733f, 1.734f, 1.735f, 1.760f, 1.761f, 1.762f, 1.763f, 1.764f, 1.765f, 1.766f, 1.767f,
                1.576f, 1.577f, 1.578f, 1.579f, 1.580f, 1.581f, 1.582f, 1.583f, 1.608f, 1.609f, 1.610f, 1.611f, 1.612f, 1.613f, 1.614f, 1.615f, 1.640f, 1.641f, 1.642f, 1.643f, 1.644f, 1.645f, 1.646f, 1.647f, 1.672f,
                1.673f, 1.674f, 1.675f, 1.676f, 1.677f, 1.678f, 1.679f, 1.704f, 1.705f, 1.706f, 1.707f, 1.708f, 1.709f, 1.710f, 1.711f, 1.736f, 1.737f, 1.738f, 1.739f, 1.740f, 1.741f, 1.742f, 1.743f, 1.768f, 1.769f,
                1.770f, 1.771f, 1.772f, 1.773f, 1.774f, 1.775f, 1.792f, 1.793f, 1.794f, 1.795f, 1.796f, 1.797f, 1.798f, 1.799f, 1.824f, 1.825f, 1.826f, 1.827f, 1.828f, 1.829f, 1.830f, 1.831f, 1.856f, 1.857f, 1.858f,
                1.859f, 1.860f, 1.861f, 1.862f, 1.863f, 1.888f, 1.889f, 1.890f, 1.891f, 1.892f, 1.893f, 1.894f, 1.895f, 1.920f, 1.921f, 1.922f, 1.923f, 1.924f, 1.925f, 1.926f, 1.927f, 1.952f, 1.953f, 1.954f, 1.955f,
                1.956f, 1.957f, 1.958f, 1.959f, 1.984f, 1.985f, 1.986f, 1.987f, 1.988f, 1.989f, 1.990f, 1.991f, 1.800f, 1.801f, 1.802f, 1.803f, 1.804f, 1.805f, 1.806f, 1.807f, 1.832f, 1.833f, 1.834f, 1.835f, 1.836f,
                1.837f, 1.838f, 1.839f, 1.864f, 1.865f, 1.866f, 1.867f, 1.868f, 1.869f, 1.870f, 1.871f, 1.896f, 1.897f, 1.898f, 1.899f, 1.900f, 1.901f, 1.902f, 1.903f, 1.928f, 1.929f, 1.930f, 1.931f, 1.932f, 1.933f,
                1.934f, 1.935f, 1.960f, 1.961f, 1.962f, 1.963f, 1.964f, 1.965f, 1.966f, 1.967f, 1.992f, 1.993f, 1.994f, 1.995f, 1.996f, 1.997f, 1.998f, 1.999f, 2.016f, 2.017f, 2.018f, 2.019f, 2.020f, 2.021f, 2.022f,
                2.023f, 2.048f, 2.049f, 2.050f, 2.051f, 2.052f, 2.053f, 2.054f, 2.055f, 2.080f, 2.081f, 2.082f, 2.083f, 2.084f, 2.085f, 2.086f, 2.087f, 2.112f, 2.113f, 2.114f, 2.115f, 2.116f, 2.117f, 2.118f, 2.119f,
                2.144f, 2.145f, 2.146f, 2.147f, 2.148f, 2.149f, 2.150f, 2.151f, 2.176f, 2.177f, 2.178f, 2.179f, 2.180f, 2.181f, 2.182f, 2.183f, 2.208f, 2.209f, 2.210f, 2.211f, 2.212f, 2.213f, 2.214f, 2.215f, 2.024f,
                2.025f, 2.026f, 2.027f, 2.028f, 2.029f, 2.030f, 2.031f, 2.056f, 2.057f, 2.058f, 2.059f, 2.060f, 2.061f, 2.062f, 2.063f, 2.088f, 2.089f, 2.090f, 2.091f, 2.092f, 2.093f, 2.094f, 2.095f, 2.120f, 2.121f,
                2.122f, 2.123f, 2.124f, 2.125f, 2.126f, 2.127f, 2.152f, 2.153f, 2.154f, 2.155f, 2.156f, 2.157f, 2.158f, 2.159f, 2.184f, 2.185f, 2.186f, 2.187f, 2.188f, 2.189f, 2.190f, 2.191f, 2.216f, 2.217f, 2.218f,
                2.219f, 2.220f, 2.221f, 2.222f, 2.223f, 1.136f, 1.137f, 1.138f, 1.139f, 1.140f, 1.141f, 1.142f, 1.143f, 1.168f, 1.169f, 1.170f, 1.171f, 1.172f, 1.173f, 1.174f, 1.175f, 1.200f, 1.201f, 1.202f, 1.203f,
                1.204f, 1.205f, 1.206f, 1.207f, 1.232f, 1.233f, 1.234f, 1.235f, 1.236f, 1.237f, 1.238f, 1.239f, 1.264f, 1.265f, 1.266f, 1.267f, 1.268f, 1.269f, 1.270f, 1.271f, 1.296f, 1.297f, 1.298f, 1.299f, 1.300f,
                1.301f, 1.302f, 1.303f, 1.328f, 1.329f, 1.330f, 1.331f, 1.332f, 1.333f, 1.334f, 1.335f, 1.144f, 1.145f, 1.146f, 1.147f, 1.148f, 1.149f, 1.150f, 1.151f, 1.176f, 1.177f, 1.178f, 1.179f, 1.180f, 1.181f,
                1.182f, 1.183f, 1.208f, 1.209f, 1.210f, 1.211f, 1.212f, 1.213f, 1.214f, 1.215f, 1.240f, 1.241f, 1.242f, 1.243f, 1.244f, 1.245f, 1.246f, 1.247f, 1.272f, 1.273f, 1.274f, 1.275f, 1.276f, 1.277f, 1.278f,
                1.279f, 1.304f, 1.305f, 1.306f, 1.307f, 1.308f, 1.309f, 1.310f, 1.311f, 1.336f, 1.337f, 1.338f, 1.339f, 1.340f, 1.341f, 1.342f, 1.343f, 1.360f, 1.361f, 1.362f, 1.363f, 1.364f, 1.365f, 1.366f, 1.367f,
                1.392f, 1.393f, 1.394f, 1.395f, 1.396f, 1.397f, 1.398f, 1.399f, 1.424f, 1.425f, 1.426f, 1.427f, 1.428f, 1.429f, 1.430f, 1.431f, 1.456f, 1.457f, 1.458f, 1.459f, 1.460f, 1.461f, 1.462f, 1.463f, 1.488f,
                1.489f, 1.490f, 1.491f, 1.492f, 1.493f, 1.494f, 1.495f, 1.520f, 1.521f, 1.522f, 1.523f, 1.524f, 1.525f, 1.526f, 1.527f, 1.552f, 1.553f, 1.554f, 1.555f, 1.556f, 1.557f, 1.558f, 1.559f, 1.368f, 1.369f,
                1.370f, 1.371f, 1.372f, 1.373f, 1.374f, 1.375f, 1.400f, 1.401f, 1.402f, 1.403f, 1.404f, 1.405f, 1.406f, 1.407f, 1.432f, 1.433f, 1.434f, 1.435f, 1.436f, 1.437f, 1.438f, 1.439f, 1.464f, 1.465f, 1.466f,
                1.467f, 1.468f, 1.469f, 1.470f, 1.471f, 1.496f, 1.497f, 1.498f, 1.499f, 1.500f, 1.501f, 1.502f, 1.503f, 1.528f, 1.529f, 1.530f, 1.531f, 1.532f, 1.533f, 1.534f, 1.535f, 1.560f, 1.561f, 1.562f, 1.563f,
                1.564f, 1.565f, 1.566f, 1.567f, 1.584f, 1.585f, 1.586f, 1.587f, 1.588f, 1.589f, 1.590f, 1.591f, 1.616f, 1.617f, 1.618f, 1.619f, 1.620f, 1.621f, 1.622f, 1.623f, 1.648f, 1.649f, 1.650f, 1.651f, 1.652f,
                1.653f, 1.654f, 1.655f, 1.680f, 1.681f, 1.682f, 1.683f, 1.684f, 1.685f, 1.686f, 1.687f, 1.712f, 1.713f, 1.714f, 1.715f, 1.716f, 1.717f, 1.718f, 1.719f, 1.744f, 1.745f, 1.746f, 1.747f, 1.748f, 1.749f,
                1.750f, 1.751f, 1.776f, 1.777f, 1.778f, 1.779f, 1.780f, 1.781f, 1.782f, 1.783f, 1.592f, 1.593f, 1.594f, 1.595f, 1.596f, 1.597f, 1.598f, 1.599f, 1.624f, 1.625f, 1.626f, 1.627f, 1.628f, 1.629f, 1.630f,
                1.631f, 1.656f, 1.657f, 1.658f, 1.659f, 1.660f, 1.661f, 1.662f, 1.663f, 1.688f, 1.689f, 1.690f, 1.691f, 1.692f, 1.693f, 1.694f, 1.695f, 1.720f, 1.721f, 1.722f, 1.723f, 1.724f, 1.725f, 1.726f, 1.727f,
                1.752f, 1.753f, 1.754f, 1.755f, 1.756f, 1.757f, 1.758f, 1.759f, 1.784f, 1.785f, 1.786f, 1.787f, 1.788f, 1.789f, 1.790f, 1.791f, 1.808f, 1.809f, 1.810f, 1.811f, 1.812f, 1.813f, 1.814f, 1.815f, 1.840f,
                1.841f, 1.842f, 1.843f, 1.844f, 1.845f, 1.846f, 1.847f, 1.872f, 1.873f, 1.874f, 1.875f, 1.876f, 1.877f, 1.878f, 1.879f, 1.904f, 1.905f, 1.906f, 1.907f, 1.908f, 1.909f, 1.910f, 1.911f, 1.936f, 1.937f,
                1.938f, 1.939f, 1.940f, 1.941f, 1.942f, 1.943f, 1.968f, 1.969f, 1.970f, 1.971f, 1.972f, 1.973f, 1.974f, 1.975f, 2.000f, 2.001f, 2.002f, 2.003f, 2.004f, 2.005f, 2.006f, 2.007f, 1.816f, 1.817f, 1.818f,
                1.819f, 1.820f, 1.821f, 1.822f, 1.823f, 1.848f, 1.849f, 1.850f, 1.851f, 1.852f, 1.853f, 1.854f, 1.855f, 1.880f, 1.881f, 1.882f, 1.883f, 1.884f, 1.885f, 1.886f, 1.887f, 1.912f, 1.913f, 1.914f, 1.915f,
                1.916f, 1.917f, 1.918f, 1.919f, 1.944f, 1.945f, 1.946f, 1.947f, 1.948f, 1.949f, 1.950f, 1.951f, 1.976f, 1.977f, 1.978f, 1.979f, 1.980f, 1.981f, 1.982f, 1.983f, 2.008f, 2.009f, 2.010f, 2.011f, 2.012f,
                2.013f, 2.014f, 2.015f, 2.032f, 2.033f, 2.034f, 2.035f, 2.036f, 2.037f, 2.038f, 2.039f, 2.064f, 2.065f, 2.066f, 2.067f, 2.068f, 2.069f, 2.070f, 2.071f, 2.096f, 2.097f, 2.098f, 2.099f, 2.100f, 2.101f,
                2.102f, 2.103f, 2.128f, 2.129f, 2.130f, 2.131f, 2.132f, 2.133f, 2.134f, 2.135f, 2.160f, 2.161f, 2.162f, 2.163f, 2.164f, 2.165f, 2.166f, 2.167f, 2.192f, 2.193f, 2.194f, 2.195f, 2.196f, 2.197f, 2.198f,
                2.199f, 2.224f, 2.225f, 2.226f, 2.227f, 2.228f, 2.229f, 2.230f, 2.231f, 2.040f, 2.041f, 2.042f, 2.043f, 2.044f, 2.045f, 2.046f, 2.047f, 2.072f, 2.073f, 2.074f, 2.075f, 2.076f, 2.077f, 2.078f, 2.079f,
                2.104f, 2.105f, 2.106f, 2.107f, 2.108f, 2.109f, 2.110f, 2.111f, 2.136f, 2.137f, 2.138f, 2.139f, 2.140f, 2.141f, 2.142f, 2.143f, 2.168f, 2.169f, 2.170f, 2.171f, 2.172f, 2.173f, 2.174f, 2.175f, 2.200f,
                2.201f, 2.202f, 2.203f, 2.204f, 2.205f, 2.206f, 2.207f, 2.232f, 2.233f, 2.234f, 2.235f, 2.236f, 2.237f, 2.238f, 2.239f, 2.240f, 2.241f, 2.242f, 2.243f, 2.244f, 2.245f, 2.246f, 2.247f, 2.272f, 2.273f,
                2.274f, 2.275f, 2.276f, 2.277f, 2.278f, 2.279f, 2.304f, 2.305f, 2.306f, 2.307f, 2.308f, 2.309f, 2.310f, 2.311f, 2.336f, 2.337f, 2.338f, 2.339f, 2.340f, 2.341f, 2.342f, 2.343f, 2.368f, 2.369f, 2.370f,
                2.371f, 2.372f, 2.373f, 2.374f, 2.375f, 2.400f, 2.401f, 2.402f, 2.403f, 2.404f, 2.405f, 2.406f, 2.407f, 2.432f, 2.433f, 2.434f, 2.435f, 2.436f, 2.437f, 2.438f, 2.439f, 2.248f, 2.249f, 2.250f, 2.251f,
                2.252f, 2.253f, 2.254f, 2.255f, 2.280f, 2.281f, 2.282f, 2.283f, 2.284f, 2.285f, 2.286f, 2.287f, 2.312f, 2.313f, 2.314f, 2.315f, 2.316f, 2.317f, 2.318f, 2.319f, 2.344f, 2.345f, 2.346f, 2.347f, 2.348f,
                2.349f, 2.350f, 2.351f, 2.376f, 2.377f, 2.378f, 2.379f, 2.380f, 2.381f, 2.382f, 2.383f, 2.408f, 2.409f, 2.410f, 2.411f, 2.412f, 2.413f, 2.414f, 2.415f, 2.440f, 2.441f, 2.442f, 2.443f, 2.444f, 2.445f,
                2.446f, 2.447f, 2.464f, 2.465f, 2.466f, 2.467f, 2.468f, 2.469f, 2.470f, 2.471f, 2.496f, 2.497f, 2.498f, 2.499f, 2.500f, 2.501f, 2.502f, 2.503f, 2.528f, 2.529f, 2.530f, 2.531f, 2.532f, 2.533f, 2.534f,
                2.535f, 2.560f, 2.561f, 2.562f, 2.563f, 2.564f, 2.565f, 2.566f, 2.567f, 2.592f, 2.593f, 2.594f, 2.595f, 2.596f, 2.597f, 2.598f, 2.599f, 2.624f, 2.625f, 2.626f, 2.627f, 2.628f, 2.629f, 2.630f, 2.631f,
                2.656f, 2.657f, 2.658f, 2.659f, 2.660f, 2.661f, 2.662f, 2.663f, 2.472f, 2.473f, 2.474f, 2.475f, 2.476f, 2.477f, 2.478f, 2.479f, 2.504f, 2.505f, 2.506f, 2.507f, 2.508f, 2.509f, 2.510f, 2.511f, 2.536f,
                2.537f, 2.538f, 2.539f, 2.540f, 2.541f, 2.542f, 2.543f, 2.568f, 2.569f, 2.570f, 2.571f, 2.572f, 2.573f, 2.574f, 2.575f, 2.600f, 2.601f, 2.602f, 2.603f, 2.604f, 2.605f, 2.606f, 2.607f, 2.632f, 2.633f,
                2.634f, 2.635f, 2.636f, 2.637f, 2.638f, 2.639f, 2.664f, 2.665f, 2.666f, 2.667f, 2.668f, 2.669f, 2.670f, 2.671f, 2.688f, 2.689f, 2.690f, 2.691f, 2.692f, 2.693f, 2.694f, 2.695f, 2.720f, 2.721f, 2.722f,
                2.723f, 2.724f, 2.725f, 2.726f, 2.727f, 2.752f, 2.753f, 2.754f, 2.755f, 2.756f, 2.757f, 2.758f, 2.759f, 2.784f, 2.785f, 2.786f, 2.787f, 2.788f, 2.789f, 2.790f, 2.791f, 2.816f, 2.817f, 2.818f, 2.819f,
                2.820f, 2.821f, 2.822f, 2.823f, 2.848f, 2.849f, 2.850f, 2.851f, 2.852f, 2.853f, 2.854f, 2.855f, 2.880f, 2.881f, 2.882f, 2.883f, 2.884f, 2.885f, 2.886f, 2.887f, 2.696f, 2.697f, 2.698f, 2.699f, 2.700f,
                2.701f, 2.702f, 2.703f, 2.728f, 2.729f, 2.730f, 2.731f, 2.732f, 2.733f, 2.734f, 2.735f, 2.760f, 2.761f, 2.762f, 2.763f, 2.764f, 2.765f, 2.766f, 2.767f, 2.792f, 2.793f, 2.794f, 2.795f, 2.796f, 2.797f,
                2.798f, 2.799f, 2.824f, 2.825f, 2.826f, 2.827f, 2.828f, 2.829f, 2.830f, 2.831f, 2.856f, 2.857f, 2.858f, 2.859f, 2.860f, 2.861f, 2.862f, 2.863f, 2.888f, 2.889f, 2.890f, 2.891f, 2.892f, 2.893f, 2.894f,
                2.895f, 2.912f, 2.913f, 2.914f, 2.915f, 2.916f, 2.917f, 2.918f, 2.919f, 2.944f, 2.945f, 2.946f, 2.947f, 2.948f, 2.949f, 2.950f, 2.951f, 2.976f, 2.977f, 2.978f, 2.979f, 2.980f, 2.981f, 2.982f, 2.983f,
                3.008f, 3.009f, 3.010f, 3.011f, 3.012f, 3.013f, 3.014f, 3.015f, 3.040f, 3.041f, 3.042f, 3.043f, 3.044f, 3.045f, 3.046f, 3.047f, 3.072f, 3.073f, 3.074f, 3.075f, 3.076f, 3.077f, 3.078f, 3.079f, 3.104f,
                3.105f, 3.106f, 3.107f, 3.108f, 3.109f, 3.110f, 3.111f, 2.920f, 2.921f, 2.922f, 2.923f, 2.924f, 2.925f, 2.926f, 2.927f, 2.952f, 2.953f, 2.954f, 2.955f, 2.956f, 2.957f, 2.958f, 2.959f, 2.984f, 2.985f,
                2.986f, 2.987f, 2.988f, 2.989f, 2.990f, 2.991f, 3.016f, 3.017f, 3.018f, 3.019f, 3.020f, 3.021f, 3.022f, 3.023f, 3.048f, 3.049f, 3.050f, 3.051f, 3.052f, 3.053f, 3.054f, 3.055f, 3.080f, 3.081f, 3.082f,
                3.083f, 3.084f, 3.085f, 3.086f, 3.087f, 3.112f, 3.113f, 3.114f, 3.115f, 3.116f, 3.117f, 3.118f, 3.119f, 3.136f, 3.137f, 3.138f, 3.139f, 3.140f, 3.141f, 3.142f, 3.143f, 3.168f, 3.169f, 3.170f, 3.171f,
                3.172f, 3.173f, 3.174f, 3.175f, 3.200f, 3.201f, 3.202f, 3.203f, 3.204f, 3.205f, 3.206f, 3.207f, 3.232f, 3.233f, 3.234f, 3.235f, 3.236f, 3.237f, 3.238f, 3.239f, 3.264f, 3.265f, 3.266f, 3.267f, 3.268f,
                3.269f, 3.270f, 3.271f, 3.296f, 3.297f, 3.298f, 3.299f, 3.300f, 3.301f, 3.302f, 3.303f, 3.328f, 3.329f, 3.330f, 3.331f, 3.332f, 3.333f, 3.334f, 3.335f, 3.144f, 3.145f, 3.146f, 3.147f, 3.148f, 3.149f,
                3.150f, 3.151f, 3.176f, 3.177f, 3.178f, 3.179f, 3.180f, 3.181f, 3.182f, 3.183f, 3.208f, 3.209f, 3.210f, 3.211f, 3.212f, 3.213f, 3.214f, 3.215f, 3.240f, 3.241f, 3.242f, 3.243f, 3.244f, 3.245f, 3.246f,
                3.247f, 3.272f, 3.273f, 3.274f, 3.275f, 3.276f, 3.277f, 3.278f, 3.279f, 3.304f, 3.305f, 3.306f, 3.307f, 3.308f, 3.309f, 3.310f, 3.311f, 3.336f, 3.337f, 3.338f, 3.339f, 3.340f, 3.341f, 3.342f, 3.343f,
                2.256f, 2.257f, 2.258f, 2.259f, 2.260f, 2.261f, 2.262f, 2.263f, 2.288f, 2.289f, 2.290f, 2.291f, 2.292f, 2.293f, 2.294f, 2.295f, 2.320f, 2.321f, 2.322f, 2.323f, 2.324f, 2.325f, 2.326f, 2.327f, 2.352f,
                2.353f, 2.354f, 2.355f, 2.356f, 2.357f, 2.358f, 2.359f, 2.384f, 2.385f, 2.386f, 2.387f, 2.388f, 2.389f, 2.390f, 2.391f, 2.416f, 2.417f, 2.418f, 2.419f, 2.420f, 2.421f, 2.422f, 2.423f, 2.448f, 2.449f,
                2.450f, 2.451f, 2.452f, 2.453f, 2.454f, 2.455f, 2.264f, 2.265f, 2.266f, 2.267f, 2.268f, 2.269f, 2.270f, 2.271f, 2.296f, 2.297f, 2.298f, 2.299f, 2.300f, 2.301f, 2.302f, 2.303f, 2.328f, 2.329f, 2.330f,
                2.331f, 2.332f, 2.333f, 2.334f, 2.335f, 2.360f, 2.361f, 2.362f, 2.363f, 2.364f, 2.365f, 2.366f, 2.367f, 2.392f, 2.393f, 2.394f, 2.395f, 2.396f, 2.397f, 2.398f, 2.399f, 2.424f, 2.425f, 2.426f, 2.427f,
                2.428f, 2.429f, 2.430f, 2.431f, 2.456f, 2.457f, 2.458f, 2.459f, 2.460f, 2.461f, 2.462f, 2.463f, 2.480f, 2.481f, 2.482f, 2.483f, 2.484f, 2.485f, 2.486f, 2.487f, 2.512f, 2.513f, 2.514f, 2.515f, 2.516f,
                2.517f, 2.518f, 2.519f, 2.544f, 2.545f, 2.546f, 2.547f, 2.548f, 2.549f, 2.550f, 2.551f, 2.576f, 2.577f, 2.578f, 2.579f, 2.580f, 2.581f, 2.582f, 2.583f, 2.608f, 2.609f, 2.610f, 2.611f, 2.612f, 2.613f,
                2.614f, 2.615f, 2.640f, 2.641f, 2.642f, 2.643f, 2.644f, 2.645f, 2.646f, 2.647f, 2.672f, 2.673f, 2.674f, 2.675f, 2.676f, 2.677f, 2.678f, 2.679f, 2.488f, 2.489f, 2.490f, 2.491f, 2.492f, 2.493f, 2.494f,
                2.495f, 2.520f, 2.521f, 2.522f, 2.523f, 2.524f, 2.525f, 2.526f, 2.527f, 2.552f, 2.553f, 2.554f, 2.555f, 2.556f, 2.557f, 2.558f, 2.559f, 2.584f, 2.585f, 2.586f, 2.587f, 2.588f, 2.589f, 2.590f, 2.591f,
                2.616f, 2.617f, 2.618f, 2.619f, 2.620f, 2.621f, 2.622f, 2.623f, 2.648f, 2.649f, 2.650f, 2.651f, 2.652f, 2.653f, 2.654f, 2.655f, 2.680f, 2.681f, 2.682f, 2.683f, 2.684f, 2.685f, 2.686f, 2.687f, 2.704f,
                2.705f, 2.706f, 2.707f, 2.708f, 2.709f, 2.710f, 2.711f, 2.736f, 2.737f, 2.738f, 2.739f, 2.740f, 2.741f, 2.742f, 2.743f, 2.768f, 2.769f, 2.770f, 2.771f, 2.772f, 2.773f, 2.774f, 2.775f, 2.800f, 2.801f,
                2.802f, 2.803f, 2.804f, 2.805f, 2.806f, 2.807f, 2.832f, 2.833f, 2.834f, 2.835f, 2.836f, 2.837f, 2.838f, 2.839f, 2.864f, 2.865f, 2.866f, 2.867f, 2.868f, 2.869f, 2.870f, 2.871f, 2.896f, 2.897f, 2.898f,
                2.899f, 2.900f, 2.901f, 2.902f, 2.903f, 2.712f, 2.713f, 2.714f, 2.715f, 2.716f, 2.717f, 2.718f, 2.719f, 2.744f, 2.745f, 2.746f, 2.747f, 2.748f, 2.749f, 2.750f, 2.751f, 2.776f, 2.777f, 2.778f, 2.779f,
                2.780f, 2.781f, 2.782f, 2.783f, 2.808f, 2.809f, 2.810f, 2.811f, 2.812f, 2.813f, 2.814f, 2.815f, 2.840f, 2.841f, 2.842f, 2.843f, 2.844f, 2.845f, 2.846f, 2.847f, 2.872f, 2.873f, 2.874f, 2.875f, 2.876f,
                2.877f, 2.878f, 2.879f, 2.904f, 2.905f, 2.906f, 2.907f, 2.908f, 2.909f, 2.910f, 2.911f, 2.928f, 2.929f, 2.930f, 2.931f, 2.932f, 2.933f, 2.934f, 2.935f, 2.960f, 2.961f, 2.962f, 2.963f, 2.964f, 2.965f,
                2.966f, 2.967f, 2.992f, 2.993f, 2.994f, 2.995f, 2.996f, 2.997f, 2.998f, 2.999f, 3.024f, 3.025f, 3.026f, 3.027f, 3.028f, 3.029f, 3.030f, 3.031f, 3.056f, 3.057f, 3.058f, 3.059f, 3.060f, 3.061f, 3.062f,
                3.063f, 3.088f, 3.089f, 3.090f, 3.091f, 3.092f, 3.093f, 3.094f, 3.095f, 3.120f, 3.121f, 3.122f, 3.123f, 3.124f, 3.125f, 3.126f, 3.127f, 2.936f, 2.937f, 2.938f, 2.939f, 2.940f, 2.941f, 2.942f, 2.943f,
                2.968f, 2.969f, 2.970f, 2.971f, 2.972f, 2.973f, 2.974f, 2.975f, 3.000f, 3.001f, 3.002f, 3.003f, 3.004f, 3.005f, 3.006f, 3.007f, 3.032f, 3.033f, 3.034f, 3.035f, 3.036f, 3.037f, 3.038f, 3.039f, 3.064f,
                3.065f, 3.066f, 3.067f, 3.068f, 3.069f, 3.070f, 3.071f, 3.096f, 3.097f, 3.098f, 3.099f, 3.100f, 3.101f, 3.102f, 3.103f, 3.128f, 3.129f, 3.130f, 3.131f, 3.132f, 3.133f, 3.134f, 3.135f, 3.152f, 3.153f,
                3.154f, 3.155f, 3.156f, 3.157f, 3.158f, 3.159f, 3.184f, 3.185f, 3.186f, 3.187f, 3.188f, 3.189f, 3.190f, 3.191f, 3.216f, 3.217f, 3.218f, 3.219f, 3.220f, 3.221f, 3.222f, 3.223f, 3.248f, 3.249f, 3.250f,
                3.251f, 3.252f, 3.253f, 3.254f, 3.255f, 3.280f, 3.281f, 3.282f, 3.283f, 3.284f, 3.285f, 3.286f, 3.287f, 3.312f, 3.313f, 3.314f, 3.315f, 3.316f, 3.317f, 3.318f, 3.319f, 3.344f, 3.345f, 3.346f, 3.347f,
                3.348f, 3.349f, 3.350f, 3.351f, 3.160f, 3.161f, 3.162f, 3.163f, 3.164f, 3.165f, 3.166f, 3.167f, 3.192f, 3.193f, 3.194f, 3.195f, 3.196f, 3.197f, 3.198f, 3.199f, 3.224f, 3.225f, 3.226f, 3.227f, 3.228f,
                3.229f, 3.230f, 3.231f, 3.256f, 3.257f, 3.258f, 3.259f, 3.260f, 3.261f, 3.262f, 3.263f, 3.288f, 3.289f, 3.290f, 3.291f, 3.292f, 3.293f, 3.294f, 3.295f, 3.320f, 3.321f, 3.322f, 3.323f, 3.324f, 3.325f,
                3.326f, 3.327f, 3.352f, 3.353f, 3.354f, 3.355f, 3.356f, 3.357f, 3.358f, 3.359f,
            };

            float[] y_actual = y.ToArray();

            AssertError.Tolerance(y_expect, y_actual, 1e-7f, 1e-5f, $"mismatch value {inchannels},{outchannels},{scale},{inwidth},{inheight},{indepth}");
        }
    }
}
