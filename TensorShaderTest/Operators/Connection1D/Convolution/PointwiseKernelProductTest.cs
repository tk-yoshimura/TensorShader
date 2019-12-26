using System;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using TensorShader.Operators.Connection1D;
using TensorShaderCudaBackend.API;

namespace TensorShaderTest.Operators.Connection1D {
    [TestClass]
    public class PointwiseKernelProductTest {
        [TestMethod]
        public void ExecuteTest() {
            float max_err = 0;

            foreach (int batch in new int[] { 1, 2 }) {
                foreach (int inchannels in new int[] { 1, 2, 3, 4, 5, 10, 15, 20 }) {
                    foreach (int outchannels in new int[] { 7, 13 }) {
                        foreach (int inwidth in new int[] { 8, 9, 13, 17 }) {
                            float[] xval = (new float[inwidth * inchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
                            float[] gyval = (new float[inwidth * outchannels * batch]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

                            Map1D x = new Map1D(inchannels, inwidth, batch, xval);
                            Map1D gy = new Map1D(outchannels, inwidth, batch, gyval);

                            Filter1D gw = Reference(x, gy);

                            OverflowCheckedTensor x_tensor = new OverflowCheckedTensor(Shape.Map1D(inchannels, inwidth, batch), xval);
                            OverflowCheckedTensor gy_tensor = new OverflowCheckedTensor(Shape.Map1D(outchannels, inwidth, batch), gyval);

                            OverflowCheckedTensor gw_tensor = new OverflowCheckedTensor(Shape.Kernel0D(inchannels, outchannels));

                            PointwiseKernelProduct ope = new PointwiseKernelProduct(inwidth, inchannels, outchannels, batch);

                            ope.Execute(x_tensor, gy_tensor, gw_tensor);

                            float[] gw_expect = gw.ToArray();
                            float[] gw_actual = gw_tensor.State;

                            CollectionAssert.AreEqual(xval, x_tensor.State);
                            CollectionAssert.AreEqual(gyval, gy_tensor.State);

                            AssertError.Tolerance(gw_expect, gw_actual, 1e-7f, 1e-5f, ref max_err, $"mismatch value {inchannels},{outchannels},{inwidth},{batch}");

                            Console.WriteLine($"pass: {inchannels},{outchannels},{inwidth},{batch}");

                        }
                    }
                }
            }

            Console.WriteLine($"maxerr:{max_err}");
        }

        [TestMethod]
        public void SpeedTest() {
            int inwidth = 512, inchannels = 32, outchannels = 63;

            OverflowCheckedTensor x_tensor = new OverflowCheckedTensor(Shape.Map1D(inchannels, inwidth));
            OverflowCheckedTensor gy_tensor = new OverflowCheckedTensor(Shape.Map1D(outchannels, inwidth));

            OverflowCheckedTensor gw_tensor = new OverflowCheckedTensor(Shape.Kernel0D(inchannels, outchannels));

            PointwiseKernelProduct ope = new PointwiseKernelProduct(inwidth, inchannels, outchannels);

            Cuda.Profiler.Initialize("../../../profiler.nvsetting", "../../nvprofiles/ptwise_convolution_1d.nvvp");
            Cuda.Profiler.Start();

            ope.Execute(x_tensor, gy_tensor, gw_tensor);
            
            Cuda.Profiler.Stop();
        }

        public static Filter1D Reference(Map1D x, Map1D gy) {
            int inchannels = x.Channels, outchannels = gy.Channels, batch = x.Batch;
            int inw = x.Width;

            Filter1D w = new Filter1D(inchannels, outchannels, 1);

            for (int th = 0; th < batch; th++) {
                for (int inch, outch = 0; outch < outchannels; outch++) {
                    for (inch = 0; inch < inchannels; inch++) {
                        double sum = 0;

                        for (int ix = 0; ix < inw; ix++) {
                            sum += x[inch, ix, th] * gy[outch, ix, th];
                        }

                        w[inch, outch, 0] += sum;
                    }
                }
            }

            return w;
        }

        [TestMethod]
        public void ReferenceTest() {
            int inchannels = 7, outchannels = 11, inwidth = 13;

            float[] xval = (new float[inwidth * inchannels]).Select((_, idx) => idx * 1e-3f).ToArray();
            float[] gyval = (new float[inwidth * outchannels]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

            Map1D x = new Map1D(inchannels, inwidth, 1, xval);
            Map1D gy = new Map1D(outchannels, inwidth, 1, gyval);

            Filter1D gw = Reference(x, gy);

            float[] gw_expect = {
                2.748200e-02f,  2.847000e-02f,  2.945800e-02f,  3.044600e-02f,  3.143400e-02f,  3.242200e-02f,  3.341000e-02f,
                2.693600e-02f,  2.791100e-02f,  2.888600e-02f,  2.986100e-02f,  3.083600e-02f,  3.181100e-02f,  3.278600e-02f,
                2.639000e-02f,  2.735200e-02f,  2.831400e-02f,  2.927600e-02f,  3.023800e-02f,  3.120000e-02f,  3.216200e-02f,
                2.584400e-02f,  2.679300e-02f,  2.774200e-02f,  2.869100e-02f,  2.964000e-02f,  3.058900e-02f,  3.153800e-02f,
                2.529800e-02f,  2.623400e-02f,  2.717000e-02f,  2.810600e-02f,  2.904200e-02f,  2.997800e-02f,  3.091400e-02f,
                2.475200e-02f,  2.567500e-02f,  2.659800e-02f,  2.752100e-02f,  2.844400e-02f,  2.936700e-02f,  3.029000e-02f,
                2.420600e-02f,  2.511600e-02f,  2.602600e-02f,  2.693600e-02f,  2.784600e-02f,  2.875600e-02f,  2.966600e-02f,
                2.366000e-02f,  2.455700e-02f,  2.545400e-02f,  2.635100e-02f,  2.724800e-02f,  2.814500e-02f,  2.904200e-02f,
                2.311400e-02f,  2.399800e-02f,  2.488200e-02f,  2.576600e-02f,  2.665000e-02f,  2.753400e-02f,  2.841800e-02f,
                2.256800e-02f,  2.343900e-02f,  2.431000e-02f,  2.518100e-02f,  2.605200e-02f,  2.692300e-02f,  2.779400e-02f,
                2.202200e-02f,  2.288000e-02f,  2.373800e-02f,  2.459600e-02f,  2.545400e-02f,  2.631200e-02f,  2.717000e-02f,
            };

            float[] gw_actual = gw.ToArray();

            AssertError.Tolerance(gw_expect, gw_actual, 1e-7f, 1e-5f, $"mismatch value {inchannels},{outchannels},{inwidth}");
        }
    }
}
