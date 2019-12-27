using System;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using TensorShader.Operators.Connection1D;
using TensorShaderCudaBackend.API;

namespace TensorShaderTest.Operators.Connection1D {
    [TestClass]
    public class ConvolutionTest {
        [TestMethod]
        public void ExecuteTest() {
            float max_err = 0;

            foreach (int batch in new int[] { 1, 2 }) {
                foreach (int inchannels in new int[] { 1, 2, 3, 4, 5, 10, 15, 20, 32, 33 }) {
                    foreach (int outchannels in new int[] { 1, 2, 3, 4, 5, 10, 15, 20, 32, 33 }) {
                        foreach (int kwidth in new int[] { 1, 3, 5 }) {
                            foreach (int inwidth in new int[] { kwidth, kwidth * 2, 8, 9, 13, 17, 25 }) {
                                int outwidth = inwidth - kwidth + 1;

                                float[] xval = (new float[inwidth * inchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
                                float[] wval = (new float[kwidth * inchannels * outchannels]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

                                Map1D x = new Map1D(inchannels, inwidth, batch, xval);
                                Filter1D w = new Filter1D(inchannels, outchannels, kwidth, wval);

                                Map1D y = Reference(x, w, kwidth);

                                OverflowCheckedTensor x_tensor = new OverflowCheckedTensor(Shape.Map1D(inchannels, inwidth, batch), xval);
                                OverflowCheckedTensor w_tensor = new OverflowCheckedTensor(Shape.Kernel1D(inchannels, outchannels, kwidth), wval);

                                OverflowCheckedTensor y_tensor = new OverflowCheckedTensor(Shape.Map1D(outchannels, outwidth, batch));

                                Convolution ope = new Convolution(inwidth, inchannels, outchannels, kwidth, batch);
                                    
                                ope.Execute(x_tensor, w_tensor, y_tensor);

                                float[] y_expect = y.ToArray();
                                float[] y_actual = y_tensor.State;

                                CollectionAssert.AreEqual(xval, x_tensor.State);
                                CollectionAssert.AreEqual(wval, w_tensor.State);

                                AssertError.Tolerance(y_expect, y_actual, 1e-7f, 1e-5f, ref max_err, $"mismatch value {inchannels},{outchannels},{kwidth},{inwidth},{batch}");

                                Console.WriteLine($"pass: {inchannels},{outchannels},{kwidth},{inwidth},{batch}");
                            }
                        }
                    }
                }
            }

            Console.WriteLine($"maxerr:{max_err}");
        }

        [TestMethod]
        public void LargeMapTest() {
            float max_err = 0;

            Random random = new Random(1234);

            int batch = 3;
            int inchannels = 49, outchannels = 50;
            int kwidth = 5; 
            int inwidth = 125; 
            int outwidth = inwidth - kwidth + 1;

            float[] xval = (new float[inwidth * inchannels * batch]).Select((_, idx) => (float)random.NextDouble() * 1e-2f).ToArray();
            float[] wval = (new float[kwidth * inchannels * outchannels]).Select((_, idx) => (float)random.NextDouble() * 1e-2f).ToArray();

            Map1D x = new Map1D(inchannels, inwidth, batch, xval);
            Filter1D w = new Filter1D(inchannels, outchannels, kwidth, wval);

            Map1D y = Reference(x, w, kwidth);

            OverflowCheckedTensor x_tensor = new OverflowCheckedTensor(Shape.Map1D(inchannels, inwidth, batch), xval);
            OverflowCheckedTensor w_tensor = new OverflowCheckedTensor(Shape.Kernel1D(inchannels, outchannels, kwidth), wval);

            OverflowCheckedTensor y_tensor = new OverflowCheckedTensor(Shape.Map1D(outchannels, outwidth, batch));

            Convolution ope = new Convolution(inwidth, inchannels, outchannels, kwidth, batch);
                                    
            ope.Execute(x_tensor, w_tensor, y_tensor);

            float[] y_expect = y.ToArray();
            float[] y_actual = y_tensor.State;

            CollectionAssert.AreEqual(xval, x_tensor.State);
            CollectionAssert.AreEqual(wval, w_tensor.State);

            AssertError.Tolerance(y_expect, y_actual, 1e-7f, 1e-5f, ref max_err, $"mismatch value {inchannels},{outchannels},{kwidth},{inwidth},{batch}");

            Console.WriteLine($"pass: {inchannels},{outchannels},{kwidth},{inwidth},{batch}");

            Console.WriteLine($"maxerr:{max_err}");
        }

        [TestMethod]
        public void SpeedTest() {
            int inwidth = 512, inchannels = 31, outchannels = 31, ksize = 3;
            int outwidth = inwidth - ksize + 1;

            OverflowCheckedTensor x_tensor = new OverflowCheckedTensor(Shape.Map1D(inchannels, inwidth));
            OverflowCheckedTensor w_tensor = new OverflowCheckedTensor(Shape.Kernel1D(inchannels, outchannels, ksize));

            OverflowCheckedTensor y_tensor = new OverflowCheckedTensor(Shape.Map1D(outchannels, outwidth));

            Convolution ope = new Convolution(inwidth, inchannels, outchannels, ksize);

            Cuda.Profiler.Initialize("../../../profiler.nvsetting", "../../nvprofiles/convolution_1d.nvvp");
            Cuda.Profiler.Start();

            ope.Execute(x_tensor, w_tensor, y_tensor);

            Cuda.Profiler.Stop();
        }

        public static Map1D Reference(Map1D x, Filter1D w, int kwidth) {
            int inchannels = x.Channels, outchannels = w.OutChannels, batch = x.Batch;
            int inw = x.Width, outw = inw - kwidth + 1;

            Map1D y = new Map1D(outchannels, outw, batch);

            for (int kx = 0; kx < kwidth; kx++) {
                for (int th = 0; th < batch; th++) {
                    for (int ox = 0; ox < outw; ox++) {
                        for (int outch = 0; outch < outchannels; outch++) {
                            double sum = y[outch, ox, th];

                            for (int inch = 0; inch < inchannels; inch++) {
                                sum += x[inch, kx + ox, th] * w[inch, outch, kx];
                            }

                            y[outch, ox, th] = sum;
                        }
                    }
                }
            }

            return y;
        }
        
        [TestMethod]
        public void ReferenceTest() {
            int inchannels = 7, outchannels = 11, kwidth = 3, inwidth = 13, batch = 2;
            int outwidth = inwidth - kwidth + 1;

            float[] xval = (new float[batch * inwidth * inchannels]).Select((_, idx) => idx * 1e-3f).ToArray();
            float[] wval = (new float[kwidth * outchannels * inchannels]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

            Map1D x = new Map1D(inchannels, inwidth, batch, xval);
            Filter1D w = new Filter1D(inchannels, outchannels, kwidth, wval);

            Map1D y = Reference(x, w, kwidth);

            float[] y_expect = {
                2.387000000e-02f, 2.240000000e-02f, 2.093000000e-02f, 1.946000000e-02f, 1.799000000e-02f, 1.652000000e-02f, 1.505000000e-02f, 1.358000000e-02f, 1.211000000e-02f, 1.064000000e-02f, 9.170000000e-03f,
                4.592000000e-02f, 4.342100000e-02f, 4.092200000e-02f, 3.842300000e-02f, 3.592400000e-02f, 3.342500000e-02f, 3.092600000e-02f, 2.842700000e-02f, 2.592800000e-02f, 2.342900000e-02f, 2.093000000e-02f,
                6.797000000e-02f, 6.444200000e-02f, 6.091400000e-02f, 5.738600000e-02f, 5.385800000e-02f, 5.033000000e-02f, 4.680200000e-02f, 4.327400000e-02f, 3.974600000e-02f, 3.621800000e-02f, 3.269000000e-02f,
                9.002000000e-02f, 8.546300000e-02f, 8.090600000e-02f, 7.634900000e-02f, 7.179200000e-02f, 6.723500000e-02f, 6.267800000e-02f, 5.812100000e-02f, 5.356400000e-02f, 4.900700000e-02f, 4.445000000e-02f,
                1.120700000e-01f, 1.064840000e-01f, 1.008980000e-01f, 9.531200000e-02f, 8.972600000e-02f, 8.414000000e-02f, 7.855400000e-02f, 7.296800000e-02f, 6.738200000e-02f, 6.179600000e-02f, 5.621000000e-02f,
                1.341200000e-01f, 1.275050000e-01f, 1.208900000e-01f, 1.142750000e-01f, 1.076600000e-01f, 1.010450000e-01f, 9.443000000e-02f, 8.781500000e-02f, 8.120000000e-02f, 7.458500000e-02f, 6.797000000e-02f,
                1.561700000e-01f, 1.485260000e-01f, 1.408820000e-01f, 1.332380000e-01f, 1.255940000e-01f, 1.179500000e-01f, 1.103060000e-01f, 1.026620000e-01f, 9.501800000e-02f, 8.737400000e-02f, 7.973000000e-02f,
                1.782200000e-01f, 1.695470000e-01f, 1.608740000e-01f, 1.522010000e-01f, 1.435280000e-01f, 1.348550000e-01f, 1.261820000e-01f, 1.175090000e-01f, 1.088360000e-01f, 1.001630000e-01f, 9.149000000e-02f,
                2.002700000e-01f, 1.905680000e-01f, 1.808660000e-01f, 1.711640000e-01f, 1.614620000e-01f, 1.517600000e-01f, 1.420580000e-01f, 1.323560000e-01f, 1.226540000e-01f, 1.129520000e-01f, 1.032500000e-01f,
                2.223200000e-01f, 2.115890000e-01f, 2.008580000e-01f, 1.901270000e-01f, 1.793960000e-01f, 1.686650000e-01f, 1.579340000e-01f, 1.472030000e-01f, 1.364720000e-01f, 1.257410000e-01f, 1.150100000e-01f,
                2.443700000e-01f, 2.326100000e-01f, 2.208500000e-01f, 2.090900000e-01f, 1.973300000e-01f, 1.855700000e-01f, 1.738100000e-01f, 1.620500000e-01f, 1.502900000e-01f, 1.385300000e-01f, 1.267700000e-01f,
                3.105200000e-01f, 2.956730000e-01f, 2.808260000e-01f, 2.659790000e-01f, 2.511320000e-01f, 2.362850000e-01f, 2.214380000e-01f, 2.065910000e-01f, 1.917440000e-01f, 1.768970000e-01f, 1.620500000e-01f,
                3.325700000e-01f, 3.166940000e-01f, 3.008180000e-01f, 2.849420000e-01f, 2.690660000e-01f, 2.531900000e-01f, 2.373140000e-01f, 2.214380000e-01f, 2.055620000e-01f, 1.896860000e-01f, 1.738100000e-01f,
                3.546200000e-01f, 3.377150000e-01f, 3.208100000e-01f, 3.039050000e-01f, 2.870000000e-01f, 2.700950000e-01f, 2.531900000e-01f, 2.362850000e-01f, 2.193800000e-01f, 2.024750000e-01f, 1.855700000e-01f,
                3.766700000e-01f, 3.587360000e-01f, 3.408020000e-01f, 3.228680000e-01f, 3.049340000e-01f, 2.870000000e-01f, 2.690660000e-01f, 2.511320000e-01f, 2.331980000e-01f, 2.152640000e-01f, 1.973300000e-01f,
                3.987200000e-01f, 3.797570000e-01f, 3.607940000e-01f, 3.418310000e-01f, 3.228680000e-01f, 3.039050000e-01f, 2.849420000e-01f, 2.659790000e-01f, 2.470160000e-01f, 2.280530000e-01f, 2.090900000e-01f,
                4.207700000e-01f, 4.007780000e-01f, 3.807860000e-01f, 3.607940000e-01f, 3.408020000e-01f, 3.208100000e-01f, 3.008180000e-01f, 2.808260000e-01f, 2.608340000e-01f, 2.408420000e-01f, 2.208500000e-01f,
                4.428200000e-01f, 4.217990000e-01f, 4.007780000e-01f, 3.797570000e-01f, 3.587360000e-01f, 3.377150000e-01f, 3.166940000e-01f, 2.956730000e-01f, 2.746520000e-01f, 2.536310000e-01f, 2.326100000e-01f,
                4.648700000e-01f, 4.428200000e-01f, 4.207700000e-01f, 3.987200000e-01f, 3.766700000e-01f, 3.546200000e-01f, 3.325700000e-01f, 3.105200000e-01f, 2.884700000e-01f, 2.664200000e-01f, 2.443700000e-01f,
                4.869200000e-01f, 4.638410000e-01f, 4.407620000e-01f, 4.176830000e-01f, 3.946040000e-01f, 3.715250000e-01f, 3.484460000e-01f, 3.253670000e-01f, 3.022880000e-01f, 2.792090000e-01f, 2.561300000e-01f,
                5.089700000e-01f, 4.848620000e-01f, 4.607540000e-01f, 4.366460000e-01f, 4.125380000e-01f, 3.884300000e-01f, 3.643220000e-01f, 3.402140000e-01f, 3.161060000e-01f, 2.919980000e-01f, 2.678900000e-01f,
                5.310200000e-01f, 5.058830000e-01f, 4.807460000e-01f, 4.556090000e-01f, 4.304720000e-01f, 4.053350000e-01f, 3.801980000e-01f, 3.550610000e-01f, 3.299240000e-01f, 3.047870000e-01f, 2.796500000e-01f,
            };

            float[] y_actual = y.ToArray();

            AssertError.Tolerance(y_expect, y_actual, 1e-7f, 1e-5f, $"mismatch value {inchannels},{outchannels},{kwidth},{inwidth},{batch}");
        }
    }
}
