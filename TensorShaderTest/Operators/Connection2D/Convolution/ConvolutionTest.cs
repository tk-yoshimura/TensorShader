using System;
using System.Diagnostics;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using TensorShader.Operators.Connection2D;
using TensorShaderCudaBackend.API;

namespace TensorShaderTest.Operators.Connection2D {
    [TestClass]
    public class ConvolutionTest {
        [TestMethod]
        public void ExecuteTest() {
            float max_err = 0;

            foreach (int batch in new int[] { 1, 2 }) {
                foreach (int inchannels in new int[] { 1, 2, 3, 4, 5, 10, 15, 20, 32, 33 }) {
                    foreach (int outchannels in new int[] { 1, 2, 3, 4, 5, 10, 15, 20, 32, 33 }) {
                        foreach (int kheight in new int[] { 1, 3, 5 }) {
                            foreach (int kwidth in new int[] { 1, 3, 5 }) {
                                foreach (int stride in new int[] { 1 }) {
                                    foreach (int inwidth in new int[] { kwidth, kwidth * 2, 8, 9, 13, 17, 25 }) {
                                        foreach (int inheight in new int[] { kheight, kheight * 2, 8, 9, 13, 17, 25 }) {
                                            int outwidth = inwidth - kwidth + 1, outheight = inheight - kheight + 1;

                                            float[] xval = (new float[inwidth * inheight * inchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
                                            float[] wval = (new float[kwidth * kheight * inchannels * outchannels]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

                                            Map2D x = new Map2D(inchannels, inwidth, inheight, batch, xval);
                                            Filter2D w = new Filter2D(inchannels, outchannels, kwidth, kheight, wval);

                                            Map2D y = Reference(x, w, kwidth, kheight, stride);

                                            OverflowCheckedTensor x_tensor = new OverflowCheckedTensor(Shape.Map2D(inchannels, inwidth, inheight, batch), xval);
                                            OverflowCheckedTensor w_tensor = new OverflowCheckedTensor(Shape.Kernel2D(inchannels, outchannels, kwidth, kheight), wval);

                                            OverflowCheckedTensor y_tensor = new OverflowCheckedTensor(Shape.Map2D(outchannels, outwidth, outheight, batch));

                                            Convolution ope = new Convolution(inwidth, inheight, inchannels, outchannels, kwidth, kheight, stride, batch);

                                            ope.Execute(x_tensor, w_tensor, y_tensor);

                                            float[] y_expect = y.ToArray();
                                            float[] y_actual = y_tensor.State;

                                            CollectionAssert.AreEqual(xval, x_tensor.State);
                                            CollectionAssert.AreEqual(wval, w_tensor.State);

                                            AssertError.Tolerance(y_expect, y_actual, 1e-7f, 1e-5f, ref max_err, $"mismatch value {inchannels},{outchannels},{kwidth},{kheight},{stride},{inwidth},{inheight},{batch}");

                                            Console.WriteLine($"pass: {inchannels},{outchannels},{kwidth},{kheight},{stride},{inwidth},{inheight},{batch}");
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

        [TestMethod]
        public void SpeedTest() {
            int inwidth = 512, inheight = 512, inchannels = 31, outchannels = 31, ksize = 3, stride = 1;
            int outwidth = (inwidth - ksize) / stride + 1, outheight = (inheight - ksize) / stride + 1;

            OverflowCheckedTensor x_tensor = new OverflowCheckedTensor(Shape.Map2D(inchannels, inwidth, inheight));
            OverflowCheckedTensor w_tensor = new OverflowCheckedTensor(Shape.Kernel2D(inchannels, outchannels, ksize, ksize));

            OverflowCheckedTensor y_tensor = new OverflowCheckedTensor(Shape.Map2D(outchannels, outwidth, outheight));

            Convolution ope = new Convolution(inwidth, inheight, inchannels, outchannels, ksize, ksize, stride);

            Cuda.Profiler.Initialize("../../../profiler.nvsetting", "../../nvprofiles/convolution2d_trans_v2.nvvp");
            Cuda.Profiler.Start();

            ope.Execute(x_tensor, w_tensor, y_tensor);

            Cuda.Profiler.Stop();
        }

        public static Map2D Reference(Map2D x, Filter2D w, int kwidth, int kheight, int stride) {
            int inchannels = x.Channels, outchannels = w.OutChannels, batch = x.Batch;
            int inw = x.Width, inh = x.Height, outw = inw - kwidth + 1, outh = inh - kheight + 1;

            Map2D y = new Map2D(outchannels, outw, outh, batch);

            for (int kx, ky = 0; ky < kheight; ky++) {
                for (kx = 0; kx < kwidth; kx++) {
                    for (int th = 0; th < batch; th++) {
                        for (int ox, oy = 0; oy < outh; oy++) {
                            for (ox = 0; ox < outw; ox++) {
                                for (int outch = 0; outch < outchannels; outch++) {
                                    double sum = y[outch, ox, oy, th];

                                    for (int inch = 0; inch < inchannels; inch++) {
                                        sum += x[inch, kx + ox * stride, ky + oy * stride, th] * w[inch, outch, kx, ky];
                                    }

                                    y[outch, ox, oy, th] = sum;
                                }
                            }
                        }
                    }
                }
            }

            return y;
        }

        public static Map2D OptimizedReference(Map2D x, Filter2D w, int kwidth, int kheight, int stride) {
            int inchannels = x.Channels, outchannels = w.OutChannels, batch = x.Batch;
            int inw = x.Width, inh = x.Height, outw = inw - kwidth + 1, outh = inh - kheight + 1;

            Map2D y = new Map2D(outchannels, outw, outh, batch);

            for (int kx, ky = 0; ky < kheight; ky++) {
                for (kx = 0; kx < kwidth; kx++) {
                    int inmap_offset = (kx + ky * inw) * inchannels;
                    int kernel_offset = (kx + ky * kwidth) * inchannels * outchannels;

                    for (int th = 0; th < batch; th++) {
                        for (int ox, oy = 0; oy < outh; oy++) {
                            for (ox = 0; ox < outw; ox++) {
                                int inmap_org = inmap_offset + (ox + oy * inw) * inchannels * stride + th * inw * inh * inchannels;
                                int outmap_idx = (ox + oy * outw) * outchannels + th * outw * outh * outchannels;
                                int kernel_idx = kernel_offset;

                                for (int outch = 0; outch < outchannels; outch++) {
                                    double sum = y[outmap_idx];

                                    int inmap_idx = inmap_org;

                                    for (int inch = 0; inch < inchannels; inch++) {
                                        sum += x[inmap_idx] * w[kernel_idx];

                                        inmap_idx++;
                                        kernel_idx++;
                                    }

                                    y[outmap_idx] = sum;

                                    outmap_idx++;
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
            int inchannels = 7, outchannels = 11, kwidth = 3, kheight = 5, stride = 2, inwidth = 13, inheight = 17, batch = 2;
            int outwidth = inwidth - kwidth + 1, outheight = inheight - kheight + 1;

            float[] xval = (new float[batch * inwidth * inheight * inchannels]).Select((_, idx) => idx * 1e-3f).ToArray();
            float[] wval = (new float[kwidth * kheight * outchannels * inchannels]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

            Map2D x = new Map2D(inchannels, inwidth, inheight, batch, xval);
            Filter2D w = new Filter2D(inchannels, outchannels, kwidth, kheight, wval);

            Map2D y = Reference(x, w, kwidth, kheight, stride);

            float[] y_expect = {
                7.885360000e+00f, 7.744240000e+00f, 7.603120000e+00f, 7.462000000e+00f, 7.320880000e+00f,
                7.179760000e+00f, 7.038640000e+00f, 6.897520000e+00f, 6.756400000e+00f, 6.615280000e+00f,
                6.474160000e+00f, 8.785000000e+00f, 8.633590000e+00f, 8.482180000e+00f, 8.330770000e+00f,
                8.179360000e+00f, 8.027950000e+00f, 7.876540000e+00f, 7.725130000e+00f, 7.573720000e+00f,
                7.422310000e+00f, 7.270900000e+00f, 9.684640000e+00f, 9.522940000e+00f, 9.361240000e+00f,
                9.199540000e+00f, 9.037840000e+00f, 8.876140000e+00f, 8.714440000e+00f, 8.552740000e+00f,
                8.391040000e+00f, 8.229340000e+00f, 8.067640000e+00f, 1.058428000e+01f, 1.041229000e+01f,
                1.024030000e+01f, 1.006831000e+01f, 9.896320000e+00f, 9.724330000e+00f, 9.552340000e+00f,
                9.380350000e+00f, 9.208360000e+00f, 9.036370000e+00f, 8.864380000e+00f, 1.148392000e+01f,
                1.130164000e+01f, 1.111936000e+01f, 1.093708000e+01f, 1.075480000e+01f, 1.057252000e+01f,
                1.039024000e+01f, 1.020796000e+01f, 1.002568000e+01f, 9.843400000e+00f, 9.661120000e+00f,
                1.238356000e+01f, 1.219099000e+01f, 1.199842000e+01f, 1.180585000e+01f, 1.161328000e+01f,
                1.142071000e+01f, 1.122814000e+01f, 1.103557000e+01f, 1.084300000e+01f, 1.065043000e+01f,
                1.045786000e+01f, 1.958068000e+01f, 1.930579000e+01f, 1.903090000e+01f, 1.875601000e+01f,
                1.848112000e+01f, 1.820623000e+01f, 1.793134000e+01f, 1.765645000e+01f, 1.738156000e+01f,
                1.710667000e+01f, 1.683178000e+01f, 2.048032000e+01f, 2.019514000e+01f, 1.990996000e+01f,
                1.962478000e+01f, 1.933960000e+01f, 1.905442000e+01f, 1.876924000e+01f, 1.848406000e+01f,
                1.819888000e+01f, 1.791370000e+01f, 1.762852000e+01f, 2.137996000e+01f, 2.108449000e+01f,
                2.078902000e+01f, 2.049355000e+01f, 2.019808000e+01f, 1.990261000e+01f, 1.960714000e+01f,
                1.931167000e+01f, 1.901620000e+01f, 1.872073000e+01f, 1.842526000e+01f, 2.227960000e+01f,
                2.197384000e+01f, 2.166808000e+01f, 2.136232000e+01f, 2.105656000e+01f, 2.075080000e+01f,
                2.044504000e+01f, 2.013928000e+01f, 1.983352000e+01f, 1.952776000e+01f, 1.922200000e+01f,
                2.317924000e+01f, 2.286319000e+01f, 2.254714000e+01f, 2.223109000e+01f, 2.191504000e+01f,
                2.159899000e+01f, 2.128294000e+01f, 2.096689000e+01f, 2.065084000e+01f, 2.033479000e+01f,
                2.001874000e+01f, 2.407888000e+01f, 2.375254000e+01f, 2.342620000e+01f, 2.309986000e+01f,
                2.277352000e+01f, 2.244718000e+01f, 2.212084000e+01f, 2.179450000e+01f, 2.146816000e+01f,
                2.114182000e+01f, 2.081548000e+01f, 3.127600000e+01f, 3.086734000e+01f, 3.045868000e+01f,
                3.005002000e+01f, 2.964136000e+01f, 2.923270000e+01f, 2.882404000e+01f, 2.841538000e+01f,
                2.800672000e+01f, 2.759806000e+01f, 2.718940000e+01f, 3.217564000e+01f, 3.175669000e+01f,
                3.133774000e+01f, 3.091879000e+01f, 3.049984000e+01f, 3.008089000e+01f, 2.966194000e+01f,
                2.924299000e+01f, 2.882404000e+01f, 2.840509000e+01f, 2.798614000e+01f, 3.307528000e+01f,
                3.264604000e+01f, 3.221680000e+01f, 3.178756000e+01f, 3.135832000e+01f, 3.092908000e+01f,
                3.049984000e+01f, 3.007060000e+01f, 2.964136000e+01f, 2.921212000e+01f, 2.878288000e+01f,
                3.397492000e+01f, 3.353539000e+01f, 3.309586000e+01f, 3.265633000e+01f, 3.221680000e+01f,
                3.177727000e+01f, 3.133774000e+01f, 3.089821000e+01f, 3.045868000e+01f, 3.001915000e+01f,
                2.957962000e+01f, 3.487456000e+01f, 3.442474000e+01f, 3.397492000e+01f, 3.352510000e+01f,
                3.307528000e+01f, 3.262546000e+01f, 3.217564000e+01f, 3.172582000e+01f, 3.127600000e+01f,
                3.082618000e+01f, 3.037636000e+01f, 3.577420000e+01f, 3.531409000e+01f, 3.485398000e+01f,
                3.439387000e+01f, 3.393376000e+01f, 3.347365000e+01f, 3.301354000e+01f, 3.255343000e+01f,
                3.209332000e+01f, 3.163321000e+01f, 3.117310000e+01f, 4.297132000e+01f, 4.242889000e+01f,
                4.188646000e+01f, 4.134403000e+01f, 4.080160000e+01f, 4.025917000e+01f, 3.971674000e+01f,
                3.917431000e+01f, 3.863188000e+01f, 3.808945000e+01f, 3.754702000e+01f, 4.387096000e+01f,
                4.331824000e+01f, 4.276552000e+01f, 4.221280000e+01f, 4.166008000e+01f, 4.110736000e+01f,
                4.055464000e+01f, 4.000192000e+01f, 3.944920000e+01f, 3.889648000e+01f, 3.834376000e+01f,
                4.477060000e+01f, 4.420759000e+01f, 4.364458000e+01f, 4.308157000e+01f, 4.251856000e+01f,
                4.195555000e+01f, 4.139254000e+01f, 4.082953000e+01f, 4.026652000e+01f, 3.970351000e+01f,
                3.914050000e+01f, 4.567024000e+01f, 4.509694000e+01f, 4.452364000e+01f, 4.395034000e+01f,
                4.337704000e+01f, 4.280374000e+01f, 4.223044000e+01f, 4.165714000e+01f, 4.108384000e+01f,
                4.051054000e+01f, 3.993724000e+01f, 4.656988000e+01f, 4.598629000e+01f, 4.540270000e+01f,
                4.481911000e+01f, 4.423552000e+01f, 4.365193000e+01f, 4.306834000e+01f, 4.248475000e+01f,
                4.190116000e+01f, 4.131757000e+01f, 4.073398000e+01f, 4.746952000e+01f, 4.687564000e+01f,
                4.628176000e+01f, 4.568788000e+01f, 4.509400000e+01f, 4.450012000e+01f, 4.390624000e+01f,
                4.331236000e+01f, 4.271848000e+01f, 4.212460000e+01f, 4.153072000e+01f, 5.466664000e+01f,
                5.399044000e+01f, 5.331424000e+01f, 5.263804000e+01f, 5.196184000e+01f, 5.128564000e+01f,
                5.060944000e+01f, 4.993324000e+01f, 4.925704000e+01f, 4.858084000e+01f, 4.790464000e+01f,
                5.556628000e+01f, 5.487979000e+01f, 5.419330000e+01f, 5.350681000e+01f, 5.282032000e+01f,
                5.213383000e+01f, 5.144734000e+01f, 5.076085000e+01f, 5.007436000e+01f, 4.938787000e+01f,
                4.870138000e+01f, 5.646592000e+01f, 5.576914000e+01f, 5.507236000e+01f, 5.437558000e+01f,
                5.367880000e+01f, 5.298202000e+01f, 5.228524000e+01f, 5.158846000e+01f, 5.089168000e+01f,
                5.019490000e+01f, 4.949812000e+01f, 5.736556000e+01f, 5.665849000e+01f, 5.595142000e+01f,
                5.524435000e+01f, 5.453728000e+01f, 5.383021000e+01f, 5.312314000e+01f, 5.241607000e+01f,
                5.170900000e+01f, 5.100193000e+01f, 5.029486000e+01f, 5.826520000e+01f, 5.754784000e+01f,
                5.683048000e+01f, 5.611312000e+01f, 5.539576000e+01f, 5.467840000e+01f, 5.396104000e+01f,
                5.324368000e+01f, 5.252632000e+01f, 5.180896000e+01f, 5.109160000e+01f, 5.916484000e+01f,
                5.843719000e+01f, 5.770954000e+01f, 5.698189000e+01f, 5.625424000e+01f, 5.552659000e+01f,
                5.479894000e+01f, 5.407129000e+01f, 5.334364000e+01f, 5.261599000e+01f, 5.188834000e+01f,
                6.636196000e+01f, 6.555199000e+01f, 6.474202000e+01f, 6.393205000e+01f, 6.312208000e+01f,
                6.231211000e+01f, 6.150214000e+01f, 6.069217000e+01f, 5.988220000e+01f, 5.907223000e+01f,
                5.826226000e+01f, 6.726160000e+01f, 6.644134000e+01f, 6.562108000e+01f, 6.480082000e+01f,
                6.398056000e+01f, 6.316030000e+01f, 6.234004000e+01f, 6.151978000e+01f, 6.069952000e+01f,
                5.987926000e+01f, 5.905900000e+01f, 6.816124000e+01f, 6.733069000e+01f, 6.650014000e+01f,
                6.566959000e+01f, 6.483904000e+01f, 6.400849000e+01f, 6.317794000e+01f, 6.234739000e+01f,
                6.151684000e+01f, 6.068629000e+01f, 5.985574000e+01f, 6.906088000e+01f, 6.822004000e+01f,
                6.737920000e+01f, 6.653836000e+01f, 6.569752000e+01f, 6.485668000e+01f, 6.401584000e+01f,
                6.317500000e+01f, 6.233416000e+01f, 6.149332000e+01f, 6.065248000e+01f, 6.996052000e+01f,
                6.910939000e+01f, 6.825826000e+01f, 6.740713000e+01f, 6.655600000e+01f, 6.570487000e+01f,
                6.485374000e+01f, 6.400261000e+01f, 6.315148000e+01f, 6.230035000e+01f, 6.144922000e+01f,
                7.086016000e+01f, 6.999874000e+01f, 6.913732000e+01f, 6.827590000e+01f, 6.741448000e+01f,
                6.655306000e+01f, 6.569164000e+01f, 6.483022000e+01f, 6.396880000e+01f, 6.310738000e+01f,
                6.224596000e+01f, 7.805728000e+01f, 7.711354000e+01f, 7.616980000e+01f, 7.522606000e+01f,
                7.428232000e+01f, 7.333858000e+01f, 7.239484000e+01f, 7.145110000e+01f, 7.050736000e+01f,
                6.956362000e+01f, 6.861988000e+01f, 7.895692000e+01f, 7.800289000e+01f, 7.704886000e+01f,
                7.609483000e+01f, 7.514080000e+01f, 7.418677000e+01f, 7.323274000e+01f, 7.227871000e+01f,
                7.132468000e+01f, 7.037065000e+01f, 6.941662000e+01f, 7.985656000e+01f, 7.889224000e+01f,
                7.792792000e+01f, 7.696360000e+01f, 7.599928000e+01f, 7.503496000e+01f, 7.407064000e+01f,
                7.310632000e+01f, 7.214200000e+01f, 7.117768000e+01f, 7.021336000e+01f, 8.075620000e+01f,
                7.978159000e+01f, 7.880698000e+01f, 7.783237000e+01f, 7.685776000e+01f, 7.588315000e+01f,
                7.490854000e+01f, 7.393393000e+01f, 7.295932000e+01f, 7.198471000e+01f, 7.101010000e+01f,
                8.165584000e+01f, 8.067094000e+01f, 7.968604000e+01f, 7.870114000e+01f, 7.771624000e+01f,
                7.673134000e+01f, 7.574644000e+01f, 7.476154000e+01f, 7.377664000e+01f, 7.279174000e+01f,
                7.180684000e+01f, 8.255548000e+01f, 8.156029000e+01f, 8.056510000e+01f, 7.956991000e+01f,
                7.857472000e+01f, 7.757953000e+01f, 7.658434000e+01f, 7.558915000e+01f, 7.459396000e+01f,
                7.359877000e+01f, 7.260358000e+01f, 1.072955800e+02f, 1.060174150e+02f, 1.047392500e+02f,
                1.034610850e+02f, 1.021829200e+02f, 1.009047550e+02f, 9.962659000e+01f, 9.834842500e+01f,
                9.707026000e+01f, 9.579209500e+01f, 9.451393000e+01f, 1.081952200e+02f, 1.069067650e+02f,
                1.056183100e+02f, 1.043298550e+02f, 1.030414000e+02f, 1.017529450e+02f, 1.004644900e+02f,
                9.917603500e+01f, 9.788758000e+01f, 9.659912500e+01f, 9.531067000e+01f, 1.090948600e+02f,
                1.077961150e+02f, 1.064973700e+02f, 1.051986250e+02f, 1.038998800e+02f, 1.026011350e+02f,
                1.013023900e+02f, 1.000036450e+02f, 9.870490000e+01f, 9.740615500e+01f, 9.610741000e+01f,
                1.099945000e+02f, 1.086854650e+02f, 1.073764300e+02f, 1.060673950e+02f, 1.047583600e+02f,
                1.034493250e+02f, 1.021402900e+02f, 1.008312550e+02f, 9.952222000e+01f, 9.821318500e+01f,
                9.690415000e+01f, 1.108941400e+02f, 1.095748150e+02f, 1.082554900e+02f, 1.069361650e+02f,
                1.056168400e+02f, 1.042975150e+02f, 1.029781900e+02f, 1.016588650e+02f, 1.003395400e+02f,
                9.902021500e+01f, 9.770089000e+01f, 1.117937800e+02f, 1.104641650e+02f, 1.091345500e+02f,
                1.078049350e+02f, 1.064753200e+02f, 1.051457050e+02f, 1.038160900e+02f, 1.024864750e+02f,
                1.011568600e+02f, 9.982724500e+01f, 9.849763000e+01f, 1.189909000e+02f, 1.175789650e+02f,
                1.161670300e+02f, 1.147550950e+02f, 1.133431600e+02f, 1.119312250e+02f, 1.105192900e+02f,
                1.091073550e+02f, 1.076954200e+02f, 1.062834850e+02f, 1.048715500e+02f, 1.198905400e+02f,
                1.184683150e+02f, 1.170460900e+02f, 1.156238650e+02f, 1.142016400e+02f, 1.127794150e+02f,
                1.113571900e+02f, 1.099349650e+02f, 1.085127400e+02f, 1.070905150e+02f, 1.056682900e+02f,
                1.207901800e+02f, 1.193576650e+02f, 1.179251500e+02f, 1.164926350e+02f, 1.150601200e+02f,
                1.136276050e+02f, 1.121950900e+02f, 1.107625750e+02f, 1.093300600e+02f, 1.078975450e+02f,
                1.064650300e+02f, 1.216898200e+02f, 1.202470150e+02f, 1.188042100e+02f, 1.173614050e+02f,
                1.159186000e+02f, 1.144757950e+02f, 1.130329900e+02f, 1.115901850e+02f, 1.101473800e+02f,
                1.087045750e+02f, 1.072617700e+02f, 1.225894600e+02f, 1.211363650e+02f, 1.196832700e+02f,
                1.182301750e+02f, 1.167770800e+02f, 1.153239850e+02f, 1.138708900e+02f, 1.124177950e+02f,
                1.109647000e+02f, 1.095116050e+02f, 1.080585100e+02f, 1.234891000e+02f, 1.220257150e+02f,
                1.205623300e+02f, 1.190989450e+02f, 1.176355600e+02f, 1.161721750e+02f, 1.147087900e+02f,
                1.132454050e+02f, 1.117820200e+02f, 1.103186350e+02f, 1.088552500e+02f, 1.306862200e+02f,
                1.291405150e+02f, 1.275948100e+02f, 1.260491050e+02f, 1.245034000e+02f, 1.229576950e+02f,
                1.214119900e+02f, 1.198662850e+02f, 1.183205800e+02f, 1.167748750e+02f, 1.152291700e+02f,
                1.315858600e+02f, 1.300298650e+02f, 1.284738700e+02f, 1.269178750e+02f, 1.253618800e+02f,
                1.238058850e+02f, 1.222498900e+02f, 1.206938950e+02f, 1.191379000e+02f, 1.175819050e+02f,
                1.160259100e+02f, 1.324855000e+02f, 1.309192150e+02f, 1.293529300e+02f, 1.277866450e+02f,
                1.262203600e+02f, 1.246540750e+02f, 1.230877900e+02f, 1.215215050e+02f, 1.199552200e+02f,
                1.183889350e+02f, 1.168226500e+02f, 1.333851400e+02f, 1.318085650e+02f, 1.302319900e+02f,
                1.286554150e+02f, 1.270788400e+02f, 1.255022650e+02f, 1.239256900e+02f, 1.223491150e+02f,
                1.207725400e+02f, 1.191959650e+02f, 1.176193900e+02f, 1.342847800e+02f, 1.326979150e+02f,
                1.311110500e+02f, 1.295241850e+02f, 1.279373200e+02f, 1.263504550e+02f, 1.247635900e+02f,
                1.231767250e+02f, 1.215898600e+02f, 1.200029950e+02f, 1.184161300e+02f, 1.351844200e+02f,
                1.335872650e+02f, 1.319901100e+02f, 1.303929550e+02f, 1.287958000e+02f, 1.271986450e+02f,
                1.256014900e+02f, 1.240043350e+02f, 1.224071800e+02f, 1.208100250e+02f, 1.192128700e+02f,
                1.423815400e+02f, 1.407020650e+02f, 1.390225900e+02f, 1.373431150e+02f, 1.356636400e+02f,
                1.339841650e+02f, 1.323046900e+02f, 1.306252150e+02f, 1.289457400e+02f, 1.272662650e+02f,
                1.255867900e+02f, 1.432811800e+02f, 1.415914150e+02f, 1.399016500e+02f, 1.382118850e+02f,
                1.365221200e+02f, 1.348323550e+02f, 1.331425900e+02f, 1.314528250e+02f, 1.297630600e+02f,
                1.280732950e+02f, 1.263835300e+02f, 1.441808200e+02f, 1.424807650e+02f, 1.407807100e+02f,
                1.390806550e+02f, 1.373806000e+02f, 1.356805450e+02f, 1.339804900e+02f, 1.322804350e+02f,
                1.305803800e+02f, 1.288803250e+02f, 1.271802700e+02f, 1.450804600e+02f, 1.433701150e+02f,
                1.416597700e+02f, 1.399494250e+02f, 1.382390800e+02f, 1.365287350e+02f, 1.348183900e+02f,
                1.331080450e+02f, 1.313977000e+02f, 1.296873550e+02f, 1.279770100e+02f, 1.459801000e+02f,
                1.442594650e+02f, 1.425388300e+02f, 1.408181950e+02f, 1.390975600e+02f, 1.373769250e+02f,
                1.356562900e+02f, 1.339356550e+02f, 1.322150200e+02f, 1.304943850e+02f, 1.287737500e+02f,
                1.468797400e+02f, 1.451488150e+02f, 1.434178900e+02f, 1.416869650e+02f, 1.399560400e+02f,
                1.382251150e+02f, 1.364941900e+02f, 1.347632650e+02f, 1.330323400e+02f, 1.313014150e+02f,
                1.295704900e+02f, 1.540768600e+02f, 1.522636150e+02f, 1.504503700e+02f, 1.486371250e+02f,
                1.468238800e+02f, 1.450106350e+02f, 1.431973900e+02f, 1.413841450e+02f, 1.395709000e+02f,
                1.377576550e+02f, 1.359444100e+02f, 1.549765000e+02f, 1.531529650e+02f, 1.513294300e+02f,
                1.495058950e+02f, 1.476823600e+02f, 1.458588250e+02f, 1.440352900e+02f, 1.422117550e+02f,
                1.403882200e+02f, 1.385646850e+02f, 1.367411500e+02f, 1.558761400e+02f, 1.540423150e+02f,
                1.522084900e+02f, 1.503746650e+02f, 1.485408400e+02f, 1.467070150e+02f, 1.448731900e+02f,
                1.430393650e+02f, 1.412055400e+02f, 1.393717150e+02f, 1.375378900e+02f, 1.567757800e+02f,
                1.549316650e+02f, 1.530875500e+02f, 1.512434350e+02f, 1.493993200e+02f, 1.475552050e+02f,
                1.457110900e+02f, 1.438669750e+02f, 1.420228600e+02f, 1.401787450e+02f, 1.383346300e+02f,
                1.576754200e+02f, 1.558210150e+02f, 1.539666100e+02f, 1.521122050e+02f, 1.502578000e+02f,
                1.484033950e+02f, 1.465489900e+02f, 1.446945850e+02f, 1.428401800e+02f, 1.409857750e+02f,
                1.391313700e+02f, 1.585750600e+02f, 1.567103650e+02f, 1.548456700e+02f, 1.529809750e+02f,
                1.511162800e+02f, 1.492515850e+02f, 1.473868900e+02f, 1.455221950e+02f, 1.436575000e+02f,
                1.417928050e+02f, 1.399281100e+02f, 1.657721800e+02f, 1.638251650e+02f, 1.618781500e+02f,
                1.599311350e+02f, 1.579841200e+02f, 1.560371050e+02f, 1.540900900e+02f, 1.521430750e+02f,
                1.501960600e+02f, 1.482490450e+02f, 1.463020300e+02f, 1.666718200e+02f, 1.647145150e+02f,
                1.627572100e+02f, 1.607999050e+02f, 1.588426000e+02f, 1.568852950e+02f, 1.549279900e+02f,
                1.529706850e+02f, 1.510133800e+02f, 1.490560750e+02f, 1.470987700e+02f, 1.675714600e+02f,
                1.656038650e+02f, 1.636362700e+02f, 1.616686750e+02f, 1.597010800e+02f, 1.577334850e+02f,
                1.557658900e+02f, 1.537982950e+02f, 1.518307000e+02f, 1.498631050e+02f, 1.478955100e+02f,
                1.684711000e+02f, 1.664932150e+02f, 1.645153300e+02f, 1.625374450e+02f, 1.605595600e+02f,
                1.585816750e+02f, 1.566037900e+02f, 1.546259050e+02f, 1.526480200e+02f, 1.506701350e+02f,
                1.486922500e+02f, 1.693707400e+02f, 1.673825650e+02f, 1.653943900e+02f, 1.634062150e+02f,
                1.614180400e+02f, 1.594298650e+02f, 1.574416900e+02f, 1.554535150e+02f, 1.534653400e+02f,
                1.514771650e+02f, 1.494889900e+02f, 1.702703800e+02f, 1.682719150e+02f, 1.662734500e+02f,
                1.642749850e+02f, 1.622765200e+02f, 1.602780550e+02f, 1.582795900e+02f, 1.562811250e+02f,
                1.542826600e+02f, 1.522841950e+02f, 1.502857300e+02f, 1.774675000e+02f, 1.753867150e+02f,
                1.733059300e+02f, 1.712251450e+02f, 1.691443600e+02f, 1.670635750e+02f, 1.649827900e+02f,
                1.629020050e+02f, 1.608212200e+02f, 1.587404350e+02f, 1.566596500e+02f, 1.783671400e+02f,
                1.762760650e+02f, 1.741849900e+02f, 1.720939150e+02f, 1.700028400e+02f, 1.679117650e+02f,
                1.658206900e+02f, 1.637296150e+02f, 1.616385400e+02f, 1.595474650e+02f, 1.574563900e+02f,
                1.792667800e+02f, 1.771654150e+02f, 1.750640500e+02f, 1.729626850e+02f, 1.708613200e+02f,
                1.687599550e+02f, 1.666585900e+02f, 1.645572250e+02f, 1.624558600e+02f, 1.603544950e+02f,
                1.582531300e+02f, 1.801664200e+02f, 1.780547650e+02f, 1.759431100e+02f, 1.738314550e+02f,
                1.717198000e+02f, 1.696081450e+02f, 1.674964900e+02f, 1.653848350e+02f, 1.632731800e+02f,
                1.611615250e+02f, 1.590498700e+02f, 1.810660600e+02f, 1.789441150e+02f, 1.768221700e+02f,
                1.747002250e+02f, 1.725782800e+02f, 1.704563350e+02f, 1.683343900e+02f, 1.662124450e+02f,
                1.640905000e+02f, 1.619685550e+02f, 1.598466100e+02f, 1.819657000e+02f, 1.798334650e+02f,
                1.777012300e+02f, 1.755689950e+02f, 1.734367600e+02f, 1.713045250e+02f, 1.691722900e+02f,
                1.670400550e+02f, 1.649078200e+02f, 1.627755850e+02f, 1.606433500e+02f
            };

            float[] y_actual = y.ToArray();

            AssertError.Tolerance(y_expect, y_actual, 1e-7f, 1e-5f, $"mismatch value {inchannels},{outchannels},{kwidth},{kheight},{stride},{inwidth},{inheight},{batch}");
        }

        [TestMethod]
        public void OptimizeTest() {
            float max_err = 0;

            foreach (int batch in new int[] { 1, 2 }) {
                foreach (int inchannels in new int[] { 1, 2, 3, 4, 5, 10, 15, 20 }) {
                    foreach (int outchannels in new int[] { 7, 13 }) {
                        foreach (int kheight in new int[] { 1, 3, 5 }) {
                            foreach (int kwidth in new int[] { 1, 3, 5 }) {
                                foreach (int stride in new int[] { 1, 2, 3 }) {
                                    foreach (int inwidth in new int[] { 8, 9, 13, 17 }) {
                                        foreach (int inheight in new int[] { 8, 9, 19, 23 }) {
                                            int outwidth = inwidth - kwidth + 1, outheight = inheight - kheight + 1;

                                            float[] xval = (new float[inwidth * inheight * inchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
                                            float[] wval = (new float[kwidth * kheight * inchannels * outchannels]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

                                            Map2D x = new Map2D(inchannels, inwidth, inheight, batch, xval);
                                            Filter2D w = new Filter2D(inchannels, outchannels, kwidth, kheight, wval);

                                            Map2D y = Reference(x, w, kwidth, kheight, stride);
                                            Map2D y_optimized = OptimizedReference(x, w, kwidth, kheight, stride);

                                            float[] y_expect = y.ToArray();
                                            float[] y_actual = y_optimized.ToArray();

                                            AssertError.Tolerance(y_expect, y_actual, 1e-7f, 1e-5f, ref max_err, $"mismatch value {inchannels},{outchannels},{kwidth},{kheight},{stride},{inwidth},{inheight},{batch}");

                                            Console.WriteLine($"pass: {inchannels},{outchannels},{kwidth},{kheight},{stride},{inwidth},{inheight},{batch}");
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
    }
}
