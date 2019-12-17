using System;
using System.Diagnostics;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using TensorShader.Operators.Connection3D;

namespace TensorShaderTest.Operators.Connection3D {
    [TestClass]
    public class ChannelwiseConvolutionTest {
        [TestMethod]
        public void ExecuteTest() {
            float max_err = 0;

            foreach (int batch in new int[] { 1, 2 }) {
                foreach (int channels in new int[] { 1, 2, 3, 4, 5, 10, 15, 20 }) {
                    foreach ((int kwidth, int kheight, int kdepth) in new (int, int, int)[] { (1, 1, 1), (3, 3, 3), (5, 5, 5), (1, 3, 5), (3, 5, 1), (5, 1, 3) }) {
                        foreach ((int inwidth, int inheight, int indepth) in new (int, int, int)[] { (13, 13, 13), (17, 17, 17), (19, 19, 19), (17, 19, 13), (13, 17, 19), (19, 13, 17) }) {
                            int outwidth = inwidth - kwidth + 1, outheight = inheight - kheight + 1, outdepth = indepth - kdepth + 1;

                            float[] xval = (new float[inwidth * inheight * indepth * channels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
                            float[] wval = (new float[kwidth * kheight * kdepth * channels]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

                            Map3D x = new Map3D(channels, inwidth, inheight, indepth, batch, xval);
                            Filter3D w = new Filter3D(channels, 1, kwidth, kheight, kdepth, wval);

                            Map3D y = Reference(x, w, kwidth, kheight, kdepth);

                            OverflowCheckedTensor x_tensor = new OverflowCheckedTensor(Shape.Map3D(channels, inwidth, inheight, indepth, batch), xval);
                            OverflowCheckedTensor w_tensor = new OverflowCheckedTensor(Shape.Kernel3D(channels, 1, kwidth, kheight, kdepth), wval);

                            OverflowCheckedTensor y_tensor = new OverflowCheckedTensor(Shape.Map3D(channels, outwidth, outheight, outdepth, batch));

                            ChannelwiseConvolution ope = new ChannelwiseConvolution(inwidth, inheight, indepth, channels, kwidth, kheight, kdepth, batch);

                            ope.Execute(x_tensor, w_tensor, y_tensor);

                            float[] y_expect = y.ToArray();
                            float[] y_actual = y_tensor.State;

                            CollectionAssert.AreEqual(xval, x_tensor.State);
                            CollectionAssert.AreEqual(wval, w_tensor.State);

                            AssertError.Tolerance(y_expect, y_actual, 1e-7f, 1e-5f, ref max_err, $"mismatch value {channels},{kwidth},{kheight},{inwidth},{inheight},{indepth},{batch}");

                            Console.WriteLine($"pass: {channels},{kwidth},{kheight},{inwidth},{inheight},{indepth},{batch}");
                        }
                    }
                }
            }

            Console.WriteLine($"maxerr:{max_err}");
        }

        [TestMethod]
        public void SpeedTest() {
            int inwidth = 64, inheight = 64, indepth = 64, channels = 31, ksize = 3;
            int outwidth = inwidth - ksize + 1, outheight = inheight - ksize + 1, outdepth = indepth - ksize + 1;

            OverflowCheckedTensor x_tensor = new OverflowCheckedTensor(Shape.Map3D(channels, inwidth, inheight, indepth));
            OverflowCheckedTensor w_tensor = new OverflowCheckedTensor(Shape.Kernel3D(channels, 1, ksize, ksize, ksize));

            OverflowCheckedTensor y_tensor = new OverflowCheckedTensor(Shape.Map3D(channels, outwidth, outheight, outdepth));

            ChannelwiseConvolution ope = new ChannelwiseConvolution(inwidth, inheight, indepth, channels, ksize, ksize, ksize);

            Stopwatch sw = new Stopwatch();
            sw.Start();

            ope.Execute(x_tensor, w_tensor, y_tensor);
            ope.Execute(x_tensor, w_tensor, y_tensor);
            ope.Execute(x_tensor, w_tensor, y_tensor);
            ope.Execute(x_tensor, w_tensor, y_tensor);

            sw.Stop();

            Console.WriteLine($"{sw.ElapsedMilliseconds / 4} msec");
        }

        public static Map3D Reference(Map3D x, Filter3D w, int kwidth, int kheight, int kdepth) {
            int channels = x.Channels, batch = x.Batch;
            int inw = x.Width, inh = x.Height, ind = x.Depth;
            int outw = inw - kwidth + 1, outh = inh - kheight + 1, outd = ind - kdepth + 1;

            Map3D y = new Map3D(channels, outw, outh, outd, batch);

            for (int kx, ky, kz = 0; kz < kdepth; kz++) {
                for (ky = 0; ky < kheight; ky++) {
                    for (kx = 0; kx < kwidth; kx++) {
                        for (int th = 0; th < batch; th++) {
                            for (int ox, oy, oz = 0; oz < outd; oz++) {
                                for (oy = 0; oy < outh; oy++) {
                                    for (ox = 0; ox < outw; ox++) {
                                        for (int ch = 0; ch < channels; ch++) {
                                            y[ch, ox, oy, oz, th] += x[ch, kx + ox, ky + oy, kz + oz, th] * w[ch, 0, kx, ky, kz];
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

        public static Map3D OptimizedReference(Map3D x, Filter3D w, int kwidth, int kheight, int kdepth) {
            int channels = x.Channels, batch = x.Batch;
            int inw = x.Width, inh = x.Height, ind = x.Depth;
            int outw = inw - kwidth + 1, outh = inh - kheight + 1, outd = ind - kdepth + 1;

            Map3D y = new Map3D(channels, outw, outh, outd, batch);

            for (int kx, ky, kz = 0; kz < kdepth; kz++) {
                for (ky = 0; ky < kheight; ky++) {
                    for (kx = 0; kx < kwidth; kx++) {
                        int inmap_offset = (kx + ky * inw + kz * inw * inh) * channels;
                        int kernel_offset = (kx + ky * kwidth + kz * kwidth * kheight) * channels;

                        for (int th = 0; th < batch; th++) {
                            for (int ox, oy, oz = 0; oz < outd; oz++) {
                                for (oy = 0; oy < outh; oy++) {
                                    for (ox = 0; ox < outw; ox++) {
                                        int inmap_idx = inmap_offset + (ox + oy * inw + oz * inw * inh) * channels + th * inw * inh * ind * channels;
                                        int outmap_idx = (ox + oy * outw + oz * outw * outh) * channels + th * outw * outh * outd * channels;
                                        int kernel_idx = kernel_offset;

                                        for (int inch = 0; inch < channels; inch++) {
                                            y[outmap_idx] += x[inmap_idx] * w[kernel_idx];

                                            inmap_idx++;
                                            outmap_idx++;
                                            kernel_idx++;
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
            int channels = 2, kwidth = 3, kheight = 5, kdepth = 7, inwidth = 13, inheight = 12, indepth = 11, batch = 2;
            int outwidth = inwidth - kwidth + 1, outheight = inheight - kheight + 1, outdepth = indepth - kdepth + 1;

            float[] xval = (new float[batch * inwidth * inheight * indepth * channels]).Select((_, idx) => idx * 1e-3f).ToArray();
            float[] wval = (new float[kwidth * kheight * kdepth * channels]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

            Map3D x = new Map3D(channels, inwidth, inheight, indepth, batch, xval);
            Filter3D w = new Filter3D(channels, 1, kwidth, kheight, kdepth, wval);

            Map3D y = Reference(x, w, kwidth, kheight, kdepth);

            float[] y_expect = {
                6.9505100e+00f,  6.8574800e+00f,  6.9946100e+00f,  6.9011600e+00f,  7.0387100e+00f,  6.9448400e+00f,
                7.0828100e+00f,  6.9885200e+00f,  7.1269100e+00f,  7.0322000e+00f,  7.1710100e+00f,  7.0758800e+00f,
                7.5238100e+00f,  7.4253200e+00f,  7.5679100e+00f,  7.4690000e+00f,  7.6120100e+00f,  7.5126800e+00f,
                7.6561100e+00f,  7.5563600e+00f,  7.7002100e+00f,  7.6000400e+00f,  7.7443100e+00f,  7.6437200e+00f,
                8.0971100e+00f,  7.9931600e+00f,  8.1412100e+00f,  8.0368400e+00f,  8.1853100e+00f,  8.0805200e+00f,
                8.2294100e+00f,  8.1242000e+00f,  8.2735100e+00f,  8.1678800e+00f,  8.3176100e+00f,  8.2115600e+00f,
                8.6704100e+00f,  8.5610000e+00f,  8.7145100e+00f,  8.6046800e+00f,  8.7586100e+00f,  8.6483600e+00f,
                8.8027100e+00f,  8.6920400e+00f,  8.8468100e+00f,  8.7357200e+00f,  8.8909100e+00f,  8.7794000e+00f,
                1.3830110e+01f,  1.3671560e+01f,  1.3874210e+01f,  1.3715240e+01f,  1.3918310e+01f,  1.3758920e+01f,
                1.3962410e+01f,  1.3802600e+01f,  1.4006510e+01f,  1.3846280e+01f,  1.4050610e+01f,  1.3889960e+01f,
                1.4403410e+01f,  1.4239400e+01f,  1.4447510e+01f,  1.4283080e+01f,  1.4491610e+01f,  1.4326760e+01f,
                1.4535710e+01f,  1.4370440e+01f,  1.4579810e+01f,  1.4414120e+01f,  1.4623910e+01f,  1.4457800e+01f,
                1.4976710e+01f,  1.4807240e+01f,  1.5020810e+01f,  1.4850920e+01f,  1.5064910e+01f,  1.4894600e+01f,
                1.5109010e+01f,  1.4938280e+01f,  1.5153110e+01f,  1.4981960e+01f,  1.5197210e+01f,  1.5025640e+01f,
                1.5550010e+01f,  1.5375080e+01f,  1.5594110e+01f,  1.5418760e+01f,  1.5638210e+01f,  1.5462440e+01f,
                1.5682310e+01f,  1.5506120e+01f,  1.5726410e+01f,  1.5549800e+01f,  1.5770510e+01f,  1.5593480e+01f,
                2.0709710e+01f,  2.0485640e+01f,  2.0753810e+01f,  2.0529320e+01f,  2.0797910e+01f,  2.0573000e+01f,
                2.0842010e+01f,  2.0616680e+01f,  2.0886110e+01f,  2.0660360e+01f,  2.0930210e+01f,  2.0704040e+01f,
                2.1283010e+01f,  2.1053480e+01f,  2.1327110e+01f,  2.1097160e+01f,  2.1371210e+01f,  2.1140840e+01f,
                2.1415310e+01f,  2.1184520e+01f,  2.1459410e+01f,  2.1228200e+01f,  2.1503510e+01f,  2.1271880e+01f,
                2.1856310e+01f,  2.1621320e+01f,  2.1900410e+01f,  2.1665000e+01f,  2.1944510e+01f,  2.1708680e+01f,
                2.1988610e+01f,  2.1752360e+01f,  2.2032710e+01f,  2.1796040e+01f,  2.2076810e+01f,  2.1839720e+01f,
                2.2429610e+01f,  2.2189160e+01f,  2.2473710e+01f,  2.2232840e+01f,  2.2517810e+01f,  2.2276520e+01f,
                2.2561910e+01f,  2.2320200e+01f,  2.2606010e+01f,  2.2363880e+01f,  2.2650110e+01f,  2.2407560e+01f,
                4.4788310e+01f,  4.4334920e+01f,  4.4832410e+01f,  4.4378600e+01f,  4.4876510e+01f,  4.4422280e+01f,
                4.4920610e+01f,  4.4465960e+01f,  4.4964710e+01f,  4.4509640e+01f,  4.5008810e+01f,  4.4553320e+01f,
                4.5361610e+01f,  4.4902760e+01f,  4.5405710e+01f,  4.4946440e+01f,  4.5449810e+01f,  4.4990120e+01f,
                4.5493910e+01f,  4.5033800e+01f,  4.5538010e+01f,  4.5077480e+01f,  4.5582110e+01f,  4.5121160e+01f,
                4.5934910e+01f,  4.5470600e+01f,  4.5979010e+01f,  4.5514280e+01f,  4.6023110e+01f,  4.5557960e+01f,
                4.6067210e+01f,  4.5601640e+01f,  4.6111310e+01f,  4.5645320e+01f,  4.6155410e+01f,  4.5689000e+01f,
                4.6508210e+01f,  4.6038440e+01f,  4.6552310e+01f,  4.6082120e+01f,  4.6596410e+01f,  4.6125800e+01f,
                4.6640510e+01f,  4.6169480e+01f,  4.6684610e+01f,  4.6213160e+01f,  4.6728710e+01f,  4.6256840e+01f,
                5.1667910e+01f,  5.1149000e+01f,  5.1712010e+01f,  5.1192680e+01f,  5.1756110e+01f,  5.1236360e+01f,
                5.1800210e+01f,  5.1280040e+01f,  5.1844310e+01f,  5.1323720e+01f,  5.1888410e+01f,  5.1367400e+01f,
                5.2241210e+01f,  5.1716840e+01f,  5.2285310e+01f,  5.1760520e+01f,  5.2329410e+01f,  5.1804200e+01f,
                5.2373510e+01f,  5.1847880e+01f,  5.2417610e+01f,  5.1891560e+01f,  5.2461710e+01f,  5.1935240e+01f,
                5.2814510e+01f,  5.2284680e+01f,  5.2858610e+01f,  5.2328360e+01f,  5.2902710e+01f,  5.2372040e+01f,
                5.2946810e+01f,  5.2415720e+01f,  5.2990910e+01f,  5.2459400e+01f,  5.3035010e+01f,  5.2503080e+01f,
                5.3387810e+01f,  5.2852520e+01f,  5.3431910e+01f,  5.2896200e+01f,  5.3476010e+01f,  5.2939880e+01f,
                5.3520110e+01f,  5.2983560e+01f,  5.3564210e+01f,  5.3027240e+01f,  5.3608310e+01f,  5.3070920e+01f,
                5.8547510e+01f,  5.7963080e+01f,  5.8591610e+01f,  5.8006760e+01f,  5.8635710e+01f,  5.8050440e+01f,
                5.8679810e+01f,  5.8094120e+01f,  5.8723910e+01f,  5.8137800e+01f,  5.8768010e+01f,  5.8181480e+01f,
                5.9120810e+01f,  5.8530920e+01f,  5.9164910e+01f,  5.8574600e+01f,  5.9209010e+01f,  5.8618280e+01f,
                5.9253110e+01f,  5.8661960e+01f,  5.9297210e+01f,  5.8705640e+01f,  5.9341310e+01f,  5.8749320e+01f,
                5.9694110e+01f,  5.9098760e+01f,  5.9738210e+01f,  5.9142440e+01f,  5.9782310e+01f,  5.9186120e+01f,
                5.9826410e+01f,  5.9229800e+01f,  5.9870510e+01f,  5.9273480e+01f,  5.9914610e+01f,  5.9317160e+01f,
                6.0267410e+01f,  5.9666600e+01f,  6.0311510e+01f,  5.9710280e+01f,  6.0355610e+01f,  5.9753960e+01f,
                6.0399710e+01f,  5.9797640e+01f,  6.0443810e+01f,  5.9841320e+01f,  6.0487910e+01f,  5.9885000e+01f,
            };

            float[] y_actual = y.ToArray();

            AssertError.Tolerance(y_expect, y_actual, 1e-7f, 1e-5f, $"mismatch value {channels},{kwidth},{kheight},{inwidth},{inheight},{batch}");
        }

        [TestMethod]
        public void OptimizeTest() {
            float max_err = 0;

            foreach (int batch in new int[] { 1, 2 }) {
                foreach (int channels in new int[] { 1, 2, 3, 4, 5, 10, 15, 20 }) {
                    foreach ((int kwidth, int kheight, int kdepth) in new (int, int, int)[] { (1, 1, 1), (3, 3, 3), (5, 5, 5), (1, 3, 5), (3, 5, 1), (5, 1, 3) }) {
                        foreach ((int inwidth, int inheight, int indepth) in new (int, int, int)[] { (13, 13, 13), (17, 17, 17), (19, 19, 19), (17, 19, 13), (13, 17, 19), (19, 13, 17) }) {
                            int outwidth = inwidth - kwidth + 1, outheight = inheight - kheight + 1, outdepth = indepth - kdepth + 1;

                            float[] xval = (new float[inwidth * inheight * indepth * channels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
                            float[] wval = (new float[kwidth * kheight * kdepth * channels]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

                            Map3D x = new Map3D(channels, inwidth, inheight, indepth, batch, xval);
                            Filter3D w = new Filter3D(channels, 1, kwidth, kheight, kdepth, wval);

                            Map3D y = Reference(x, w, kwidth, kheight, kdepth);
                            Map3D y_optimized = OptimizedReference(x, w, kwidth, kheight, kdepth);

                            float[] y_expect = y.ToArray();
                            float[] y_actual = y_optimized.ToArray();

                            AssertError.Tolerance(y_expect, y_actual, 1e-7f, 1e-5f, ref max_err, $"mismatch value {channels},{kwidth},{kheight},{inwidth},{inheight},{batch}");

                            Console.WriteLine($"pass: {channels},{kwidth},{kheight},{kdepth},{inwidth},{inheight},{indepth},{batch}");
                        }
                    }
                }
            }

            Console.WriteLine($"maxerr:{max_err}");
        }
    }
}
