using System;
using System.Diagnostics;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using TensorShader.Operators.Connection2D;
using TensorShaderCudaBackend.API;

namespace TensorShaderTest.Operators.Connection2D {
    [TestClass]
    public class ChannelwiseConvolutionTest {
        [TestMethod]
        public void ExecuteTest() {
            float max_err = 0;

            foreach (int batch in new int[] { 1, 2 }) {
                foreach (int channels in new int[] { 1, 2, 3, 4, 5, 10, 15, 20 }) {
                    foreach (int kheight in new int[] { 1, 3, 5 }) {
                        foreach (int kwidth in new int[] { 1, 3, 5 }) {
                            foreach (int inwidth in new int[] { 8, 9, 13, 17 }) {
                                foreach (int inheight in new int[] { 8, 9, 19, 23 }) {
                                    int outwidth = inwidth - kwidth + 1, outheight = inheight - kheight + 1;

                                    float[] xval = (new float[inwidth * inheight * channels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
                                    float[] wval = (new float[kwidth * kheight * channels]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

                                    Map2D x = new Map2D(channels, inwidth, inheight, batch, xval);
                                    Filter2D w = new Filter2D(channels, 1, kwidth, kheight, wval);

                                    Map2D y = Reference(x, w, kwidth, kheight);

                                    OverflowCheckedTensor x_tensor = new OverflowCheckedTensor(Shape.Map2D(channels, inwidth, inheight, batch), xval);
                                    OverflowCheckedTensor w_tensor = new OverflowCheckedTensor(Shape.Kernel2D(channels, 1, kwidth, kheight), wval);

                                    OverflowCheckedTensor y_tensor = new OverflowCheckedTensor(Shape.Map2D(channels, outwidth, outheight, batch));

                                    ChannelwiseConvolution ope = new ChannelwiseConvolution(inwidth, inheight, channels, kwidth, kheight, batch);

                                    ope.Execute(x_tensor, w_tensor, y_tensor);

                                    float[] y_expect = y.ToArray();
                                    float[] y_actual = y_tensor.State;

                                    CollectionAssert.AreEqual(xval, x_tensor.State);
                                    CollectionAssert.AreEqual(wval, w_tensor.State);

                                    AssertError.Tolerance(y_expect, y_actual, 1e-7f, 1e-5f, ref max_err, $"mismatch value {channels},{kwidth},{inwidth},{inheight},{batch}");

                                    Console.WriteLine($"pass: {channels},{kwidth},{kheight},{inwidth},{inheight},{batch}");
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
            int inwidth = 512, inheight = 512, channels = 31, ksize = 3;
            int outwidth = inwidth - ksize + 1, outheight = inheight - ksize + 1;

            OverflowCheckedTensor x_tensor = new OverflowCheckedTensor(Shape.Map2D(channels, inwidth, inheight));
            OverflowCheckedTensor w_tensor = new OverflowCheckedTensor(Shape.Kernel2D(channels, 1, ksize, ksize));

            OverflowCheckedTensor y_tensor = new OverflowCheckedTensor(Shape.Map2D(channels, outwidth, outheight));

            ChannelwiseConvolution ope = new ChannelwiseConvolution(inwidth, inheight, channels, ksize, ksize);

            Cuda.Profiler.Initialize("../../../profiler.nvsetting", "../../nvprofiles/chwise_convolution_2d.nvvp");
            Cuda.Profiler.Start();

            ope.Execute(x_tensor, w_tensor, y_tensor);
            
            Cuda.Profiler.Stop();
        }

        public static Map2D Reference(Map2D x, Filter2D w, int kwidth, int kheight) {
            int channels = x.Channels, batch = x.Batch;
            int inw = x.Width, inh = x.Height, outw = inw - kwidth + 1, outh = inh - kheight + 1;

            Map2D y = new Map2D(channels, outw, outh, batch);

            for (int kx, ky = 0; ky < kheight; ky++) {
                for (kx = 0; kx < kwidth; kx++) {
                    for (int th = 0; th < batch; th++) {
                        for (int ox, oy = 0; oy < outh; oy++) {
                            for (ox = 0; ox < outw; ox++) {
                                for (int ch = 0; ch < channels; ch++) {
                                    y[ch, ox, oy, th] += x[ch, kx + ox, ky + oy, th] * w[ch, 0, kx, ky];
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
            int channels = 7, kwidth = 3, kheight = 5, inwidth = 13, inheight = 17, batch = 2;
            int outwidth = inwidth - kwidth + 1, outheight = inheight - kheight + 1;

            float[] xval = (new float[batch * inwidth * inheight * channels]).Select((_, idx) => idx * 1e-3f).ToArray();
            float[] wval = (new float[kwidth * kheight * channels]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

            Map2D x = new Map2D(channels, inwidth, inheight, batch, xval);
            Filter2D w = new Filter2D(channels, 1, kwidth, kheight, wval);

            Map2D y = Reference(x, w, kwidth, kheight);

            float[] y_expect = {
                9.8105000e-02f,  9.6080000e-02f,  9.4025000e-02f,  9.1940000e-02f,  8.9825000e-02f,  8.7680000e-02f,
                8.5505000e-02f,  1.0965500e-01f,  1.0742000e-01f,  1.0515500e-01f,  1.0286000e-01f,  1.0053500e-01f,
                9.8180000e-02f,  9.5795000e-02f,  1.2120500e-01f,  1.1876000e-01f,  1.1628500e-01f,  1.1378000e-01f,
                1.1124500e-01f,  1.0868000e-01f,  1.0608500e-01f,  1.3275500e-01f,  1.3010000e-01f,  1.2741500e-01f,
                1.2470000e-01f,  1.2195500e-01f,  1.1918000e-01f,  1.1637500e-01f,  1.4430500e-01f,  1.4144000e-01f,
                1.3854500e-01f,  1.3562000e-01f,  1.3266500e-01f,  1.2968000e-01f,  1.2666500e-01f,  1.5585500e-01f,
                1.5278000e-01f,  1.4967500e-01f,  1.4654000e-01f,  1.4337500e-01f,  1.4018000e-01f,  1.3695500e-01f,
                2.4825500e-01f,  2.4350000e-01f,  2.3871500e-01f,  2.3390000e-01f,  2.2905500e-01f,  2.2418000e-01f,
                2.1927500e-01f,  2.5980500e-01f,  2.5484000e-01f,  2.4984500e-01f,  2.4482000e-01f,  2.3976500e-01f,
                2.3468000e-01f,  2.2956500e-01f,  2.7135500e-01f,  2.6618000e-01f,  2.6097500e-01f,  2.5574000e-01f,
                2.5047500e-01f,  2.4518000e-01f,  2.3985500e-01f,  2.8290500e-01f,  2.7752000e-01f,  2.7210500e-01f,
                2.6666000e-01f,  2.6118500e-01f,  2.5568000e-01f,  2.5014500e-01f,  2.9445500e-01f,  2.8886000e-01f,
                2.8323500e-01f,  2.7758000e-01f,  2.7189500e-01f,  2.6618000e-01f,  2.6043500e-01f,  3.0600500e-01f,
                3.0020000e-01f,  2.9436500e-01f,  2.8850000e-01f,  2.8260500e-01f,  2.7668000e-01f,  2.7072500e-01f,
                3.9840500e-01f,  3.9092000e-01f,  3.8340500e-01f,  3.7586000e-01f,  3.6828500e-01f,  3.6068000e-01f,
                3.5304500e-01f,  4.0995500e-01f,  4.0226000e-01f,  3.9453500e-01f,  3.8678000e-01f,  3.7899500e-01f,
                3.7118000e-01f,  3.6333500e-01f,  4.2150500e-01f,  4.1360000e-01f,  4.0566500e-01f,  3.9770000e-01f,
                3.8970500e-01f,  3.8168000e-01f,  3.7362500e-01f,  4.3305500e-01f,  4.2494000e-01f,  4.1679500e-01f,
                4.0862000e-01f,  4.0041500e-01f,  3.9218000e-01f,  3.8391500e-01f,  4.4460500e-01f,  4.3628000e-01f,
                4.2792500e-01f,  4.1954000e-01f,  4.1112500e-01f,  4.0268000e-01f,  3.9420500e-01f,  4.5615500e-01f,
                4.4762000e-01f,  4.3905500e-01f,  4.3046000e-01f,  4.2183500e-01f,  4.1318000e-01f,  4.0449500e-01f,
                5.4855500e-01f,  5.3834000e-01f,  5.2809500e-01f,  5.1782000e-01f,  5.0751500e-01f,  4.9718000e-01f,
                4.8681500e-01f,  5.6010500e-01f,  5.4968000e-01f,  5.3922500e-01f,  5.2874000e-01f,  5.1822500e-01f,
                5.0768000e-01f,  4.9710500e-01f,  5.7165500e-01f,  5.6102000e-01f,  5.5035500e-01f,  5.3966000e-01f,
                5.2893500e-01f,  5.1818000e-01f,  5.0739500e-01f,  5.8320500e-01f,  5.7236000e-01f,  5.6148500e-01f,
                5.5058000e-01f,  5.3964500e-01f,  5.2868000e-01f,  5.1768500e-01f,  5.9475500e-01f,  5.8370000e-01f,
                5.7261500e-01f,  5.6150000e-01f,  5.5035500e-01f,  5.3918000e-01f,  5.2797500e-01f,  6.0630500e-01f,
                5.9504000e-01f,  5.8374500e-01f,  5.7242000e-01f,  5.6106500e-01f,  5.4968000e-01f,  5.3826500e-01f,
                6.9870500e-01f,  6.8576000e-01f,  6.7278500e-01f,  6.5978000e-01f,  6.4674500e-01f,  6.3368000e-01f,
                6.2058500e-01f,  7.1025500e-01f,  6.9710000e-01f,  6.8391500e-01f,  6.7070000e-01f,  6.5745500e-01f,
                6.4418000e-01f,  6.3087500e-01f,  7.2180500e-01f,  7.0844000e-01f,  6.9504500e-01f,  6.8162000e-01f,
                6.6816500e-01f,  6.5468000e-01f,  6.4116500e-01f,  7.3335500e-01f,  7.1978000e-01f,  7.0617500e-01f,
                6.9254000e-01f,  6.7887500e-01f,  6.6518000e-01f,  6.5145500e-01f,  7.4490500e-01f,  7.3112000e-01f,
                7.1730500e-01f,  7.0346000e-01f,  6.8958500e-01f,  6.7568000e-01f,  6.6174500e-01f,  7.5645500e-01f,
                7.4246000e-01f,  7.2843500e-01f,  7.1438000e-01f,  7.0029500e-01f,  6.8618000e-01f,  6.7203500e-01f,
                8.4885500e-01f,  8.3318000e-01f,  8.1747500e-01f,  8.0174000e-01f,  7.8597500e-01f,  7.7018000e-01f,
                7.5435500e-01f,  8.6040500e-01f,  8.4452000e-01f,  8.2860500e-01f,  8.1266000e-01f,  7.9668500e-01f,
                7.8068000e-01f,  7.6464500e-01f,  8.7195500e-01f,  8.5586000e-01f,  8.3973500e-01f,  8.2358000e-01f,
                8.0739500e-01f,  7.9118000e-01f,  7.7493500e-01f,  8.8350500e-01f,  8.6720000e-01f,  8.5086500e-01f,
                8.3450000e-01f,  8.1810500e-01f,  8.0168000e-01f,  7.8522500e-01f,  8.9505500e-01f,  8.7854000e-01f,
                8.6199500e-01f,  8.4542000e-01f,  8.2881500e-01f,  8.1218000e-01f,  7.9551500e-01f,  9.0660500e-01f,
                8.8988000e-01f,  8.7312500e-01f,  8.5634000e-01f,  8.3952500e-01f,  8.2268000e-01f,  8.0580500e-01f,
                9.9900500e-01f,  9.8060000e-01f,  9.6216500e-01f,  9.4370000e-01f,  9.2520500e-01f,  9.0668000e-01f,
                8.8812500e-01f,  1.0105550e+00f,  9.9194000e-01f,  9.7329500e-01f,  9.5462000e-01f,  9.3591500e-01f,
                9.1718000e-01f,  8.9841500e-01f,  1.0221050e+00f,  1.0032800e+00f,  9.8442500e-01f,  9.6554000e-01f,
                9.4662500e-01f,  9.2768000e-01f,  9.0870500e-01f,  1.0336550e+00f,  1.0146200e+00f,  9.9555500e-01f,
                9.7646000e-01f,  9.5733500e-01f,  9.3818000e-01f,  9.1899500e-01f,  1.0452050e+00f,  1.0259600e+00f,
                1.0066850e+00f,  9.8738000e-01f,  9.6804500e-01f,  9.4868000e-01f,  9.2928500e-01f,  1.0567550e+00f,
                1.0373000e+00f,  1.0178150e+00f,  9.9830000e-01f,  9.7875500e-01f,  9.5918000e-01f,  9.3957500e-01f,
                1.3743800e+00f,  1.3491500e+00f,  1.3238900e+00f,  1.2986000e+00f,  1.2732800e+00f,  1.2479300e+00f,
                1.2225500e+00f,  1.3859300e+00f,  1.3604900e+00f,  1.3350200e+00f,  1.3095200e+00f,  1.2839900e+00f,
                1.2584300e+00f,  1.2328400e+00f,  1.3974800e+00f,  1.3718300e+00f,  1.3461500e+00f,  1.3204400e+00f,
                1.2947000e+00f,  1.2689300e+00f,  1.2431300e+00f,  1.4090300e+00f,  1.3831700e+00f,  1.3572800e+00f,
                1.3313600e+00f,  1.3054100e+00f,  1.2794300e+00f,  1.2534200e+00f,  1.4205800e+00f,  1.3945100e+00f,
                1.3684100e+00f,  1.3422800e+00f,  1.3161200e+00f,  1.2899300e+00f,  1.2637100e+00f,  1.4321300e+00f,
                1.4058500e+00f,  1.3795400e+00f,  1.3532000e+00f,  1.3268300e+00f,  1.3004300e+00f,  1.2740000e+00f,
                1.5245300e+00f,  1.4965700e+00f,  1.4685800e+00f,  1.4405600e+00f,  1.4125100e+00f,  1.3844300e+00f,
                1.3563200e+00f,  1.5360800e+00f,  1.5079100e+00f,  1.4797100e+00f,  1.4514800e+00f,  1.4232200e+00f,
                1.3949300e+00f,  1.3666100e+00f,  1.5476300e+00f,  1.5192500e+00f,  1.4908400e+00f,  1.4624000e+00f,
                1.4339300e+00f,  1.4054300e+00f,  1.3769000e+00f,  1.5591800e+00f,  1.5305900e+00f,  1.5019700e+00f,
                1.4733200e+00f,  1.4446400e+00f,  1.4159300e+00f,  1.3871900e+00f,  1.5707300e+00f,  1.5419300e+00f,
                1.5131000e+00f,  1.4842400e+00f,  1.4553500e+00f,  1.4264300e+00f,  1.3974800e+00f,  1.5822800e+00f,
                1.5532700e+00f,  1.5242300e+00f,  1.4951600e+00f,  1.4660600e+00f,  1.4369300e+00f,  1.4077700e+00f,
                1.6746800e+00f,  1.6439900e+00f,  1.6132700e+00f,  1.5825200e+00f,  1.5517400e+00f,  1.5209300e+00f,
                1.4900900e+00f,  1.6862300e+00f,  1.6553300e+00f,  1.6244000e+00f,  1.5934400e+00f,  1.5624500e+00f,
                1.5314300e+00f,  1.5003800e+00f,  1.6977800e+00f,  1.6666700e+00f,  1.6355300e+00f,  1.6043600e+00f,
                1.5731600e+00f,  1.5419300e+00f,  1.5106700e+00f,  1.7093300e+00f,  1.6780100e+00f,  1.6466600e+00f,
                1.6152800e+00f,  1.5838700e+00f,  1.5524300e+00f,  1.5209600e+00f,  1.7208800e+00f,  1.6893500e+00f,
                1.6577900e+00f,  1.6262000e+00f,  1.5945800e+00f,  1.5629300e+00f,  1.5312500e+00f,  1.7324300e+00f,
                1.7006900e+00f,  1.6689200e+00f,  1.6371200e+00f,  1.6052900e+00f,  1.5734300e+00f,  1.5415400e+00f,
                1.8248300e+00f,  1.7914100e+00f,  1.7579600e+00f,  1.7244800e+00f,  1.6909700e+00f,  1.6574300e+00f,
                1.6238600e+00f,  1.8363800e+00f,  1.8027500e+00f,  1.7690900e+00f,  1.7354000e+00f,  1.7016800e+00f,
                1.6679300e+00f,  1.6341500e+00f,  1.8479300e+00f,  1.8140900e+00f,  1.7802200e+00f,  1.7463200e+00f,
                1.7123900e+00f,  1.6784300e+00f,  1.6444400e+00f,  1.8594800e+00f,  1.8254300e+00f,  1.7913500e+00f,
                1.7572400e+00f,  1.7231000e+00f,  1.6889300e+00f,  1.6547300e+00f,  1.8710300e+00f,  1.8367700e+00f,
                1.8024800e+00f,  1.7681600e+00f,  1.7338100e+00f,  1.6994300e+00f,  1.6650200e+00f,  1.8825800e+00f,
                1.8481100e+00f,  1.8136100e+00f,  1.7790800e+00f,  1.7445200e+00f,  1.7099300e+00f,  1.6753100e+00f,
                1.9749800e+00f,  1.9388300e+00f,  1.9026500e+00f,  1.8664400e+00f,  1.8302000e+00f,  1.7939300e+00f,
                1.7576300e+00f,  1.9865300e+00f,  1.9501700e+00f,  1.9137800e+00f,  1.8773600e+00f,  1.8409100e+00f,
                1.8044300e+00f,  1.7679200e+00f,  1.9980800e+00f,  1.9615100e+00f,  1.9249100e+00f,  1.8882800e+00f,
                1.8516200e+00f,  1.8149300e+00f,  1.7782100e+00f,  2.0096300e+00f,  1.9728500e+00f,  1.9360400e+00f,
                1.8992000e+00f,  1.8623300e+00f,  1.8254300e+00f,  1.7885000e+00f,  2.0211800e+00f,  1.9841900e+00f,
                1.9471700e+00f,  1.9101200e+00f,  1.8730400e+00f,  1.8359300e+00f,  1.7987900e+00f,  2.0327300e+00f,
                1.9955300e+00f,  1.9583000e+00f,  1.9210400e+00f,  1.8837500e+00f,  1.8464300e+00f,  1.8090800e+00f,
                2.1251300e+00f,  2.0862500e+00f,  2.0473400e+00f,  2.0084000e+00f,  1.9694300e+00f,  1.9304300e+00f,
                1.8914000e+00f,  2.1366800e+00f,  2.0975900e+00f,  2.0584700e+00f,  2.0193200e+00f,  1.9801400e+00f,
                1.9409300e+00f,  1.9016900e+00f,  2.1482300e+00f,  2.1089300e+00f,  2.0696000e+00f,  2.0302400e+00f,
                1.9908500e+00f,  1.9514300e+00f,  1.9119800e+00f,  2.1597800e+00f,  2.1202700e+00f,  2.0807300e+00f,
                2.0411600e+00f,  2.0015600e+00f,  1.9619300e+00f,  1.9222700e+00f,  2.1713300e+00f,  2.1316100e+00f,
                2.0918600e+00f,  2.0520800e+00f,  2.0122700e+00f,  1.9724300e+00f,  1.9325600e+00f,  2.1828800e+00f,
                2.1429500e+00f,  2.1029900e+00f,  2.0630000e+00f,  2.0229800e+00f,  1.9829300e+00f,  1.9428500e+00f,
                2.2752800e+00f,  2.2336700e+00f,  2.1920300e+00f,  2.1503600e+00f,  2.1086600e+00f,  2.0669300e+00f,
                2.0251700e+00f,  2.2868300e+00f,  2.2450100e+00f,  2.2031600e+00f,  2.1612800e+00f,  2.1193700e+00f,
                2.0774300e+00f,  2.0354600e+00f,  2.2983800e+00f,  2.2563500e+00f,  2.2142900e+00f,  2.1722000e+00f,
                2.1300800e+00f,  2.0879300e+00f,  2.0457500e+00f,  2.3099300e+00f,  2.2676900e+00f,  2.2254200e+00f,
                2.1831200e+00f,  2.1407900e+00f,  2.0984300e+00f,  2.0560400e+00f,  2.3214800e+00f,  2.2790300e+00f,
                2.2365500e+00f,  2.1940400e+00f,  2.1515000e+00f,  2.1089300e+00f,  2.0663300e+00f,  2.3330300e+00f,
                2.2903700e+00f,  2.2476800e+00f,  2.2049600e+00f,  2.1622100e+00f,  2.1194300e+00f,  2.0766200e+00f,
            };

            float[] y_actual = y.ToArray();

            AssertError.Tolerance(y_expect, y_actual, 1e-7f, 1e-5f, $"mismatch value {channels},{kwidth},{inwidth},{inheight},{batch}");
        }
    }
}
