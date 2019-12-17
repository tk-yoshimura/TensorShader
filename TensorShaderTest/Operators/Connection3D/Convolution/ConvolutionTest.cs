using System;
using System.Diagnostics;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using TensorShader.Operators.Connection3D;
using TensorShaderCudaBackend.API;

namespace TensorShaderTest.Operators.Connection3D {
    [TestClass]
    public class ConvolutionTest {
        [TestMethod]
        public void ExecuteTest() {
            float max_err = 0;

            TestCaseIterator iterator = new TestCaseIterator(
                times: 5,
                new int[] { 1, 2 },
                new int[] { 1, 2, 3, 4, 5, 10, 15, 20, 32, 33 },
                new int[] { 1, 2, 3, 4, 5, 10, 15, 20, 32, 33 },
                new int[] { 1, 3, 5 },
                new int[] { 1, 3, 5 },
                new int[] { 1, 3, 5 },
                new int[] { 5, 7, 12, 13, 17, 19 }, 
                new int[] { 5, 7, 12, 13, 17, 19 },
                new int[] { 5, 7, 12, 13, 17, 19 }
            );

            foreach(int[] testcase in iterator) {  
                int batch = testcase[0], inchannels = testcase[1], outchannels = testcase[2];
                int kwidth = testcase[3], kheight = testcase[4], kdepth = testcase[5];
                int inwidth = testcase[6], inheight = testcase[7], indepth = testcase[8];
                                
                int outwidth = inwidth - kwidth + 1, outheight = inheight - kheight + 1, outdepth = indepth - kdepth + 1;

                float[] xval = (new float[inwidth * inheight * indepth * inchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
                float[] wval = (new float[kwidth * kheight * kdepth * inchannels * outchannels]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

                Map3D x = new Map3D(inchannels, inwidth, inheight, indepth, batch, xval);
                Filter3D w = new Filter3D(inchannels, outchannels, kwidth, kheight, kdepth, wval);

                Map3D y = Reference(x, w, kwidth, kheight, kdepth);

                OverflowCheckedTensor x_tensor = new OverflowCheckedTensor(Shape.Map3D(inchannels, inwidth, inheight, indepth, batch), xval);
                OverflowCheckedTensor w_tensor = new OverflowCheckedTensor(Shape.Kernel3D(inchannels, outchannels, kwidth, kheight, kdepth), wval);

                OverflowCheckedTensor y_tensor = new OverflowCheckedTensor(Shape.Map3D(outchannels, outwidth, outheight, outdepth, batch));

                Convolution ope = new Convolution(inwidth, inheight, indepth, inchannels, outchannels, kwidth, kheight, kdepth, batch);

                ope.Execute(x_tensor, w_tensor, y_tensor);

                float[] y_expect = y.ToArray();
                float[] y_actual = y_tensor.State;

                CollectionAssert.AreEqual(xval, x_tensor.State);
                CollectionAssert.AreEqual(wval, w_tensor.State);

                AssertError.Tolerance(y_expect, y_actual, 1e-7f, 1e-5f, ref max_err, $"mismatch value {inchannels},{outchannels},{kwidth},{kheight},{kdepth},{inwidth},{inheight},{batch}");

                Console.WriteLine($"pass: {inchannels},{outchannels},{kwidth},{kheight},{kdepth},{inwidth},{inheight},{batch}");
            }

            Console.WriteLine($"maxerr:{max_err}");
        }

        [TestMethod]
        public void SpeedTest() {
            int inwidth = 64, inheight = 64, indepth = 64, inchannels = 31, outchannels = 31, ksize = 3;
            int outwidth = inwidth - ksize + 1, outheight = inheight - ksize + 1, outdepth = indepth - ksize + 1;

            OverflowCheckedTensor x_tensor = new OverflowCheckedTensor(Shape.Map3D(inchannels, inwidth, inheight, indepth));
            OverflowCheckedTensor w_tensor = new OverflowCheckedTensor(Shape.Kernel3D(inchannels, outchannels, ksize, ksize, ksize));

            OverflowCheckedTensor y_tensor = new OverflowCheckedTensor(Shape.Map3D(outchannels, outwidth, outheight, outdepth));

            Convolution ope = new Convolution(inwidth, inheight, indepth, inchannels, outchannels, ksize, ksize, ksize);

            Cuda.Profiler.Initialize("../../../profiler.nvsetting", "../../nvprofiles/convolution3d_trans_v2.nvvp");
            Cuda.Profiler.Start();

            ope.Execute(x_tensor, w_tensor, y_tensor);

            Cuda.Profiler.Stop();
        }

        public static Map3D Reference(Map3D x, Filter3D w, int kwidth, int kheight, int kdepth) {
            int inchannels = x.Channels, outchannels = w.OutChannels, batch = x.Batch;
            int inw = x.Width, inh = x.Height, ind = x.Depth;
            int outw = inw - kwidth + 1, outh = inh - kheight + 1, outd = ind - kdepth + 1;

            Map3D y = new Map3D(outchannels, outw, outh, outd, batch);

            for (int kx, ky, kz = 0; kz < kdepth; kz++) {
                for (ky = 0; ky < kheight; ky++) {
                    for (kx = 0; kx < kwidth; kx++) {
                        for (int th = 0; th < batch; th++) {
                            for (int ox, oy, oz = 0; oz < outd; oz++) {
                                for (oy = 0; oy < outh; oy++) {
                                    for (ox = 0; ox < outw; ox++) {
                                        for (int outch = 0; outch < outchannels; outch++) {
                                            double sum = y[outch, ox, oy, oz, th];

                                            for (int inch = 0; inch < inchannels; inch++) {
                                                sum += x[inch, kx + ox, ky + oy, kz + oz, th] * w[inch, outch, kx, ky, kz];
                                            }

                                            y[outch, ox, oy, oz, th] = sum;
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
            int inchannels = x.Channels, outchannels = w.OutChannels, batch = x.Batch;
            int inw = x.Width, inh = x.Height, ind = x.Depth;
            int outw = inw - kwidth + 1, outh = inh - kheight + 1, outd = ind - kdepth + 1;

            Map3D y = new Map3D(outchannels, outw, outh, outd, batch);

            for (int kx, ky, kz = 0; kz < kdepth; kz++) {
                for (ky = 0; ky < kheight; ky++) {
                    for (kx = 0; kx < kwidth; kx++) {
                        int inmap_offset = (kx + ky * inw + kz * inw * inh) * inchannels;
                        int kernel_offset = (kx + ky * kwidth + kz * kwidth * kheight) * inchannels * outchannels;

                        for (int th = 0; th < batch; th++) {
                            for (int ox, oy, oz = 0; oz < outd; oz++) {
                                for (oy = 0; oy < outh; oy++) {
                                    for (ox = 0; ox < outw; ox++) {
                                        int inmap_org = inmap_offset + (ox + oy * inw + oz * inw * inh) * inchannels + th * inw * inh * ind * inchannels;
                                        int outmap_idx = (ox + oy * outw + oz * outw * outh) * outchannels + th * outw * outh * outd * outchannels;
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
                }
            }

            return y;
        }

        [TestMethod]
        public void ReferenceTest() {
            int inchannels = 2, outchannels = 3, kwidth = 3, kheight = 5, kdepth = 7, inwidth = 13, inheight = 12, indepth = 11, batch = 2;
            int outwidth = inwidth - kwidth + 1, outheight = inheight - kheight + 1, outdepth = indepth - kdepth + 1;

            float[] xval = (new float[batch * inwidth * inheight * indepth * inchannels]).Select((_, idx) => idx * 1e-3f).ToArray();
            float[] wval = (new float[kwidth * kheight * kdepth * outchannels * inchannels]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

            Map3D x = new Map3D(inchannels, inwidth, inheight, indepth, batch, xval);
            Filter3D w = new Filter3D(inchannels, outchannels, kwidth, kheight, kdepth, wval);

            Map3D y = Reference(x, w, kwidth, kheight, kdepth);

            float[] y_expect = {
                4.204809000e+01f,  4.163208000e+01f,  4.121607000e+01f,  4.231395000e+01f,
                4.189626000e+01f,  4.147857000e+01f,  4.257981000e+01f,  4.216044000e+01f,
                4.174107000e+01f,  4.284567000e+01f,  4.242462000e+01f,  4.200357000e+01f,
                4.311153000e+01f,  4.268880000e+01f,  4.226607000e+01f,  4.337739000e+01f,
                4.295298000e+01f,  4.252857000e+01f,  4.550427000e+01f,  4.506642000e+01f,
                4.462857000e+01f,  4.577013000e+01f,  4.533060000e+01f,  4.489107000e+01f,
                4.603599000e+01f,  4.559478000e+01f,  4.515357000e+01f,  4.630185000e+01f,
                4.585896000e+01f,  4.541607000e+01f,  4.656771000e+01f,  4.612314000e+01f,
                4.567857000e+01f,  4.683357000e+01f,  4.638732000e+01f,  4.594107000e+01f,
                4.896045000e+01f,  4.850076000e+01f,  4.804107000e+01f,  4.922631000e+01f,
                4.876494000e+01f,  4.830357000e+01f,  4.949217000e+01f,  4.902912000e+01f,
                4.856607000e+01f,  4.975803000e+01f,  4.929330000e+01f,  4.882857000e+01f,
                5.002389000e+01f,  4.955748000e+01f,  4.909107000e+01f,  5.028975000e+01f,
                4.982166000e+01f,  4.935357000e+01f,  5.241663000e+01f,  5.193510000e+01f,
                5.145357000e+01f,  5.268249000e+01f,  5.219928000e+01f,  5.171607000e+01f,
                5.294835000e+01f,  5.246346000e+01f,  5.197857000e+01f,  5.321421000e+01f,
                5.272764000e+01f,  5.224107000e+01f,  5.348007000e+01f,  5.299182000e+01f,
                5.250357000e+01f,  5.374593000e+01f,  5.325600000e+01f,  5.276607000e+01f,
                8.352225000e+01f,  8.284416000e+01f,  8.216607000e+01f,  8.378811000e+01f,
                8.310834000e+01f,  8.242857000e+01f,  8.405397000e+01f,  8.337252000e+01f,
                8.269107000e+01f,  8.431983000e+01f,  8.363670000e+01f,  8.295357000e+01f,
                8.458569000e+01f,  8.390088000e+01f,  8.321607000e+01f,  8.485155000e+01f,
                8.416506000e+01f,  8.347857000e+01f,  8.697843000e+01f,  8.627850000e+01f,
                8.557857000e+01f,  8.724429000e+01f,  8.654268000e+01f,  8.584107000e+01f,
                8.751015000e+01f,  8.680686000e+01f,  8.610357000e+01f,  8.777601000e+01f,
                8.707104000e+01f,  8.636607000e+01f,  8.804187000e+01f,  8.733522000e+01f,
                8.662857000e+01f,  8.830773000e+01f,  8.759940000e+01f,  8.689107000e+01f,
                9.043461000e+01f,  8.971284000e+01f,  8.899107000e+01f,  9.070047000e+01f,
                8.997702000e+01f,  8.925357000e+01f,  9.096633000e+01f,  9.024120000e+01f,
                8.951607000e+01f,  9.123219000e+01f,  9.050538000e+01f,  8.977857000e+01f,
                9.149805000e+01f,  9.076956000e+01f,  9.004107000e+01f,  9.176391000e+01f,
                9.103374000e+01f,  9.030357000e+01f,  9.389079000e+01f,  9.314718000e+01f,
                9.240357000e+01f,  9.415665000e+01f,  9.341136000e+01f,  9.266607000e+01f,
                9.442251000e+01f,  9.367554000e+01f,  9.292857000e+01f,  9.468837000e+01f,
                9.393972000e+01f,  9.319107000e+01f,  9.495423000e+01f,  9.420390000e+01f,
                9.345357000e+01f,  9.522009000e+01f,  9.446808000e+01f,  9.371607000e+01f,
                1.249964100e+02f,  1.240562400e+02f,  1.231160700e+02f,  1.252622700e+02f,
                1.243204200e+02f,  1.233785700e+02f,  1.255281300e+02f,  1.245846000e+02f,
                1.236410700e+02f,  1.257939900e+02f,  1.248487800e+02f,  1.239035700e+02f,
                1.260598500e+02f,  1.251129600e+02f,  1.241660700e+02f,  1.263257100e+02f,
                1.253771400e+02f,  1.244285700e+02f,  1.284525900e+02f,  1.274905800e+02f,
                1.265285700e+02f,  1.287184500e+02f,  1.277547600e+02f,  1.267910700e+02f,
                1.289843100e+02f,  1.280189400e+02f,  1.270535700e+02f,  1.292501700e+02f,
                1.282831200e+02f,  1.273160700e+02f,  1.295160300e+02f,  1.285473000e+02f,
                1.275785700e+02f,  1.297818900e+02f,  1.288114800e+02f,  1.278410700e+02f,
                1.319087700e+02f,  1.309249200e+02f,  1.299410700e+02f,  1.321746300e+02f,
                1.311891000e+02f,  1.302035700e+02f,  1.324404900e+02f,  1.314532800e+02f,
                1.304660700e+02f,  1.327063500e+02f,  1.317174600e+02f,  1.307285700e+02f,
                1.329722100e+02f,  1.319816400e+02f,  1.309910700e+02f,  1.332380700e+02f,
                1.322458200e+02f,  1.312535700e+02f,  1.353649500e+02f,  1.343592600e+02f,
                1.333535700e+02f,  1.356308100e+02f,  1.346234400e+02f,  1.336160700e+02f,
                1.358966700e+02f,  1.348876200e+02f,  1.338785700e+02f,  1.361625300e+02f,
                1.351518000e+02f,  1.341410700e+02f,  1.364283900e+02f,  1.354159800e+02f,
                1.344035700e+02f,  1.366942500e+02f,  1.356801600e+02f,  1.346660700e+02f,
                2.701559700e+02f,  2.682985200e+02f,  2.664410700e+02f,  2.704218300e+02f,
                2.685627000e+02f,  2.667035700e+02f,  2.706876900e+02f,  2.688268800e+02f,
                2.669660700e+02f,  2.709535500e+02f,  2.690910600e+02f,  2.672285700e+02f,
                2.712194100e+02f,  2.693552400e+02f,  2.674910700e+02f,  2.714852700e+02f,
                2.696194200e+02f,  2.677535700e+02f,  2.736121500e+02f,  2.717328600e+02f,
                2.698535700e+02f,  2.738780100e+02f,  2.719970400e+02f,  2.701160700e+02f,
                2.741438700e+02f,  2.722612200e+02f,  2.703785700e+02f,  2.744097300e+02f,
                2.725254000e+02f,  2.706410700e+02f,  2.746755900e+02f,  2.727895800e+02f,
                2.709035700e+02f,  2.749414500e+02f,  2.730537600e+02f,  2.711660700e+02f,
                2.770683300e+02f,  2.751672000e+02f,  2.732660700e+02f,  2.773341900e+02f,
                2.754313800e+02f,  2.735285700e+02f,  2.776000500e+02f,  2.756955600e+02f,
                2.737910700e+02f,  2.778659100e+02f,  2.759597400e+02f,  2.740535700e+02f,
                2.781317700e+02f,  2.762239200e+02f,  2.743160700e+02f,  2.783976300e+02f,
                2.764881000e+02f,  2.745785700e+02f,  2.805245100e+02f,  2.786015400e+02f,
                2.766785700e+02f,  2.807903700e+02f,  2.788657200e+02f,  2.769410700e+02f,
                2.810562300e+02f,  2.791299000e+02f,  2.772035700e+02f,  2.813220900e+02f,
                2.793940800e+02f,  2.774660700e+02f,  2.815879500e+02f,  2.796582600e+02f,
                2.777285700e+02f,  2.818538100e+02f,  2.799224400e+02f,  2.779910700e+02f,
                3.116301300e+02f,  3.095106000e+02f,  3.073910700e+02f,  3.118959900e+02f,
                3.097747800e+02f,  3.076535700e+02f,  3.121618500e+02f,  3.100389600e+02f,
                3.079160700e+02f,  3.124277100e+02f,  3.103031400e+02f,  3.081785700e+02f,
                3.126935700e+02f,  3.105673200e+02f,  3.084410700e+02f,  3.129594300e+02f,
                3.108315000e+02f,  3.087035700e+02f,  3.150863100e+02f,  3.129449400e+02f,
                3.108035700e+02f,  3.153521700e+02f,  3.132091200e+02f,  3.110660700e+02f,
                3.156180300e+02f,  3.134733000e+02f,  3.113285700e+02f,  3.158838900e+02f,
                3.137374800e+02f,  3.115910700e+02f,  3.161497500e+02f,  3.140016600e+02f,
                3.118535700e+02f,  3.164156100e+02f,  3.142658400e+02f,  3.121160700e+02f,
                3.185424900e+02f,  3.163792800e+02f,  3.142160700e+02f,  3.188083500e+02f,
                3.166434600e+02f,  3.144785700e+02f,  3.190742100e+02f,  3.169076400e+02f,
                3.147410700e+02f,  3.193400700e+02f,  3.171718200e+02f,  3.150035700e+02f,
                3.196059300e+02f,  3.174360000e+02f,  3.152660700e+02f,  3.198717900e+02f,
                3.177001800e+02f,  3.155285700e+02f,  3.219986700e+02f,  3.198136200e+02f,
                3.176285700e+02f,  3.222645300e+02f,  3.200778000e+02f,  3.178910700e+02f,
                3.225303900e+02f,  3.203419800e+02f,  3.181535700e+02f,  3.227962500e+02f,
                3.206061600e+02f,  3.184160700e+02f,  3.230621100e+02f,  3.208703400e+02f,
                3.186785700e+02f,  3.233279700e+02f,  3.211345200e+02f,  3.189410700e+02f,
                3.531042900e+02f,  3.507226800e+02f,  3.483410700e+02f,  3.533701500e+02f,
                3.509868600e+02f,  3.486035700e+02f,  3.536360100e+02f,  3.512510400e+02f,
                3.488660700e+02f,  3.539018700e+02f,  3.515152200e+02f,  3.491285700e+02f,
                3.541677300e+02f,  3.517794000e+02f,  3.493910700e+02f,  3.544335900e+02f,
                3.520435800e+02f,  3.496535700e+02f,  3.565604700e+02f,  3.541570200e+02f,
                3.517535700e+02f,  3.568263300e+02f,  3.544212000e+02f,  3.520160700e+02f,
                3.570921900e+02f,  3.546853800e+02f,  3.522785700e+02f,  3.573580500e+02f,
                3.549495600e+02f,  3.525410700e+02f,  3.576239100e+02f,  3.552137400e+02f,
                3.528035700e+02f,  3.578897700e+02f,  3.554779200e+02f,  3.530660700e+02f,
                3.600166500e+02f,  3.575913600e+02f,  3.551660700e+02f,  3.602825100e+02f,
                3.578555400e+02f,  3.554285700e+02f,  3.605483700e+02f,  3.581197200e+02f,
                3.556910700e+02f,  3.608142300e+02f,  3.583839000e+02f,  3.559535700e+02f,
                3.610800900e+02f,  3.586480800e+02f,  3.562160700e+02f,  3.613459500e+02f,
                3.589122600e+02f,  3.564785700e+02f,  3.634728300e+02f,  3.610257000e+02f,
                3.585785700e+02f,  3.637386900e+02f,  3.612898800e+02f,  3.588410700e+02f,
                3.640045500e+02f,  3.615540600e+02f,  3.591035700e+02f,  3.642704100e+02f,
                3.618182400e+02f,  3.593660700e+02f,  3.645362700e+02f,  3.620824200e+02f,
                3.596285700e+02f,  3.648021300e+02f,  3.623466000e+02f,  3.598910700e+02f,
            };

            float[] y_actual = y.ToArray();

            AssertError.Tolerance(y_expect, y_actual, 1e-7f, 1e-5f, $"mismatch value {inchannels},{outchannels},{kwidth},{kheight},{kdepth},{inwidth},{inheight},{batch}");
        }

        [TestMethod]
        public void OptimizeTest() {
            float max_err = 0;

            foreach (int batch in new int[] { 1, 2 }) {
                foreach (int inchannels in new int[] { 1, 2, 3, 4, 5, 10, 15, 20 }) {
                    foreach (int outchannels in new int[] { 7, 13 }) {
                        foreach ((int kwidth, int kheight, int kdepth) in new (int, int, int)[] { (1, 1, 1), (3, 3, 3), (5, 5, 5), (1, 3, 5), (3, 5, 1), (5, 1, 3) }) {
                            foreach ((int inwidth, int inheight, int indepth) in new (int, int, int)[] { (13, 13, 13), (17, 17, 17), (19, 19, 19), (17, 19, 13), (13, 17, 19), (19, 13, 17) }) {
                                int outwidth = inwidth - kwidth + 1, outheight = inheight - kheight + 1, outdepth = indepth - kdepth + 1;

                                float[] xval = (new float[inwidth * inheight * indepth * inchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
                                float[] wval = (new float[kwidth * kheight * kdepth * inchannels * outchannels]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

                                Map3D x = new Map3D(inchannels, inwidth, inheight, indepth, batch, xval);
                                Filter3D w = new Filter3D(inchannels, outchannels, kwidth, kheight, kdepth, wval);

                                Map3D y = Reference(x, w, kwidth, kheight, kdepth);
                                Map3D y_optimized = OptimizedReference(x, w, kwidth, kheight, kdepth);

                                float[] y_expect = y.ToArray();
                                float[] y_actual = y_optimized.ToArray();

                                AssertError.Tolerance(y_expect, y_actual, 1e-7f, 1e-5f, ref max_err, $"mismatch value {inchannels},{outchannels},{kwidth},{kheight},{kdepth},{inwidth},{inheight},{batch}");

                                Console.WriteLine($"pass: {inchannels},{outchannels},{kwidth},{kheight},{kdepth},{inwidth},{inheight},{indepth},{batch}");
                            }
                        }
                    }
                }
            }

            Console.WriteLine($"maxerr:{max_err}");
        }
    }
}
