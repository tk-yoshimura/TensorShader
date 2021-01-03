using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;
using TensorShader;
using TensorShaderUtil.DataSplitUtil;

using static TensorShader.Field;

namespace TensorShaderUtilTest.DataSplitUtil {
    [TestClass]
    public class PatchworkFlowTest {
        [TestMethod]
        public void Execute1DTest() {
            Shape[] shapes = new Shape[]{
                Shape.Map1D(3, 29, 2),
                Shape.Map1D(3, 30, 2),
                Shape.Map1D(3, 32, 2),
                Shape.Map1D(3, 59, 2),
                Shape.Map1D(3, 64, 2),
                Shape.Map1D(3, 96, 2),
            };

            VariableField input = Shape.Map1D(3, 32, 2);
            StoreField output = input + 1;

            (Flow flow, _) = Flow.Inference(output);

            PatchworkFlow patchwork = new PatchworkFlow(flow, input, output, new int[] { 4 });

            patchwork.ProgressEvent += Patchwork_ProgressEvent;

            Random random = new Random();

            foreach (Shape shape in shapes) {
                NdimArray<float> inmap = (shape, (new float[shape.Length]).Select((_) => (float)random.Next(1, 16)).ToArray());
                NdimArray<float> outmap = patchwork.Execute(inmap);

                CollectionAssert.AreEqual((inmap + 1f).Value, outmap.Value);
            }
        }

        [TestMethod]
        public void Execute1Dx2Test() {
            Shape[] shapes = new Shape[]{
                Shape.Map1D(3, 29, 2),
                Shape.Map1D(3, 30, 2),
                Shape.Map1D(3, 32, 2),
                Shape.Map1D(3, 59, 2),
                Shape.Map1D(3, 64, 2),
                Shape.Map1D(3, 96, 2),
            };

            VariableField input = Shape.Map1D(3, 32, 2);
            StoreField output = NeighborZoom1D(input + 1);;

            (Flow flow, _) = Flow.Inference(output);

            PatchworkFlow patchwork = new PatchworkFlow(flow, input, output, new int[] { 4 });

            patchwork.ProgressEvent += Patchwork_ProgressEvent;

            Random random = new Random();

            foreach (Shape shape in shapes) {
                NdimArray<float> inmap = (shape, (new float[shape.Length]).Select((_) => (float)random.Next(1, 16)).ToArray());
                NdimArray<float> outmap = patchwork.Execute(inmap);

                for (int th = 0; th < outmap.Batch; th++) {
                    for (int x = 0; x < outmap.Width; x++) {
                        for (int c = 0; c < outmap.Channels; c++) {
                            Assert.AreEqual(inmap[c, x / 2, th] + 1, outmap[c, x, th], $"{shape}, {c}, {x}, {th}");
                        }
                    }
                }
            }
        }

        [TestMethod]
        public void Execute1Dd2Test() {
            Shape[] shapes = new Shape[]{
                Shape.Map1D(3, 29, 2),
                Shape.Map1D(3, 30, 2),
                Shape.Map1D(3, 32, 2),
                Shape.Map1D(3, 59, 2),
                Shape.Map1D(3, 64, 2),
                Shape.Map1D(3, 96, 2),
            };

            VariableField input = Shape.Map1D(3, 32, 2);
            StoreField output = AveragePooling1D(input + 1, 2);

            (Flow flow, _) = Flow.Inference(output);

            PatchworkFlow patchwork = new PatchworkFlow(flow, input, output, new int[] { 4 });

            patchwork.ProgressEvent += Patchwork_ProgressEvent;

            Random random = new Random();

            foreach (Shape shape in shapes) {
                NdimArray<float> inmap = (shape, (new float[shape.Length]).Select((_) => (float)random.Next(1, 16)).ToArray());
                NdimArray<float> outmap = patchwork.Execute(inmap);

                for (int th = 0; th < outmap.Batch; th++) {
                    for (int x = 0; x < inmap.Width / 2; x++) {
                        for (int c = 0; c < outmap.Channels; c++) {
                            Assert.AreEqual(
                                (inmap[c, 2 * x, th] + inmap[c, 2 * x + 1, th]) / 2 + 1, 
                                outmap[c, x, th], $"{shape}, {c}, {x}, {th}"
                            );
                        }
                    }
                }
            }
        }

        [TestMethod]
        public void Execute1Dd3Test() {
            Shape[] shapes = new Shape[]{
                Shape.Map1D(3, 29, 2),
                Shape.Map1D(3, 30, 2),
                Shape.Map1D(3, 32, 2),
                Shape.Map1D(3, 59, 2),
                Shape.Map1D(3, 64, 2),
                Shape.Map1D(3, 96, 2),
            };

            VariableField input = Shape.Map1D(3, 30, 2);
            StoreField output = AveragePooling1D(input + 1, 3);

            (Flow flow, _) = Flow.Inference(output);

            PatchworkFlow patchwork = new PatchworkFlow(flow, input, output, new int[] { 4 });

            patchwork.ProgressEvent += Patchwork_ProgressEvent;

            Random random = new Random();

            foreach (Shape shape in shapes) {
                NdimArray<float> inmap = (shape, (new float[shape.Length]).Select((_) => (float)random.Next(1, 16)).ToArray());
                NdimArray<float> outmap = patchwork.Execute(inmap);

                for (int th = 0; th < outmap.Batch; th++) {
                    for (int x = 0; x < inmap.Width / 3; x++) {
                        for (int c = 0; c < outmap.Channels; c++) {
                            Assert.AreEqual(
                                (inmap[c, 3 * x, th] + inmap[c, 3 * x + 1, th] + inmap[c, 3 * x + 2, th]) / 3 + 1, 
                                outmap[c, x, th], 1e-4f, $"{shape}, {c}, {x}, {th}"
                            );
                        }
                    }
                }
            }
        }

        [TestMethod]
        public void Execute2DTest() {
            Shape[] shapes = new Shape[]{
                Shape.Map2D(3, 29,  41, 2),
                Shape.Map2D(3, 30,  42, 2),
                Shape.Map2D(3, 32,  64, 2),
                Shape.Map2D(3, 59, 128, 2),
                Shape.Map2D(3, 64, 128, 2),
                Shape.Map2D(3, 96, 128, 2),
                Shape.Map2D(3, 96,  42, 2),
            };

            VariableField input = Shape.Map2D(3, 32, 64, 2);
            StoreField output = input + 1;

            (Flow flow, _) = Flow.Inference(output);

            PatchworkFlow patchwork = new PatchworkFlow(flow, input, output, new int[] { 4, 3 });

            patchwork.ProgressEvent += Patchwork_ProgressEvent;

            Random random = new Random();

            foreach (Shape shape in shapes) {
                NdimArray<float> inmap = (shape, (new float[shape.Length]).Select((_) => (float)random.Next(1, 16)).ToArray());
                NdimArray<float> outmap = patchwork.Execute(inmap);

                CollectionAssert.AreEqual((inmap + 1f).Value, outmap.Value);
            }
        }

        [TestMethod]
        public void Execute2Dx2Test() {
            Shape[] shapes = new Shape[]{
                Shape.Map2D(3, 29,  41, 2),
                Shape.Map2D(3, 30,  42, 2),
                Shape.Map2D(3, 32,  64, 2),
                Shape.Map2D(3, 59, 128, 2),
                Shape.Map2D(3, 64, 128, 2),
                Shape.Map2D(3, 96, 128, 2),
                Shape.Map2D(3, 96,  42, 2),
            };

            VariableField input = Shape.Map2D(3, 32, 64, 2);
            StoreField output = NeighborZoom2D(input + 1);

            (Flow flow, _) = Flow.Inference(output);

            PatchworkFlow patchwork = new PatchworkFlow(flow, input, output, new int[] { 4, 3 });

            patchwork.ProgressEvent += Patchwork_ProgressEvent;

            Random random = new Random();

            foreach (Shape shape in shapes) {
                NdimArray<float> inmap = (shape, (new float[shape.Length]).Select((_) => (float)random.Next(1, 16)).ToArray());
                NdimArray<float> outmap = patchwork.Execute(inmap);

                for (int th = 0; th < outmap.Batch; th++) {
                    for (int y = 0; y < outmap.Height; y++) { 
                        for (int x = 0; x < outmap.Width; x++) {
                            for (int c = 0; c < outmap.Channels; c++) {
                                Assert.AreEqual(inmap[c, x / 2, y / 2, th] + 1, outmap[c, x, y, th], $"{shape}, {c}, {x}, {y}, {th}");
                            }
                        }
                    }
                }
            }
        }

        [TestMethod]
        public void Execute2Dd2Test() {
            Shape[] shapes = new Shape[]{
                Shape.Map2D(3, 29,  41, 2),
                Shape.Map2D(3, 30,  42, 2),
                Shape.Map2D(3, 32,  64, 2),
                Shape.Map2D(3, 59, 128, 2),
                Shape.Map2D(3, 64, 128, 2),
                Shape.Map2D(3, 96, 128, 2),
                Shape.Map2D(3, 96,  42, 2),
            };

            VariableField input = Shape.Map2D(3, 32, 64, 2);
            StoreField output = AveragePooling2D(input + 1, 2);

            (Flow flow, _) = Flow.Inference(output);

            PatchworkFlow patchwork = new PatchworkFlow(flow, input, output, new int[] { 4, 3 });

            patchwork.ProgressEvent += Patchwork_ProgressEvent;

            Random random = new Random();

            foreach (Shape shape in shapes) {
                NdimArray<float> inmap = (shape, (new float[shape.Length]).Select((_) => (float)random.Next(1, 16)).ToArray());
                NdimArray<float> outmap = patchwork.Execute(inmap);

                for (int th = 0; th < outmap.Batch; th++) {
                    for (int y = 0; y < inmap.Height / 2; y++) { 
                        for (int x = 0; x < inmap.Width / 2; x++) {
                            for (int c = 0; c < outmap.Channels; c++) {
                                Assert.AreEqual(
                                    (inmap[c, 2 * x, 2 * y, th] + inmap[c, 2 * x + 1, 2 * y, th] + inmap[c, 2 * x, 2 * y + 1, th] + inmap[c, 2 * x + 1, 2 * y + 1, th]) / 4 + 1, 
                                    outmap[c, x, y, th], $"{shape}, {c}, {x}, {y}, {th}"
                                );
                            }
                        }
                    }
                }
            }
        }

        [TestMethod]
        public void Execute2Dd3Test() {
            Shape[] shapes = new Shape[]{
                Shape.Map2D(3, 29,  41, 2),
                Shape.Map2D(3, 30,  42, 2),
                Shape.Map2D(3, 32,  64, 2),
                Shape.Map2D(3, 59, 128, 2),
                Shape.Map2D(3, 64, 128, 2),
                Shape.Map2D(3, 96, 128, 2),
                Shape.Map2D(3, 96,  42, 2),
            };

            VariableField input = Shape.Map2D(3, 30, 60, 2);
            StoreField output = AveragePooling2D(input + 1, 3);

            (Flow flow, _) = Flow.Inference(output);

            PatchworkFlow patchwork = new PatchworkFlow(flow, input, output, new int[] { 4, 3 });

            patchwork.ProgressEvent += Patchwork_ProgressEvent;

            Random random = new Random();

            foreach (Shape shape in shapes) {
                NdimArray<float> inmap = (shape, (new float[shape.Length]).Select((_) => (float)random.Next(1, 16)).ToArray());
                NdimArray<float> outmap = patchwork.Execute(inmap);

                for (int th = 0; th < outmap.Batch; th++) {
                    for (int y = 0; y < inmap.Height / 3; y++) { 
                        for (int x = 0; x < inmap.Width / 3; x++) {
                            for (int c = 0; c < outmap.Channels; c++) {
                                Assert.AreEqual(
                                    (inmap[c, 3 * x, 3 * y, th] + inmap[c, 3 * x + 1, 3 * y, th] + inmap[c, 3 * x + 2, 3 * y, th] + 
                                     inmap[c, 3 * x, 3 * y + 1, th] + inmap[c, 3 * x + 1, 3 * y + 1, th] + inmap[c, 3 * x + 2, 3 * y + 1, th] + 
                                     inmap[c, 3 * x, 3 * y + 2, th] + inmap[c, 3 * x + 1, 3 * y + 2, th] + inmap[c, 3 * x + 2, 3 * y + 2, th]) / 9 + 1, 
                                    outmap[c, x, y, th], 1e-4f, $"{shape}, {c}, {x}, {y}, {th}"
                                );
                            }
                        }
                    }
                }
            }
        }

        private void Patchwork_ProgressEvent(int progress, int all) {
            Console.WriteLine($"{progress}/{all}");
        }
    }
}
