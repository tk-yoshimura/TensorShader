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
        public void Execute1DeTest() {
            Shape[] shapes = new Shape[]{
                Shape.Map1D(1, 59, 2),
                Shape.Map1D(1, 64, 2),
                Shape.Map1D(1, 96, 2),
            };

            VariableField input = Shape.Map1D(1, 32, 2);
            VariableField kernel = (Shape.Kernel1D(1, 1, 3), new float[] { 1, 0, -1 });
            StoreField output = Convolution1D(EdgePadding1D(input, 1), kernel);

            (Flow flow, _) = Flow.Inference(output);
            Random random = new Random(1234);

            {
                PatchworkFlow patchwork = new PatchworkFlow(flow, input, output, 1);

                patchwork.ProgressEvent += Patchwork_ProgressEvent;

                foreach (Shape shape in shapes) {
                    NdimArray<float> inmap = (shape, (new float[shape.Length]).Select((_) => (float)random.Next(1, 16)).ToArray());
                    NdimArray<float> outmap = patchwork.Execute(inmap);

                    NdimArray<float> inmap_padded = NdimArray<float>.EdgePadding1D(inmap, 1);
                    NdimArray<float> diff =
                        NdimArray<float>.Slice1D(inmap_padded, 0, inmap.Width) -
                        NdimArray<float>.Slice1D(inmap_padded, 2, inmap.Width);

                    CollectionAssert.AreEqual(diff.Value, outmap.Value);
                }
            }

            {
                PatchworkFlow patchwork = new PatchworkFlow(flow, input, output, 0);

                patchwork.ProgressEvent += Patchwork_ProgressEvent;

                foreach (Shape shape in shapes) {
                    NdimArray<float> inmap = (shape, (new float[shape.Length]).Select((_) => (float)random.Next(1, 16)).ToArray());
                    NdimArray<float> outmap = patchwork.Execute(inmap);

                    NdimArray<float> inmap_padded = NdimArray<float>.EdgePadding1D(inmap, 1);
                    NdimArray<float> diff =
                        NdimArray<float>.Slice1D(inmap_padded, 0, inmap.Width) -
                        NdimArray<float>.Slice1D(inmap_padded, 2, inmap.Width);

                    CollectionAssert.AreNotEqual(diff.Value, outmap.Value);
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

        [TestMethod]
        public void Execute2DeTest() {
            Shape[] shapes = new Shape[]{
                Shape.Map2D(1, 59, 128, 2),
                Shape.Map2D(1, 64, 128, 2),
                Shape.Map2D(1, 96, 128, 2),
                Shape.Map2D(1, 96,  42, 2),
            };

            VariableField input = Shape.Map2D(1, 32, 64, 2);
            VariableField kernel = (Shape.Kernel2D(1, 1, 3, 3), new float[] { 1, 0, 0, 0, 0, 0, 0, 0, -1 });
            StoreField output = Convolution2D(EdgePadding2D(input, 1), kernel);

            (Flow flow, _) = Flow.Inference(output);
            Random random = new Random(1234);

            {
                PatchworkFlow patchwork = new PatchworkFlow(flow, input, output, 1);

                patchwork.ProgressEvent += Patchwork_ProgressEvent;

                foreach (Shape shape in shapes) {
                    NdimArray<float> inmap = (shape, (new float[shape.Length]).Select((_) => (float)random.Next(1, 16)).ToArray());
                    NdimArray<float> outmap = patchwork.Execute(inmap);

                    NdimArray<float> inmap_padded = NdimArray<float>.EdgePadding2D(inmap, 1);
                    NdimArray<float> diff =
                        NdimArray<float>.Slice2D(inmap_padded, 0, inmap.Width, 0, inmap.Height) -
                        NdimArray<float>.Slice2D(inmap_padded, 2, inmap.Width, 2, inmap.Height);

                    CollectionAssert.AreEqual(diff.Value, outmap.Value);
                }
            }

            {
                PatchworkFlow patchwork = new PatchworkFlow(flow, input, output, 0);

                patchwork.ProgressEvent += Patchwork_ProgressEvent;

                foreach (Shape shape in shapes) {
                    NdimArray<float> inmap = (shape, (new float[shape.Length]).Select((_) => (float)random.Next(1, 16)).ToArray());
                    NdimArray<float> outmap = patchwork.Execute(inmap);

                    NdimArray<float> inmap_padded = NdimArray<float>.EdgePadding2D(inmap, 1);
                    NdimArray<float> diff =
                        NdimArray<float>.Slice2D(inmap_padded, 0, inmap.Width, 0, inmap.Height) -
                        NdimArray<float>.Slice2D(inmap_padded, 2, inmap.Width, 2, inmap.Height);

                    CollectionAssert.AreNotEqual(diff.Value, outmap.Value);
                }
            }
        }

        [TestMethod]
        public void Execute3DTest() {
            Shape[] shapes = new Shape[]{
                Shape.Map3D(3, 14, 20, 15, 2),
                Shape.Map3D(3, 15, 21, 16, 2),
                Shape.Map3D(3, 16, 32, 18, 2),
                Shape.Map3D(3, 29, 64, 24, 2),
                Shape.Map3D(3, 32, 64, 24, 2),
                Shape.Map3D(3, 48, 64, 20, 2),
                Shape.Map3D(3, 48, 21, 10, 2),
            };

            VariableField input = Shape.Map3D(3, 16, 32, 18, 2);
            StoreField output = input + 1;

            (Flow flow, _) = Flow.Inference(output);

            PatchworkFlow patchwork = new PatchworkFlow(flow, input, output, new int[] { 4, 3, 2 });

            patchwork.ProgressEvent += Patchwork_ProgressEvent;

            Random random = new Random();

            foreach (Shape shape in shapes) {
                NdimArray<float> inmap = (shape, (new float[shape.Length]).Select((_) => (float)random.Next(1, 16)).ToArray());
                NdimArray<float> outmap = patchwork.Execute(inmap);

                CollectionAssert.AreEqual((inmap + 1f).Value, outmap.Value);
            }
        }

        [TestMethod]
        public void Execute3Dx2Test() {
            Shape[] shapes = new Shape[]{
                Shape.Map3D(3, 14, 20, 15, 2),
                Shape.Map3D(3, 15, 21, 16, 2),
                Shape.Map3D(3, 16, 32, 18, 2),
                Shape.Map3D(3, 29, 64, 24, 2),
                Shape.Map3D(3, 32, 64, 24, 2),
                Shape.Map3D(3, 48, 64, 20, 2),
                Shape.Map3D(3, 48, 21, 10, 2),
            };

            VariableField input = Shape.Map3D(3, 16, 32, 18, 2);
            StoreField output = NeighborZoom3D(input + 1);

            (Flow flow, _) = Flow.Inference(output);

            PatchworkFlow patchwork = new PatchworkFlow(flow, input, output, new int[] { 4, 3, 2 });

            patchwork.ProgressEvent += Patchwork_ProgressEvent;

            Random random = new Random();

            foreach (Shape shape in shapes) {
                NdimArray<float> inmap = (shape, (new float[shape.Length]).Select((_) => (float)random.Next(1, 16)).ToArray());
                NdimArray<float> outmap = patchwork.Execute(inmap);

                for (int th = 0; th < outmap.Batch; th++) {
                    for (int z = 0; z < outmap.Depth; z++) {
                        for (int y = 0; y < outmap.Height; y++) {
                            for (int x = 0; x < outmap.Width; x++) {
                                for (int c = 0; c < outmap.Channels; c++) {
                                    Assert.AreEqual(inmap[c, x / 2, y / 2, z / 2, th] + 1, outmap[c, x, y, z, th], $"{shape}, {c}, {x}, {y}, {z}, {th}");
                                }
                            }
                        }
                    }
                }
            }
        }

        [TestMethod]
        public void Execute3Dd2Test() {
            Shape[] shapes = new Shape[]{
                Shape.Map3D(3, 14, 20, 15, 2),
                Shape.Map3D(3, 15, 21, 16, 2),
                Shape.Map3D(3, 16, 32, 18, 2),
                Shape.Map3D(3, 29, 64, 24, 2),
                Shape.Map3D(3, 32, 64, 24, 2),
                Shape.Map3D(3, 48, 64, 20, 2),
                Shape.Map3D(3, 48, 21, 10, 2),
            };

            VariableField input = Shape.Map3D(3, 16, 32, 18, 2);
            StoreField output = AveragePooling3D(input + 1, 2);

            (Flow flow, _) = Flow.Inference(output);

            PatchworkFlow patchwork = new PatchworkFlow(flow, input, output, new int[] { 4, 3, 2 });

            patchwork.ProgressEvent += Patchwork_ProgressEvent;

            Random random = new Random();

            foreach (Shape shape in shapes) {
                NdimArray<float> inmap = (shape, (new float[shape.Length]).Select((_) => (float)random.Next(1, 16)).ToArray());
                NdimArray<float> outmap = patchwork.Execute(inmap);

                for (int th = 0; th < outmap.Batch; th++) {
                    for (int z = 0; z < inmap.Depth / 2; z++) {
                        for (int y = 0; y < inmap.Height / 2; y++) {
                            for (int x = 0; x < inmap.Width / 2; x++) {
                                for (int c = 0; c < outmap.Channels; c++) {
                                    Assert.AreEqual(
                                        (inmap[c, 2 * x, 2 * y, 2 * z, th] + inmap[c, 2 * x + 1, 2 * y, 2 * z, th] + 
                                         inmap[c, 2 * x, 2 * y + 1, 2 * z, th] + inmap[c, 2 * x + 1, 2 * y + 1, 2 * z, th] + 
                                         inmap[c, 2 * x, 2 * y, 2 * z + 1, th] + inmap[c, 2 * x + 1, 2 * y, 2 * z + 1, th] + 
                                         inmap[c, 2 * x, 2 * y + 1, 2 * z + 1, th] + inmap[c, 2 * x + 1, 2 * y + 1, 2 * z + 1, th]) / 8 + 1,
                                        outmap[c, x, y, z, th], $"{shape}, {c}, {x}, {y}, {z}, {th}"
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }

        [TestMethod]
        public void Execute3Dd3Test() {
            Shape[] shapes = new Shape[]{
                Shape.Map3D(3, 14, 20, 15, 2),
                Shape.Map3D(3, 15, 21, 16, 2),
                Shape.Map3D(3, 15, 30, 18, 2),
                Shape.Map3D(3, 29, 64, 24, 2),
                Shape.Map3D(3, 32, 64, 24, 2),
                Shape.Map3D(3, 48, 64, 20, 2),
                Shape.Map3D(3, 48, 21, 10, 2),
            };

            VariableField input = Shape.Map3D(3, 15, 30, 18, 2);
            StoreField output = AveragePooling3D(input + 1, 3);

            (Flow flow, _) = Flow.Inference(output);

            PatchworkFlow patchwork = new PatchworkFlow(flow, input, output, new int[] { 4, 3, 2 });

            patchwork.ProgressEvent += Patchwork_ProgressEvent;

            Random random = new Random();

            foreach (Shape shape in shapes) {
                NdimArray<float> inmap = (shape, (new float[shape.Length]).Select((_) => (float)random.Next(1, 16)).ToArray());
                NdimArray<float> outmap = patchwork.Execute(inmap);

                for (int th = 0; th < outmap.Batch; th++) {
                    for (int z = 0; z < inmap.Depth / 3; z++) {
                        for (int y = 0; y < inmap.Height / 3; y++) {
                            for (int x = 0; x < inmap.Width / 3; x++) {
                                for (int c = 0; c < outmap.Channels; c++) {
                                    Assert.AreEqual(
                                        (inmap[c, 3 * x, 3 * y, 3 * z, th] + inmap[c, 3 * x + 1, 3 * y, 3 * z, th] + inmap[c, 3 * x + 2, 3 * y, 3 * z, th] +
                                         inmap[c, 3 * x, 3 * y + 1, 3 * z, th] + inmap[c, 3 * x + 1, 3 * y + 1, 3 * z, th] + inmap[c, 3 * x + 2, 3 * y + 1, 3 * z, th] +
                                         inmap[c, 3 * x, 3 * y + 2, 3 * z, th] + inmap[c, 3 * x + 1, 3 * y + 2, 3 * z, th] + inmap[c, 3 * x + 2, 3 * y + 2, 3 * z, th] + 
                                         inmap[c, 3 * x, 3 * y, 3 * z + 1, th] + inmap[c, 3 * x + 1, 3 * y, 3 * z + 1, th] + inmap[c, 3 * x + 2, 3 * y, 3 * z + 1, th] +
                                         inmap[c, 3 * x, 3 * y + 1, 3 * z + 1, th] + inmap[c, 3 * x + 1, 3 * y + 1, 3 * z + 1, th] + inmap[c, 3 * x + 2, 3 * y + 1, 3 * z + 1, th] +
                                         inmap[c, 3 * x, 3 * y + 2, 3 * z + 1, th] + inmap[c, 3 * x + 1, 3 * y + 2, 3 * z + 1, th] + inmap[c, 3 * x + 2, 3 * y + 2, 3 * z + 1, th] + 
                                         inmap[c, 3 * x, 3 * y, 3 * z + 2, th] + inmap[c, 3 * x + 1, 3 * y, 3 * z + 2, th] + inmap[c, 3 * x + 2, 3 * y, 3 * z + 2, th] +
                                         inmap[c, 3 * x, 3 * y + 1, 3 * z + 2, th] + inmap[c, 3 * x + 1, 3 * y + 1, 3 * z + 2, th] + inmap[c, 3 * x + 2, 3 * y + 1, 3 * z + 2, th] +
                                         inmap[c, 3 * x, 3 * y + 2, 3 * z + 2, th] + inmap[c, 3 * x + 1, 3 * y + 2, 3 * z + 2, th] + inmap[c, 3 * x + 2, 3 * y + 2, 3 * z + 2, th]) / 27 + 1,
                                        outmap[c, x, y, z, th], 1e-4f, $"{shape}, {c}, {x}, {y}, {z}, {th}"
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }

        [TestMethod]
        public void Execute3DeTest() {
            Shape[] shapes = new Shape[]{
                Shape.Map3D(1, 29, 64, 24, 2),
                Shape.Map3D(1, 32, 64, 24, 2),
                Shape.Map3D(1, 48, 64, 20, 2),
                Shape.Map3D(1, 48, 21, 10, 2),
            };

            VariableField input = Shape.Map3D(1, 32, 64, 20, 2);
            VariableField kernel = (Shape.Kernel3D(1, 1, 3, 3, 3), new float[] { 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1 });
            StoreField output = Convolution3D(EdgePadding3D(input, 1), kernel);

            (Flow flow, _) = Flow.Inference(output);
            Random random = new Random(1234);

            {
                PatchworkFlow patchwork = new PatchworkFlow(flow, input, output, 1);

                patchwork.ProgressEvent += Patchwork_ProgressEvent;

                foreach (Shape shape in shapes) {
                    NdimArray<float> inmap = (shape, (new float[shape.Length]).Select((_) => (float)random.Next(1, 16)).ToArray());
                    NdimArray<float> outmap = patchwork.Execute(inmap);

                    NdimArray<float> inmap_padded = NdimArray<float>.EdgePadding3D(inmap, 1);
                    NdimArray<float> diff =
                        NdimArray<float>.Slice3D(inmap_padded, 0, inmap.Width, 0, inmap.Height, 0, inmap.Depth) -
                        NdimArray<float>.Slice3D(inmap_padded, 2, inmap.Width, 2, inmap.Height, 2, inmap.Depth);

                    CollectionAssert.AreEqual(diff.Value, outmap.Value);
                }
            }

            {
                PatchworkFlow patchwork = new PatchworkFlow(flow, input, output, 0);

                patchwork.ProgressEvent += Patchwork_ProgressEvent;

                foreach (Shape shape in shapes) {
                    NdimArray<float> inmap = (shape, (new float[shape.Length]).Select((_) => (float)random.Next(1, 16)).ToArray());
                    NdimArray<float> outmap = patchwork.Execute(inmap);

                    NdimArray<float> inmap_padded = NdimArray<float>.EdgePadding3D(inmap, 1);
                    NdimArray<float> diff =
                        NdimArray<float>.Slice3D(inmap_padded, 0, inmap.Width, 0, inmap.Height, 0, inmap.Depth) -
                        NdimArray<float>.Slice3D(inmap_padded, 2, inmap.Width, 2, inmap.Height, 2, inmap.Depth);

                    CollectionAssert.AreNotEqual(diff.Value, outmap.Value);
                }
            }
        }

        private void Patchwork_ProgressEvent(int progress, int all) {
            Console.WriteLine($"{progress}/{all}");
        }
    }
}
