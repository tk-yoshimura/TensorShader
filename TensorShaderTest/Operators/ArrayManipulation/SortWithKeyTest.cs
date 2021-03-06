using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;
using TensorShader;
using TensorShader.Operators.ArrayManipulation;
using TensorShaderCudaBackend.API;

namespace TensorShaderTest.Operators.ArrayManipulation {
    [TestClass]
    public class SortWithKeyTest {
        [TestMethod]
        public void ExecuteTest() {
            Random rd = new(1234);

            foreach (Shape shape in new Shape[]{
                   new Shape(ShapeType.Map, 2, 512, 512, 6),
                   new Shape(ShapeType.Map, 3, 512, 512, 4),
                   new Shape(ShapeType.Map, 4, 512, 512, 2),
                   new Shape(ShapeType.Map, 5, 512, 512, 1),
                   new Shape(ShapeType.Map, 6, 512, 512, 1),
                   new Shape(ShapeType.Map, 7, 512, 512, 1),
                   new Shape(ShapeType.Map, 8, 512, 512, 1),
                   new Shape(ShapeType.Map, 9, 512, 512, 1),
                   new Shape(ShapeType.Map, 10, 512, 512, 1),
                   new Shape(ShapeType.Map, 12, 512, 512, 1),
                   new Shape(ShapeType.Map, 25, 512, 512, 1),
                   new Shape(ShapeType.Map, 5, 16384, 3, 1),
                   new Shape(ShapeType.Map, 5, 3, 16384, 1),
                   new Shape(ShapeType.Map, 16384, 2, 3, 1),
                   new Shape(ShapeType.Map, 16, 19, 23, 8, 1, 5, 6),
                   new Shape(ShapeType.Map, 17, 9, 2, 4, 1, 3, 67) }) {

                for (int axis = 0; axis < shape.Ndim; axis++) {
                    int stride = 1, axislength = shape[axis];
                    for (int i = 0; i < axis; i++) {
                        stride *= shape[i];
                    }

                    float[] x1 = (new float[shape.Length]).Select((_, idx) => (float)(((idx * 4969 % 17 + 3) * (idx * 6577 % 13 + 5) + idx) % 8)).ToArray();
                    float[] x2 = (new float[shape.Length]).Select((_, idx) => (float)(idx / stride % axislength)).ToArray();

                    OverflowCheckedTensor inkey = new(shape, x1);
                    OverflowCheckedTensor inval = new(shape, x2);
                    OverflowCheckedTensor outkey = new(shape);
                    OverflowCheckedTensor outval = new(shape);

                    SortWithKey ope = new(shape, axis);

                    ope.Execute(inkey, inval, outkey, outval);

                    CollectionAssert.AreEqual(x1, inkey.State.Value);
                    CollectionAssert.AreEqual(x2, inval.State.Value);

                    float[] y = outkey.State.Value;
                    int p = 0;

                    for (int i = 0; i < shape.Length / axislength; i++, p = i / stride * stride * axislength + i % stride) {
                        for (int j = 1; j < axislength; j++) {
                            if (y[p + (j - 1) * stride] > y[p + j * stride]) {
                                Assert.Fail($"{shape}, axis{axis} outkey");
                            }
                        }
                    }

                    Assert.AreEqual(shape.Length, p);

                    ope.Execute(outval, outkey, inval, inkey);

                    float[] y2 = inval.State.Value;
                    p = 0;

                    for (int i = 0; i < shape.Length / axislength; i++, p = i / stride * stride * axislength + i % stride) {
                        for (int j = 1; j < axislength; j++) {
                            if (y2[p + (j - 1) * stride] > y2[p + j * stride]) {
                                Assert.Fail($"{shape}, axis{axis} inval");
                            }
                        }
                    }

                    Assert.AreEqual(shape.Length, p);

                    float[] y3 = inkey.State.Value;

                    CollectionAssert.AreEqual(x1, y3, $"axis:{axis} inkey");

                    Console.WriteLine($"pass : axis{axis}");
                }
            }
        }

        [TestMethod]
        public void SpeedTest() {
            Shape shape = new(ShapeType.Map, 4096, 500);
            int length = shape.Length, axis = 0, stride = 1, axislength = shape[axis];

            float[] xval = (new float[length]).Select((_, idx) => (float)(((idx * 4969 % 17 + 3) * (idx * 6577 % 13 + 5) + idx) % 8)).ToArray();
            float[] ival = (new float[shape.Length]).Select((_, idx) => (float)(idx / stride % shape[axis])).ToArray();

            OverflowCheckedTensor x_key = new(shape, xval);
            OverflowCheckedTensor x_val = new(shape, ival);

            OverflowCheckedTensor y_key = new(shape);
            OverflowCheckedTensor y_val = new(shape);

            SortWithKey ope = new(shape, axis);

            Cuda.Profiler.Initialize("../../../../profiler.nvsetting", "../../nvprofiles/sortwithkey_random.nvvp");
            Cuda.Profiler.Start();

            ope.Execute(x_key, x_val, y_key, y_val);

            Cuda.Profiler.Stop();

            float[] v = y_key.State.Value;

            for (int i = 1; i < axislength; i++) {
                Assert.IsTrue(v[i - 1] <= v[i], $"{i}: {v[i - 1]}, {v[i]}");
                Assert.IsTrue(v[i - 1 + axislength] <= v[i + axislength], $"{i + axislength}: {v[i - 1 + axislength]}, {v[i + axislength]}");
                Assert.IsTrue(v[i - 1 + axislength * 2] <= v[i + axislength * 2], $"{i + axislength * 2}: {v[i - 1 + axislength * 2]}, {v[i + axislength * 2]}");
                Assert.IsTrue(v[i - 1 + axislength * 3] <= v[i + axislength * 3], $"{i + axislength * 3}: {v[i - 1 + axislength * 3]}, {v[i + axislength * 3]}");
                Assert.IsTrue(v[i - 1 + axislength * 4] <= v[i + axislength * 4], $"{i + axislength * 4}: {v[i - 1 + axislength * 4]}, {v[i + axislength * 4]}");
            }
        }

        [TestMethod]
        public void MaxAxisLengthInSMemTest() {
            Shape shape = new(ShapeType.Map, (int)TensorShaderCudaBackend.Shaders.ArrayManipulation.SortWithKeyUseSharedMemory.MaxAxisLength, 500);
            int length = shape.Length, axis = 0, stride = 1, axislength = shape[axis];

            float[] xval = (new float[length]).Select((_, idx) => (float)(((idx * 4969 % 17 + 3) * (idx * 6577 % 13 + 5) + idx) % 8)).ToArray();
            float[] ival = (new float[shape.Length]).Select((_, idx) => (float)(idx / stride % shape[axis])).ToArray();

            OverflowCheckedTensor x_key = new(shape, xval);
            OverflowCheckedTensor x_val = new(shape, ival);

            OverflowCheckedTensor y_key = new(shape);
            OverflowCheckedTensor y_val = new(shape);

            SortWithKey ope = new(shape, axis);

            Cuda.Profiler.Initialize("../../../../profiler.nvsetting", "../../nvprofiles/sortwithkey_maxlength.nvvp");
            Cuda.Profiler.Start();

            ope.Execute(x_key, x_val, y_key, y_val);

            Cuda.Profiler.Stop();

            float[] v = y_key.State.Value;

            for (int i = 1; i < axislength; i++) {
                Assert.IsTrue(v[i - 1] <= v[i], $"{i}: {v[i - 1]}, {v[i]}");
                Assert.IsTrue(v[i - 1 + axislength] <= v[i + axislength], $"{i + axislength}: {v[i - 1 + axislength]}, {v[i + axislength]}");
                Assert.IsTrue(v[i - 1 + axislength * 2] <= v[i + axislength * 2], $"{i + axislength * 2}: {v[i - 1 + axislength * 2]}, {v[i + axislength * 2]}");
                Assert.IsTrue(v[i - 1 + axislength * 3] <= v[i + axislength * 3], $"{i + axislength * 3}: {v[i - 1 + axislength * 3]}, {v[i + axislength * 3]}");
                Assert.IsTrue(v[i - 1 + axislength * 4] <= v[i + axislength * 4], $"{i + axislength * 4}: {v[i - 1 + axislength * 4]}, {v[i + axislength * 4]}");
            }
        }

        [TestMethod]
        public void OverMaxAxisLengthInSMemTest() {
            Shape shape = new(ShapeType.Map, (int)TensorShaderCudaBackend.Shaders.ArrayManipulation.SortWithKeyUseSharedMemory.MaxAxisLength + 1, 500);
            int length = shape.Length, axis = 0, stride = 1, axislength = shape[axis];

            float[] xval = (new float[length]).Select((_, idx) => (float)(((idx * 4969 % 17 + 3) * (idx * 6577 % 13 + 5) + idx) % 8)).ToArray();
            float[] ival = (new float[shape.Length]).Select((_, idx) => (float)(idx / stride % shape[axis])).ToArray();

            OverflowCheckedTensor x_key = new(shape, xval);
            OverflowCheckedTensor x_val = new(shape, ival);

            OverflowCheckedTensor y_key = new(shape);
            OverflowCheckedTensor y_val = new(shape);

            SortWithKey ope = new(shape, axis);

            Cuda.Profiler.Initialize("../../../../profiler.nvsetting", "../../nvprofiles/sortwithkey_overmaxlength.nvvp");
            Cuda.Profiler.Start();

            ope.Execute(x_key, x_val, y_key, y_val);

            Cuda.Profiler.Stop();

            float[] v = y_key.State.Value;

            for (int i = 1; i < axislength; i++) {
                Assert.IsTrue(v[i - 1] <= v[i], $"{i}: {v[i - 1]}, {v[i]}");
                Assert.IsTrue(v[i - 1 + axislength] <= v[i + axislength], $"{i + axislength}: {v[i - 1 + axislength]}, {v[i + axislength]}");
                Assert.IsTrue(v[i - 1 + axislength * 2] <= v[i + axislength * 2], $"{i + axislength * 2}: {v[i - 1 + axislength * 2]}, {v[i + axislength * 2]}");
                Assert.IsTrue(v[i - 1 + axislength * 3] <= v[i + axislength * 3], $"{i + axislength * 3}: {v[i - 1 + axislength * 3]}, {v[i + axislength * 3]}");
                Assert.IsTrue(v[i - 1 + axislength * 4] <= v[i + axislength * 4], $"{i + axislength * 4}: {v[i - 1 + axislength * 4]}, {v[i + axislength * 4]}");
            }
        }

        [TestMethod]
        public void ReverseTest() {
            Shape shape = new(ShapeType.Map, 4096, 500);
            int length = shape.Length, axis = 0, stride = 1, axislength = shape[axis];

            float[] xval = (new float[length]).Select((_, idx) => (float)idx).Reverse().ToArray();
            float[] ival = (new float[shape.Length]).Select((_, idx) => (float)(idx / stride % shape[axis])).ToArray();

            OverflowCheckedTensor x_key = new(shape, xval);
            OverflowCheckedTensor x_val = new(shape, ival);

            OverflowCheckedTensor y_key = new(shape);
            OverflowCheckedTensor y_val = new(shape);

            SortWithKey ope = new(shape, axis);

            Cuda.Profiler.Initialize("../../../../profiler.nvsetting", "../../nvprofiles/sortwithkey_reverse.nvvp");
            Cuda.Profiler.Start();

            ope.Execute(x_key, x_val, y_key, y_val);

            Cuda.Profiler.Stop();

            float[] v = y_key.State.Value;

            for (int i = 1; i < axislength; i++) {
                Assert.IsTrue(v[i - 1] <= v[i], $"{i}: {v[i - 1]}, {v[i]}");
                Assert.IsTrue(v[i - 1 + axislength] <= v[i + axislength], $"{i + axislength}: {v[i - 1 + axislength]}, {v[i + axislength]}");
                Assert.IsTrue(v[i - 1 + axislength * 2] <= v[i + axislength * 2], $"{i + axislength * 2}: {v[i - 1 + axislength * 2]}, {v[i + axislength * 2]}");
                Assert.IsTrue(v[i - 1 + axislength * 3] <= v[i + axislength * 3], $"{i + axislength * 3}: {v[i - 1 + axislength * 3]}, {v[i + axislength * 3]}");
                Assert.IsTrue(v[i - 1 + axislength * 4] <= v[i + axislength * 4], $"{i + axislength * 4}: {v[i - 1 + axislength * 4]}, {v[i + axislength * 4]}");
            }
        }
    }
}
