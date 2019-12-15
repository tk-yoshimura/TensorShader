using System;
using System.Diagnostics;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using TensorShader.Operators.Aggregation;
using TensorShaderCudaBackend.API;

namespace TensorShaderTest.Operators.Aggregation {
    [TestClass]
    public class MaxTest {
        [TestMethod]
        public void ExecuteTest() {
            Random rd = new Random(1234);

            int width = 4, height = 5;

            foreach (int length in new int[] { 1, 2, 3, 4, 5, 6, 13, 16, 17, 19, 64, 2048 }) {
                foreach (int ch in new int[] { 1, 2, 3, 4, 5, 6, 13, 16, 17, 19, 64 }) {
                    foreach (int batch in new int[] { 1, 2, 3 }) {
                        float[] x = (new float[ch * width * height * length * batch]).Select((_) => (float)rd.NextDouble() * 2 - 1).ToArray();

                        Shape shape = Shape.Map3D(ch, width, height, length, batch);

                        OverflowCheckedTensor v1 = new OverflowCheckedTensor(shape, x);

                        /*axis = 0*/
                        {
                            OverflowCheckedTensor v2 = new OverflowCheckedTensor(Shape.Map3D(1, width, height, length, batch));

                            Max ope = new Max(shape, axis: 0);

                            ope.Execute(v1, v2);

                            float[] y = v2.State;

                            for (int th = 0; th < batch; th++) {
                                for (int k = 0; k < length; k++) {
                                    for (int j = 0; j < height; j++) {
                                        for (int i = 0; i < width; i++) {
                                            float vmax = float.MinValue;

                                            for (int f = 0; f < ch; f++) {
                                                int idx = f + ch * (i + width * (j + height * (k + length * th)));

                                                vmax = Math.Max(x[idx], vmax);
                                            }

                                            Assert.AreEqual(vmax, y[i + width * (j + height * (k + length * th))], 1e-4f,
                                                $"axis:0, width:{width}, height:{height}, length:{length}, ch:{ch}, batch:{batch}");

                                        }
                                    }
                                }
                            }
                        }

                        /*axis = 1*/
                        {
                            OverflowCheckedTensor v2 = new OverflowCheckedTensor(Shape.Map3D(ch, 1, height, length, batch));

                            Max ope = new Max(shape, axis: 1);

                            ope.Execute(v1, v2);

                            float[] y = v2.State;

                            for (int th = 0; th < batch; th++) {
                                for (int k = 0; k < length; k++) {
                                    for (int j = 0; j < height; j++) {
                                        for (int f = 0; f < ch; f++) {
                                            float vmax = float.MinValue;

                                            for (int i = 0; i < width; i++) {
                                                int idx = f + ch * (i + width * (j + height * (k + length * th)));

                                                vmax = Math.Max(x[idx], vmax);
                                            }

                                            Assert.AreEqual(vmax, y[f + ch * (j + height * (k + length * th))], 1e-4f,
                                                $"axis:1, width:{width}, height:{height}, length:{length}, ch:{ch}, batch:{batch}");

                                        }
                                    }
                                }
                            }
                        }

                        /*axis = 2*/
                        {
                            OverflowCheckedTensor v2 = new OverflowCheckedTensor(Shape.Map3D(ch, width, 1, length, batch));

                            Max ope = new Max(shape, axis: 2);

                            ope.Execute(v1, v2);

                            float[] y = v2.State;

                            for (int th = 0; th < batch; th++) {
                                for (int k = 0; k < length; k++) {
                                    for (int i = 0; i < width; i++) {
                                        for (int f = 0; f < ch; f++) {
                                            float vmax = float.MinValue;

                                            for (int j = 0; j < height; j++) {
                                                int idx = f + ch * (i + width * (j + height * (k + length * th)));

                                                vmax = Math.Max(x[idx], vmax);
                                            }

                                            Assert.AreEqual(vmax, y[f + ch * (i + width * (k + length * th))], 1e-4f,
                                                $"axis:2, width:{width}, height:{height}, length:{length}, ch:{ch}, batch:{batch}");

                                        }
                                    }
                                }
                            }
                        }

                        /*axis = 3*/
                        {
                            OverflowCheckedTensor v2 = new OverflowCheckedTensor(Shape.Map3D(ch, width, height, 1, batch));

                            Max ope = new Max(shape, axis: 3);

                            ope.Execute(v1, v2);

                            float[] y = v2.State;

                            for (int th = 0; th < batch; th++) {
                                for (int j = 0; j < height; j++) {
                                    for (int i = 0; i < width; i++) {
                                        for (int f = 0; f < ch; f++) {
                                            float vmax = float.MinValue;

                                            for (int k = 0; k < length; k++) {
                                                int idx = f + ch * (i + width * (j + height * (k + length * th)));

                                                vmax = Math.Max(x[idx], vmax);
                                            }

                                            Assert.AreEqual(vmax, y[f + ch * (i + width * (j + height * th))], 1e-4f,
                                                $"axis:3, width:{width}, height:{height}, length:{length}, ch:{ch}, batch:{batch}");

                                        }
                                    }
                                }
                            }
                        }

                        /*axis = 4*/
                        {
                            OverflowCheckedTensor v2 = new OverflowCheckedTensor(Shape.Map3D(ch, width, height, length, 1));

                            Max ope = new Max(shape, axis: 4);

                            ope.Execute(v1, v2);

                            float[] y = v2.State;

                            for (int k = 0; k < length; k++) {
                                for (int j = 0; j < height; j++) {
                                    for (int i = 0; i < width; i++) {
                                        for (int f = 0; f < ch; f++) {
                                            float vmax = float.MinValue;

                                            for (int th = 0; th < batch; th++) {
                                                int idx = f + ch * (i + width * (j + height * (k + length * th)));

                                                vmax = Math.Max(x[idx], vmax);
                                            }

                                            Assert.AreEqual(vmax, y[f + ch * (i + width * (j + height * k))], 1e-4f,
                                                $"axis:4, width:{width}, height:{height}, length:{length}, ch:{ch}, batch:{batch}");

                                        }
                                    }
                                }
                            }
                        }

                    }
                }
            }
        }

        [TestMethod]
        public void SpeedTest() {
            int length = 8192, ch = 256;

            Shape shape = Shape.Map1D(ch, length);

            OverflowCheckedTensor v1 = new OverflowCheckedTensor(shape);
            OverflowCheckedTensor v2 = new OverflowCheckedTensor(Shape.Map1D(ch, 1));

            Max ope = new Max(shape, axis: 1);

            Cuda.Profiler.Initialize("../../../profiler.nvsetting", "../../nvprofiles/aggregate_max.nvvp");
            Cuda.Profiler.Start();

            ope.Execute(v1, v2);

            Cuda.Profiler.Stop();
        }
    }
}
