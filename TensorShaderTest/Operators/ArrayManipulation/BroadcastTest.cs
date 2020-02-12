using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using TensorShader.Operators.ArrayManipulation;

namespace TensorShaderTest.Operators.ArrayManipulation {
    [TestClass]
    public class BroadcastTest {
        [TestMethod]
        public void ExecuteTest() {
            Random rd = new Random(1234);

            List<Shape> shapes = new List<Shape> {
                Shape.Scalar,
                Shape.Vector(1), Shape.Vector(2),
                Shape.Map0D(1, 1), Shape.Map0D(2, 1), Shape.Map0D(1, 3), Shape.Map0D(2, 3),
                Shape.Map1D(1, 1, 1), Shape.Map1D(2, 1, 1), Shape.Map1D(1, 3, 1), Shape.Map1D(2, 3, 1),
                Shape.Map1D(1, 1, 5), Shape.Map1D(2, 1, 5), Shape.Map1D(1, 3, 5), Shape.Map1D(2, 3, 5),
                Shape.Map2D(1, 1, 1, 1), Shape.Map2D(2, 1, 1, 1), Shape.Map2D(1, 3, 1, 1), Shape.Map2D(2, 3, 1, 1),
                Shape.Map2D(1, 1, 5, 1), Shape.Map2D(2, 1, 5, 1), Shape.Map2D(1, 3, 5, 1), Shape.Map2D(2, 3, 5, 1),
                Shape.Map2D(1, 1, 1, 7), Shape.Map2D(2, 1, 1, 7), Shape.Map2D(1, 3, 1, 7), Shape.Map2D(2, 3, 1, 7),
                Shape.Map2D(1, 1, 5, 7), Shape.Map2D(2, 1, 5, 7), Shape.Map2D(1, 3, 5, 7), Shape.Map2D(2, 3, 5, 7),
                Shape.Map3D(1, 1, 1, 1, 1), Shape.Map3D(2, 1, 1, 1, 1), Shape.Map3D(1, 3, 1, 1, 1), Shape.Map3D(2, 3, 1, 1, 1),
                Shape.Map3D(1, 1, 5, 1, 1), Shape.Map3D(2, 1, 5, 1, 1), Shape.Map3D(1, 3, 5, 1, 1), Shape.Map3D(2, 3, 5, 1, 1),
                Shape.Map3D(1, 1, 1, 7, 1), Shape.Map3D(2, 1, 1, 7, 1), Shape.Map3D(1, 3, 1, 7, 1), Shape.Map3D(2, 3, 1, 7, 1),
                Shape.Map3D(1, 1, 5, 7, 1), Shape.Map3D(2, 1, 5, 7, 1), Shape.Map3D(1, 3, 5, 7, 1), Shape.Map3D(2, 3, 5, 7, 1),
                Shape.Map3D(1, 1, 1, 1, 9), Shape.Map3D(2, 1, 1, 1, 9), Shape.Map3D(1, 3, 1, 1, 9), Shape.Map3D(2, 3, 1, 1, 9),
                Shape.Map3D(1, 1, 5, 1, 9), Shape.Map3D(2, 1, 5, 1, 9), Shape.Map3D(1, 3, 5, 1, 9), Shape.Map3D(2, 3, 5, 1, 9),
                Shape.Map3D(1, 1, 1, 7, 9), Shape.Map3D(2, 1, 1, 7, 9), Shape.Map3D(1, 3, 1, 7, 9), Shape.Map3D(2, 3, 1, 7, 9),
                Shape.Map3D(1, 1, 5, 7, 9), Shape.Map3D(2, 1, 5, 7, 9), Shape.Map3D(1, 3, 5, 7, 9), Shape.Map3D(2, 3, 5, 7, 9),
                new Shape(ShapeType.Map, 1, 1, 1, 1, 1, 1), new Shape(ShapeType.Map, 2, 1, 1, 1, 1, 1), new Shape(ShapeType.Map, 1, 3, 1, 1, 1, 1), new Shape(ShapeType.Map, 2, 3, 1, 1, 1, 1),
                new Shape(ShapeType.Map, 1, 1, 5, 1, 1, 1), new Shape(ShapeType.Map, 2, 1, 5, 1, 1, 1), new Shape(ShapeType.Map, 1, 3, 5, 1, 1, 1), new Shape(ShapeType.Map, 2, 3, 5, 1, 1, 1),
                new Shape(ShapeType.Map, 1, 1, 1, 7, 1, 1), new Shape(ShapeType.Map, 2, 1, 1, 7, 1, 1), new Shape(ShapeType.Map, 1, 3, 1, 7, 1, 1), new Shape(ShapeType.Map, 2, 3, 1, 7, 1, 1),
                new Shape(ShapeType.Map, 1, 1, 5, 7, 1, 1), new Shape(ShapeType.Map, 2, 1, 5, 7, 1, 1), new Shape(ShapeType.Map, 1, 3, 5, 7, 1, 1), new Shape(ShapeType.Map, 2, 3, 5, 7, 1, 1),
                new Shape(ShapeType.Map, 1, 1, 1, 1, 9, 1), new Shape(ShapeType.Map, 2, 1, 1, 1, 9, 1), new Shape(ShapeType.Map, 1, 3, 1, 1, 9, 1), new Shape(ShapeType.Map, 2, 3, 1, 1, 9, 1),
                new Shape(ShapeType.Map, 1, 1, 5, 1, 9, 1), new Shape(ShapeType.Map, 2, 1, 5, 1, 9, 1), new Shape(ShapeType.Map, 1, 3, 5, 1, 9, 1), new Shape(ShapeType.Map, 2, 3, 5, 1, 9, 1),
                new Shape(ShapeType.Map, 1, 1, 1, 7, 9, 1), new Shape(ShapeType.Map, 2, 1, 1, 7, 9, 1), new Shape(ShapeType.Map, 1, 3, 1, 7, 9, 1), new Shape(ShapeType.Map, 2, 3, 1, 7, 9, 1),
                new Shape(ShapeType.Map, 1, 1, 5, 7, 9, 1), new Shape(ShapeType.Map, 2, 1, 5, 7, 9, 1), new Shape(ShapeType.Map, 1, 3, 5, 7, 9, 1), new Shape(ShapeType.Map, 2, 3, 5, 7, 9, 1),
                new Shape(ShapeType.Map, 1, 1, 1, 1, 1, 11), new Shape(ShapeType.Map, 2, 1, 1, 1, 1, 11), new Shape(ShapeType.Map, 1, 3, 1, 1, 1, 11), new Shape(ShapeType.Map, 2, 3, 1, 1, 1, 11),
                new Shape(ShapeType.Map, 1, 1, 5, 1, 1, 11), new Shape(ShapeType.Map, 2, 1, 5, 1, 1, 11), new Shape(ShapeType.Map, 1, 3, 5, 1, 1, 11), new Shape(ShapeType.Map, 2, 3, 5, 1, 1, 11),
                new Shape(ShapeType.Map, 1, 1, 1, 7, 1, 11), new Shape(ShapeType.Map, 2, 1, 1, 7, 1, 11), new Shape(ShapeType.Map, 1, 3, 1, 7, 1, 11), new Shape(ShapeType.Map, 2, 3, 1, 7, 1, 11),
                new Shape(ShapeType.Map, 1, 1, 5, 7, 1, 11), new Shape(ShapeType.Map, 2, 1, 5, 7, 1, 11), new Shape(ShapeType.Map, 1, 3, 5, 7, 1, 11), new Shape(ShapeType.Map, 2, 3, 5, 7, 1, 11),
                new Shape(ShapeType.Map, 1, 1, 1, 1, 9, 11), new Shape(ShapeType.Map, 2, 1, 1, 1, 9, 11), new Shape(ShapeType.Map, 1, 3, 1, 1, 9, 11), new Shape(ShapeType.Map, 2, 3, 1, 1, 9, 11),
                new Shape(ShapeType.Map, 1, 1, 5, 1, 9, 11), new Shape(ShapeType.Map, 2, 1, 5, 1, 9, 11), new Shape(ShapeType.Map, 1, 3, 5, 1, 9, 11), new Shape(ShapeType.Map, 2, 3, 5, 1, 9, 11),
                new Shape(ShapeType.Map, 1, 1, 1, 7, 9, 11), new Shape(ShapeType.Map, 2, 1, 1, 7, 9, 11), new Shape(ShapeType.Map, 1, 3, 1, 7, 9, 11), new Shape(ShapeType.Map, 2, 3, 1, 7, 9, 11),
                new Shape(ShapeType.Map, 1, 1, 5, 7, 9, 11), new Shape(ShapeType.Map, 2, 1, 5, 7, 9, 11), new Shape(ShapeType.Map, 1, 3, 5, 7, 9, 11), new Shape(ShapeType.Map, 2, 3, 5, 7, 9, 11),
            };

            foreach (Shape inshape in shapes) {
                foreach (Shape outshape in shapes) {
                    if (inshape.Ndim > outshape.Ndim) {
                        continue;
                    }

                    bool is_creatable = true;

                    if (Broadcast.BroadcastShape(inshape, outshape) != outshape) {
                        continue;
                    }

                    if (is_creatable) {
                        Broadcast broadcast = null;

                        try {
                            broadcast = new Broadcast(inshape, outshape);
                        }
                        catch (Exception e) {
                            Assert.Fail($"fail create instance {inshape}, {outshape} : {e.Message}");
                        }

                        float[] x = (new float[inshape.Length]).Select((_) => (float)rd.NextDouble()).ToArray();
                        OverflowCheckedTensor intensor = new OverflowCheckedTensor(inshape, x);
                        OverflowCheckedTensor outtensor = new OverflowCheckedTensor(outshape);

                        broadcast.Execute(intensor, outtensor);

                        float[] y = outtensor.State;

                        int[] il = (new int[5]).Select((_, idx) => outshape.Ndim > idx ? outshape[idx] : 1).ToArray();
                        int[] jl = (new int[5]).Select((_, idx) => inshape.Ndim > idx ? inshape[idx] : 1).ToArray();

                        int i = 0;

                        for (int i4 = 0, j4 = 0; i4 < il[4]; i4++, j4 += (jl[4] > 1) ? 1 : 0) {
                            for (int i3 = 0, j3 = 0; i3 < il[3]; i3++, j3 += (jl[3] > 1) ? 1 : 0) {
                                for (int i2 = 0, j2 = 0; i2 < il[2]; i2++, j2 += (jl[2] > 1) ? 1 : 0) {
                                    for (int i1 = 0, j1 = 0; i1 < il[1]; i1++, j1 += (jl[1] > 1) ? 1 : 0) {
                                        for (int i0 = 0, j0 = 0; i0 < il[0]; i0++, j0 += (jl[0] > 1) ? 1 : 0) {
                                            int j = j0 + jl[0]
                                                 * (j1 + jl[1]
                                                 * (j2 + jl[2]
                                                 * (j3 + jl[3] * j4)));

                                            Assert.AreEqual(x[j], y[i], $"fail broadcast {inshape}, {outshape}");

                                            i++;
                                        }
                                    }
                                }
                            }
                        }

                        Console.WriteLine($"pass : {inshape}, {outshape}");
                    }
                }
            }
        }

        [TestMethod]
        public void ReferenceTest() {
            Shape inshape = new Shape(ShapeType.Map, 3, 5, 1, 1, 2);
            Shape outshape = new Shape(ShapeType.Map, 3, 5, 2, 3, 2);

            float[] x = (new float[inshape.Length]).Select((_, idx) => (float)idx).ToArray();

            OverflowCheckedTensor intensor = new OverflowCheckedTensor(inshape, x);
            OverflowCheckedTensor outtensor = new OverflowCheckedTensor(outshape);

            Broadcast broadcast = new Broadcast(inshape, outshape);

            broadcast.Execute(intensor, outtensor);

            float[] y = outtensor.State;

            float[] y_expect = {
                 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14, 0, 1, 2, 3, 4, 5, 6, 7, 8,
                 9,10,11,12,13,14, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14, 0, 1, 2,
                 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,
                12,13,14, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,17,18,19,20,
                21,22,23,24,25,26,27,28,29,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,
                15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,15,16,17,18,19,20,21,22,23,
                24,25,26,27,28,29,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,15,16,17,
                18,19,20,21,22,23,24,25,26,27,28,29
            };

            CollectionAssert.AreEqual(y_expect, y);
        }
    }
}
