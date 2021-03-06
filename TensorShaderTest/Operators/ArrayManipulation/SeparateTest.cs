using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;
using TensorShader;
using TensorShader.Operators.ArrayManipulation;
using TensorShaderCudaBackend.API;

namespace TensorShaderTest.Operators.ArrayManipulation {
    [TestClass]
    public class SeparateTest {
        [TestMethod]
        public void ExecuteTest() {
            Random rd = new(1234);

            foreach (Shape outshape in new Shape[]{
                new Shape(ShapeType.Map, 13),
                new Shape(ShapeType.Map, 24),
                new Shape(ShapeType.Map, 13, 14),
                new Shape(ShapeType.Map, 24, 21),
                new Shape(ShapeType.Map, 14, 19, 13),
                new Shape(ShapeType.Map, 13, 14, 19),
                new Shape(ShapeType.Map, 13, 14, 19, 21),
                new Shape(ShapeType.Map, 19, 21, 13, 14),
                new Shape(ShapeType.Map, 13, 14, 19, 21, 24),
                new Shape(ShapeType.Map, 24, 19, 14, 13, 21),
                new Shape(ShapeType.Map, 19, 13, 21, 24, 14)}) {
                for (int axis = 0; axis < outshape.Ndim; axis++) {
                    int length = outshape[axis];

                    for (int n = 1; n <= 5; n++) {
                        int[] c = (new int[n]).Select((_) => 1).ToArray();

                        for (int j = n; j < length; j++) {
                            c[rd.Next(c.Length)]++;
                        }

                        float[][] xs = new float[n][];
                        Shape[] inshapes = new Shape[n];
                        OverflowCheckedTensor[] intensors = new OverflowCheckedTensor[n];
                        OverflowCheckedTensor[] outtensors = new OverflowCheckedTensor[n];

                        for (int i = 0; i < n; i++) {
                            int[] s = outshape;
                            s[axis] = c[i];
                            inshapes[i] = new Shape(ShapeType.Map, s);

                            xs[i] = (new float[inshapes[i].Length]).Select((_) => (float)rd.NextDouble() * (i + 1)).ToArray();

                            intensors[i] = new OverflowCheckedTensor(inshapes[i], xs[i]);
                            outtensors[i] = new OverflowCheckedTensor(inshapes[i]);
                        }

                        OverflowCheckedTensor outtensor = new(outshape);

                        Concat concat = new(inshapes, outshape, axis);
                        concat.Execute(intensors.Concat(new Tensor[] { outtensor }).ToArray());

                        float[] y = outtensor.State.Value;

                        Separate separate = new(outshape, inshapes, axis);
                        separate.Execute((new Tensor[] { outtensor }).Concat(outtensors).ToArray());

                        CollectionAssert.AreEqual(y, outtensor.State.Value);

                        for (int i = 0; i < n; i++) {
                            CollectionAssert.AreEqual(xs[i], outtensors[i].State.Value);
                        }

                        Console.WriteLine($"pass {outshape} axis:{axis} -> {string.Join(", ", inshapes.Select((shape) => shape.ToString()))}");
                    }
                };
            }
        }

        [TestMethod]
        public void ExceptionTest() {
            Assert.ThrowsException<ArgumentException>(() => {
                Shape outshape = new(ShapeType.Map, 3, 7, 19, 5, 11);
                Shape inshape1 = new(ShapeType.Map, 3, 7, 6, 5, 11);
                Shape inshape2 = new(ShapeType.Map, 3, 7, 9, 5, 11);
                Shape inshape3 = new(ShapeType.Map, 3, 7, 5, 5, 11);

                OverflowCheckedTensor v1 = new(inshape1);
                OverflowCheckedTensor v2 = new(inshape2);
                OverflowCheckedTensor v3 = new(inshape3);

                OverflowCheckedTensor vc = new(outshape);

                Separate ope = new(vc.Shape, new Shape[] { v1.Shape, v2.Shape, v3.Shape }, axis: 2);
            }
            );

            Assert.ThrowsException<ArgumentException>(() => {
                Shape outshape = new(ShapeType.Map, 3, 7, 19, 5, 11);
                Shape inshape1 = new(ShapeType.Map, 3, 7, 6, 5, 11);
                Shape inshape2 = new(ShapeType.Map, 3, 7, 9, 5, 11);
                Shape inshape3 = new(ShapeType.Map, 3, 7, 4, 5, 11);

                OverflowCheckedTensor v1 = new(inshape1);
                OverflowCheckedTensor v2 = new(inshape2);
                OverflowCheckedTensor v3 = new(inshape3);

                OverflowCheckedTensor vc = new(outshape);

                Separate ope = new(vc.Shape, new Shape[] { v1.Shape, v2.Shape, v3.Shape }, axis: 1);
            }
            );

            Assert.ThrowsException<ArgumentException>(() => {
                Shape outshape = new(ShapeType.Map, 3, 7, 19, 5, 11);
                Shape inshape1 = new(ShapeType.Map, 3, 7, 6, 5, 11);
                Shape inshape2 = new(ShapeType.Map, 3, 6, 9, 5, 11);
                Shape inshape3 = new(ShapeType.Map, 3, 7, 4, 5, 11);

                OverflowCheckedTensor v1 = new(inshape1);
                OverflowCheckedTensor v2 = new(inshape2);
                OverflowCheckedTensor v3 = new(inshape3);

                OverflowCheckedTensor vc = new(outshape);

                Separate ope = new(vc.Shape, new Shape[] { v1.Shape, v2.Shape, v3.Shape }, axis: 2);
            }
            );

            Assert.ThrowsException<ArgumentException>(() => {
                Shape outshape = new(ShapeType.Map, 3, 7, 19, 5, 11);
                Shape inshape1 = new(ShapeType.Map, 3, 7, 6, 5, 11);
                Shape inshape2 = new(ShapeType.Map, 3, 7, 9, 5, 11);
                Shape inshape3 = new(ShapeType.Map, 3, 7, 4, 5);

                OverflowCheckedTensor v1 = new(inshape1);
                OverflowCheckedTensor v2 = new(inshape2);
                OverflowCheckedTensor v3 = new(inshape3);

                OverflowCheckedTensor vc = new(outshape);

                Separate ope = new(vc.Shape, new Shape[] { v1.Shape, v2.Shape, v3.Shape }, axis: 2);
            }
            );

            Assert.ThrowsException<ArgumentException>(() => {
                Shape outshape = new(ShapeType.Map, 3, 7, 19, 5, 11, 12);
                Shape inshape1 = new(ShapeType.Map, 3, 7, 6, 5, 11);
                Shape inshape2 = new(ShapeType.Map, 3, 7, 9, 5, 11);
                Shape inshape3 = new(ShapeType.Map, 3, 7, 4, 5, 11);

                OverflowCheckedTensor v1 = new(inshape1);
                OverflowCheckedTensor v2 = new(inshape2);
                OverflowCheckedTensor v3 = new(inshape3);

                OverflowCheckedTensor vc = new(outshape);

                Separate ope = new(vc.Shape, new Shape[] { v1.Shape, v2.Shape, v3.Shape }, axis: 5);
            }
            );

            Assert.ThrowsException<ArgumentException>(() => {
                Shape outshape = new(ShapeType.Map, 3, 7, 19, 5, 11);
                Shape inshape1 = new(ShapeType.Map, 3, 7, 6, 5, 11);
                Shape inshape2 = new(ShapeType.Map, 3, 7, 9, 5, 11);
                Shape inshape3 = new(ShapeType.Map, 3, 7, 4, 5, 11);

                OverflowCheckedTensor v1 = new(inshape1);
                OverflowCheckedTensor v2 = new(inshape2);
                OverflowCheckedTensor v3 = new(inshape3);

                OverflowCheckedTensor vc = new(outshape);

                Separate ope = new(vc.Shape, new Shape[] { v1.Shape, v2.Shape, v3.Shape }, axis: 5);
            }
            );
        }

        [TestMethod]
        public void SpeedTest() {
            Shape inshape = new(ShapeType.Map, 3, 7, 19, 5, 11);
            Shape outshape1 = new(ShapeType.Map, 3, 7, 6, 5, 11);
            Shape outshape2 = new(ShapeType.Map, 3, 7, 9, 5, 11);
            Shape outshape3 = new(ShapeType.Map, 3, 7, 4, 5, 11);

            OverflowCheckedTensor vc = new(inshape);

            OverflowCheckedTensor v1 = new(outshape1);
            OverflowCheckedTensor v2 = new(outshape2);
            OverflowCheckedTensor v3 = new(outshape3);

            Separate ope = new(vc.Shape, new Shape[] { v1.Shape, v2.Shape, v3.Shape }, axis: 2);

            Cuda.Profiler.Initialize("../../../../profiler.nvsetting", "../../nvprofiles/separate.nvvp");
            Cuda.Profiler.Start();

            ope.Execute(vc, v1, v2, v3);

            Cuda.Profiler.Stop();
        }
    }
}
