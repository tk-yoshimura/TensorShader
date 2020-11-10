using System;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;

namespace TensorShaderTest.Functions.Aggregation {
    [TestClass]
    public class AverageTest {
        [TestMethod]
        public void ExecuteTest() {
            Random rd = new Random(1234);

            int length = 6, ch = 4, batch = 3, width = 4, height = 5;

            float[] x = (new float[ch * width * height * length * batch]).Select((_) => (float)rd.NextDouble()).ToArray();

            Shape shape = Shape.Map3D(ch, width, height, length, batch);

            Tensor v1 = (shape, x);

            /*axis = 0, keepdims = true*/
            {
                Tensor v2 = Tensor.Average(v1, axes: new int[] { 0 }, keepdims: true);

                Assert.AreEqual(Shape.Map3D(1, width, height, length, batch), v2.Shape);
            }

            /*axis = 0, keepdims = false*/
            {
                Tensor v2 = Tensor.Average(v1, axes: new int[] { 0 }, keepdims: false);

                Assert.AreEqual(Shape.Map2D(width, height, length, batch), v2.Shape);
            }

            /*axis = 1, keepdims = true*/
            {
                Tensor v2 = Tensor.Average(v1, axes: new int[] { 1 }, keepdims: true);

                Assert.AreEqual(Shape.Map3D(ch, 1, height, length, batch), v2.Shape);
            }

            /*axis = 1, keepdims = false*/
            {
                Tensor v2 = Tensor.Average(v1, axes: new int[] { 1 }, keepdims: false);

                Assert.AreEqual(Shape.Map2D(ch, height, length, batch), v2.Shape);
            }

            /*axis = 2, keepdims = true*/
            {
                Tensor v2 = Tensor.Average(v1, axes: new int[] { 2 }, keepdims: true);

                Assert.AreEqual(Shape.Map3D(ch, width, 1, length, batch), v2.Shape);
            }

            /*axis = 2, keepdims = false*/
            {
                Tensor v2 = Tensor.Average(v1, axes: new int[] { 2 }, keepdims: false);

                Assert.AreEqual(Shape.Map2D(ch, width, length, batch), v2.Shape);
            }

            /*axis = 3, keepdims = true*/
            {
                Tensor v2 = Tensor.Average(v1, axes: new int[] { 3 }, keepdims: true);

                Assert.AreEqual(Shape.Map3D(ch, width, height, 1, batch), v2.Shape);
            }

            /*axis = 3, keepdims = false*/
            {
                Tensor v2 = Tensor.Average(v1, axes: new int[] { 3 }, keepdims: false);

                Assert.AreEqual(Shape.Map2D(ch, width, height, batch), v2.Shape);
            }

            /*axis = 4, keepdims = true*/
            {
                Tensor v2 = Tensor.Average(v1, axes: new int[] { 4 }, keepdims: true);

                Assert.AreEqual(Shape.Map3D(ch, width, height, length, 1), v2.Shape);
            }

            /*axis = 4, keepdims = false*/
            {
                Tensor v2 = Tensor.Average(v1, axes: new int[] { 4 }, keepdims: false);

                Assert.AreEqual(Shape.Map2D(ch, width, height, length), v2.Shape);
            }

            /*axis = 0, 2, 4, keepdims = true*/
            {
                Tensor v2 = Tensor.Average(v1, axes: new int[] { 0, 2, 4 }, keepdims: true);

                Assert.AreEqual(Shape.Map3D(1, width, 1, length, 1), v2.Shape);
            }

            /*axis = 0, 2, 4, keepdims = false*/
            {
                Tensor v2 = Tensor.Average(v1, axes: new int[] { 0, 2, 4 }, keepdims: false);

                Assert.AreEqual(Shape.Map0D(width, length), v2.Shape);
            }

            /*axis = 1, 2, 3, keepdims = true*/
            {
                Tensor v2 = Tensor.Average(v1, axes: new int[] { 1, 2, 3 }, keepdims: true);

                Assert.AreEqual(Shape.Map3D(ch, 1, 1, 1, batch), v2.Shape);
            }

            /*axis = 1, 2, 3, keepdims = false*/
            {
                Tensor v2 = Tensor.Average(v1, axes: new int[] { 1, 2, 3 }, keepdims: false);

                Assert.AreEqual(Shape.Map0D(ch, batch), v2.Shape);
            }

            /*axis = 1, 2, 3, keepdims = true*/
            {
                Tensor v2 = Tensor.Average(v1, axes: new int[] { 1, 2, 3, 4 }, keepdims: true);

                Assert.AreEqual(Shape.Map3D(ch, 1, 1, 1, 1), v2.Shape);
            }

            /*axis = 1, 2, 3, keepdims = false*/
            {
                Tensor v2 = Tensor.Average(v1, axes: new int[] { 1, 2, 3, 4 }, keepdims: false);

                Assert.AreEqual(Shape.Vector(ch), v2.Shape);
            }

            /*axis = all, keepdims = true*/
            {
                Tensor v2 = Tensor.Average(v1, keepdims: true);

                Assert.AreEqual(Shape.Map3D(1, 1, 1, 1, 1), v2.Shape);
            }

            /*axis = all, keepdims = false*/
            {
                Tensor v2 = Tensor.Average(v1, keepdims: false);

                Assert.AreEqual(Shape.Scalar, v2.Shape);

                float average_expect = x.Average();
                float average_actual = v2.State[0];

                Assert.AreEqual(average_expect, average_actual, 1e-6f);
            }
        }
    }
}
