using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using static TensorShader.Field;

namespace TensorShaderTest.Links.Evaluation.Classify {
    [TestClass]
    public class ConfusionMatrixTest {
        [TestMethod]
        public void ReferenceMethod() {
            int channels = 3, batch = 16;

            float[] xval =
                { 1, 2, 3, //2 -  0
                  2, 1, 3, //2 -  1
                  3, 2, 1, //0 -  2
                  1, 1, 3, //2 +  1
                  3, 1, 2, //0 -  2
                  2, 1, 3, //2 +  2
                  1, 2, 1, //1 -  0
                  2, 1, 3, //2 +  2
                  1, 2, 1, //1 -  0
                  3, 1, 2, //0 -  1
                  2, 1, 3, //2 -  0
                  3, 2, 1, //0 -  1
                  1, 2, 1, //1 -  2
                  2, 1, 3, //2 -  1
                  1, 2, 1, //1 -  2
                  3, 1, 2, //0 +  0
            };

            float[] tval =
                { 0,
                  1,
                  2,
                  1,
                  2,
                  2,
                  0,
                  2,
                  0,
                  1,
                  0,
                  1,
                  2,
                  1,
                  2,
                  0
            };

            Tensor xtensor = new Tensor(Shape.Map0D(channels, batch), xval);
            Tensor ttensor = new Tensor(Shape.Vector(batch), tval);

            VariableField x = xtensor;
            VariableField t = ttensor;

            Field mat = ConfusionMatrix(x, t);
            OutputNode matnode = mat.Value.Save();
            OutputNode expnode = Sum(mat, axes: new int[] { Axis.ConfusionMatrix.Actual }).Value.Save();
            OutputNode actnode = Sum(mat, axes: new int[] { Axis.ConfusionMatrix.Expect }).Value.Save();

            Flow flow = Flow.Inference(matnode, expnode, actnode);

            flow.Execute();

            float[] mat_actual = matnode.Tensor.State;

            Assert.AreEqual(channels * channels, mat_actual.Length);
            Assert.AreEqual(2, matnode.Shape.Ndim);
            Assert.AreEqual(batch, mat_actual.Sum());
            CollectionAssert.AreEqual(mat_expect, mat_actual);
            CollectionAssert.AreEqual(new float[] { 5, 5, 6 }, expnode.Tensor.State);
            CollectionAssert.AreEqual(new float[] { 5, 4, 7 }, actnode.Tensor.State);
        }

        float[] mat_expect =
                { 1,
                  2,
                  2,
                  2,
                  0,
                  3,
                  2,
                  2,
                  2,
            };

        //0 +  0
        //1 -  0
        //1 -  0
        //2 -  0
        //2 -  0
        //0 -  1
        //0 -  1
        //2 -  1
        //2 -  1
        //2 -  1
        //0 -  2
        //0 -  2
        //1 -  2
        //1 -  2
        //2 +  2
        //2 +  2
    }
}
