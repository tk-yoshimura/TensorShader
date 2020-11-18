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

            VariableField x = (Shape.Map0D(channels, batch), xval);
            VariableField t = (Shape.Vector(batch), tval);

            StoreField mat = ConfusionMatrix(x, t);
            StoreField exp = Sum(mat, Axis.ConfusionMatrix.Actual);
            StoreField act = Sum(mat, Axis.ConfusionMatrix.Expect);

            (Flow flow, _) = Flow.Inference(mat, exp, act);

            flow.Execute();

            float[] mat_actual = mat.State.Value;

            Assert.AreEqual(channels * channels, mat_actual.Length);
            Assert.AreEqual(2, mat.Shape.Ndim);
            Assert.AreEqual(batch, mat_actual.Sum());
            CollectionAssert.AreEqual(mat_expect, mat_actual);
            CollectionAssert.AreEqual(new float[] { 5, 5, 6 }, exp.State.Value);
            CollectionAssert.AreEqual(new float[] { 5, 4, 7 }, act.State.Value);
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
