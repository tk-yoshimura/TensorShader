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
            StoreField mat_store = mat;
            StoreField exp_store = Sum(mat, Axis.ConfusionMatrix.Actual );
            StoreField act_store = Sum(mat, Axis.ConfusionMatrix.Expect );

            (Flow flow, _) = Flow.Inference(mat_store, exp_store, act_store);

            flow.Execute();

            float[] mat_actual = mat_store.State;

            Assert.AreEqual(channels * channels, mat_actual.Length);
            Assert.AreEqual(2, mat_store.Shape.Ndim);
            Assert.AreEqual(batch, mat_actual.Sum());
            CollectionAssert.AreEqual(mat_expect, mat_actual);
            CollectionAssert.AreEqual(new float[] { 5, 5, 6 }, exp_store.State);
            CollectionAssert.AreEqual(new float[] { 5, 4, 7 }, act_store.State);
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
