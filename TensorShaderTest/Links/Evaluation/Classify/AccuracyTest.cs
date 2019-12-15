using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using static TensorShader.Field;

namespace TensorShaderTest.Links.Evaluation.Classify {
    [TestClass]
    public class AccuracyTest {
        [TestMethod]
        public void ReferenceMethod() {
            int channels = 3, batch = 16;

            float[] xval =
                { 1, 2, 3, //2 -  0
                  2, 1, 3, //2 -  1
                  3, 2, 1, //0 -  2
                  1, 2, 1, //1 +  1
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

            Field acc = Accuracy(x, t);
            OutputNode accnode = acc.Value.Save();

            Flow flow = Flow.Inference(accnode);

            flow.Execute();

            float[] acc_actual = accnode.Tensor.State;

            Assert.AreEqual(1, acc_actual.Length);
            Assert.AreEqual(4 / 16.0f, acc_actual[0]);
        }
    }
}
