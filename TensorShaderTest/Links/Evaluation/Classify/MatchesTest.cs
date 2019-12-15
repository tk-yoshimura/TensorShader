using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using static TensorShader.Field;

namespace TensorShaderTest.Links.Evaluation.Classify {
    [TestClass]
    public class MatchesTest {
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

            Field matches = Matches(x, t);
            OutputNode matches_node = matches.Value.Save();

            Flow flow = Flow.Inference(matches_node);

            flow.Execute();

            float[] matches_actual = matches_node.Tensor.State;

            Assert.AreEqual(Shape.Vector(channels), matches_node.Shape);
            CollectionAssert.AreEqual(new float[] { 1, 1, 2 }, matches_actual);
        }
    }
}
