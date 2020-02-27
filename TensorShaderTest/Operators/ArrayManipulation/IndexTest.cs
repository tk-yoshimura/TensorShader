using System;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;

namespace TensorShaderTest.Operators.ArrayManipulation {
    [TestClass]
    public class IndexTest {
        [TestMethod]
        public void ExecuteTest() {
            Random rd = new Random(1234);

            foreach (Shape shape in new Shape[]{
                   new Shape(ShapeType.Map, 16, 19, 23, 8, 1, 5, 6),
                   new Shape(ShapeType.Map, 17, 9, 2, 4, 1, 3, 67) }) {

                for (int axis = 0; axis < shape.Ndim; axis++) {
                    int stride = 1, length = shape[axis];
                    for (int i = 0; i < axis; i++) {
                        stride *= shape[i];
                    }

                    float[] x1 = (new float[shape.Length]).Select((_, idx) => (float)(idx / stride % length)).ToArray();

                    OverflowCheckedTensor tensor = new OverflowCheckedTensor(shape);

                    TensorShader.Operators.ArrayManipulation.Index ope
                        = new TensorShader.Operators.ArrayManipulation.Index(shape, axis);

                    ope.Execute(tensor);

                    CollectionAssert.AreEqual(x1, tensor.State);
                }
            }
        }
    }
}
