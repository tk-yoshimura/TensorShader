using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using TensorShader;
using TensorShader.Updaters.OptimizeMethod;
using TensorShaderUtil.ParameterUtil;

namespace TensorShaderUtilTest.ParameterUtil {
    [TestClass]
    public class ParametersValueTest {
        [TestMethod]
        public void ExecuteTest() {
            ParameterField p1 = new Tensor(Shape.Scalar);
            ParameterField p2 = new Tensor(Shape.Scalar);
            ParameterField p3 = new Tensor(Shape.Scalar);

            StoreField s = p1 + p2 + p3;

            (Flow flow, Parameters parameters) = Flow.Optimize(s);
            parameters.AddUpdater((parameter) => new Adam(parameter, 1e-2f));
            ParametersValue<float> adam_alpha = new ParametersValue<float>(parameters, "Adam.Alpha");

            Adam adam1 = (Adam)p1.Updaters[0];
            Adam adam2 = (Adam)p2.Updaters[0];
            Adam adam3 = (Adam)p3.Updaters[0];

            Assert.AreEqual(1e-2f, adam_alpha.Value);

            adam_alpha.Value = 1e-1f;

            Assert.AreEqual(1e-1f, adam_alpha.Value);

            adam_alpha.Value = 1e-3f;

            Assert.AreEqual(1e-3f, adam1.Alpha);
            Assert.AreEqual(1e-3f, adam2.Alpha);
            Assert.AreEqual(1e-3f, adam3.Alpha);

            adam_alpha.Value *= 0.5f;

            Assert.AreEqual(5e-4f, adam1.Alpha);
            Assert.AreEqual(5e-4f, adam2.Alpha);
            Assert.AreEqual(5e-4f, adam3.Alpha);

            adam1.Alpha = 1e-3f;

            Assert.ThrowsException<ArgumentException>(() => { 
                _ = adam_alpha.Value;
            });
        }
    }
}
