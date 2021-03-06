using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using TensorShader.Updaters.OptimizeMethod;
using static TensorShader.Field;

namespace TensorShaderTest {
    [TestClass]
    public class UpdaterTest {
        [TestMethod]
        public void MinimumTest() {
            Tensor errtensor = -3;

            {
                ParameterField ferr = new(errtensor);

                (Flow flow, Parameters parameters) = Flow.Optimize(ferr);
                parameters.AddUpdater((parameter) => new SGD(parameter, 0.1f));

                flow.Execute();
                parameters.Update();

                Assert.AreEqual(-2.7f, (float)ferr.State);
                Assert.AreEqual(-3f, (float)ferr.GradState);

                Snapshot snapshot = parameters.Save();

                ferr.State = 1.5f;

                parameters.Load(snapshot);

                Assert.AreEqual(-2.7f, (float)ferr.State);
            }
        }

        [TestMethod]
        public void In1Out1Test() {
            Tensor intensor = -1.5f;
            Tensor outtensor = 2;

            {
                ParameterField f1 = new(intensor);
                VariableField fo = new(outtensor);

                Field f2 = Abs(f1);
                Field ferr = f2 - fo;

                (Flow flow, Parameters parameters) = Flow.Optimize(ferr);
                parameters.AddUpdater((parameter) => new SGD(parameter, 0.1f));

                flow.Execute();
                parameters.Update();

                Assert.AreEqual(-1.55f, (float)f1.State);

                Snapshot snapshot = parameters.Save();

                f1.State = 1.5f;

                parameters.Load(snapshot);

                Assert.AreEqual(-1.55f, (float)f1.State);
            }
        }

        [TestMethod]
        public void In2Out1Test() {
            Tensor in1tensor = 1;
            Tensor in2tensor = 2;
            Tensor outtensor = -3;

            {
                ParameterField f1 = new(in1tensor);
                ParameterField f2 = new(in2tensor);
                VariableField fo = new(outtensor);

                Field f3 = f1 + f2;
                Field ferr = f3 - fo;

                (Flow flow, Parameters parameters) = Flow.Optimize(ferr);
                parameters.AddUpdater((parameter) => new SGD(parameter, 0.1f));

                flow.Execute();
                parameters.Update();

                Assert.AreEqual(0.4f, f1.State, 1e-6f);
                Assert.AreEqual(1.4f, f2.State, 1e-6f);

                Snapshot snapshot = parameters.Save();

                f1.State = 1.5f;
                f2.State = 1.5f;

                parameters.Load(snapshot);

                Assert.AreEqual(0.4f, f1.State, 1e-6f);
                Assert.AreEqual(1.4f, f2.State, 1e-6f);
            }
        }

        [TestMethod]
        public void RejoinTest() {
            Tensor intensor = -1.5f;
            Tensor outtensor = -2;

            {
                ParameterField f1 = new(intensor);
                VariableField fo = new(outtensor);

                (Field f2, Field f3) = (f1, -f1);
                Field f4 = f2 * f3;
                Field ferr = f4 - fo;

                (Flow flow, Parameters parameters) = Flow.Optimize(ferr);
                parameters.AddUpdater((parameter) => new SGD(parameter, 0.1f));

                flow.Execute();
                parameters.Update();

                Assert.AreEqual((-2.25f + 2) * 1.5f * 2, (float)f1.GradState);

                Assert.AreEqual(-1.5f - (-2.25f + 2) * 1.5f * 2 * 0.1f, f1.State, 1e-6f);

                Snapshot snapshot = parameters.Save();

                f1.State = 1.5f;
                f1.GradState = 1.5f;

                parameters.Load(snapshot);

                Assert.AreEqual(-1.5f - (-2.25f + 2) * 1.5f * 2 * 0.1f, f1.State, 1e-6f);
                Assert.AreEqual((-2.25f + 2) * 1.5f * 2, (float)f1.GradState);
            }
        }

        [TestMethod]
        public void ChangeUpdaterValueTest() {
            Tensor in1tensor = 1;
            Tensor in2tensor = 2;
            Tensor outtensor = -3;

            {
                ParameterField f1 = new(in1tensor);
                ParameterField f2 = new(in2tensor);
                VariableField fo = new(outtensor);

                Field f3 = f1 + f2;
                Field ferr = f3 - fo;

                (Flow flow, Parameters parameters) = Flow.Optimize(ferr);
                parameters.AddUpdater((parameter) => new SGD(parameter, 0.1f));

                Assert.AreEqual(0.1f, (float)parameters["SGD.Lambda"], 1e-6f);

                parameters["SGD.Lambda"] = 0.2;

                Assert.AreEqual(0.2f, (float)parameters["SGD.Lambda"], 1e-6f);

                flow.Execute();
                parameters.Update();

                Assert.AreEqual(-0.2f, f1.State, 1e-6f);
                Assert.AreEqual(0.8f, f2.State, 1e-6f);

                Snapshot snapshot = parameters.Save();

                parameters["SGD.Lambda"] = 0.1;
                f1.State = 1.5f;
                f2.State = 1.5f;

                Assert.AreEqual(0.1f, (float)parameters["SGD.Lambda"], 1e-6f);

                parameters.Load(snapshot);

                Assert.AreEqual(-0.2f, f1.State, 1e-6f);
                Assert.AreEqual(0.8f, f2.State, 1e-6f);

                parameters.Where((parameter) => parameter == f1)["SGD.Lambda"] = 0.3;
                Assert.ThrowsException<System.ArgumentException>(() =>
                    _ = (float)parameters["SGD.Lambda"]
                );
            }
        }
    }
}
