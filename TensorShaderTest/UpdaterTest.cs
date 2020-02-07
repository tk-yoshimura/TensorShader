using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using TensorShader.Updaters.OptimizeMethod;
using static TensorShader.Field;

namespace TensorShaderTest {
    [TestClass]
    public class UpdaterTest {
        [TestMethod]
        public void MinimumTest() {
            Tensor errtensor = new Tensor(Shape.Scalar(), new float[] { -3 });

            {
                ParameterField ferr = new ParameterField(errtensor);

                (Flow flow, Parameters parameters) = Flow.Optimize(ferr);
                parameters.AddUpdater((parameter) => new SGD(parameter, 0.1f));

                flow.Execute();
                parameters.Update();

                Assert.AreEqual(-2.7f, ferr.State[0]);
                Assert.AreEqual(-3f, ferr.GradState[0]);

                Snapshot snapshot = parameters.Save();

                ferr.State = new float[] { 1.5f };

                parameters.Load(snapshot);

                Assert.AreEqual(-2.7f, ferr.State[0]);
            }
        }

        [TestMethod]
        public void In1Out1Test() {
            Tensor intensor = new Tensor(Shape.Scalar(), new float[] { -1.5f });
            Tensor outtensor = new Tensor(Shape.Scalar(), new float[] { 2 });

            {
                ParameterField f1 = new ParameterField(intensor);
                VariableField fo = new VariableField(outtensor);

                Field f2 = Abs(f1);
                Field ferr = f2 - fo;

                (Flow flow, Parameters parameters) = Flow.Optimize(ferr);
                parameters.AddUpdater((parameter) => new SGD(parameter, 0.1f));

                flow.Execute();
                parameters.Update();

                Assert.AreEqual(-1.55f, f1.State[0]);

                Snapshot snapshot = parameters.Save();

                f1.State = new float[] { 1.5f };

                parameters.Load(snapshot);

                Assert.AreEqual(-1.55f, f1.State[0]);
            }
        }

        [TestMethod]
        public void In2Out1Test() {
            Tensor in1tensor = new Tensor(Shape.Scalar(), new float[] { 1 });
            Tensor in2tensor = new Tensor(Shape.Scalar(), new float[] { 2 });
            Tensor outtensor = new Tensor(Shape.Scalar(), new float[] { -3 });

            {
                ParameterField f1 = new ParameterField(in1tensor);
                ParameterField f2 = new ParameterField(in2tensor);
                VariableField fo = new VariableField(outtensor);

                Field f3 = f1 + f2;
                Field ferr = f3 - fo;

                (Flow flow, Parameters parameters) = Flow.Optimize(ferr);
                parameters.AddUpdater((parameter) => new SGD(parameter, 0.1f));

                flow.Execute();
                parameters.Update();

                Assert.AreEqual(0.4f, f1.State[0], 1e-6f);
                Assert.AreEqual(1.4f, f2.State[0], 1e-6f);

                Snapshot snapshot = parameters.Save();

                f1.State = new float[] { 1.5f };
                f2.State = new float[] { 1.5f };

                parameters.Load(snapshot);

                Assert.AreEqual(0.4f, f1.State[0], 1e-6f);
                Assert.AreEqual(1.4f, f2.State[0], 1e-6f);
            }
        }

        [TestMethod]
        public void RejoinTest() {
            Tensor intensor = new Tensor(Shape.Scalar(), new float[] { -1.5f });
            Tensor outtensor = new Tensor(Shape.Scalar(), new float[] { -2 });

            {
                ParameterField f1 = new ParameterField(intensor);
                VariableField fo = new VariableField(outtensor);

                (Field f2, Field f3) = (f1, -f1);
                Field f4 = f2 * f3;
                Field ferr = f4 - fo;

                (Flow flow, Parameters parameters) = Flow.Optimize(ferr);
                parameters.AddUpdater((parameter) => new SGD(parameter, 0.1f));

                flow.Execute();
                parameters.Update();

                Assert.AreEqual((-2.25f + 2) * 1.5f * 2, f1.GradState[0]);

                Assert.AreEqual(-1.5f - (-2.25f + 2) * 1.5f * 2 * 0.1f, f1.State[0], 1e-6f);

                Snapshot snapshot = parameters.Save();

                f1.State = new float[] { 1.5f };
                f1.GradState = new float[] { 1.5f };

                parameters.Load(snapshot);

                Assert.AreEqual(-1.5f - (-2.25f + 2) * 1.5f * 2 * 0.1f, f1.State[0], 1e-6f);
                Assert.AreEqual((-2.25f + 2) * 1.5f * 2, f1.GradState[0]);
            }
        }

        [TestMethod]
        public void ChangeUpdaterValueTest() {
            Tensor in1tensor = new Tensor(Shape.Scalar(), new float[] { 1 });
            Tensor in2tensor = new Tensor(Shape.Scalar(), new float[] { 2 });
            Tensor outtensor = new Tensor(Shape.Scalar(), new float[] { -3 });

            {
                ParameterField f1 = new ParameterField(in1tensor);
                ParameterField f2 = new ParameterField(in2tensor);
                VariableField fo = new VariableField(outtensor);

                Field f3 = f1 + f2;
                Field ferr = f3 - fo;

                (Flow flow, Parameters parameters) = Flow.Optimize(ferr);
                parameters.AddUpdater((parameter) => new SGD(parameter, 0.1f));

                Assert.AreEqual(0.1f, (float)parameters["SGD.Lambda"], 1e-6f);

                parameters["SGD.Lambda"] = 0.2;

                Assert.AreEqual(0.2f, (float)parameters["SGD.Lambda"], 1e-6f);

                flow.Execute();
                parameters.Update();

                Assert.AreEqual(-0.2f, f1.State[0], 1e-6f);
                Assert.AreEqual(0.8f, f2.State[0], 1e-6f);

                Snapshot snapshot = parameters.Save();

                parameters["SGD.Lambda"] = 0.1;
                f1.State = new float[] { 1.5f };
                f2.State = new float[] { 1.5f };

                Assert.AreEqual(0.1f, (float)parameters["SGD.Lambda"], 1e-6f);

                parameters.Load(snapshot);

                Assert.AreEqual(-0.2f, f1.State[0], 1e-6f);
                Assert.AreEqual(0.8f, f2.State[0], 1e-6f);

                parameters.Where((parameter) => parameter == f1)["SGD.Lambda"] = 0.3;
                Assert.ThrowsException<System.ArgumentException>(() =>
                    _ = (float)parameters["SGD.Lambda"]
                );
            }
        }
    }
}
