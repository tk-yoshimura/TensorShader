using System;
using System.Collections.Generic;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using TensorShader.Updaters.OptimizeMethod;
using static TensorShader.Field;

namespace TensorShaderTest.Updaters.OptimizeMethod {
    [TestClass]
    public class AdamTest {
        [TestMethod]
        public void ReferenceTest() {
            const int loops = 10;

            Tensor x_tensor = new Tensor(Shape.Scalar(), new float[] { 0.7f });
            Tensor y_tensor = new Tensor(Shape.Scalar(), new float[] { 0.8f });

            ParameterField x = x_tensor, y = y_tensor;

            Field f = Square(x) + Square(y);
            Field g = Square(Sin(x + Sin(y))) + Square(Sin(y + Sin(x)));

            StoreField loss = f / 20 + g;

            (Flow flow, Parameters parameters) = Flow.Optimize(loss);
            parameters.AddUpdater((parameter) => new Adam(parameter, alpha: 0.1f));

            List<float> losses = new List<float>(), xs = new List<float>(), ys = new List<float>();

            for (int i = 0; i < loops; i++) {
                flow.Execute();
                parameters.Update();

                losses.Add(loss.State[0]);
                xs.Add(x_tensor.State[0]);
                ys.Add(y_tensor.State[0]);
            }

            float[] losses_expect = {
                2.017204e+00f,  1.848747e+00f,  1.581739e+00f,  1.235519e+00f,  8.501037e-01f,
                4.878383e-01f,  2.126677e-01f,  5.384384e-02f,  7.368052e-04f,  2.728836e-02f,
            };

            float[] xs_expect = {
                6.000000e-01f,  5.027176e-01f,  4.046842e-01f,  3.056116e-01f,  2.077064e-01f,
                1.152703e-01f,  3.180198e-02f,  -4.215108e-02f,  -1.078023e-01f,  -1.662601e-01f,
            };

            float[] ys_expect = {
                7.000000e-01f,  6.028399e-01f,  5.049215e-01f,  4.059139e-01f,  3.079818e-01f,
                2.154228e-01f,  1.317901e-01f,  5.767954e-02f,  -8.111567e-03f,  -6.669372e-02f,
            };

            for (int i = 0; i < loops; i++) {
                Assert.AreEqual(losses_expect[i], losses[i], Math.Abs(losses_expect[i]) * 1e-2f);
                Assert.AreEqual(xs_expect[i], xs[i], Math.Abs(xs[i]) * 1e-2f);
                Assert.AreEqual(ys_expect[i], ys[i], Math.Abs(ys[i]) * 1e-2f);
            }
        }

        [TestMethod]
        public void InitializeTest() {
            const int loops = 10;

            Tensor x_tensor = new Tensor(Shape.Scalar(), new float[] { 0.7f });
            Tensor y_tensor = new Tensor(Shape.Scalar(), new float[] { 0.8f });

            ParameterField x = x_tensor, y = y_tensor;

            Field f = Square(x) + Square(y);
            Field g = Square(Sin(x + Sin(y))) + Square(Sin(y + Sin(x)));

            StoreField loss = f / 20 + g;

            (Flow flow, Parameters parameters) = Flow.Optimize(loss);
            parameters.AddUpdater((parameter) => new Adam(parameter, alpha: 0.1f));

            Snapshot snapshot = parameters.Save();

            List<float> losses_first = new List<float>(), xs_first = new List<float>(), ys_first = new List<float>();

            for (int i = 0; i < loops; i++) {
                flow.Execute();
                parameters.Update();

                losses_first.Add(loss.State[0]);
                xs_first.Add(x_tensor.State[0]);
                ys_first.Add(y_tensor.State[0]);
            }

            x_tensor.State = new float[] { 0.7f };
            y_tensor.State = new float[] { 0.8f };

            parameters.InitializeUpdater();

            List<float> losses_second = new List<float>(), xs_second = new List<float>(), ys_second = new List<float>();

            for (int i = 0; i < loops; i++) {
                flow.Execute();
                parameters.Update();

                losses_second.Add(loss.State[0]);
                xs_second.Add(x_tensor.State[0]);
                ys_second.Add(y_tensor.State[0]);
            }

            CollectionAssert.AreEqual(losses_first, losses_second);
            CollectionAssert.AreEqual(xs_first, xs_second);
            CollectionAssert.AreEqual(ys_first, ys_second);

            parameters.Load(snapshot);

            List<float> losses_third = new List<float>(), xs_third = new List<float>(), ys_third = new List<float>();

            for (int i = 0; i < loops; i++) {
                flow.Execute();
                parameters.Update();

                losses_third.Add(loss.State[0]);
                xs_third.Add(x_tensor.State[0]);
                ys_third.Add(y_tensor.State[0]);
            }

            CollectionAssert.AreEqual(losses_first, losses_third);
            CollectionAssert.AreEqual(xs_first, xs_third);
            CollectionAssert.AreEqual(ys_first, ys_third);
        }
    }
}
