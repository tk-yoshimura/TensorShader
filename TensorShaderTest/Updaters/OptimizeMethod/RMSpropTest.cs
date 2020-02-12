using System;
using System.Collections.Generic;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using TensorShader.Updaters.OptimizeMethod;
using static TensorShader.Field;

namespace TensorShaderTest.Updaters.OptimizeMethod {
    [TestClass]
    public class RMSpropTest {
        [TestMethod]
        public void ReferenceTest() {
            const int loops = 10;

            ParameterField x = new Tensor(Shape.Scalar, new float[] { 0.7f });
            ParameterField y = new Tensor(Shape.Scalar, new float[] { 0.8f });

            Field f = Square(x) + Square(y);
            Field g = Square(Sin(x + Sin(y))) + Square(Sin(y + Sin(x)));

            StoreField loss = f / 20 + g;

            (Flow flow, Parameters parameters) = Flow.Optimize(loss);
            parameters.AddUpdater((parameter) => new RMSprop(parameter, lambda: 0.1f, rho: 0.9f));

            List<float> losses = new List<float>(), xs = new List<float>(), ys = new List<float>();

            for (int i = 0; i < loops; i++) {
                flow.Execute();
                parameters.Update();

                losses.Add(loss.State[0]);
                xs.Add(x.State[0]);
                ys.Add(y.State[0]);
            }

            float[] losses_expect = {
                2.017204e+00f,  1.154929e+00f,  1.717709e-01f,  1.181439e-01f,  9.004968e-02f,
                7.239353e-02f,  6.016864e-02f,  5.116799e-02f,  4.425157e-02f,  3.876602e-02f,
            };

            float[] xs_expect = {
                3.837722e-01f,  9.862978e-02f,  7.299351e-02f,  5.738019e-02f,  4.636137e-02f,
                3.796008e-02f,  3.123579e-02f,  2.566988e-02f,  2.094758e-02f,  1.686437e-02f,
            };

            float[] ys_expect = {
                4.837722e-01f,  1.973386e-01f,  1.711202e-01f,  1.551072e-01f,  1.437824e-01f,
                1.351323e-01f,  1.281980e-01f,  1.224498e-01f,  1.175662e-01f,  1.133380e-01f,
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

            ParameterField x = new Tensor(Shape.Scalar, new float[] { 0.7f });
            ParameterField y = new Tensor(Shape.Scalar, new float[] { 0.8f });

            Field f = Square(x) + Square(y);
            Field g = Square(Sin(x + Sin(y))) + Square(Sin(y + Sin(x)));

            StoreField loss = f / 20 + g;

            (Flow flow, Parameters parameters) = Flow.Optimize(loss);
            parameters.AddUpdater((parameter) => new RMSprop(parameter, lambda: 0.1f, rho: 0.9f));

            Snapshot snapshot = parameters.Save();

            List<float> losses_first = new List<float>(), xs_first = new List<float>(), ys_first = new List<float>();

            for (int i = 0; i < loops; i++) {
                flow.Execute();
                parameters.Update();

                losses_first.Add(loss.State[0]);
                xs_first.Add(x.State[0]);
                ys_first.Add(y.State[0]);
            }

            x.State = new float[] { 0.7f };
            y.State = new float[] { 0.8f };

            parameters.InitializeUpdater();

            List<float> losses_second = new List<float>(), xs_second = new List<float>(), ys_second = new List<float>();

            for (int i = 0; i < loops; i++) {
                flow.Execute();
                parameters.Update();

                losses_second.Add(loss.State[0]);
                xs_second.Add(x.State[0]);
                ys_second.Add(y.State[0]);
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
                xs_third.Add(x.State[0]);
                ys_third.Add(y.State[0]);
            }

            CollectionAssert.AreEqual(losses_first, losses_third);
            CollectionAssert.AreEqual(xs_first, xs_third);
            CollectionAssert.AreEqual(ys_first, ys_third);
        }
    }
}
