using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using TensorShader;
using TensorShader.Updaters.OptimizeMethod;
using static TensorShader.Field;

namespace TensorShaderTest.Updaters.OptimizeMethod {
    [TestClass]
    public class RMSpropTest {
        [TestMethod]
        public void ReferenceTest() {
            const int loops = 10;

            ParameterField x = 0.7f;
            ParameterField y = 0.8f;

            Field f = Square(x) + Square(y);
            Field g = Square(Sin(x + Sin(y))) + Square(Sin(y + Sin(x)));

            StoreField loss = f / 20 + g;

            (Flow flow, Parameters parameters) = Flow.Optimize(loss);
            parameters.AddUpdater((parameter) => new RMSprop(parameter, lambda: 0.1f, rho: 0.9f));

            List<float> losses = new(), xs = new(), ys = new();

            for (int i = 0; i < loops; i++) {
                flow.Execute();
                parameters.Update();

                losses.Add(loss.State);
                xs.Add(x.State);
                ys.Add(y.State);
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

            ParameterField x = 0.7f;
            ParameterField y = 0.8f;

            Field f = Square(x) + Square(y);
            Field g = Square(Sin(x + Sin(y))) + Square(Sin(y + Sin(x)));

            StoreField loss = f / 20 + g;

            (Flow flow, Parameters parameters) = Flow.Optimize(loss);
            parameters.AddUpdater((parameter) => new RMSprop(parameter, lambda: 0.1f, rho: 0.9f));

            Snapshot snapshot = parameters.Save();

            List<float> losses_first = new(), xs_first = new(), ys_first = new();

            for (int i = 0; i < loops; i++) {
                flow.Execute();
                parameters.Update();

                losses_first.Add(loss.State);
                xs_first.Add(x.State);
                ys_first.Add(y.State);
            }

            x.State = 0.7f;
            y.State = 0.8f;

            parameters.InitializeUpdater();

            List<float> losses_second = new(), xs_second = new(), ys_second = new();

            for (int i = 0; i < loops; i++) {
                flow.Execute();
                parameters.Update();

                losses_second.Add(loss.State);
                xs_second.Add(x.State);
                ys_second.Add(y.State);
            }

            CollectionAssert.AreEqual(losses_first, losses_second);
            CollectionAssert.AreEqual(xs_first, xs_second);
            CollectionAssert.AreEqual(ys_first, ys_second);

            parameters.Load(snapshot);

            List<float> losses_third = new(), xs_third = new(), ys_third = new();

            for (int i = 0; i < loops; i++) {
                flow.Execute();
                parameters.Update();

                losses_third.Add(loss.State);
                xs_third.Add(x.State);
                ys_third.Add(y.State);
            }

            CollectionAssert.AreEqual(losses_first, losses_third);
            CollectionAssert.AreEqual(xs_first, xs_third);
            CollectionAssert.AreEqual(ys_first, ys_third);
        }
    }
}
