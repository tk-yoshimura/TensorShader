using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using TensorShader;
using TensorShader.Updaters.OptimizeMethod;
using static TensorShader.Field;

namespace TensorShaderTest.Updaters.OptimizeMethod {
    [TestClass]
    public class AdaGradTest {
        [TestMethod]
        public void ReferenceTest() {
            const int loops = 10;

            ParameterField x = 0.7f;
            ParameterField y = 0.8f;

            Field f = Square(x) + Square(y);
            Field g = Square(Sin(x + Sin(y))) + Square(Sin(y + Sin(x)));

            StoreField loss = f / 20 + g;

            (Flow flow, Parameters parameters) = Flow.Optimize(loss);
            parameters.AddUpdater((parameter) => new AdaGrad(parameter, lambda: 0.1f));

            List<float> losses = new(), xs = new(), ys = new();

            for (int i = 0; i < loops; i++) {
                flow.Execute();
                parameters.Update();

                losses.Add(loss.State);
                xs.Add(x.State);
                ys.Add(y.State);
            }

            float[] losses_expect = {
                2.017204e+00f,  1.848749e+00f,  1.610362e+00f,  1.362630e+00f,  1.142554e+00f,
                9.620425e-01f,  8.188430e-01f,  7.060527e-01f,  6.166833e-01f,  5.450422e-01f,
            };

            float[] xs_expect = {
                6.000009e-01f,  5.119559e-01f,  4.392103e-01f,  3.809867e-01f,  3.349419e-01f,
                2.984151e-01f,  2.690911e-01f,  2.451812e-01f,  2.253696e-01f,  2.087025e-01f,
            };

            float[] ys_expect = {
                7.000009e-01f,  6.117151e-01f,  5.386798e-01f,  4.801518e-01f,  4.338083e-01f,
                3.970002e-01f,  3.674175e-01f,  3.432719e-01f,  3.232467e-01f,  3.063857e-01f,
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
            parameters.AddUpdater((parameter) => new AdaGrad(parameter, lambda: 0.1f));

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
