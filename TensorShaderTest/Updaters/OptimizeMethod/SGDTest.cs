using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using TensorShader;
using TensorShader.Updaters.OptimizeMethod;
using static TensorShader.Field;

namespace TensorShaderTest.Updaters.OptimizeMethod {
    [TestClass]
    public class SGDTest {
        [TestMethod]
        public void ReferenceTest() {
            const int loops = 10;

            ParameterField x = 0.7f;
            ParameterField y = 0.8f;

            Field f = Square(x) + Square(y);
            Field g = Square(Sin(x + Sin(y))) + Square(Sin(y + Sin(x)));

            StoreField loss = f / 20 + g;

            (Flow flow, Parameters parameters) = Flow.Optimize(loss);
            parameters.AddUpdater((parameter) => new SGD(parameter, lambda: 0.01f));

            List<float> losses = new(), xs = new(), ys = new();

            for (int i = 0; i < loops; i++) {
                flow.Execute();
                parameters.Update();

                losses.Add(loss.State);
                xs.Add(x.State);
                ys.Add(y.State);
            }

            float[] losses_expect = {
                2.017204e+00f,  2.004166e+00f,  1.987980e+00f,  1.967951e+00f,  1.943281e+00f,
                1.913086e+00f,  1.876436e+00f,  1.832431e+00f,  1.780311e+00f,  1.719609e+00f,
            };

            float[] xs_expect = {
                6.886300e-01f,  6.760122e-01f,  6.620414e-01f,  6.466209e-01f,  6.296709e-01f,
                6.111408e-01f,  5.910241e-01f,  5.693740e-01f,  5.463198e-01f,  5.220780e-01f,
            };

            float[] ys_expect = {
                7.890884e-01f,  7.769550e-01f,  7.634956e-01f,  7.486135e-01f,  7.322285e-01f,
                7.142882e-01f,  6.947826e-01f,  6.737600e-01f,  6.513427e-01f,  6.277383e-01f,
            };

            for (int i = 0; i < loops; i++) {
                Assert.AreEqual(losses_expect[i], losses[i], Math.Abs(losses_expect[i]) * 1e-4f);
                Assert.AreEqual(xs_expect[i], xs[i], Math.Abs(xs[i]) * 1e-4f);
                Assert.AreEqual(ys_expect[i], ys[i], Math.Abs(ys[i]) * 1e-4f);
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
            parameters.AddUpdater((parameter) => new SGD(parameter, lambda: 0.01f));

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
