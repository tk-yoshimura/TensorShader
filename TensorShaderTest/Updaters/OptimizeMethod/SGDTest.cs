using System;
using System.Collections.Generic;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using TensorShader.Updaters.OptimizeMethod;
using static TensorShader.Field;

namespace TensorShaderTest.Updaters.OptimizeMethod {
    [TestClass]
    public class SGDTest {
        [TestMethod]
        public void ReferenceTest() {
            const int loops = 10;

            ParameterField x = new Tensor(Shape.Scalar, new float[] { 0.7f });
            ParameterField y = new Tensor(Shape.Scalar, new float[] { 0.8f });

            Field f = Square(x) + Square(y);
            Field g = Square(Sin(x + Sin(y))) + Square(Sin(y + Sin(x)));

            StoreField loss = f / 20 + g;

            (Flow flow, Parameters parameters) = Flow.Optimize(loss);
            parameters.AddUpdater((parameter) => new SGD(parameter, lambda: 0.01f));

            List<float> losses = new List<float>(), xs = new List<float>(), ys = new List<float>();

            for (int i = 0; i < loops; i++) {
                flow.Execute();
                parameters.Update();

                losses.Add(loss.State[0]);
                xs.Add(x.State[0]);
                ys.Add(y.State[0]);
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

            ParameterField x = new Tensor(Shape.Scalar, new float[] { 0.7f });
            ParameterField y = new Tensor(Shape.Scalar, new float[] { 0.8f });

            Field f = Square(x) + Square(y);
            Field g = Square(Sin(x + Sin(y))) + Square(Sin(y + Sin(x)));

            StoreField loss = f / 20 + g;

            (Flow flow, Parameters parameters) = Flow.Optimize(loss);
            parameters.AddUpdater((parameter) => new SGD(parameter, lambda: 0.01f));

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
