using System;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using static TensorShader.Field;

namespace TensorShaderTest.Links.Aggregation {
    [TestClass]
    public class VarianceTest {
        [TestMethod]
        public void ExecuteTest() {
            int channels = 7, width = 3, height = 5, batch = 2;

            float[] xval = (new float[channels * width * height * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
            Tensor xtensor = new Tensor(Shape.Map2D(channels, width, height, batch), xval);

            float[] gxval_true = null, gxval_false = null;

            try {
                float[] yval = (new float[channels * width * height * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
                Tensor ytensor = new Tensor(Shape.Map2D(channels, width, height, batch), yval);

                ParameterField x = xtensor;
                VariableField y_actual = ytensor;

                Field y_expect = Variance(x, new int[] { }, keepdims: true);
                Field err = Abs(y_expect - y_actual);

                (Flow flow, Parameters parameters) = Flow.Optimize(err);

                flow.Execute();

                gxval_true = x.GradTensor.State;
            }
            catch (Exception e) {
                Assert.Fail("axis:none keepdims:true  " + e.Message);
            }

            try {
                float[] yval = (new float[channels * width * height * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
                Tensor ytensor = new Tensor(Shape.Map2D(channels, width, height, batch), yval);

                ParameterField x = xtensor;
                VariableField y_actual = ytensor;

                Field y_expect = Variance(x, new int[] { }, keepdims: false);
                Field err = Abs(y_expect - y_actual);

                (Flow flow, Parameters parameters) = Flow.Optimize(err);

                flow.Execute();

                gxval_false = x.GradTensor.State;

                CollectionAssert.AreEqual(gxval_true, gxval_false);
            }
            catch (Exception e) {
                Assert.Fail("axis:none keepdims:false  " + e.Message);
            }

            try {
                float[] yval = (new float[width * height * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
                Tensor ytensor = new Tensor(Shape.Map2D(1, width, height, batch), yval);

                ParameterField x = xtensor;
                VariableField y_actual = ytensor;

                Field y_expect = Variance(x, new int[] { Axis.Map2D.Channels }, keepdims: true);
                Field err = Abs(y_expect - y_actual);

                (Flow flow, Parameters parameters) = Flow.Optimize(err);

                flow.Execute();

                gxval_true = x.GradTensor.State;
            }
            catch (Exception e) {
                Assert.Fail("axis:0 keepdims:true  " + e.Message);
            }

            try {
                float[] yval = (new float[width * height * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
                Tensor ytensor = new Tensor(Shape.Map1D(width, height, batch), yval);

                ParameterField x = xtensor;
                VariableField y_actual = ytensor;

                Field y_expect = Variance(x, new int[] { Axis.Map2D.Channels }, keepdims: false);
                Field err = Abs(y_expect - y_actual);

                (Flow flow, Parameters parameters) = Flow.Optimize(err);

                flow.Execute();

                gxval_false = x.GradTensor.State;

                CollectionAssert.AreEqual(gxval_true, gxval_false);
            }
            catch (Exception e) {
                Assert.Fail("axis:0 keepdims:false  " + e.Message);
            }

            try {
                float[] yval = (new float[channels * height * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
                Tensor ytensor = new Tensor(Shape.Map2D(channels, 1, height, batch), yval);

                ParameterField x = xtensor;
                VariableField y_actual = ytensor;

                Field y_expect = Variance(x, new int[] { Axis.Map2D.Width }, keepdims: true);
                Field err = Abs(y_expect - y_actual);

                (Flow flow, Parameters parameters) = Flow.Optimize(err);

                flow.Execute();

                gxval_true = x.GradTensor.State;
            }
            catch (Exception e) {
                Assert.Fail("axis:1 keepdims:true  " + e.Message);
            }

            try {
                float[] yval = (new float[channels * height * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
                Tensor ytensor = new Tensor(Shape.Map1D(channels, height, batch), yval);

                ParameterField x = xtensor;
                VariableField y_actual = ytensor;

                Field y_expect = Variance(x, new int[] { Axis.Map2D.Width }, keepdims: false);
                Field err = Abs(y_expect - y_actual);

                (Flow flow, Parameters parameters) = Flow.Optimize(err);

                flow.Execute();

                gxval_false = x.GradTensor.State;

                CollectionAssert.AreEqual(gxval_true, gxval_false);
            }
            catch (Exception e) {
                Assert.Fail("axis:1 keepdims:false  " + e.Message);
            }

            try {
                float[] yval = (new float[channels * width * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
                Tensor ytensor = new Tensor(Shape.Map2D(channels, width, 1, batch), yval);

                ParameterField x = xtensor;
                VariableField y_actual = ytensor;

                Field y_expect = Variance(x, new int[] { Axis.Map2D.Height }, keepdims: true);
                Field err = Abs(y_expect - y_actual);

                (Flow flow, Parameters parameters) = Flow.Optimize(err);

                flow.Execute();

                gxval_true = x.GradTensor.State;
            }
            catch (Exception e) {
                Assert.Fail("axis:2 keepdims:true  " + e.Message);
            }

            try {
                float[] yval = (new float[channels * width * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
                Tensor ytensor = new Tensor(Shape.Map1D(channels, width, batch), yval);

                ParameterField x = xtensor;
                VariableField y_actual = ytensor;

                Field y_expect = Variance(x, new int[] { Axis.Map2D.Height }, keepdims: false);
                Field err = Abs(y_expect - y_actual);

                (Flow flow, Parameters parameters) = Flow.Optimize(err);

                flow.Execute();

                gxval_false = x.GradTensor.State;

                CollectionAssert.AreEqual(gxval_true, gxval_false);
            }
            catch (Exception e) {
                Assert.Fail("axis:2 keepdims:false  " + e.Message);
            }

            try {
                float[] yval = (new float[channels * width * height]).Select((_, idx) => idx * 1e-3f).ToArray();
                Tensor ytensor = new Tensor(Shape.Map2D(channels, width, height, 1), yval);

                ParameterField x = xtensor;
                VariableField y_actual = ytensor;

                Field y_expect = Variance(x, new int[] { Axis.Map2D.Batch }, keepdims: true);
                Field err = Abs(y_expect - y_actual);

                (Flow flow, Parameters parameters) = Flow.Optimize(err);

                flow.Execute();

                gxval_true = x.GradTensor.State;
            }
            catch (Exception e) {
                Assert.Fail("axis:3 keepdims:true  " + e.Message);
            }

            try {
                float[] yval = (new float[channels * width * height]).Select((_, idx) => idx * 1e-3f).ToArray();
                Tensor ytensor = new Tensor(Shape.Map1D(channels, width, height), yval);

                ParameterField x = xtensor;
                VariableField y_actual = ytensor;

                Field y_expect = Variance(x, new int[] { Axis.Map2D.Batch }, keepdims: false);
                Field err = Abs(y_expect - y_actual);

                (Flow flow, Parameters parameters) = Flow.Optimize(err);

                flow.Execute();

                gxval_false = x.GradTensor.State;

                CollectionAssert.AreEqual(gxval_true, gxval_false);
            }
            catch (Exception e) {
                Assert.Fail("axis:3 keepdims:false  " + e.Message);
            }

            try {
                float[] yval = (new float[height * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
                Tensor ytensor = new Tensor(Shape.Map2D(1, 1, height, batch), yval);

                ParameterField x = xtensor;
                VariableField y_actual = ytensor;

                Field y_expect = Variance(x, new int[] { Axis.Map2D.Channels, Axis.Map2D.Width }, keepdims: true);
                Field err = Abs(y_expect - y_actual);

                (Flow flow, Parameters parameters) = Flow.Optimize(err);

                flow.Execute();

                gxval_true = x.GradTensor.State;
            }
            catch (Exception e) {
                Assert.Fail("axis:0,1 keepdims:true  " + e.Message);
            }

            try {
                float[] yval = (new float[height * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
                Tensor ytensor = new Tensor(Shape.Map0D(height, batch), yval);

                ParameterField x = xtensor;
                VariableField y_actual = ytensor;

                Field y_expect = Variance(x, new int[] { Axis.Map2D.Channels, Axis.Map2D.Width }, keepdims: false);
                Field err = Abs(y_expect - y_actual);

                (Flow flow, Parameters parameters) = Flow.Optimize(err);

                flow.Execute();

                gxval_false = x.GradTensor.State;

                CollectionAssert.AreEqual(gxval_true, gxval_false);
            }
            catch (Exception e) {
                Assert.Fail("axis:0,1 keepdims:false  " + e.Message);
            }

            try {
                float[] yval = (new float[width * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
                Tensor ytensor = new Tensor(Shape.Map2D(1, width, 1, batch), yval);

                ParameterField x = xtensor;
                VariableField y_actual = ytensor;

                Field y_expect = Variance(x, new int[] { Axis.Map2D.Channels, Axis.Map2D.Height }, keepdims: true);
                Field err = Abs(y_expect - y_actual);

                (Flow flow, Parameters parameters) = Flow.Optimize(err);

                flow.Execute();

                gxval_true = x.GradTensor.State;
            }
            catch (Exception e) {
                Assert.Fail("axis:0,2 keepdims:true  " + e.Message);
            }

            try {
                float[] yval = (new float[width * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
                Tensor ytensor = new Tensor(Shape.Map0D(width, batch), yval);

                ParameterField x = xtensor;
                VariableField y_actual = ytensor;

                Field y_expect = Variance(x, new int[] { Axis.Map2D.Channels, Axis.Map2D.Height }, keepdims: false);
                Field err = Abs(y_expect - y_actual);

                (Flow flow, Parameters parameters) = Flow.Optimize(err);

                flow.Execute();

                gxval_false = x.GradTensor.State;

                CollectionAssert.AreEqual(gxval_true, gxval_false);
            }
            catch (Exception e) {
                Assert.Fail("axis:0,2 keepdims:false  " + e.Message);
            }

            try {
                float[] yval = (new float[width * height]).Select((_, idx) => idx * 1e-3f).ToArray();
                Tensor ytensor = new Tensor(Shape.Map2D(1, width, height, 1), yval);

                ParameterField x = xtensor;
                VariableField y_actual = ytensor;

                Field y_expect = Variance(x, new int[] { Axis.Map2D.Channels, Axis.Map2D.Batch }, keepdims: true);
                Field err = Abs(y_expect - y_actual);

                (Flow flow, Parameters parameters) = Flow.Optimize(err);

                flow.Execute();

                gxval_true = x.GradTensor.State;
            }
            catch (Exception e) {
                Assert.Fail("axis:0,3 keepdims:true  " + e.Message);
            }

            try {
                float[] yval = (new float[width * height]).Select((_, idx) => idx * 1e-3f).ToArray();
                Tensor ytensor = new Tensor(Shape.Map0D(width, height), yval);

                ParameterField x = xtensor;
                VariableField y_actual = ytensor;

                Field y_expect = Variance(x, new int[] { Axis.Map2D.Channels, Axis.Map2D.Batch }, keepdims: false);
                Field err = Abs(y_expect - y_actual);

                (Flow flow, Parameters parameters) = Flow.Optimize(err);

                flow.Execute();

                gxval_false = x.GradTensor.State;

                CollectionAssert.AreEqual(gxval_true, gxval_false);
            }
            catch (Exception e) {
                Assert.Fail("axis:0,3 keepdims:false  " + e.Message);
            }

            try {
                float[] yval = (new float[channels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
                Tensor ytensor = new Tensor(Shape.Map2D(channels, 1, 1, batch), yval);

                ParameterField x = xtensor;
                VariableField y_actual = ytensor;

                Field y_expect = Variance(x, new int[] { Axis.Map2D.Width, Axis.Map2D.Height }, keepdims: true);
                Field err = Abs(y_expect - y_actual);

                (Flow flow, Parameters parameters) = Flow.Optimize(err);

                flow.Execute();

                gxval_true = x.GradTensor.State;
            }
            catch (Exception e) {
                Assert.Fail("axis:1,2 keepdims:true  " + e.Message);
            }

            try {
                float[] yval = (new float[channels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
                Tensor ytensor = new Tensor(Shape.Map0D(channels, batch), yval);

                ParameterField x = xtensor;
                VariableField y_actual = ytensor;

                Field y_expect = Variance(x, new int[] { Axis.Map2D.Width, Axis.Map2D.Height }, keepdims: false);
                Field err = Abs(y_expect - y_actual);

                (Flow flow, Parameters parameters) = Flow.Optimize(err);

                flow.Execute();

                gxval_false = x.GradTensor.State;

                CollectionAssert.AreEqual(gxval_true, gxval_false);
            }
            catch (Exception e) {
                Assert.Fail("axis:1,2 keepdims:false  " + e.Message);
            }

            try {
                float[] yval = (new float[channels * width]).Select((_, idx) => idx * 1e-3f).ToArray();
                Tensor ytensor = new Tensor(Shape.Map2D(channels, width, 1, 1), yval);

                ParameterField x = xtensor;
                VariableField y_actual = ytensor;

                Field y_expect = Variance(x, new int[] { Axis.Map2D.Height, Axis.Map2D.Batch }, keepdims: true);
                Field err = Abs(y_expect - y_actual);

                (Flow flow, Parameters parameters) = Flow.Optimize(err);

                flow.Execute();

                gxval_true = x.GradTensor.State;
            }
            catch (Exception e) {
                Assert.Fail("axis:2,3 keepdims:true  " + e.Message);
            }

            try {
                float[] yval = (new float[channels * width]).Select((_, idx) => idx * 1e-3f).ToArray();
                Tensor ytensor = new Tensor(Shape.Map0D(channels, width), yval);

                ParameterField x = xtensor;
                VariableField y_actual = ytensor;

                Field y_expect = Variance(x, new int[] { Axis.Map2D.Height, Axis.Map2D.Batch }, keepdims: false);
                Field err = Abs(y_expect - y_actual);

                (Flow flow, Parameters parameters) = Flow.Optimize(err);

                flow.Execute();

                gxval_false = x.GradTensor.State;

                CollectionAssert.AreEqual(gxval_true, gxval_false);
            }
            catch (Exception e) {
                Assert.Fail("axis:2,3 keepdims:false  " + e.Message);
            }

            try {
                float[] yval = (new float[batch]).Select((_, idx) => idx * 1e-3f).ToArray();
                Tensor ytensor = new Tensor(Shape.Map2D(1, 1, 1, batch), yval);

                ParameterField x = xtensor;
                VariableField y_actual = ytensor;

                Field y_expect = Variance(x, new int[] { Axis.Map2D.Channels, Axis.Map2D.Width, Axis.Map2D.Height }, keepdims: true);
                Field err = Abs(y_expect - y_actual);

                (Flow flow, Parameters parameters) = Flow.Optimize(err);

                flow.Execute();

                gxval_true = x.GradTensor.State;
            }
            catch (Exception e) {
                Assert.Fail("axis:0,1,2 keepdims:true  " + e.Message);
            }

            try {
                float[] yval = (new float[batch]).Select((_, idx) => idx * 1e-3f).ToArray();
                Tensor ytensor = new Tensor(Shape.Vector(batch), yval);

                ParameterField x = xtensor;
                VariableField y_actual = ytensor;

                Field y_expect = Variance(x, new int[] { Axis.Map2D.Channels, Axis.Map2D.Width, Axis.Map2D.Height }, keepdims: false);
                Field err = Abs(y_expect - y_actual);

                (Flow flow, Parameters parameters) = Flow.Optimize(err);

                flow.Execute();

                gxval_false = x.GradTensor.State;

                CollectionAssert.AreEqual(gxval_true, gxval_false);
            }
            catch (Exception e) {
                Assert.Fail("axis:0,1,2 keepdims:false  " + e.Message);
            }

            try {
                float[] yval = (new float[height]).Select((_, idx) => idx * 1e-3f).ToArray();
                Tensor ytensor = new Tensor(Shape.Map2D(1, 1, height, 1), yval);

                ParameterField x = xtensor;
                VariableField y_actual = ytensor;

                Field y_expect = Variance(x, new int[] { Axis.Map2D.Channels, Axis.Map2D.Width, Axis.Map2D.Batch }, keepdims: true);
                Field err = Abs(y_expect - y_actual);

                (Flow flow, Parameters parameters) = Flow.Optimize(err);

                flow.Execute();

                gxval_true = x.GradTensor.State;
            }
            catch (Exception e) {
                Assert.Fail("axis:0,1,3 keepdims:true  " + e.Message);
            }

            try {
                float[] yval = (new float[height]).Select((_, idx) => idx * 1e-3f).ToArray();
                Tensor ytensor = new Tensor(Shape.Vector(height), yval);

                ParameterField x = xtensor;
                VariableField y_actual = ytensor;

                Field y_expect = Variance(x, new int[] { Axis.Map2D.Channels, Axis.Map2D.Width, Axis.Map2D.Batch }, keepdims: false);
                Field err = Abs(y_expect - y_actual);

                (Flow flow, Parameters parameters) = Flow.Optimize(err);

                flow.Execute();

                gxval_false = x.GradTensor.State;

                CollectionAssert.AreEqual(gxval_true, gxval_false);
            }
            catch (Exception e) {
                Assert.Fail("axis:0,1,3 keepdims:false  " + e.Message);
            }

            try {
                float[] yval = (new float[width]).Select((_, idx) => idx * 1e-3f).ToArray();
                Tensor ytensor = new Tensor(Shape.Map2D(1, width, 1, 1), yval);

                ParameterField x = xtensor;
                VariableField y_actual = ytensor;

                Field y_expect = Variance(x, new int[] { Axis.Map2D.Channels, Axis.Map2D.Height, Axis.Map2D.Batch }, keepdims: true);
                Field err = Abs(y_expect - y_actual);

                (Flow flow, Parameters parameters) = Flow.Optimize(err);

                flow.Execute();

                gxval_true = x.GradTensor.State;
            }
            catch (Exception e) {
                Assert.Fail("axis:0,2,3 keepdims:true  " + e.Message);
            }

            try {
                float[] yval = (new float[width]).Select((_, idx) => idx * 1e-3f).ToArray();
                Tensor ytensor = new Tensor(Shape.Vector(width), yval);

                ParameterField x = xtensor;
                VariableField y_actual = ytensor;

                Field y_expect = Variance(x, new int[] { Axis.Map2D.Channels, Axis.Map2D.Height, Axis.Map2D.Batch }, keepdims: false);
                Field err = Abs(y_expect - y_actual);

                (Flow flow, Parameters parameters) = Flow.Optimize(err);

                flow.Execute();

                gxval_false = x.GradTensor.State;

                CollectionAssert.AreEqual(gxval_true, gxval_false);
            }
            catch (Exception e) {
                Assert.Fail("axis:0,2,3 keepdims:false  " + e.Message);
            }

            try {
                float[] yval = (new float[channels]).Select((_, idx) => idx * 1e-3f).ToArray();
                Tensor ytensor = new Tensor(Shape.Map2D(channels, 1, 1, 1), yval);

                ParameterField x = xtensor;
                VariableField y_actual = ytensor;

                Field y_expect = Variance(x, new int[] { Axis.Map2D.Width, Axis.Map2D.Height, Axis.Map2D.Batch }, keepdims: true);
                Field err = Abs(y_expect - y_actual);

                (Flow flow, Parameters parameters) = Flow.Optimize(err);

                flow.Execute();

                gxval_true = x.GradTensor.State;
            }
            catch (Exception e) {
                Assert.Fail("axis:1,2,3 keepdims:true  " + e.Message);
            }

            try {
                float[] yval = (new float[channels]).Select((_, idx) => idx * 1e-3f).ToArray();
                Tensor ytensor = new Tensor(Shape.Vector(channels), yval);

                ParameterField x = xtensor;
                VariableField y_actual = ytensor;

                Field y_expect = Variance(x, new int[] { Axis.Map2D.Width, Axis.Map2D.Height, Axis.Map2D.Batch }, keepdims: false);
                Field err = Abs(y_expect - y_actual);

                (Flow flow, Parameters parameters) = Flow.Optimize(err);

                flow.Execute();

                gxval_false = x.GradTensor.State;

                CollectionAssert.AreEqual(gxval_true, gxval_false);
            }
            catch (Exception e) {
                Assert.Fail("axis:1,2,3 keepdims:false  " + e.Message);
            }

            try {
                float[] yval = (new float[1]).Select((_, idx) => idx * 1e-3f).ToArray();
                Tensor ytensor = new Tensor(Shape.Map2D(1, 1, 1, 1), yval);

                ParameterField x = xtensor;
                VariableField y_actual = ytensor;

                Field y_expect = Variance(x, axes: null, keepdims: true);
                Field err = Abs(y_expect - y_actual);

                (Flow flow, Parameters parameters) = Flow.Optimize(err);

                flow.Execute();

                gxval_true = x.GradTensor.State;
            }
            catch (Exception e) {
                Assert.Fail("axis:all keepdims:true  " + e.Message);
            }

            try {
                float[] yval = (new float[1]).Select((_, idx) => idx * 1e-3f).ToArray();
                Tensor ytensor = new Tensor(Shape.Scalar(), yval);

                ParameterField x = xtensor;
                VariableField y_actual = ytensor;

                Field y_expect = Variance(x, axes: null, keepdims: false);
                Field err = Abs(y_expect - y_actual);

                (Flow flow, Parameters parameters) = Flow.Optimize(err);

                flow.Execute();

                gxval_false = x.GradTensor.State;

                CollectionAssert.AreEqual(gxval_true, gxval_false);
            }
            catch (Exception e) {
                Assert.Fail("axis:all keepdims:false  " + e.Message);
            }
        }

        [TestMethod]
        public void ReferenceTest() {
            int channels = 7, width = 3, height = 5, batch = 2;

            float[] xval = (new float[channels * width * height * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
            Tensor xtensor = new Tensor(Shape.Map2D(channels, width, height, batch), xval);

            float[] yval = (new float[width * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
            Tensor ytensor = new Tensor(Shape.Map0D(width, batch), yval);

            ParameterField x = xtensor;
            VariableField y_actual = ytensor;

            Field y_expect = Variance(x, new int[] { Axis.Map2D.Channels, Axis.Map2D.Height }, keepdims: false);
            Field err = Abs(y_expect - y_actual);

            (Flow flow, Parameters parameters) = Flow.Optimize(err);

            flow.Execute();

            float[] gx_actual = x.GradTensor.State;

            AssertError.Tolerance(gx_expect, gx_actual, 1e-9f, 1e-5f, $"not equal gx");
        }

        float[] gx_expect = {
            -2.2783e-06f, -2.2277e-06f, -2.1770e-06f, -2.1264e-06f, -2.0758e-06f,
            -2.0251e-06f, -1.9745e-06f, 2.9314e-07f, 2.8663e-07f, 2.8011e-07f,
            2.7360e-07f, 2.6709e-07f, 2.6057e-07f, 2.5406e-07f, 2.8646e-06f,
            2.8009e-06f, 2.7373e-06f, 2.6736e-06f, 2.6099e-06f, 2.5463e-06f,
            2.4826e-06f, -1.2151e-06f, -1.1645e-06f, -1.1138e-06f, -1.0632e-06f,
            -1.0126e-06f, -9.6194e-07f, -9.1131e-07f, 1.5634e-07f, 1.4983e-07f,
            1.4331e-07f, 1.3680e-07f, 1.3029e-07f, 1.2377e-07f, 1.1726e-07f,
            1.5278e-06f, 1.4641e-06f, 1.4005e-06f, 1.3368e-06f, 1.2731e-06f,
            1.2095e-06f, 1.1458e-06f, -1.5189e-07f, -1.0126e-07f, -5.0629e-08f,
            0.0000e+00f, 5.0629e-08f, 1.0126e-07f, 1.5189e-07f, 1.9543e-08f,
            1.3029e-08f, 6.5143e-09f, -1.0588e-22f, -6.5143e-09f, -1.3029e-08f,
            -1.9543e-08f, 1.9097e-07f, 1.2731e-07f, 6.3657e-08f, 0.0000e+00f,
            -6.3657e-08f, -1.2731e-07f, -1.9097e-07f, 9.1131e-07f, 9.6194e-07f,
            1.0126e-06f, 1.0632e-06f, 1.1138e-06f, 1.1645e-06f, 1.2151e-06f,
            -1.1726e-07f, -1.2377e-07f, -1.3029e-07f, -1.3680e-07f, -1.4331e-07f,
            -1.4983e-07f, -1.5634e-07f, -1.1458e-06f, -1.2095e-06f, -1.2731e-06f,
            -1.3368e-06f, -1.4005e-06f, -1.4641e-06f, -1.5278e-06f, 1.9745e-06f,
            2.0251e-06f, 2.0758e-06f, 2.1264e-06f, 2.1770e-06f, 2.2277e-06f,
            2.2783e-06f, -2.5406e-07f, -2.6057e-07f, -2.6709e-07f, -2.7360e-07f,
            -2.8011e-07f, -2.8663e-07f, -2.9314e-07f, -2.4826e-06f, -2.5463e-06f,
            -2.6099e-06f, -2.6736e-06f, -2.7373e-06f, -2.8009e-06f, -2.8646e-06f,
            5.4360e-06f, 5.3152e-06f, 5.1944e-06f, 5.0736e-06f, 4.9528e-06f,
            4.8320e-06f, 4.7112e-06f, 8.0074e-06f, 7.8295e-06f, 7.6515e-06f,
            7.4736e-06f, 7.2957e-06f, 7.1177e-06f, 6.9398e-06f, 1.0579e-05f,
            1.0344e-05f, 1.0109e-05f, 9.8736e-06f, 9.6385e-06f, 9.4034e-06f,
            9.1683e-06f, 2.8992e-06f, 2.7784e-06f, 2.6576e-06f, 2.5368e-06f,
            2.4160e-06f, 2.2952e-06f, 2.1744e-06f, 4.2706e-06f, 4.0927e-06f,
            3.9147e-06f, 3.7368e-06f, 3.5589e-06f, 3.3809e-06f, 3.2030e-06f,
            5.6421e-06f, 5.4070e-06f, 5.1719e-06f, 4.9368e-06f, 4.7017e-06f,
            4.4666e-06f, 4.2315e-06f, 3.6240e-07f, 2.4160e-07f, 1.2080e-07f,
            0.0000e+00f, -1.2080e-07f, -2.4160e-07f, -3.6240e-07f, 5.3383e-07f,
            3.5589e-07f, 1.7794e-07f, -3.3881e-21f, -1.7794e-07f, -3.5589e-07f,
            -5.3383e-07f, 7.0526e-07f, 4.7017e-07f, 2.3509e-07f, 0.0000e+00f,
            -2.3509e-07f, -4.7017e-07f, -7.0526e-07f, -2.1744e-06f, -2.2952e-06f,
            -2.4160e-06f, -2.5368e-06f, -2.6576e-06f, -2.7784e-06f, -2.8992e-06f,
            -3.2030e-06f, -3.3809e-06f, -3.5589e-06f, -3.7368e-06f, -3.9147e-06f,
            -4.0927e-06f, -4.2706e-06f, -4.2315e-06f, -4.4666e-06f, -4.7017e-06f,
            -4.9368e-06f, -5.1719e-06f, -5.4070e-06f, -5.6421e-06f, -4.7112e-06f,
            -4.8320e-06f, -4.9528e-06f, -5.0736e-06f, -5.1944e-06f, -5.3152e-06f,
            -5.4360e-06f, -6.9398e-06f, -7.1177e-06f, -7.2957e-06f, -7.4736e-06f,
            -7.6515e-06f, -7.8295e-06f, -8.0074e-06f, -9.1683e-06f, -9.4034e-06f,
            -9.6385e-06f, -9.8736e-06f, -1.0109e-05f, -1.0344e-05f, -1.0579e-05f,
        };
    }
}
