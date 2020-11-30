using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;
using TensorShader;
using static TensorShader.Field;

namespace TensorShaderTest.Links.Aggregation {
    [TestClass]
    public class SumTest {
        [TestMethod]
        public void ExecuteTest() {
            int channels = 7, width = 3, height = 5, batch = 2;

            float[] xval = (new float[channels * width * height * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
            Tensor xtensor = (Shape.Map2D(channels, width, height, batch), xval);

            float[] gxval_true = null, gxval_false = null;

            try {
                float[] yval = (new float[channels * width * height * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
                Tensor ytensor = (Shape.Map2D(channels, width, height, batch), yval);

                ParameterField x = xtensor;
                VariableField y_actual = ytensor;

                Field y_expect = Sum(x, new int[] { }, keepdims: true);
                Field err = Abs(y_expect - y_actual);

                (Flow flow, Parameters parameters) = Flow.Optimize(err);

                flow.Execute();

                gxval_true = x.GradState.Value;
            }
            catch (Exception e) {
                Assert.Fail("axis:none keepdims:true  " + e.Message);
            }

            try {
                float[] yval = (new float[channels * width * height * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
                Tensor ytensor = (Shape.Map2D(channels, width, height, batch), yval);

                ParameterField x = xtensor;
                VariableField y_actual = ytensor;

                Field y_expect = Sum(x, new int[] { }, keepdims: false);
                Field err = Abs(y_expect - y_actual);

                (Flow flow, Parameters parameters) = Flow.Optimize(err);

                flow.Execute();

                gxval_false = x.GradState.Value;

                CollectionAssert.AreEqual(gxval_true, gxval_false);
            }
            catch (Exception e) {
                Assert.Fail("axis:none keepdims:false  " + e.Message);
            }

            try {
                float[] yval = (new float[width * height * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
                Tensor ytensor = (Shape.Map2D(1, width, height, batch), yval);

                ParameterField x = xtensor;
                VariableField y_actual = ytensor;

                Field y_expect = Sum(x, new int[] { Axis.Map2D.Channels }, keepdims: true);
                Field err = Abs(y_expect - y_actual);

                (Flow flow, Parameters parameters) = Flow.Optimize(err);

                flow.Execute();

                gxval_true = x.GradState.Value;
            }
            catch (Exception e) {
                Assert.Fail("axis:0 keepdims:true  " + e.Message);
            }

            try {
                float[] yval = (new float[width * height * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
                Tensor ytensor = (Shape.Map1D(width, height, batch), yval);

                ParameterField x = xtensor;
                VariableField y_actual = ytensor;

                Field y_expect = Sum(x, new int[] { Axis.Map2D.Channels }, keepdims: false);
                Field err = Abs(y_expect - y_actual);

                (Flow flow, Parameters parameters) = Flow.Optimize(err);

                flow.Execute();

                gxval_false = x.GradState.Value;

                CollectionAssert.AreEqual(gxval_true, gxval_false);
            }
            catch (Exception e) {
                Assert.Fail("axis:0 keepdims:false  " + e.Message);
            }

            try {
                float[] yval = (new float[channels * height * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
                Tensor ytensor = (Shape.Map2D(channels, 1, height, batch), yval);

                ParameterField x = xtensor;
                VariableField y_actual = ytensor;

                Field y_expect = Sum(x, new int[] { Axis.Map2D.Width }, keepdims: true);
                Field err = Abs(y_expect - y_actual);

                (Flow flow, Parameters parameters) = Flow.Optimize(err);

                flow.Execute();

                gxval_true = x.GradState.Value;
            }
            catch (Exception e) {
                Assert.Fail("axis:1 keepdims:true  " + e.Message);
            }

            try {
                float[] yval = (new float[channels * height * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
                Tensor ytensor = (Shape.Map1D(channels, height, batch), yval);

                ParameterField x = xtensor;
                VariableField y_actual = ytensor;

                Field y_expect = Sum(x, new int[] { Axis.Map2D.Width }, keepdims: false);
                Field err = Abs(y_expect - y_actual);

                (Flow flow, Parameters parameters) = Flow.Optimize(err);

                flow.Execute();

                gxval_false = x.GradState.Value;

                CollectionAssert.AreEqual(gxval_true, gxval_false);
            }
            catch (Exception e) {
                Assert.Fail("axis:1 keepdims:false  " + e.Message);
            }

            try {
                float[] yval = (new float[channels * width * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
                Tensor ytensor = (Shape.Map2D(channels, width, 1, batch), yval);

                ParameterField x = xtensor;
                VariableField y_actual = ytensor;

                Field y_expect = Sum(x, new int[] { Axis.Map2D.Height }, keepdims: true);
                Field err = Abs(y_expect - y_actual);

                (Flow flow, Parameters parameters) = Flow.Optimize(err);

                flow.Execute();

                gxval_true = x.GradState.Value;
            }
            catch (Exception e) {
                Assert.Fail("axis:2 keepdims:true  " + e.Message);
            }

            try {
                float[] yval = (new float[channels * width * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
                Tensor ytensor = (Shape.Map1D(channels, width, batch), yval);

                ParameterField x = xtensor;
                VariableField y_actual = ytensor;

                Field y_expect = Sum(x, new int[] { Axis.Map2D.Height }, keepdims: false);
                Field err = Abs(y_expect - y_actual);

                (Flow flow, Parameters parameters) = Flow.Optimize(err);

                flow.Execute();

                gxval_false = x.GradState.Value;

                CollectionAssert.AreEqual(gxval_true, gxval_false);
            }
            catch (Exception e) {
                Assert.Fail("axis:2 keepdims:false  " + e.Message);
            }

            try {
                float[] yval = (new float[channels * width * height]).Select((_, idx) => idx * 1e-3f).ToArray();
                Tensor ytensor = (Shape.Map2D(channels, width, height, 1), yval);

                ParameterField x = xtensor;
                VariableField y_actual = ytensor;

                Field y_expect = Sum(x, new int[] { Axis.Map2D.Batch }, keepdims: true);
                Field err = Abs(y_expect - y_actual);

                (Flow flow, Parameters parameters) = Flow.Optimize(err);

                flow.Execute();

                gxval_true = x.GradState.Value;
            }
            catch (Exception e) {
                Assert.Fail("axis:3 keepdims:true  " + e.Message);
            }

            try {
                float[] yval = (new float[channels * width * height]).Select((_, idx) => idx * 1e-3f).ToArray();
                Tensor ytensor = (Shape.Map1D(channels, width, height), yval);

                ParameterField x = xtensor;
                VariableField y_actual = ytensor;

                Field y_expect = Sum(x, new int[] { Axis.Map2D.Batch }, keepdims: false);
                Field err = Abs(y_expect - y_actual);

                (Flow flow, Parameters parameters) = Flow.Optimize(err);

                flow.Execute();

                gxval_false = x.GradState.Value;

                CollectionAssert.AreEqual(gxval_true, gxval_false);
            }
            catch (Exception e) {
                Assert.Fail("axis:3 keepdims:false  " + e.Message);
            }

            try {
                float[] yval = (new float[height * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
                Tensor ytensor = (Shape.Map2D(1, 1, height, batch), yval);

                ParameterField x = xtensor;
                VariableField y_actual = ytensor;

                Field y_expect = Sum(x, new int[] { Axis.Map2D.Channels, Axis.Map2D.Width }, keepdims: true);
                Field err = Abs(y_expect - y_actual);

                (Flow flow, Parameters parameters) = Flow.Optimize(err);

                flow.Execute();

                gxval_true = x.GradState.Value;
            }
            catch (Exception e) {
                Assert.Fail("axis:0,1 keepdims:true  " + e.Message);
            }

            try {
                float[] yval = (new float[height * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
                Tensor ytensor = (Shape.Map0D(height, batch), yval);

                ParameterField x = xtensor;
                VariableField y_actual = ytensor;

                Field y_expect = Sum(x, new int[] { Axis.Map2D.Channels, Axis.Map2D.Width }, keepdims: false);
                Field err = Abs(y_expect - y_actual);

                (Flow flow, Parameters parameters) = Flow.Optimize(err);

                flow.Execute();

                gxval_false = x.GradState.Value;

                CollectionAssert.AreEqual(gxval_true, gxval_false);
            }
            catch (Exception e) {
                Assert.Fail("axis:0,1 keepdims:false  " + e.Message);
            }

            try {
                float[] yval = (new float[width * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
                Tensor ytensor = (Shape.Map2D(1, width, 1, batch), yval);

                ParameterField x = xtensor;
                VariableField y_actual = ytensor;

                Field y_expect = Sum(x, new int[] { Axis.Map2D.Channels, Axis.Map2D.Height }, keepdims: true);
                Field err = Abs(y_expect - y_actual);

                (Flow flow, Parameters parameters) = Flow.Optimize(err);

                flow.Execute();

                gxval_true = x.GradState.Value;
            }
            catch (Exception e) {
                Assert.Fail("axis:0,2 keepdims:true  " + e.Message);
            }

            try {
                float[] yval = (new float[width * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
                Tensor ytensor = (Shape.Map0D(width, batch), yval);

                ParameterField x = xtensor;
                VariableField y_actual = ytensor;

                Field y_expect = Sum(x, new int[] { Axis.Map2D.Channels, Axis.Map2D.Height }, keepdims: false);
                Field err = Abs(y_expect - y_actual);

                (Flow flow, Parameters parameters) = Flow.Optimize(err);

                flow.Execute();

                gxval_false = x.GradState.Value;

                CollectionAssert.AreEqual(gxval_true, gxval_false);
            }
            catch (Exception e) {
                Assert.Fail("axis:0,2 keepdims:false  " + e.Message);
            }

            try {
                float[] yval = (new float[width * height]).Select((_, idx) => idx * 1e-3f).ToArray();
                Tensor ytensor = (Shape.Map2D(1, width, height, 1), yval);

                ParameterField x = xtensor;
                VariableField y_actual = ytensor;

                Field y_expect = Sum(x, new int[] { Axis.Map2D.Channels, Axis.Map2D.Batch }, keepdims: true);
                Field err = Abs(y_expect - y_actual);

                (Flow flow, Parameters parameters) = Flow.Optimize(err);

                flow.Execute();

                gxval_true = x.GradState.Value;
            }
            catch (Exception e) {
                Assert.Fail("axis:0,3 keepdims:true  " + e.Message);
            }

            try {
                float[] yval = (new float[width * height]).Select((_, idx) => idx * 1e-3f).ToArray();
                Tensor ytensor = (Shape.Map0D(width, height), yval);

                ParameterField x = xtensor;
                VariableField y_actual = ytensor;

                Field y_expect = Sum(x, new int[] { Axis.Map2D.Channels, Axis.Map2D.Batch }, keepdims: false);
                Field err = Abs(y_expect - y_actual);

                (Flow flow, Parameters parameters) = Flow.Optimize(err);

                flow.Execute();

                gxval_false = x.GradState.Value;

                CollectionAssert.AreEqual(gxval_true, gxval_false);
            }
            catch (Exception e) {
                Assert.Fail("axis:0,3 keepdims:false  " + e.Message);
            }

            try {
                float[] yval = (new float[channels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
                Tensor ytensor = (Shape.Map2D(channels, 1, 1, batch), yval);

                ParameterField x = xtensor;
                VariableField y_actual = ytensor;

                Field y_expect = Sum(x, new int[] { Axis.Map2D.Width, Axis.Map2D.Height }, keepdims: true);
                Field err = Abs(y_expect - y_actual);

                (Flow flow, Parameters parameters) = Flow.Optimize(err);

                flow.Execute();

                gxval_true = x.GradState.Value;
            }
            catch (Exception e) {
                Assert.Fail("axis:1,2 keepdims:true  " + e.Message);
            }

            try {
                float[] yval = (new float[channels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
                Tensor ytensor = (Shape.Map0D(channels, batch), yval);

                ParameterField x = xtensor;
                VariableField y_actual = ytensor;

                Field y_expect = Sum(x, new int[] { Axis.Map2D.Width, Axis.Map2D.Height }, keepdims: false);
                Field err = Abs(y_expect - y_actual);

                (Flow flow, Parameters parameters) = Flow.Optimize(err);

                flow.Execute();

                gxval_false = x.GradState.Value;

                CollectionAssert.AreEqual(gxval_true, gxval_false);
            }
            catch (Exception e) {
                Assert.Fail("axis:1,2 keepdims:false  " + e.Message);
            }

            try {
                float[] yval = (new float[channels * width]).Select((_, idx) => idx * 1e-3f).ToArray();
                Tensor ytensor = (Shape.Map2D(channels, width, 1, 1), yval);

                ParameterField x = xtensor;
                VariableField y_actual = ytensor;

                Field y_expect = Sum(x, new int[] { Axis.Map2D.Height, Axis.Map2D.Batch }, keepdims: true);
                Field err = Abs(y_expect - y_actual);

                (Flow flow, Parameters parameters) = Flow.Optimize(err);

                flow.Execute();

                gxval_true = x.GradState.Value;
            }
            catch (Exception e) {
                Assert.Fail("axis:2,3 keepdims:true  " + e.Message);
            }

            try {
                float[] yval = (new float[channels * width]).Select((_, idx) => idx * 1e-3f).ToArray();
                Tensor ytensor = (Shape.Map0D(channels, width), yval);

                ParameterField x = xtensor;
                VariableField y_actual = ytensor;

                Field y_expect = Sum(x, new int[] { Axis.Map2D.Height, Axis.Map2D.Batch }, keepdims: false);
                Field err = Abs(y_expect - y_actual);

                (Flow flow, Parameters parameters) = Flow.Optimize(err);

                flow.Execute();

                gxval_false = x.GradState.Value;

                CollectionAssert.AreEqual(gxval_true, gxval_false);
            }
            catch (Exception e) {
                Assert.Fail("axis:2,3 keepdims:false  " + e.Message);
            }

            try {
                float[] yval = (new float[batch]).Select((_, idx) => idx * 1e-3f).ToArray();
                Tensor ytensor = (Shape.Map2D(1, 1, 1, batch), yval);

                ParameterField x = xtensor;
                VariableField y_actual = ytensor;

                Field y_expect = Sum(x, new int[] { Axis.Map2D.Channels, Axis.Map2D.Width, Axis.Map2D.Height }, keepdims: true);
                Field err = Abs(y_expect - y_actual);

                (Flow flow, Parameters parameters) = Flow.Optimize(err);

                flow.Execute();

                gxval_true = x.GradState.Value;
            }
            catch (Exception e) {
                Assert.Fail("axis:0,1,2 keepdims:true  " + e.Message);
            }

            try {
                float[] yval = (new float[batch]).Select((_, idx) => idx * 1e-3f).ToArray();
                Tensor ytensor = (Shape.Vector(batch), yval);

                ParameterField x = xtensor;
                VariableField y_actual = ytensor;

                Field y_expect = Sum(x, new int[] { Axis.Map2D.Channels, Axis.Map2D.Width, Axis.Map2D.Height }, keepdims: false);
                Field err = Abs(y_expect - y_actual);

                (Flow flow, Parameters parameters) = Flow.Optimize(err);

                flow.Execute();

                gxval_false = x.GradState.Value;

                CollectionAssert.AreEqual(gxval_true, gxval_false);
            }
            catch (Exception e) {
                Assert.Fail("axis:0,1,2 keepdims:false  " + e.Message);
            }

            try {
                float[] yval = (new float[height]).Select((_, idx) => idx * 1e-3f).ToArray();
                Tensor ytensor = (Shape.Map2D(1, 1, height, 1), yval);

                ParameterField x = xtensor;
                VariableField y_actual = ytensor;

                Field y_expect = Sum(x, new int[] { Axis.Map2D.Channels, Axis.Map2D.Width, Axis.Map2D.Batch }, keepdims: true);
                Field err = Abs(y_expect - y_actual);

                (Flow flow, Parameters parameters) = Flow.Optimize(err);

                flow.Execute();

                gxval_true = x.GradState.Value;
            }
            catch (Exception e) {
                Assert.Fail("axis:0,1,3 keepdims:true  " + e.Message);
            }

            try {
                float[] yval = (new float[height]).Select((_, idx) => idx * 1e-3f).ToArray();
                Tensor ytensor = (Shape.Vector(height), yval);

                ParameterField x = xtensor;
                VariableField y_actual = ytensor;

                Field y_expect = Sum(x, new int[] { Axis.Map2D.Channels, Axis.Map2D.Width, Axis.Map2D.Batch }, keepdims: false);
                Field err = Abs(y_expect - y_actual);

                (Flow flow, Parameters parameters) = Flow.Optimize(err);

                flow.Execute();

                gxval_false = x.GradState.Value;

                CollectionAssert.AreEqual(gxval_true, gxval_false);
            }
            catch (Exception e) {
                Assert.Fail("axis:0,1,3 keepdims:false  " + e.Message);
            }

            try {
                float[] yval = (new float[width]).Select((_, idx) => idx * 1e-3f).ToArray();
                Tensor ytensor = (Shape.Map2D(1, width, 1, 1), yval);

                ParameterField x = xtensor;
                VariableField y_actual = ytensor;

                Field y_expect = Sum(x, new int[] { Axis.Map2D.Channels, Axis.Map2D.Height, Axis.Map2D.Batch }, keepdims: true);
                Field err = Abs(y_expect - y_actual);

                (Flow flow, Parameters parameters) = Flow.Optimize(err);

                flow.Execute();

                gxval_true = x.GradState.Value;
            }
            catch (Exception e) {
                Assert.Fail("axis:0,2,3 keepdims:true  " + e.Message);
            }

            try {
                float[] yval = (new float[width]).Select((_, idx) => idx * 1e-3f).ToArray();
                Tensor ytensor = (Shape.Vector(width), yval);

                ParameterField x = xtensor;
                VariableField y_actual = ytensor;

                Field y_expect = Sum(x, new int[] { Axis.Map2D.Channels, Axis.Map2D.Height, Axis.Map2D.Batch }, keepdims: false);
                Field err = Abs(y_expect - y_actual);

                (Flow flow, Parameters parameters) = Flow.Optimize(err);

                flow.Execute();

                gxval_false = x.GradState.Value;

                CollectionAssert.AreEqual(gxval_true, gxval_false);
            }
            catch (Exception e) {
                Assert.Fail("axis:0,2,3 keepdims:false  " + e.Message);
            }

            try {
                float[] yval = (new float[channels]).Select((_, idx) => idx * 1e-3f).ToArray();
                Tensor ytensor = (Shape.Map2D(channels, 1, 1, 1), yval);

                ParameterField x = xtensor;
                VariableField y_actual = ytensor;

                Field y_expect = Sum(x, new int[] { Axis.Map2D.Width, Axis.Map2D.Height, Axis.Map2D.Batch }, keepdims: true);
                Field err = Abs(y_expect - y_actual);

                (Flow flow, Parameters parameters) = Flow.Optimize(err);

                flow.Execute();

                gxval_true = x.GradState.Value;
            }
            catch (Exception e) {
                Assert.Fail("axis:1,2,3 keepdims:true  " + e.Message);
            }

            try {
                float[] yval = (new float[channels]).Select((_, idx) => idx * 1e-3f).ToArray();
                Tensor ytensor = (Shape.Vector(channels), yval);

                ParameterField x = xtensor;
                VariableField y_actual = ytensor;

                Field y_expect = Sum(x, new int[] { Axis.Map2D.Width, Axis.Map2D.Height, Axis.Map2D.Batch }, keepdims: false);
                Field err = Abs(y_expect - y_actual);

                (Flow flow, Parameters parameters) = Flow.Optimize(err);

                flow.Execute();

                gxval_false = x.GradState.Value;

                CollectionAssert.AreEqual(gxval_true, gxval_false);
            }
            catch (Exception e) {
                Assert.Fail("axis:1,2,3 keepdims:false  " + e.Message);
            }

            try {
                float[] yval = (new float[1]).Select((_, idx) => idx * 1e-3f).ToArray();
                Tensor ytensor = (Shape.Map2D(1, 1, 1, 1), yval);

                ParameterField x = xtensor;
                VariableField y_actual = ytensor;

                Field y_expect = Sum(x, axes: null, keepdims: true);
                Field err = Abs(y_expect - y_actual);

                (Flow flow, Parameters parameters) = Flow.Optimize(err);

                flow.Execute();

                gxval_true = x.GradState.Value;
            }
            catch (Exception e) {
                Assert.Fail("axis:all keepdims:true  " + e.Message);
            }

            try {
                float[] yval = (new float[1]).Select((_, idx) => idx * 1e-3f).ToArray();
                Tensor ytensor = (Shape.Scalar, yval);

                ParameterField x = xtensor;
                VariableField y_actual = ytensor;

                Field y_expect = Sum(x, axes: null, keepdims: false);
                Field err = Abs(y_expect - y_actual);

                (Flow flow, Parameters parameters) = Flow.Optimize(err);

                flow.Execute();

                gxval_false = x.GradState.Value;

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
            Tensor xtensor = (Shape.Map2D(channels, width, height, batch), xval);

            float[] yval = (new float[width * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
            Tensor ytensor = (Shape.Map0D(width, batch), yval);

            ParameterField x = xtensor;
            VariableField y_actual = ytensor;

            Field y_expect = Sum(x, new int[] { Axis.Map2D.Channels, Axis.Map2D.Height }, keepdims: false);
            Field err = Abs(y_expect - y_actual);

            (Flow flow, Parameters parameters) = Flow.Optimize(err);

            flow.Execute();

            float[] gx_actual = x.GradState.Value;

            AssertError.Tolerance(gx_expect, gx_actual, 1e-7f, 1e-5f, $"not equal gx");
        }

        readonly float[] gx_expect = {
            1.5750e+00f, 1.5750e+00f, 1.5750e+00f, 1.5750e+00f, 1.5750e+00f,
            1.5750e+00f, 1.5750e+00f, 1.8190e+00f, 1.8190e+00f, 1.8190e+00f,
            1.8190e+00f, 1.8190e+00f, 1.8190e+00f, 1.8190e+00f, 2.0630e+00f,
            2.0630e+00f, 2.0630e+00f, 2.0630e+00f, 2.0630e+00f, 2.0630e+00f,
            2.0630e+00f, 1.5750e+00f, 1.5750e+00f, 1.5750e+00f, 1.5750e+00f,
            1.5750e+00f, 1.5750e+00f, 1.5750e+00f, 1.8190e+00f, 1.8190e+00f,
            1.8190e+00f, 1.8190e+00f, 1.8190e+00f, 1.8190e+00f, 1.8190e+00f,
            2.0630e+00f, 2.0630e+00f, 2.0630e+00f, 2.0630e+00f, 2.0630e+00f,
            2.0630e+00f, 2.0630e+00f, 1.5750e+00f, 1.5750e+00f, 1.5750e+00f,
            1.5750e+00f, 1.5750e+00f, 1.5750e+00f, 1.5750e+00f, 1.8190e+00f,
            1.8190e+00f, 1.8190e+00f, 1.8190e+00f, 1.8190e+00f, 1.8190e+00f,
            1.8190e+00f, 2.0630e+00f, 2.0630e+00f, 2.0630e+00f, 2.0630e+00f,
            2.0630e+00f, 2.0630e+00f, 2.0630e+00f, 1.5750e+00f, 1.5750e+00f,
            1.5750e+00f, 1.5750e+00f, 1.5750e+00f, 1.5750e+00f, 1.5750e+00f,
            1.8190e+00f, 1.8190e+00f, 1.8190e+00f, 1.8190e+00f, 1.8190e+00f,
            1.8190e+00f, 1.8190e+00f, 2.0630e+00f, 2.0630e+00f, 2.0630e+00f,
            2.0630e+00f, 2.0630e+00f, 2.0630e+00f, 2.0630e+00f, 1.5750e+00f,
            1.5750e+00f, 1.5750e+00f, 1.5750e+00f, 1.5750e+00f, 1.5750e+00f,
            1.5750e+00f, 1.8190e+00f, 1.8190e+00f, 1.8190e+00f, 1.8190e+00f,
            1.8190e+00f, 1.8190e+00f, 1.8190e+00f, 2.0630e+00f, 2.0630e+00f,
            2.0630e+00f, 2.0630e+00f, 2.0630e+00f, 2.0630e+00f, 2.0630e+00f,
            5.2470e+00f, 5.2470e+00f, 5.2470e+00f, 5.2470e+00f, 5.2470e+00f,
            5.2470e+00f, 5.2470e+00f, 5.4910e+00f, 5.4910e+00f, 5.4910e+00f,
            5.4910e+00f, 5.4910e+00f, 5.4910e+00f, 5.4910e+00f, 5.7350e+00f,
            5.7350e+00f, 5.7350e+00f, 5.7350e+00f, 5.7350e+00f, 5.7350e+00f,
            5.7350e+00f, 5.2470e+00f, 5.2470e+00f, 5.2470e+00f, 5.2470e+00f,
            5.2470e+00f, 5.2470e+00f, 5.2470e+00f, 5.4910e+00f, 5.4910e+00f,
            5.4910e+00f, 5.4910e+00f, 5.4910e+00f, 5.4910e+00f, 5.4910e+00f,
            5.7350e+00f, 5.7350e+00f, 5.7350e+00f, 5.7350e+00f, 5.7350e+00f,
            5.7350e+00f, 5.7350e+00f, 5.2470e+00f, 5.2470e+00f, 5.2470e+00f,
            5.2470e+00f, 5.2470e+00f, 5.2470e+00f, 5.2470e+00f, 5.4910e+00f,
            5.4910e+00f, 5.4910e+00f, 5.4910e+00f, 5.4910e+00f, 5.4910e+00f,
            5.4910e+00f, 5.7350e+00f, 5.7350e+00f, 5.7350e+00f, 5.7350e+00f,
            5.7350e+00f, 5.7350e+00f, 5.7350e+00f, 5.2470e+00f, 5.2470e+00f,
            5.2470e+00f, 5.2470e+00f, 5.2470e+00f, 5.2470e+00f, 5.2470e+00f,
            5.4910e+00f, 5.4910e+00f, 5.4910e+00f, 5.4910e+00f, 5.4910e+00f,
            5.4910e+00f, 5.4910e+00f, 5.7350e+00f, 5.7350e+00f, 5.7350e+00f,
            5.7350e+00f, 5.7350e+00f, 5.7350e+00f, 5.7350e+00f, 5.2470e+00f,
            5.2470e+00f, 5.2470e+00f, 5.2470e+00f, 5.2470e+00f, 5.2470e+00f,
            5.2470e+00f, 5.4910e+00f, 5.4910e+00f, 5.4910e+00f, 5.4910e+00f,
            5.4910e+00f, 5.4910e+00f, 5.4910e+00f, 5.7350e+00f, 5.7350e+00f,
            5.7350e+00f, 5.7350e+00f, 5.7350e+00f, 5.7350e+00f, 5.7350e+00f,
        };
    }
}
