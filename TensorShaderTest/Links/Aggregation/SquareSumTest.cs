using System;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using static TensorShader.Field;

namespace TensorShaderTest.Links.Aggregation {
    [TestClass]
    public class SquareSumTest {
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

                Field y_expect = SquareSum(x, new int[] { }, keepdims: true);
                Field err = Abs(y_expect - y_actual);

                (Flow flow, Parameters parameters) = Flow.Optimize(err);

                flow.Execute();

                gxval_true = x.GradState;
            }
            catch (Exception e) {
                Assert.Fail("axis:none keepdims:true  " + e.Message);
            }

            try {
                float[] yval = (new float[channels * width * height * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
                Tensor ytensor = (Shape.Map2D(channels, width, height, batch), yval);

                ParameterField x = xtensor;
                VariableField y_actual = ytensor;

                Field y_expect = SquareSum(x, new int[] { }, keepdims: false);
                Field err = Abs(y_expect - y_actual);

                (Flow flow, Parameters parameters) = Flow.Optimize(err);

                flow.Execute();

                gxval_false = x.GradState;

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

                Field y_expect = SquareSum(x, new int[] { Axis.Map2D.Channels }, keepdims: true);
                Field err = Abs(y_expect - y_actual);

                (Flow flow, Parameters parameters) = Flow.Optimize(err);

                flow.Execute();

                gxval_true = x.GradState;
            }
            catch (Exception e) {
                Assert.Fail("axis:0 keepdims:true  " + e.Message);
            }

            try {
                float[] yval = (new float[width * height * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
                Tensor ytensor = (Shape.Map1D(width, height, batch), yval);

                ParameterField x = xtensor;
                VariableField y_actual = ytensor;

                Field y_expect = SquareSum(x, new int[] { Axis.Map2D.Channels }, keepdims: false);
                Field err = Abs(y_expect - y_actual);

                (Flow flow, Parameters parameters) = Flow.Optimize(err);

                flow.Execute();

                gxval_false = x.GradState;

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

                Field y_expect = SquareSum(x, new int[] { Axis.Map2D.Width }, keepdims: true);
                Field err = Abs(y_expect - y_actual);

                (Flow flow, Parameters parameters) = Flow.Optimize(err);

                flow.Execute();

                gxval_true = x.GradState;
            }
            catch (Exception e) {
                Assert.Fail("axis:1 keepdims:true  " + e.Message);
            }

            try {
                float[] yval = (new float[channels * height * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
                Tensor ytensor = (Shape.Map1D(channels, height, batch), yval);

                ParameterField x = xtensor;
                VariableField y_actual = ytensor;

                Field y_expect = SquareSum(x, new int[] { Axis.Map2D.Width }, keepdims: false);
                Field err = Abs(y_expect - y_actual);

                (Flow flow, Parameters parameters) = Flow.Optimize(err);

                flow.Execute();

                gxval_false = x.GradState;

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

                Field y_expect = SquareSum(x, new int[] { Axis.Map2D.Height }, keepdims: true);
                Field err = Abs(y_expect - y_actual);

                (Flow flow, Parameters parameters) = Flow.Optimize(err);

                flow.Execute();

                gxval_true = x.GradState;
            }
            catch (Exception e) {
                Assert.Fail("axis:2 keepdims:true  " + e.Message);
            }

            try {
                float[] yval = (new float[channels * width * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
                Tensor ytensor = (Shape.Map1D(channels, width, batch), yval);

                ParameterField x = xtensor;
                VariableField y_actual = ytensor;

                Field y_expect = SquareSum(x, new int[] { Axis.Map2D.Height }, keepdims: false);
                Field err = Abs(y_expect - y_actual);

                (Flow flow, Parameters parameters) = Flow.Optimize(err);

                flow.Execute();

                gxval_false = x.GradState;

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

                Field y_expect = SquareSum(x, new int[] { Axis.Map2D.Batch }, keepdims: true);
                Field err = Abs(y_expect - y_actual);

                (Flow flow, Parameters parameters) = Flow.Optimize(err);

                flow.Execute();

                gxval_true = x.GradState;
            }
            catch (Exception e) {
                Assert.Fail("axis:3 keepdims:true  " + e.Message);
            }

            try {
                float[] yval = (new float[channels * width * height]).Select((_, idx) => idx * 1e-3f).ToArray();
                Tensor ytensor = (Shape.Map1D(channels, width, height), yval);

                ParameterField x = xtensor;
                VariableField y_actual = ytensor;

                Field y_expect = SquareSum(x, new int[] { Axis.Map2D.Batch }, keepdims: false);
                Field err = Abs(y_expect - y_actual);

                (Flow flow, Parameters parameters) = Flow.Optimize(err);

                flow.Execute();

                gxval_false = x.GradState;

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

                Field y_expect = SquareSum(x, new int[] { Axis.Map2D.Channels, Axis.Map2D.Width }, keepdims: true);
                Field err = Abs(y_expect - y_actual);

                (Flow flow, Parameters parameters) = Flow.Optimize(err);

                flow.Execute();

                gxval_true = x.GradState;
            }
            catch (Exception e) {
                Assert.Fail("axis:0,1 keepdims:true  " + e.Message);
            }

            try {
                float[] yval = (new float[height * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
                Tensor ytensor = (Shape.Map0D(height, batch), yval);

                ParameterField x = xtensor;
                VariableField y_actual = ytensor;

                Field y_expect = SquareSum(x, new int[] { Axis.Map2D.Channels, Axis.Map2D.Width }, keepdims: false);
                Field err = Abs(y_expect - y_actual);

                (Flow flow, Parameters parameters) = Flow.Optimize(err);

                flow.Execute();

                gxval_false = x.GradState;

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

                Field y_expect = SquareSum(x, new int[] { Axis.Map2D.Channels, Axis.Map2D.Height }, keepdims: true);
                Field err = Abs(y_expect - y_actual);

                (Flow flow, Parameters parameters) = Flow.Optimize(err);

                flow.Execute();

                gxval_true = x.GradState;
            }
            catch (Exception e) {
                Assert.Fail("axis:0,2 keepdims:true  " + e.Message);
            }

            try {
                float[] yval = (new float[width * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
                Tensor ytensor = (Shape.Map0D(width, batch), yval);

                ParameterField x = xtensor;
                VariableField y_actual = ytensor;

                Field y_expect = SquareSum(x, new int[] { Axis.Map2D.Channels, Axis.Map2D.Height }, keepdims: false);
                Field err = Abs(y_expect - y_actual);

                (Flow flow, Parameters parameters) = Flow.Optimize(err);

                flow.Execute();

                gxval_false = x.GradState;

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

                Field y_expect = SquareSum(x, new int[] { Axis.Map2D.Channels, Axis.Map2D.Batch }, keepdims: true);
                Field err = Abs(y_expect - y_actual);

                (Flow flow, Parameters parameters) = Flow.Optimize(err);

                flow.Execute();

                gxval_true = x.GradState;
            }
            catch (Exception e) {
                Assert.Fail("axis:0,3 keepdims:true  " + e.Message);
            }

            try {
                float[] yval = (new float[width * height]).Select((_, idx) => idx * 1e-3f).ToArray();
                Tensor ytensor = (Shape.Map0D(width, height), yval);

                ParameterField x = xtensor;
                VariableField y_actual = ytensor;

                Field y_expect = SquareSum(x, new int[] { Axis.Map2D.Channels, Axis.Map2D.Batch }, keepdims: false);
                Field err = Abs(y_expect - y_actual);

                (Flow flow, Parameters parameters) = Flow.Optimize(err);

                flow.Execute();

                gxval_false = x.GradState;

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

                Field y_expect = SquareSum(x, new int[] { Axis.Map2D.Width, Axis.Map2D.Height }, keepdims: true);
                Field err = Abs(y_expect - y_actual);

                (Flow flow, Parameters parameters) = Flow.Optimize(err);

                flow.Execute();

                gxval_true = x.GradState;
            }
            catch (Exception e) {
                Assert.Fail("axis:1,2 keepdims:true  " + e.Message);
            }

            try {
                float[] yval = (new float[channels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
                Tensor ytensor = (Shape.Map0D(channels, batch), yval);

                ParameterField x = xtensor;
                VariableField y_actual = ytensor;

                Field y_expect = SquareSum(x, new int[] { Axis.Map2D.Width, Axis.Map2D.Height }, keepdims: false);
                Field err = Abs(y_expect - y_actual);

                (Flow flow, Parameters parameters) = Flow.Optimize(err);

                flow.Execute();

                gxval_false = x.GradState;

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

                Field y_expect = SquareSum(x, new int[] { Axis.Map2D.Height, Axis.Map2D.Batch }, keepdims: true);
                Field err = Abs(y_expect - y_actual);

                (Flow flow, Parameters parameters) = Flow.Optimize(err);

                flow.Execute();

                gxval_true = x.GradState;
            }
            catch (Exception e) {
                Assert.Fail("axis:2,3 keepdims:true  " + e.Message);
            }

            try {
                float[] yval = (new float[channels * width]).Select((_, idx) => idx * 1e-3f).ToArray();
                Tensor ytensor = (Shape.Map0D(channels, width), yval);

                ParameterField x = xtensor;
                VariableField y_actual = ytensor;

                Field y_expect = SquareSum(x, new int[] { Axis.Map2D.Height, Axis.Map2D.Batch }, keepdims: false);
                Field err = Abs(y_expect - y_actual);

                (Flow flow, Parameters parameters) = Flow.Optimize(err);

                flow.Execute();

                gxval_false = x.GradState;

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

                Field y_expect = SquareSum(x, new int[] { Axis.Map2D.Channels, Axis.Map2D.Width, Axis.Map2D.Height }, keepdims: true);
                Field err = Abs(y_expect - y_actual);

                (Flow flow, Parameters parameters) = Flow.Optimize(err);

                flow.Execute();

                gxval_true = x.GradState;
            }
            catch (Exception e) {
                Assert.Fail("axis:0,1,2 keepdims:true  " + e.Message);
            }

            try {
                float[] yval = (new float[batch]).Select((_, idx) => idx * 1e-3f).ToArray();
                Tensor ytensor = (Shape.Vector(batch), yval);

                ParameterField x = xtensor;
                VariableField y_actual = ytensor;

                Field y_expect = SquareSum(x, new int[] { Axis.Map2D.Channels, Axis.Map2D.Width, Axis.Map2D.Height }, keepdims: false);
                Field err = Abs(y_expect - y_actual);

                (Flow flow, Parameters parameters) = Flow.Optimize(err);

                flow.Execute();

                gxval_false = x.GradState;

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

                Field y_expect = SquareSum(x, new int[] { Axis.Map2D.Channels, Axis.Map2D.Width, Axis.Map2D.Batch }, keepdims: true);
                Field err = Abs(y_expect - y_actual);

                (Flow flow, Parameters parameters) = Flow.Optimize(err);

                flow.Execute();

                gxval_true = x.GradState;
            }
            catch (Exception e) {
                Assert.Fail("axis:0,1,3 keepdims:true  " + e.Message);
            }

            try {
                float[] yval = (new float[height]).Select((_, idx) => idx * 1e-3f).ToArray();
                Tensor ytensor = (Shape.Vector(height), yval);

                ParameterField x = xtensor;
                VariableField y_actual = ytensor;

                Field y_expect = SquareSum(x, new int[] { Axis.Map2D.Channels, Axis.Map2D.Width, Axis.Map2D.Batch }, keepdims: false);
                Field err = Abs(y_expect - y_actual);

                (Flow flow, Parameters parameters) = Flow.Optimize(err);

                flow.Execute();

                gxval_false = x.GradState;

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

                Field y_expect = SquareSum(x, new int[] { Axis.Map2D.Channels, Axis.Map2D.Height, Axis.Map2D.Batch }, keepdims: true);
                Field err = Abs(y_expect - y_actual);

                (Flow flow, Parameters parameters) = Flow.Optimize(err);

                flow.Execute();

                gxval_true = x.GradState;
            }
            catch (Exception e) {
                Assert.Fail("axis:0,2,3 keepdims:true  " + e.Message);
            }

            try {
                float[] yval = (new float[width]).Select((_, idx) => idx * 1e-3f).ToArray();
                Tensor ytensor = (Shape.Vector(width), yval);

                ParameterField x = xtensor;
                VariableField y_actual = ytensor;

                Field y_expect = SquareSum(x, new int[] { Axis.Map2D.Channels, Axis.Map2D.Height, Axis.Map2D.Batch }, keepdims: false);
                Field err = Abs(y_expect - y_actual);

                (Flow flow, Parameters parameters) = Flow.Optimize(err);

                flow.Execute();

                gxval_false = x.GradState;

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

                Field y_expect = SquareSum(x, new int[] { Axis.Map2D.Width, Axis.Map2D.Height, Axis.Map2D.Batch }, keepdims: true);
                Field err = Abs(y_expect - y_actual);

                (Flow flow, Parameters parameters) = Flow.Optimize(err);

                flow.Execute();

                gxval_true = x.GradState;
            }
            catch (Exception e) {
                Assert.Fail("axis:1,2,3 keepdims:true  " + e.Message);
            }

            try {
                float[] yval = (new float[channels]).Select((_, idx) => idx * 1e-3f).ToArray();
                Tensor ytensor = (Shape.Vector(channels), yval);

                ParameterField x = xtensor;
                VariableField y_actual = ytensor;

                Field y_expect = SquareSum(x, new int[] { Axis.Map2D.Width, Axis.Map2D.Height, Axis.Map2D.Batch }, keepdims: false);
                Field err = Abs(y_expect - y_actual);

                (Flow flow, Parameters parameters) = Flow.Optimize(err);

                flow.Execute();

                gxval_false = x.GradState;

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

                Field y_expect = SquareSum(x, axes: null, keepdims: true);
                Field err = Abs(y_expect - y_actual);

                (Flow flow, Parameters parameters) = Flow.Optimize(err);

                flow.Execute();

                gxval_true = x.GradState;
            }
            catch (Exception e) {
                Assert.Fail("axis:all keepdims:true  " + e.Message);
            }

            try {
                float[] yval = (new float[1]).Select((_, idx) => idx * 1e-3f).ToArray();
                Tensor ytensor = (Shape.Scalar, yval);

                ParameterField x = xtensor;
                VariableField y_actual = ytensor;

                Field y_expect = SquareSum(x, axes: null, keepdims: false);
                Field err = Abs(y_expect - y_actual);

                (Flow flow, Parameters parameters) = Flow.Optimize(err);

                flow.Execute();

                gxval_false = x.GradState;

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

            Field y_expect = SquareSum(x, new int[] { Axis.Map2D.Channels, Axis.Map2D.Height }, keepdims: false);
            Field err = Abs(y_expect - y_actual);

            (Flow flow, Parameters parameters) = Flow.Optimize(err);

            flow.Execute();

            float[] gx_actual = x.GradState;

            AssertError.Tolerance(gx_expect, gx_actual, 1e-6f, 1e-4f, $"not equal gx"); /*backward tolerance*/
        }

        float[] gx_expect = {
            0.0000e+00f, 2.0377e-04f, 4.0754e-04f, 6.1131e-04f, 8.1508e-04f,
            1.0188e-03f, 1.2226e-03f, 1.7451e-03f, 1.9944e-03f, 2.2437e-03f,
            2.4930e-03f, 2.7423e-03f, 2.9916e-03f, 3.2409e-03f, 4.2237e-03f,
            4.5254e-03f, 4.8270e-03f, 5.1287e-03f, 5.4304e-03f, 5.7321e-03f,
            6.0338e-03f, 4.2792e-03f, 4.4829e-03f, 4.6867e-03f, 4.8905e-03f,
            5.0943e-03f, 5.2980e-03f, 5.5018e-03f, 6.9804e-03f, 7.2297e-03f,
            7.4790e-03f, 7.7283e-03f, 7.9776e-03f, 8.2269e-03f, 8.4762e-03f,
            1.0559e-02f, 1.0861e-02f, 1.1163e-02f, 1.1464e-02f, 1.1766e-02f,
            1.2068e-02f, 1.2369e-02f, 8.5583e-03f, 8.7621e-03f, 8.9659e-03f,
            9.1696e-03f, 9.3734e-03f, 9.5772e-03f, 9.7810e-03f, 1.2216e-02f,
            1.2465e-02f, 1.2714e-02f, 1.2964e-02f, 1.3213e-02f, 1.3462e-02f,
            1.3712e-02f, 1.6895e-02f, 1.7196e-02f, 1.7498e-02f, 1.7800e-02f,
            1.8101e-02f, 1.8403e-02f, 1.8705e-02f, 1.2838e-02f, 1.3041e-02f,
            1.3245e-02f, 1.3449e-02f, 1.3653e-02f, 1.3856e-02f, 1.4060e-02f,
            1.7451e-02f, 1.7700e-02f, 1.7950e-02f, 1.8199e-02f, 1.8448e-02f,
            1.8698e-02f, 1.8947e-02f, 2.3230e-02f, 2.3532e-02f, 2.3834e-02f,
            2.4135e-02f, 2.4437e-02f, 2.4739e-02f, 2.5040e-02f, 1.7117e-02f,
            1.7320e-02f, 1.7524e-02f, 1.7728e-02f, 1.7932e-02f, 1.8136e-02f,
            1.8339e-02f, 2.2686e-02f, 2.2936e-02f, 2.3185e-02f, 2.3434e-02f,
            2.3684e-02f, 2.3933e-02f, 2.4182e-02f, 2.9566e-02f, 2.9867e-02f,
            3.0169e-02f, 3.0471e-02f, 3.0772e-02f, 3.1074e-02f, 3.1376e-02f,
            1.7126e-01f, 1.7289e-01f, 1.7452e-01f, 1.7615e-01f, 1.7778e-01f,
            1.7941e-01f, 1.8104e-01f, 1.9930e-01f, 2.0108e-01f, 2.0286e-01f,
            2.0464e-01f, 2.0642e-01f, 2.0820e-01f, 2.0998e-01f, 2.3023e-01f,
            2.3217e-01f, 2.3410e-01f, 2.3604e-01f, 2.3797e-01f, 2.3991e-01f,
            2.4184e-01f, 2.0551e-01f, 2.0714e-01f, 2.0877e-01f, 2.1040e-01f,
            2.1203e-01f, 2.1366e-01f, 2.1529e-01f, 2.3667e-01f, 2.3845e-01f,
            2.4023e-01f, 2.4201e-01f, 2.4378e-01f, 2.4556e-01f, 2.4734e-01f,
            2.7086e-01f, 2.7280e-01f, 2.7473e-01f, 2.7667e-01f, 2.7860e-01f,
            2.8054e-01f, 2.8247e-01f, 2.3976e-01f, 2.4139e-01f, 2.4302e-01f,
            2.4465e-01f, 2.4628e-01f, 2.4792e-01f, 2.4955e-01f, 2.7404e-01f,
            2.7581e-01f, 2.7759e-01f, 2.7937e-01f, 2.8115e-01f, 2.8293e-01f,
            2.8471e-01f, 3.1149e-01f, 3.1343e-01f, 3.1536e-01f, 3.1730e-01f,
            3.1923e-01f, 3.2117e-01f, 3.2310e-01f, 2.7401e-01f, 2.7564e-01f,
            2.7727e-01f, 2.7890e-01f, 2.8054e-01f, 2.8217e-01f, 2.8380e-01f,
            3.1140e-01f, 3.1318e-01f, 3.1496e-01f, 3.1674e-01f, 3.1852e-01f,
            3.2030e-01f, 3.2208e-01f, 3.5212e-01f, 3.5406e-01f, 3.5599e-01f,
            3.5793e-01f, 3.5986e-01f, 3.6180e-01f, 3.6373e-01f, 3.0826e-01f,
            3.0989e-01f, 3.1152e-01f, 3.1316e-01f, 3.1479e-01f, 3.1642e-01f,
            3.1805e-01f, 3.4877e-01f, 3.5055e-01f, 3.5233e-01f, 3.5411e-01f,
            3.5589e-01f, 3.5767e-01f, 3.5945e-01f, 3.9275e-01f, 3.9469e-01f,
            3.9662e-01f, 3.9856e-01f, 4.0049e-01f, 4.0243e-01f, 4.0436e-01f,
        };
    }
}
