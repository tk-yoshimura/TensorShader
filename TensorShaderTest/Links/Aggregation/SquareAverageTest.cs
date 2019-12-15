using System;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using static TensorShader.Field;

namespace TensorShaderTest.Links.Aggregation {
    [TestClass]
    public class SquareAverageTest {
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

                Field y_expect = SquareAverage(x, new int[] { }, keepdims: true);
                Field err = Abs(y_expect - y_actual);

                (Flow flow, Parameters Parameters) = Flow.Optimize(err);

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

                Field y_expect = SquareAverage(x, new int[] { }, keepdims: false);
                Field err = Abs(y_expect - y_actual);

                (Flow flow, Parameters Parameters) = Flow.Optimize(err);

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

                Field y_expect = SquareAverage(x, new int[] { Axis.Map2D.Channels }, keepdims: true);
                Field err = Abs(y_expect - y_actual);

                (Flow flow, Parameters Parameters) = Flow.Optimize(err);

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

                Field y_expect = SquareAverage(x, new int[] { Axis.Map2D.Channels }, keepdims: false);
                Field err = Abs(y_expect - y_actual);

                (Flow flow, Parameters Parameters) = Flow.Optimize(err);

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

                Field y_expect = SquareAverage(x, new int[] { Axis.Map2D.Width }, keepdims: true);
                Field err = Abs(y_expect - y_actual);

                (Flow flow, Parameters Parameters) = Flow.Optimize(err);

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

                Field y_expect = SquareAverage(x, new int[] { Axis.Map2D.Width }, keepdims: false);
                Field err = Abs(y_expect - y_actual);

                (Flow flow, Parameters Parameters) = Flow.Optimize(err);

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

                Field y_expect = SquareAverage(x, new int[] { Axis.Map2D.Height }, keepdims: true);
                Field err = Abs(y_expect - y_actual);

                (Flow flow, Parameters Parameters) = Flow.Optimize(err);

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

                Field y_expect = SquareAverage(x, new int[] { Axis.Map2D.Height }, keepdims: false);
                Field err = Abs(y_expect - y_actual);

                (Flow flow, Parameters Parameters) = Flow.Optimize(err);

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

                Field y_expect = SquareAverage(x, new int[] { Axis.Map2D.Batch }, keepdims: true);
                Field err = Abs(y_expect - y_actual);

                (Flow flow, Parameters Parameters) = Flow.Optimize(err);

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

                Field y_expect = SquareAverage(x, new int[] { Axis.Map2D.Batch }, keepdims: false);
                Field err = Abs(y_expect - y_actual);

                (Flow flow, Parameters Parameters) = Flow.Optimize(err);

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

                Field y_expect = SquareAverage(x, new int[] { Axis.Map2D.Channels, Axis.Map2D.Width }, keepdims: true);
                Field err = Abs(y_expect - y_actual);

                (Flow flow, Parameters Parameters) = Flow.Optimize(err);

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

                Field y_expect = SquareAverage(x, new int[] { Axis.Map2D.Channels, Axis.Map2D.Width }, keepdims: false);
                Field err = Abs(y_expect - y_actual);

                (Flow flow, Parameters Parameters) = Flow.Optimize(err);

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

                Field y_expect = SquareAverage(x, new int[] { Axis.Map2D.Channels, Axis.Map2D.Height }, keepdims: true);
                Field err = Abs(y_expect - y_actual);

                (Flow flow, Parameters Parameters) = Flow.Optimize(err);

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

                Field y_expect = SquareAverage(x, new int[] { Axis.Map2D.Channels, Axis.Map2D.Height }, keepdims: false);
                Field err = Abs(y_expect - y_actual);

                (Flow flow, Parameters Parameters) = Flow.Optimize(err);

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

                Field y_expect = SquareAverage(x, new int[] { Axis.Map2D.Channels, Axis.Map2D.Batch }, keepdims: true);
                Field err = Abs(y_expect - y_actual);

                (Flow flow, Parameters Parameters) = Flow.Optimize(err);

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

                Field y_expect = SquareAverage(x, new int[] { Axis.Map2D.Channels, Axis.Map2D.Batch }, keepdims: false);
                Field err = Abs(y_expect - y_actual);

                (Flow flow, Parameters Parameters) = Flow.Optimize(err);

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

                Field y_expect = SquareAverage(x, new int[] { Axis.Map2D.Width, Axis.Map2D.Height }, keepdims: true);
                Field err = Abs(y_expect - y_actual);

                (Flow flow, Parameters Parameters) = Flow.Optimize(err);

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

                Field y_expect = SquareAverage(x, new int[] { Axis.Map2D.Width, Axis.Map2D.Height }, keepdims: false);
                Field err = Abs(y_expect - y_actual);

                (Flow flow, Parameters Parameters) = Flow.Optimize(err);

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

                Field y_expect = SquareAverage(x, new int[] { Axis.Map2D.Height, Axis.Map2D.Batch }, keepdims: true);
                Field err = Abs(y_expect - y_actual);

                (Flow flow, Parameters Parameters) = Flow.Optimize(err);

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

                Field y_expect = SquareAverage(x, new int[] { Axis.Map2D.Height, Axis.Map2D.Batch }, keepdims: false);
                Field err = Abs(y_expect - y_actual);

                (Flow flow, Parameters Parameters) = Flow.Optimize(err);

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

                Field y_expect = SquareAverage(x, new int[] { Axis.Map2D.Channels, Axis.Map2D.Width, Axis.Map2D.Height }, keepdims: true);
                Field err = Abs(y_expect - y_actual);

                (Flow flow, Parameters Parameters) = Flow.Optimize(err);

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

                Field y_expect = SquareAverage(x, new int[] { Axis.Map2D.Channels, Axis.Map2D.Width, Axis.Map2D.Height }, keepdims: false);
                Field err = Abs(y_expect - y_actual);

                (Flow flow, Parameters Parameters) = Flow.Optimize(err);

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

                Field y_expect = SquareAverage(x, new int[] { Axis.Map2D.Channels, Axis.Map2D.Width, Axis.Map2D.Batch }, keepdims: true);
                Field err = Abs(y_expect - y_actual);

                (Flow flow, Parameters Parameters) = Flow.Optimize(err);

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

                Field y_expect = SquareAverage(x, new int[] { Axis.Map2D.Channels, Axis.Map2D.Width, Axis.Map2D.Batch }, keepdims: false);
                Field err = Abs(y_expect - y_actual);

                (Flow flow, Parameters Parameters) = Flow.Optimize(err);

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

                Field y_expect = SquareAverage(x, new int[] { Axis.Map2D.Channels, Axis.Map2D.Height, Axis.Map2D.Batch }, keepdims: true);
                Field err = Abs(y_expect - y_actual);

                (Flow flow, Parameters Parameters) = Flow.Optimize(err);

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

                Field y_expect = SquareAverage(x, new int[] { Axis.Map2D.Channels, Axis.Map2D.Height, Axis.Map2D.Batch }, keepdims: false);
                Field err = Abs(y_expect - y_actual);

                (Flow flow, Parameters Parameters) = Flow.Optimize(err);

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

                Field y_expect = SquareAverage(x, new int[] { Axis.Map2D.Width, Axis.Map2D.Height, Axis.Map2D.Batch }, keepdims: true);
                Field err = Abs(y_expect - y_actual);

                (Flow flow, Parameters Parameters) = Flow.Optimize(err);

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

                Field y_expect = SquareAverage(x, new int[] { Axis.Map2D.Width, Axis.Map2D.Height, Axis.Map2D.Batch }, keepdims: false);
                Field err = Abs(y_expect - y_actual);

                (Flow flow, Parameters Parameters) = Flow.Optimize(err);

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

                Field y_expect = SquareAverage(x, axes: null, keepdims: true);
                Field err = Abs(y_expect - y_actual);

                (Flow flow, Parameters Parameters) = Flow.Optimize(err);

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

                Field y_expect = SquareAverage(x, axes: null, keepdims: false);
                Field err = Abs(y_expect - y_actual);

                (Flow flow, Parameters Parameters) = Flow.Optimize(err);

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

            Field y_expect = SquareAverage(x, new int[] { Axis.Map2D.Channels, Axis.Map2D.Height }, keepdims: false);
            Field err = Abs(y_expect - y_actual);

            (Flow flow, Parameters Parameters) = Flow.Optimize(err);

            flow.Execute();

            float[] gx_actual = x.GradTensor.State;

            AssertError.Tolerance(gx_expect, gx_actual, 1e-7f, 1e-5f, $"not equal gx");
        }

        float[] gx_expect = {
            0.0000e+00f, 1.6634e-07f, 3.3269e-07f, 4.9903e-07f, 6.6537e-07f,
            8.3171e-07f, 9.9806e-07f, 1.0360e-06f, 1.1840e-06f, 1.3320e-06f,
            1.4800e-06f, 1.6280e-06f, 1.7760e-06f, 1.9240e-06f, 1.8936e-06f,
            2.0289e-06f, 2.1641e-06f, 2.2994e-06f, 2.4346e-06f, 2.5699e-06f,
            2.7051e-06f, 3.4932e-06f, 3.6595e-06f, 3.8259e-06f, 3.9922e-06f,
            4.1586e-06f, 4.3249e-06f, 4.4913e-06f, 4.1440e-06f, 4.2920e-06f,
            4.4400e-06f, 4.5880e-06f, 4.7360e-06f, 4.8840e-06f, 5.0320e-06f,
            4.7340e-06f, 4.8693e-06f, 5.0045e-06f, 5.1398e-06f, 5.2750e-06f,
            5.4103e-06f, 5.5455e-06f, 6.9864e-06f, 7.1527e-06f, 7.3191e-06f,
            7.4854e-06f, 7.6518e-06f, 7.8181e-06f, 7.9845e-06f, 7.2520e-06f,
            7.4000e-06f, 7.5480e-06f, 7.6960e-06f, 7.8440e-06f, 7.9920e-06f,
            8.1400e-06f, 7.5744e-06f, 7.7097e-06f, 7.8449e-06f, 7.9802e-06f,
            8.1154e-06f, 8.2507e-06f, 8.3859e-06f, 1.0480e-05f, 1.0646e-05f,
            1.0812e-05f, 1.0979e-05f, 1.1145e-05f, 1.1311e-05f, 1.1478e-05f,
            1.0360e-05f, 1.0508e-05f, 1.0656e-05f, 1.0804e-05f, 1.0952e-05f,
            1.1100e-05f, 1.1248e-05f, 1.0415e-05f, 1.0550e-05f, 1.0685e-05f,
            1.0821e-05f, 1.0956e-05f, 1.1091e-05f, 1.1226e-05f, 1.3973e-05f,
            1.4139e-05f, 1.4305e-05f, 1.4472e-05f, 1.4638e-05f, 1.4805e-05f,
            1.4971e-05f, 1.3468e-05f, 1.3616e-05f, 1.3764e-05f, 1.3912e-05f,
            1.4060e-05f, 1.4208e-05f, 1.4356e-05f, 1.3255e-05f, 1.3390e-05f,
            1.3526e-05f, 1.3661e-05f, 1.3796e-05f, 1.3931e-05f, 1.4067e-05f,
            1.2232e-04f, 1.2348e-04f, 1.2465e-04f, 1.2581e-04f, 1.2698e-04f,
            1.2814e-04f, 1.2931e-04f, 1.3782e-04f, 1.3905e-04f, 1.4029e-04f,
            1.4152e-04f, 1.4275e-04f, 1.4398e-04f, 1.4521e-04f, 1.5492e-04f,
            1.5622e-04f, 1.5752e-04f, 1.5882e-04f, 1.6012e-04f, 1.6143e-04f,
            1.6273e-04f, 1.4678e-04f, 1.4794e-04f, 1.4911e-04f, 1.5027e-04f,
            1.5144e-04f, 1.5260e-04f, 1.5377e-04f, 1.6367e-04f, 1.6490e-04f,
            1.6613e-04f, 1.6736e-04f, 1.6859e-04f, 1.6982e-04f, 1.7105e-04f,
            1.8226e-04f, 1.8356e-04f, 1.8486e-04f, 1.8616e-04f, 1.8746e-04f,
            1.8877e-04f, 1.9007e-04f, 1.7124e-04f, 1.7241e-04f, 1.7357e-04f,
            1.7474e-04f, 1.7590e-04f, 1.7707e-04f, 1.7823e-04f, 1.8951e-04f,
            1.9074e-04f, 1.9197e-04f, 1.9320e-04f, 1.9443e-04f, 1.9566e-04f,
            1.9689e-04f, 2.0959e-04f, 2.1090e-04f, 2.1220e-04f, 2.1350e-04f,
            2.1480e-04f, 2.1610e-04f, 2.1741e-04f, 1.9571e-04f, 1.9687e-04f,
            1.9804e-04f, 1.9920e-04f, 2.0037e-04f, 2.0153e-04f, 2.0270e-04f,
            2.1535e-04f, 2.1658e-04f, 2.1781e-04f, 2.1904e-04f, 2.2027e-04f,
            2.2150e-04f, 2.2273e-04f, 2.3693e-04f, 2.3823e-04f, 2.3954e-04f,
            2.4084e-04f, 2.4214e-04f, 2.4344e-04f, 2.4474e-04f, 2.2017e-04f,
            2.2133e-04f, 2.2250e-04f, 2.2366e-04f, 2.2483e-04f, 2.2599e-04f,
            2.2716e-04f, 2.4119e-04f, 2.4242e-04f, 2.4365e-04f, 2.4488e-04f,
            2.4611e-04f, 2.4734e-04f, 2.4858e-04f, 2.6427e-04f, 2.6557e-04f,
            2.6687e-04f, 2.6818e-04f, 2.6948e-04f, 2.7078e-04f, 2.7208e-04f,
        };
    }
}
