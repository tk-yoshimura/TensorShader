using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using static TensorShader.Field;

namespace TensorShaderTest {
    [TestClass]
    public class FieldTest {
        [TestMethod]
        public void MinimumTest() {
            Tensor intensor = (Shape.Scalar, new float[] { 1 });

            {
                ParameterField f1 = new ParameterField(intensor);

                (Flow flow, List<ParameterField> parameters) = Flow.Optimize(f1);

                flow.Execute();

                Assert.AreEqual(2, flow.NodeCount);
                Assert.AreEqual(1, flow.InTensorCount);
                Assert.AreEqual(1, flow.OutTensorCount);

                CollectionAssert.AreEquivalent(new ParameterField[] { f1 }, parameters);
                Assert.AreEqual(1, f1.GradState[0]);
                Assert.AreEqual(intensor, f1.ValueTensor);
            }

            {
                VariableField f1 = new VariableField(intensor);

                (Flow flow, List<ParameterField> parameters) = Flow.Optimize(f1);

                flow.Execute();

                Assert.AreEqual(1, flow.NodeCount);
                Assert.AreEqual(1, flow.InTensorCount);
                Assert.AreEqual(0, flow.OutTensorCount);

                Assert.AreEqual(0, parameters.Count);
                Assert.AreEqual(intensor, f1.ValueTensor);
            }

            {
                ParameterField f1 = new ParameterField(intensor);
                StoreField n1 = f1;

                (Flow flow, _) = Flow.Inference(n1);

                flow.Execute();

                Assert.AreEqual(2, flow.NodeCount);
                Assert.AreEqual(1, flow.InTensorCount);
                Assert.AreEqual(1, flow.OutTensorCount);

                Assert.AreEqual(1, n1.State[0]);
            }

            {
                VariableField f1 = intensor;
                StoreField n1 = f1;

                (Flow flow, _) = Flow.Inference(n1);

                flow.Execute();

                Assert.AreEqual(2, flow.NodeCount);
                Assert.AreEqual(1, flow.InTensorCount);
                Assert.AreEqual(1, flow.OutTensorCount);

                Assert.AreEqual(null, f1.Grad);
                Assert.AreEqual(1, n1.State[0]);
            }

            {
                ParameterField f1 = new ParameterField(intensor);

                (Flow optimize_flow, List<ParameterField> parameters) = Flow.Optimize(f1);

                optimize_flow.Execute();

                StoreField n1 = f1;

                (Flow inference_flow, _) = Flow.Inference(n1);

                inference_flow.Execute();

                Assert.AreEqual(2, optimize_flow.NodeCount);
                Assert.AreEqual(1, optimize_flow.InTensorCount);
                Assert.AreEqual(1, optimize_flow.OutTensorCount);

                CollectionAssert.AreEquivalent(new ParameterField[] { f1 }, parameters);
                Assert.AreEqual(1, f1.GradState[0]);
                Assert.AreEqual(intensor, f1.ValueTensor);

                Assert.AreEqual(2, inference_flow.NodeCount);
                Assert.AreEqual(1, inference_flow.InTensorCount);
                Assert.AreEqual(1, inference_flow.OutTensorCount);

                Assert.AreEqual(1, n1.State[0]);

                foreach (var item in inference_flow.InTensors) {
                    Assert.IsTrue(optimize_flow.InTensors.ContainsKey(item.Key));
                    Assert.AreEqual(item.Value, optimize_flow.InTensors[item.Key]);
                }
            }

            {
                VariableField f1 = new VariableField(intensor);

                (Flow optimize_flow, List<ParameterField> parameters) = Flow.Optimize(f1);

                optimize_flow.Execute();

                StoreField n1 = f1;

                (Flow inference_flow, _) = Flow.Inference(n1);

                inference_flow.Execute();

                Assert.AreEqual(1, optimize_flow.NodeCount);
                Assert.AreEqual(1, optimize_flow.InTensorCount);
                Assert.AreEqual(0, optimize_flow.OutTensorCount);

                Assert.AreEqual(0, parameters.Count);
                Assert.AreEqual(intensor, f1.ValueTensor);

                Assert.AreEqual(2, inference_flow.NodeCount);
                Assert.AreEqual(1, inference_flow.InTensorCount);
                Assert.AreEqual(1, inference_flow.OutTensorCount);

                Assert.AreEqual(1, n1.State[0]);

                foreach (var item in inference_flow.InTensors) {
                    Assert.IsTrue(optimize_flow.InTensors.ContainsKey(item.Key));
                    Assert.AreEqual(item.Value, optimize_flow.InTensors[item.Key]);
                }
            }

            {
                ParameterField f1 = new ParameterField(intensor);
                StoreField n1 = f1;

                (Flow inference_flow, _) = Flow.Inference(n1);

                inference_flow.Execute();

                (Flow optimize_flow, List<ParameterField> parameters) = Flow.Optimize(f1);

                optimize_flow.Execute();

                Assert.AreEqual(3, optimize_flow.NodeCount);
                Assert.AreEqual(1, optimize_flow.InTensorCount);
                Assert.AreEqual(2, optimize_flow.OutTensorCount);

                CollectionAssert.AreEquivalent(new ParameterField[] { f1 }, parameters);
                Assert.AreEqual(1, f1.GradState[0]);
                Assert.AreEqual(intensor, f1.ValueTensor);

                Assert.AreEqual(2, inference_flow.NodeCount);
                Assert.AreEqual(1, inference_flow.InTensorCount);
                Assert.AreEqual(1, inference_flow.OutTensorCount);

                Assert.AreEqual(1, n1.State[0]);

                foreach (var item in inference_flow.InTensors) {
                    Assert.IsTrue(optimize_flow.InTensors.ContainsKey(item.Key));
                    Assert.AreEqual(item.Value, optimize_flow.InTensors[item.Key]);
                }
            }

            {
                VariableField f1 = new VariableField(intensor);
                StoreField n1 = f1;

                (Flow inference_flow, _) = Flow.Inference(n1);

                inference_flow.Execute();

                (Flow optimize_flow, List<ParameterField> parameters) = Flow.Optimize(f1);

                optimize_flow.Execute();

                Assert.AreEqual(2, optimize_flow.NodeCount);
                Assert.AreEqual(1, optimize_flow.InTensorCount);
                Assert.AreEqual(1, optimize_flow.OutTensorCount);

                Assert.AreEqual(0, parameters.Count);
                Assert.AreEqual(intensor, f1.ValueTensor);

                Assert.AreEqual(2, inference_flow.NodeCount);
                Assert.AreEqual(1, inference_flow.InTensorCount);
                Assert.AreEqual(1, inference_flow.OutTensorCount);

                Assert.AreEqual(1, n1.State[0]);

                foreach (var item in inference_flow.InTensors) {
                    Assert.IsTrue(optimize_flow.InTensors.ContainsKey(item.Key));
                    Assert.AreEqual(item.Value, optimize_flow.InTensors[item.Key]);
                }
            }
        }

        [TestMethod]
        public void In1Out1Test() {
            Tensor intensor = (Shape.Scalar, new float[] { -1.5f });
            Tensor outtensor = (Shape.Scalar, new float[] { 2 });

            {
                ParameterField f1 = new ParameterField(intensor);
                VariableField fo = outtensor;

                Field f2 = Abs(f1);
                Field ferr = f2 - fo;

                (Flow flow, List<ParameterField> parameters) = Flow.Optimize(ferr);

                flow.Execute();

                Assert.AreEqual(11, flow.NodeCount);
                Assert.AreEqual(7, flow.VariableNodeCount);
                Assert.AreEqual(4, flow.FunctionNodeCount);
                Assert.AreEqual(2, flow.InTensorCount);
                Assert.AreEqual(1, flow.OutTensorCount);

                CollectionAssert.AreEquivalent(new ParameterField[] { f1 }, parameters);
                Assert.AreEqual(intensor, f1.ValueTensor);
                Assert.AreEqual(0.5f, f1.GradState[0]);
            }

            {
                ParameterField f1 = new ParameterField(intensor);

                Field f3 = Abs(f1);
                StoreField n3 = f3;

                (Flow flow, _) = Flow.Inference(n3);

                flow.Execute();

                Assert.AreEqual(4, flow.NodeCount);
                Assert.AreEqual(3, flow.VariableNodeCount);
                Assert.AreEqual(1, flow.FunctionNodeCount);
                Assert.AreEqual(1, flow.InTensorCount);
                Assert.AreEqual(1, flow.OutTensorCount);

                Assert.AreEqual(intensor, f1.ValueTensor);
                Assert.AreEqual(1.5f, n3.State[0]);
            }

            {
                ParameterField f1 = new ParameterField(intensor);
                VariableField fo = outtensor;

                Field f2 = Abs(f1);
                Field ferr = f2 - fo;
                StoreField n3 = f2;

                (Flow optimize_flow, List<ParameterField> parameters) = Flow.Optimize(ferr);

                optimize_flow.Execute();

                (Flow inference_flow, _) = Flow.Inference(n3);

                inference_flow.Execute();

                Assert.AreEqual(12, optimize_flow.NodeCount);
                Assert.AreEqual(8, optimize_flow.VariableNodeCount);
                Assert.AreEqual(4, optimize_flow.FunctionNodeCount);
                Assert.AreEqual(2, optimize_flow.InTensorCount);
                Assert.AreEqual(2, optimize_flow.OutTensorCount);

                CollectionAssert.AreEquivalent(new ParameterField[] { f1 }, parameters);
                Assert.AreEqual(intensor, f1.ValueTensor);
                Assert.AreEqual(0.5f, f1.GradState[0]);

                Assert.AreEqual(4, inference_flow.NodeCount);
                Assert.AreEqual(3, inference_flow.VariableNodeCount);
                Assert.AreEqual(1, inference_flow.FunctionNodeCount);
                Assert.AreEqual(1, inference_flow.InTensorCount);
                Assert.AreEqual(1, inference_flow.OutTensorCount);

                Assert.AreEqual(intensor, f1.ValueTensor);
                Assert.AreEqual(1.5f, n3.State[0]);

                foreach (var item in inference_flow.InTensors) {
                    Assert.IsTrue(optimize_flow.InTensors.ContainsKey(item.Key));
                    Assert.AreEqual(item.Value, optimize_flow.InTensors[item.Key]);
                }
            }

            {
                ParameterField f1 = new ParameterField(intensor);
                VariableField fo = outtensor;

                Field f2 = Abs(f1);
                Field ferr = f2 - fo;
                StoreField n3 = f2;

                (Flow inference_flow, _) = Flow.Inference(n3);

                inference_flow.Execute();

                (Flow optimize_flow, List<ParameterField> parameters) = Flow.Optimize(ferr);

                optimize_flow.Execute();

                Assert.AreEqual(12, optimize_flow.NodeCount);
                Assert.AreEqual(8, optimize_flow.VariableNodeCount);
                Assert.AreEqual(4, optimize_flow.FunctionNodeCount);
                Assert.AreEqual(2, optimize_flow.InTensorCount);
                Assert.AreEqual(2, optimize_flow.OutTensorCount);

                CollectionAssert.AreEquivalent(new ParameterField[] { f1 }, parameters);
                Assert.AreEqual(intensor, f1.ValueTensor);
                Assert.AreEqual(0.5f, f1.GradState[0]);

                Assert.AreEqual(4, inference_flow.NodeCount);
                Assert.AreEqual(3, inference_flow.VariableNodeCount);
                Assert.AreEqual(1, inference_flow.FunctionNodeCount);
                Assert.AreEqual(1, inference_flow.InTensorCount);
                Assert.AreEqual(1, inference_flow.OutTensorCount);

                Assert.AreEqual(intensor, f1.ValueTensor);
                Assert.AreEqual(1.5f, n3.State[0]);

                foreach (var item in inference_flow.InTensors) {
                    Assert.IsTrue(optimize_flow.InTensors.ContainsKey(item.Key));
                    Assert.AreEqual(item.Value, optimize_flow.InTensors[item.Key]);
                }
            }
        }

        [TestMethod]
        public void In2Out1Test() {
            Tensor in1tensor = (Shape.Scalar, new float[] { 1 });
            Tensor in2tensor = (Shape.Scalar, new float[] { 2 });
            Tensor outtensor = (Shape.Scalar, new float[] { -3 });

            {
                ParameterField f1 = new ParameterField(in1tensor);
                ParameterField f2 = new ParameterField(in2tensor);
                VariableField fo = new VariableField(outtensor);

                Field f3 = f1 + f2;
                Field ferr = f3 - fo;

                (Flow flow, List<ParameterField> parameters) = Flow.Optimize(ferr);

                flow.Execute();

                Assert.AreEqual(9, flow.NodeCount);
                Assert.AreEqual(7, flow.VariableNodeCount);
                Assert.AreEqual(2, flow.FunctionNodeCount);
                Assert.AreEqual(3, flow.InTensorCount);
                Assert.AreEqual(2, flow.OutTensorCount);

                CollectionAssert.AreEquivalent(new ParameterField[] { f1, f2 }, parameters);
                Assert.AreEqual(in1tensor, f1.ValueTensor);
                Assert.AreEqual(in2tensor, f2.ValueTensor);
                Assert.AreEqual(6, f1.GradState[0]);
                Assert.AreEqual(6, f2.GradState[0]);
            }

            {
                ParameterField f1 = new ParameterField(in1tensor);
                VariableField f2 = new VariableField(in2tensor);
                VariableField fo = new VariableField(outtensor);

                Field f3 = f1 + f2;
                Field ferr = f3 - fo;

                (Flow flow, List<ParameterField> parameters) = Flow.Optimize(ferr);

                flow.Execute();

                Assert.AreEqual(8, flow.NodeCount);
                Assert.AreEqual(6, flow.VariableNodeCount);
                Assert.AreEqual(2, flow.FunctionNodeCount);
                Assert.AreEqual(3, flow.InTensorCount);
                Assert.AreEqual(1, flow.OutTensorCount);

                CollectionAssert.AreEquivalent(new ParameterField[] { f1 }, parameters);
                Assert.AreEqual(in1tensor, f1.ValueTensor);
                Assert.AreEqual(in2tensor, f2.ValueTensor);
                Assert.AreEqual(6, f1.GradState[0]);
                Assert.AreEqual(null, f2.Grad);
            }

            {
                ParameterField f1 = new ParameterField(in1tensor);
                VariableField f2 = new VariableField(in2tensor);

                Field f3 = f1 + f2;
                StoreField n3 = f3;

                (Flow flow, _) = Flow.Inference(n3);

                flow.Execute();

                Assert.AreEqual(5, flow.NodeCount);
                Assert.AreEqual(4, flow.VariableNodeCount);
                Assert.AreEqual(1, flow.FunctionNodeCount);
                Assert.AreEqual(2, flow.InTensorCount);
                Assert.AreEqual(1, flow.OutTensorCount);

                Assert.AreEqual(in1tensor, f1.ValueTensor);
                Assert.AreEqual(in2tensor, f2.ValueTensor);
                Assert.AreEqual(3, n3.State[0]);
            }

            {
                ParameterField f1 = new ParameterField(in1tensor);
                VariableField f2 = new VariableField(in2tensor);
                VariableField fo = new VariableField(outtensor);

                Field f3 = f1 + f2;
                Field ferr = f3 - fo;
                StoreField n3 = f3;

                (Flow optimize_flow, List<ParameterField> parameters) = Flow.Optimize(ferr);

                optimize_flow.Execute();

                Assert.AreEqual(9, optimize_flow.NodeCount);
                Assert.AreEqual(7, optimize_flow.VariableNodeCount);
                Assert.AreEqual(2, optimize_flow.FunctionNodeCount);
                Assert.AreEqual(3, optimize_flow.InTensorCount);
                Assert.AreEqual(2, optimize_flow.OutTensorCount);

                CollectionAssert.AreEquivalent(new ParameterField[] { f1 }, parameters);
                Assert.AreEqual(in1tensor, f1.ValueTensor);
                Assert.AreEqual(in2tensor, f2.ValueTensor);
                Assert.AreEqual(6, f1.GradState[0]);
                Assert.AreEqual(null, f2.Grad);

                (Flow inference_flow, _) = Flow.Inference(n3);

                inference_flow.Execute();

                Assert.AreEqual(5, inference_flow.NodeCount);
                Assert.AreEqual(4, inference_flow.VariableNodeCount);
                Assert.AreEqual(1, inference_flow.FunctionNodeCount);
                Assert.AreEqual(2, inference_flow.InTensorCount);
                Assert.AreEqual(1, inference_flow.OutTensorCount);

                Assert.AreEqual(in1tensor, f1.ValueTensor);
                Assert.AreEqual(in2tensor, f2.ValueTensor);
                Assert.AreEqual(3, n3.State[0]);

                foreach (var item in inference_flow.InTensors) {
                    Assert.IsTrue(optimize_flow.InTensors.ContainsKey(item.Key));
                    Assert.AreEqual(item.Value, optimize_flow.InTensors[item.Key]);
                }
            }

            {
                ParameterField f1 = new ParameterField(in1tensor);
                VariableField f2 = new VariableField(in2tensor);
                VariableField fo = new VariableField(outtensor);

                Field f3 = f1 + f2;
                Field ferr = f3 - fo;
                StoreField n3 = f3;

                (Flow inference_flow, _) = Flow.Inference(n3);

                inference_flow.Execute();

                (Flow optimize_flow, List<ParameterField> parameters) = Flow.Optimize(ferr);

                optimize_flow.Execute();

                Assert.AreEqual(9, optimize_flow.NodeCount);
                Assert.AreEqual(7, optimize_flow.VariableNodeCount);
                Assert.AreEqual(2, optimize_flow.FunctionNodeCount);
                Assert.AreEqual(3, optimize_flow.InTensorCount);
                Assert.AreEqual(2, optimize_flow.OutTensorCount);

                CollectionAssert.AreEquivalent(new ParameterField[] { f1 }, parameters);
                Assert.AreEqual(in1tensor, f1.ValueTensor);
                Assert.AreEqual(in2tensor, f2.ValueTensor);
                Assert.AreEqual(6, f1.GradState[0]);
                Assert.AreEqual(null, f2.Grad);

                Assert.AreEqual(5, inference_flow.NodeCount);
                Assert.AreEqual(4, inference_flow.VariableNodeCount);
                Assert.AreEqual(1, inference_flow.FunctionNodeCount);
                Assert.AreEqual(2, inference_flow.InTensorCount);
                Assert.AreEqual(1, inference_flow.OutTensorCount);

                Assert.AreEqual(in1tensor, f1.ValueTensor);
                Assert.AreEqual(in2tensor, f2.ValueTensor);
                Assert.AreEqual(3, n3.State[0]);

                foreach (var item in inference_flow.InTensors) {
                    Assert.IsTrue(optimize_flow.InTensors.ContainsKey(item.Key));
                    Assert.AreEqual(item.Value, optimize_flow.InTensors[item.Key]);
                }
            }
        }

        [TestMethod]
        public void In1Out2Test() {
            Tensor intensor = (Shape.Scalar, new float[] { -1.5f });
            Tensor outtensor = (Shape.Scalar, new float[] { 2 });

            {
                ParameterField f1 = new ParameterField(intensor);
                VariableField fo = new VariableField(outtensor);

                Field f2 = Abs(f1);
                Field ferr1 = f2 - fo;

                Field f3 = f2 * f2;
                Field ferr2 = f3 - fo * fo;

                (Flow flow, List<ParameterField> parameters) = Flow.Optimize(ferr1, ferr2);

                flow.Execute();

                Assert.AreEqual(23, flow.NodeCount);
                Assert.AreEqual(13, flow.VariableNodeCount);
                Assert.AreEqual(10, flow.FunctionNodeCount);
                Assert.AreEqual(2, flow.InTensorCount);
                Assert.AreEqual(1, flow.OutTensorCount);

                CollectionAssert.AreEquivalent(new ParameterField[] { f1 }, parameters);
                Assert.AreEqual(intensor, f1.ValueTensor);
                Assert.AreEqual(Math.Sign(-1.5f) * ((1.5f - 2f) + (1.5f * 1.5f - 2f * 2f) * 1.5f * 2), f1.GradState[0]);
            }

            {
                ParameterField f1 = new ParameterField(intensor);
                VariableField fo = new VariableField(outtensor);

                Field f2 = Abs(f1);
                Field ferr1 = f2 - fo;

                Field f3 = f2 * f2;
                Field ferr2 = f3 - fo * fo;

                (Flow flow, List<ParameterField> parameters) = Flow.Optimize(ferr1, ferr2);

                flow.Execute();

                Assert.AreEqual(23, flow.NodeCount);
                Assert.AreEqual(13, flow.VariableNodeCount);
                Assert.AreEqual(10, flow.FunctionNodeCount);
                Assert.AreEqual(2, flow.InTensorCount);
                Assert.AreEqual(1, flow.OutTensorCount);

                CollectionAssert.AreEquivalent(new ParameterField[] { f1 }, parameters);
                Assert.AreEqual(intensor, f1.ValueTensor);
                Assert.AreEqual(Math.Sign(-1.5f) * ((1.5f - 2f) + (1.5f * 1.5f - 2f * 2f) * 1.5f * 2), f1.GradState[0]);
            }

            {
                ParameterField f1 = new ParameterField(intensor);

                Field f2 = Abs(f1);
                Field f3 = f2 * f2;

                StoreField n2 = f2;
                StoreField n3 = f3;

                (Flow flow, _) = Flow.Inference(n2, n3);

                flow.Execute();

                Assert.AreEqual(7, flow.NodeCount);
                Assert.AreEqual(5, flow.VariableNodeCount);
                Assert.AreEqual(2, flow.FunctionNodeCount);
                Assert.AreEqual(1, flow.InTensorCount);
                Assert.AreEqual(2, flow.OutTensorCount);

                Assert.AreEqual(intensor, f1.ValueTensor);
                Assert.AreEqual(1.5f, n2.State[0]);
                Assert.AreEqual(1.5f * 1.5f, n3.State[0]);
            }

            {
                ParameterField f1 = new ParameterField(intensor);

                Field f2 = Abs(f1);
                Field f3 = f2 * f2;

                StoreField n2 = f2;
                StoreField n3 = f3;

                (Flow flow, _) = Flow.Inference(n3, n2);

                flow.Execute();

                Assert.AreEqual(7, flow.NodeCount);
                Assert.AreEqual(5, flow.VariableNodeCount);
                Assert.AreEqual(2, flow.FunctionNodeCount);
                Assert.AreEqual(1, flow.InTensorCount);
                Assert.AreEqual(2, flow.OutTensorCount);

                Assert.AreEqual(intensor, f1.ValueTensor);
                Assert.AreEqual(1.5f, n2.State[0]);
                Assert.AreEqual(1.5f * 1.5f, n3.State[0]);
            }

            {
                ParameterField f1 = new ParameterField(intensor);
                VariableField fo = new VariableField(outtensor);

                Field f2 = Abs(f1);
                Field ferr1 = f2 - fo;

                Field f3 = f2 * f2;
                Field ferr2 = f3 - fo * fo;

                (Flow optimize_flow, List<ParameterField> parameters) = Flow.Optimize(ferr1, ferr2);

                optimize_flow.Execute();

                Assert.AreEqual(23, optimize_flow.NodeCount);
                Assert.AreEqual(13, optimize_flow.VariableNodeCount);
                Assert.AreEqual(10, optimize_flow.FunctionNodeCount);
                Assert.AreEqual(2, optimize_flow.InTensorCount);
                Assert.AreEqual(1, optimize_flow.OutTensorCount);

                CollectionAssert.AreEquivalent(new ParameterField[] { f1 }, parameters);
                Assert.AreEqual(intensor, f1.ValueTensor);
                Assert.AreEqual(Math.Sign(-1.5f) * ((1.5f - 2f) + (1.5f * 1.5f - 2f * 2f) * 1.5f * 2), f1.GradState[0]);

                StoreField n2 = f2;
                StoreField n3 = f3;

                (Flow inference_flow, _) = Flow.Inference(n2, n3);

                inference_flow.Execute();

                Assert.AreEqual(7, inference_flow.NodeCount);
                Assert.AreEqual(5, inference_flow.VariableNodeCount);
                Assert.AreEqual(2, inference_flow.FunctionNodeCount);
                Assert.AreEqual(1, inference_flow.InTensorCount);
                Assert.AreEqual(2, inference_flow.OutTensorCount);

                Assert.AreEqual(intensor, f1.ValueTensor);
                Assert.AreEqual(1.5f, n2.State[0]);
                Assert.AreEqual(1.5f * 1.5f, n3.State[0]);

                foreach (var item in inference_flow.InTensors) {
                    Assert.IsTrue(optimize_flow.InTensors.ContainsKey(item.Key));
                    Assert.AreEqual(item.Value, optimize_flow.InTensors[item.Key]);
                }
            }

            {
                ParameterField f1 = new ParameterField(intensor);
                VariableField fo = new VariableField(outtensor);

                Field f2 = Abs(f1);
                Field ferr1 = f2 - fo;

                Field f3 = f2 * f2;
                Field ferr2 = f3 - fo * fo;

                (Flow optimize_flow, List<ParameterField> parameters) = Flow.Optimize(ferr2, ferr1);

                optimize_flow.Execute();

                Assert.AreEqual(23, optimize_flow.NodeCount);
                Assert.AreEqual(13, optimize_flow.VariableNodeCount);
                Assert.AreEqual(10, optimize_flow.FunctionNodeCount);
                Assert.AreEqual(2, optimize_flow.InTensorCount);
                Assert.AreEqual(1, optimize_flow.OutTensorCount);

                CollectionAssert.AreEquivalent(new ParameterField[] { f1 }, parameters);
                Assert.AreEqual(intensor, f1.ValueTensor);
                Assert.AreEqual(Math.Sign(-1.5f) * ((1.5f - 2f) + (1.5f * 1.5f - 2f * 2f) * 1.5f * 2), f1.GradState[0]);

                StoreField n2 = f2;
                StoreField n3 = f3;

                (Flow inference_flow, _) = Flow.Inference(n3, n2);

                inference_flow.Execute();

                Assert.AreEqual(7, inference_flow.NodeCount);
                Assert.AreEqual(5, inference_flow.VariableNodeCount);
                Assert.AreEqual(2, inference_flow.FunctionNodeCount);
                Assert.AreEqual(1, inference_flow.InTensorCount);
                Assert.AreEqual(2, inference_flow.OutTensorCount);

                Assert.AreEqual(intensor, f1.ValueTensor);
                Assert.AreEqual(1.5f, n2.State[0]);
                Assert.AreEqual(1.5f * 1.5f, n3.State[0]);

                foreach (var item in inference_flow.InTensors) {
                    Assert.IsTrue(optimize_flow.InTensors.ContainsKey(item.Key));
                    Assert.AreEqual(item.Value, optimize_flow.InTensors[item.Key]);
                }
            }
        }

        [TestMethod]
        public void In2Out2Test() {
            Tensor intensor = (Shape.Scalar, new float[] { -1.5f });
            Tensor outtensor = (Shape.Scalar, new float[] { 2 });

            {
                ParameterField f1 = new ParameterField(intensor);
                ParameterField fo = new ParameterField(outtensor);

                Field f2 = Abs(f1);
                Field ferr1 = f2 - fo;

                Field f3 = f2 * f2;
                Field ferr2 = f3 - fo * fo;

                (Flow flow, List<ParameterField> parameters) = Flow.Optimize(ferr1, ferr2);

                flow.Execute();

                Assert.AreEqual(34, flow.NodeCount);
                Assert.AreEqual(19, flow.VariableNodeCount);
                Assert.AreEqual(15, flow.FunctionNodeCount);
                Assert.AreEqual(2, flow.InTensorCount);
                Assert.AreEqual(2, flow.OutTensorCount);

                CollectionAssert.AreEquivalent(new ParameterField[] { f1, fo }, parameters);
                Assert.AreEqual(intensor, f1.ValueTensor);
                Assert.AreEqual(Math.Sign(-1.5f) * ((1.5f - 2f) + (1.5f * 1.5f - 2f * 2f) * 1.5f * 2), f1.GradState[0]);
                Assert.AreEqual(-(1.5f - 2f) - (1.5f * 1.5f - 2f * 2f) * 2f * 2, fo.GradState[0]);
            }
        }

        [TestMethod]
        public void ParallelTest() {
            Tensor intensor1 = (Shape.Scalar, new float[] { -1.5f });
            Tensor outtensor1 = (Shape.Scalar, new float[] { 2 });

            Tensor intensor2 = (Shape.Scalar, new float[] { -2.5f });
            Tensor outtensor2 = (Shape.Scalar, new float[] { 4 });

            Tensor intensor3 = (Shape.Scalar, new float[] { -4.5f });
            Tensor outtensor3 = (Shape.Scalar, new float[] { 8 });

            {
                ParameterField fi1 = new ParameterField(intensor1);
                VariableField fo1 = new VariableField(outtensor1);

                ParameterField fi2 = new ParameterField(intensor2);
                VariableField fo2 = new VariableField(outtensor2);

                ParameterField fi3 = new ParameterField(intensor3);
                VariableField fo3 = new VariableField(outtensor3);

                Field ferr1 = fo1 - Abs(fi1);
                Field ferr2 = fo2 - Abs(fi2);
                Field ferr3 = fo3 - Abs(fi3);

                (Flow flow, List<ParameterField> parameters) = Flow.Optimize(ferr1, ferr2, ferr3);

                flow.Execute();

                Assert.AreEqual(39, flow.NodeCount);
                Assert.AreEqual(24, flow.VariableNodeCount);
                Assert.AreEqual(15, flow.FunctionNodeCount);
                Assert.AreEqual(6, flow.InTensorCount);
                Assert.AreEqual(3, flow.OutTensorCount);

                CollectionAssert.AreEquivalent(new ParameterField[] { fi1, fi2, fi3 }, parameters);
                Assert.AreEqual(0.5f, fi1.GradState[0]);
                Assert.AreEqual(1.5f, fi2.GradState[0]);
                Assert.AreEqual(3.5f, fi3.GradState[0]);

                Assert.AreEqual(null, fo1.Grad);
                Assert.AreEqual(null, fo2.Grad);
                Assert.AreEqual(null, fo3.Grad);
            }

            {
                VariableField fi1 = new VariableField(intensor1);
                VariableField fo1 = new VariableField(outtensor1);

                VariableField fi2 = new VariableField(intensor2);
                VariableField fo2 = new VariableField(outtensor2);

                VariableField fi3 = new VariableField(intensor3);
                VariableField fo3 = new VariableField(outtensor3);

                StoreField n1 = fo1 - Abs(fi1);
                StoreField n2 = fo2 - Abs(fi2);
                StoreField n3 = fo3 - Abs(fi3);

                (Flow flow, _) = Flow.Inference(n1, n2, n3);

                flow.Execute();

                Assert.AreEqual(21, flow.NodeCount);
                Assert.AreEqual(15, flow.VariableNodeCount);
                Assert.AreEqual(6, flow.FunctionNodeCount);
                Assert.AreEqual(6, flow.InTensorCount);
                Assert.AreEqual(3, flow.OutTensorCount);

                Assert.AreEqual(0.5f, n1.State[0]);
                Assert.AreEqual(1.5f, n2.State[0]);
                Assert.AreEqual(3.5f, n3.State[0]);
            }

            {
                ParameterField fi1 = new ParameterField(intensor1);
                VariableField fo1 = new VariableField(outtensor1);

                ParameterField fi2 = new ParameterField(intensor2);
                VariableField fo2 = new VariableField(outtensor2);

                ParameterField fi3 = new ParameterField(intensor3);
                VariableField fo3 = new VariableField(outtensor3);

                Field ferr1 = fo1 - Abs(fi1);
                Field ferr2 = fo2 - Abs(fi2);
                Field ferr3 = fo3 - Abs(fi3);

                (Flow optimize_flow, List<ParameterField> parameters) = Flow.Optimize(ferr1, ferr2, ferr3);

                optimize_flow.Execute();

                Assert.AreEqual(39, optimize_flow.NodeCount);
                Assert.AreEqual(24, optimize_flow.VariableNodeCount);
                Assert.AreEqual(15, optimize_flow.FunctionNodeCount);
                Assert.AreEqual(6, optimize_flow.InTensorCount);
                Assert.AreEqual(3, optimize_flow.OutTensorCount);

                CollectionAssert.AreEquivalent(new ParameterField[] { fi1, fi2, fi3 }, parameters);
                Assert.AreEqual(0.5f, fi1.GradState[0]);
                Assert.AreEqual(1.5f, fi2.GradState[0]);
                Assert.AreEqual(3.5f, fi3.GradState[0]);

                StoreField n1 = ferr1;
                StoreField n2 = ferr2;

                (Flow inference_flow, _) = Flow.Inference(n1, n2);

                inference_flow.Execute();

                Assert.AreEqual(14, inference_flow.NodeCount);
                Assert.AreEqual(10, inference_flow.VariableNodeCount);
                Assert.AreEqual(4, inference_flow.FunctionNodeCount);
                Assert.AreEqual(4, inference_flow.InTensorCount);
                Assert.AreEqual(2, inference_flow.OutTensorCount);

                Assert.AreEqual(0.5f, n1.State[0]);
                Assert.AreEqual(1.5f, n2.State[0]);

                foreach (var item in inference_flow.InTensors) {
                    Assert.IsTrue(optimize_flow.InTensors.ContainsKey(item.Key));
                    Assert.AreEqual(item.Value, optimize_flow.InTensors[item.Key]);
                }
            }
        }

        [TestMethod]
        public void RejoinTest() {
            Tensor intensor = (Shape.Scalar, new float[] { -1.5f });
            Tensor outtensor = (Shape.Scalar, new float[] { -2 });

            {
                ParameterField f1 = new ParameterField(intensor);
                VariableField fo = new VariableField(outtensor);

                (Field f2, Field f3) = (f1, -f1);
                Field f4 = f2 * f3;
                Field ferr = f4 - fo;

                (Flow flow, List<ParameterField> parameters) = Flow.Optimize(ferr);

                flow.Execute();

                Assert.AreEqual(17, flow.NodeCount);
                Assert.AreEqual(10, flow.VariableNodeCount);
                Assert.AreEqual(7, flow.FunctionNodeCount);
                Assert.AreEqual(2, flow.InTensorCount);
                Assert.AreEqual(1, flow.OutTensorCount);

                CollectionAssert.AreEquivalent(new ParameterField[] { f1 }, parameters);
                Assert.AreEqual(intensor, f1.ValueTensor);
                Assert.AreEqual((-2.25f + 2) * 1.5f * 2, f1.GradState[0]);
            }

            {
                ParameterField f1 = new ParameterField(intensor);

                (Field f2, Field f3) = (f1, -f1);
                Field f4 = f2 * f3;

                StoreField n4 = f4;

                (Flow flow, _) = Flow.Inference(n4);

                flow.Execute();

                Assert.AreEqual(6, flow.NodeCount);
                Assert.AreEqual(4, flow.VariableNodeCount);
                Assert.AreEqual(2, flow.FunctionNodeCount);
                Assert.AreEqual(1, flow.InTensorCount);
                Assert.AreEqual(1, flow.OutTensorCount);

                Assert.AreEqual(intensor, f1.ValueTensor);
                Assert.AreEqual(-2.25f, n4.State[0]);
            }

            {
                ParameterField f1 = new ParameterField(intensor);
                VariableField fo = new VariableField(outtensor);

                (Field f2, Field f3) = (f1, -f1);
                Field f4 = f2 * f3;
                Field ferr = f4 - fo;

                (Flow optimize_flow, List<ParameterField> parameters) = Flow.Optimize(ferr);

                optimize_flow.Execute();

                Assert.AreEqual(17, optimize_flow.NodeCount);
                Assert.AreEqual(10, optimize_flow.VariableNodeCount);
                Assert.AreEqual(7, optimize_flow.FunctionNodeCount);
                Assert.AreEqual(2, optimize_flow.InTensorCount);
                Assert.AreEqual(1, optimize_flow.OutTensorCount);

                CollectionAssert.AreEquivalent(new ParameterField[] { f1 }, parameters);
                Assert.AreEqual(intensor, f1.ValueTensor);
                Assert.AreEqual((-2.25f + 2) * 1.5f * 2, f1.GradState[0]);

                StoreField n4 = f4;

                (Flow inference_flow, _) = Flow.Inference(n4);

                inference_flow.Execute();

                Assert.AreEqual(6, inference_flow.NodeCount);
                Assert.AreEqual(4, inference_flow.VariableNodeCount);
                Assert.AreEqual(2, inference_flow.FunctionNodeCount);
                Assert.AreEqual(1, inference_flow.InTensorCount);
                Assert.AreEqual(1, inference_flow.OutTensorCount);

                Assert.AreEqual(intensor, f1.ValueTensor);
                Assert.AreEqual(-2.25f, n4.State[0]);

                foreach (var item in inference_flow.InTensors) {
                    Assert.IsTrue(optimize_flow.InTensors.ContainsKey(item.Key));
                    Assert.AreEqual(item.Value, optimize_flow.InTensors[item.Key]);
                }
            }
        }

        [TestMethod]
        public void BifurcTest() {
            Tensor intensor = (Shape.Scalar, new float[] { -1.5f });

            {
                ParameterField f1 = intensor;

                (Field f2, Field f3) = (-(-f1), -f1);

                (Flow flow, List<ParameterField> parameters) = Flow.Optimize(f2, f3);

                flow.Execute();

                Assert.AreEqual(16, flow.NodeCount);
                Assert.AreEqual(9, flow.VariableNodeCount);
                Assert.AreEqual(7, flow.FunctionNodeCount);
                Assert.AreEqual(1, flow.InTensorCount);
                Assert.AreEqual(1, flow.OutTensorCount);

                CollectionAssert.AreEquivalent(new ParameterField[] { f1 }, parameters);
                Assert.AreEqual(intensor, f1.ValueTensor);
                Assert.AreEqual(-1.5f - 1.5f, f1.GradState[0]);
            }

            {
                VariableField f1 = intensor;

                (Field f2, Field f3) = (-(-f1), -f1);
                StoreField n2 = f2;
                StoreField n3 = f3;

                (Flow flow, _) = Flow.Inference(n2, n3);

                flow.Execute();

                Assert.AreEqual(9, flow.NodeCount);
                Assert.AreEqual(6, flow.VariableNodeCount);
                Assert.AreEqual(3, flow.FunctionNodeCount);
                Assert.AreEqual(1, flow.InTensorCount);
                Assert.AreEqual(2, flow.OutTensorCount);

                Assert.AreEqual(-1.5f, n2.State[0]);
                Assert.AreEqual(1.5f, n3.State[0]);
            }

            {
                ParameterField f1 = intensor;

                (Field f2, Field f3) = (-(-f1), -f1);

                (Flow optimize_flow, List<ParameterField> parameters) = Flow.Optimize(f2, f3);

                optimize_flow.Execute();

                StoreField n2 = f2;
                StoreField n3 = f3;

                (Flow inference_flow, _) = Flow.Inference(n2, n3);

                inference_flow.Execute();

                Assert.AreEqual(16, optimize_flow.NodeCount);
                Assert.AreEqual(9, optimize_flow.VariableNodeCount);
                Assert.AreEqual(7, optimize_flow.FunctionNodeCount);
                Assert.AreEqual(1, optimize_flow.InTensorCount);
                Assert.AreEqual(1, optimize_flow.OutTensorCount);

                CollectionAssert.AreEquivalent(new ParameterField[] { f1 }, parameters);
                Assert.AreEqual(intensor, f1.ValueTensor);
                Assert.AreEqual(-1.5f - 1.5f, f1.GradState[0]);

                Assert.AreEqual(9, inference_flow.NodeCount);
                Assert.AreEqual(6, inference_flow.VariableNodeCount);
                Assert.AreEqual(3, inference_flow.FunctionNodeCount);
                Assert.AreEqual(1, inference_flow.InTensorCount);
                Assert.AreEqual(2, inference_flow.OutTensorCount);

                Assert.AreEqual(-1.5f, n2.State[0]);
                Assert.AreEqual(1.5f, n3.State[0]);

                foreach (var item in inference_flow.InTensors) {
                    Assert.IsTrue(optimize_flow.InTensors.ContainsKey(item.Key));
                    Assert.AreEqual(item.Value, optimize_flow.InTensors[item.Key]);
                }
            }

            {
                ParameterField f1 = intensor;

                (Field f2, Field f3) = (-(-f1), -f1);
                StoreField n2 = f2;
                StoreField n3 = f3;

                (Flow inference_flow, _) = Flow.Inference(n2, n3);

                inference_flow.Execute();

                (Flow optimize_flow, List<ParameterField> parameters) = Flow.Optimize(f2, f3);

                optimize_flow.Execute();

                Assert.AreEqual(18, optimize_flow.NodeCount);
                Assert.AreEqual(11, optimize_flow.VariableNodeCount);
                Assert.AreEqual(7, optimize_flow.FunctionNodeCount);
                Assert.AreEqual(1, optimize_flow.InTensorCount);
                Assert.AreEqual(3, optimize_flow.OutTensorCount);

                CollectionAssert.AreEquivalent(new ParameterField[] { f1 }, parameters);
                Assert.AreEqual(intensor, f1.ValueTensor);
                Assert.AreEqual(-1.5f - 1.5f, f1.GradState[0]);

                Assert.AreEqual(9, inference_flow.NodeCount);
                Assert.AreEqual(6, inference_flow.VariableNodeCount);
                Assert.AreEqual(3, inference_flow.FunctionNodeCount);
                Assert.AreEqual(1, inference_flow.InTensorCount);
                Assert.AreEqual(2, inference_flow.OutTensorCount);

                Assert.AreEqual(-1.5f, n2.State[0]);
                Assert.AreEqual(1.5f, n3.State[0]);

                foreach (var item in inference_flow.InTensors) {
                    Assert.IsTrue(optimize_flow.InTensors.ContainsKey(item.Key));
                    Assert.AreEqual(item.Value, optimize_flow.InTensors[item.Key]);
                }
            }

            {
                ParameterField f1 = intensor;

                (Field f2, Field f3) = (-(-f1), -f1);

                (Flow optimize_flow, List<ParameterField> parameters) = Flow.Optimize(f3, f2);

                optimize_flow.Execute();

                StoreField n2 = f2;
                StoreField n3 = f3;

                (Flow inference_flow, _) = Flow.Inference(n3, n2);

                inference_flow.Execute();

                Assert.AreEqual(16, optimize_flow.NodeCount);
                Assert.AreEqual(9, optimize_flow.VariableNodeCount);
                Assert.AreEqual(7, optimize_flow.FunctionNodeCount);
                Assert.AreEqual(1, optimize_flow.InTensorCount);
                Assert.AreEqual(1, optimize_flow.OutTensorCount);

                CollectionAssert.AreEquivalent(new ParameterField[] { f1 }, parameters);
                Assert.AreEqual(intensor, f1.ValueTensor);
                Assert.AreEqual(-1.5f - 1.5f, f1.GradState[0]);

                Assert.AreEqual(9, inference_flow.NodeCount);
                Assert.AreEqual(6, inference_flow.VariableNodeCount);
                Assert.AreEqual(3, inference_flow.FunctionNodeCount);
                Assert.AreEqual(1, inference_flow.InTensorCount);
                Assert.AreEqual(2, inference_flow.OutTensorCount);

                Assert.AreEqual(-1.5f, n2.State[0]);
                Assert.AreEqual(1.5f, n3.State[0]);

                foreach (var item in inference_flow.InTensors) {
                    Assert.IsTrue(optimize_flow.InTensors.ContainsKey(item.Key));
                    Assert.AreEqual(item.Value, optimize_flow.InTensors[item.Key]);
                }
            }
        }

        [TestMethod]
        public void BroadcastTest() {
            Tensor intensor1 = (Shape.Vector(5), new float[] { 0f, 1f, 2f, 3f, 4f });
            Tensor intensor2 = (Shape.Scalar, new float[] { 2f });
            Tensor outtensor1 = (Shape.Vector(5), new float[] { 4f, 3f, 2f, 1f, 0f });
            Tensor outtensor2 = (Shape.Vector(5), new float[] { 0f, 2f, 4f, 6f, 8f });

            {
                ParameterField f1 = intensor1;
                ParameterField f2 = intensor2;
                VariableField fo = outtensor1;

                Field f3 = f1 + f2;
                Field ferr = f3 - fo;
                (Flow flow, List<ParameterField> _) = Flow.Optimize(ferr);

                flow.Execute();

                Assert.AreEqual(13, flow.NodeCount);
                Assert.AreEqual(9, flow.VariableNodeCount);
                Assert.AreEqual(4, flow.FunctionNodeCount);
                Assert.AreEqual(3, flow.InTensorCount);
                Assert.AreEqual(2, flow.OutTensorCount);

                float[] y1 = f1.GradState;
                float[] y2 = f2.GradState;

                CollectionAssert.AreEqual(new float[] { -2f, 0f, 2f, 4f, 6f }, y1);
                CollectionAssert.AreEqual(new float[] { 10f }, y2);
            }

            {
                ParameterField f1 = intensor1;
                ParameterField f2 = intensor2;
                VariableField fo = outtensor1;

                Field ferr = f1 + (f2 - fo);
                (Flow flow, List<ParameterField> _) = Flow.Optimize(ferr);

                flow.Execute();

                Assert.AreEqual(13, flow.NodeCount);
                Assert.AreEqual(9, flow.VariableNodeCount);
                Assert.AreEqual(4, flow.FunctionNodeCount);
                Assert.AreEqual(3, flow.InTensorCount);
                Assert.AreEqual(2, flow.OutTensorCount);

                float[] y1 = f1.GradState;
                float[] y2 = f2.GradState;

                CollectionAssert.AreEqual(new float[] { -2f, 0f, 2f, 4f, 6f }, y1);
                CollectionAssert.AreEqual(new float[] { 10f }, y2);
            }

            {
                ParameterField f1 = intensor1;
                ParameterField f2 = intensor2;
                VariableField fo = outtensor1;

                Field f3 = f1 * f2;
                Field ferr = f3 - fo;
                (Flow flow, List<ParameterField> _) = Flow.Optimize(ferr);

                flow.Execute();

                Assert.AreEqual(17, flow.NodeCount);
                Assert.AreEqual(11, flow.VariableNodeCount);
                Assert.AreEqual(6, flow.FunctionNodeCount);
                Assert.AreEqual(3, flow.InTensorCount);
                Assert.AreEqual(2, flow.OutTensorCount);

                float[] y1 = f1.GradState;
                float[] y2 = f2.GradState;

                CollectionAssert.AreEqual(new float[] { -8f, -2f, 4f, 10f, 16f }, y1);
                CollectionAssert.AreEqual(new float[] { 50f }, y2);
            }

            {
                ParameterField f1 = intensor1;
                ParameterField f2 = intensor2;
                VariableField fo = outtensor2;

                Field f3 = f1 * f2;
                Field ferr = f3 - fo;
                (Flow flow, List<ParameterField> _) = Flow.Optimize(ferr);

                flow.Execute();

                Assert.AreEqual(17, flow.NodeCount);
                Assert.AreEqual(11, flow.VariableNodeCount);
                Assert.AreEqual(6, flow.FunctionNodeCount);
                Assert.AreEqual(3, flow.InTensorCount);
                Assert.AreEqual(2, flow.OutTensorCount);

                float[] y1 = f1.GradState;
                float[] y2 = f2.GradState;

                CollectionAssert.AreEqual(new float[] { 0f, 0f, 0f, 0f, 0f }, y1);
                CollectionAssert.AreEqual(new float[] { 0f }, y2);
            }

            {
                ParameterField f1 = intensor1;
                ParameterField f2 = intensor2;
                VariableField fo = outtensor1;

                Field f3 = f1 + f2;
                Field ferr = f3 - fo;
                StoreField ave_ferr = Average(ferr);
                (Flow flow, List<ParameterField> _) = Flow.Optimize(ferr);

                flow.Execute();

                Assert.AreEqual(18, flow.NodeCount);
                Assert.AreEqual(12, flow.VariableNodeCount);
                Assert.AreEqual(6, flow.FunctionNodeCount);
                Assert.AreEqual(3, flow.InTensorCount);
                Assert.AreEqual(3, flow.OutTensorCount);

                float[] y1 = f1.GradState;
                float[] y2 = f2.GradState;

                CollectionAssert.AreEqual(new float[] { -2f, 0f, 2f, 4f, 6f }, y1);
                CollectionAssert.AreEqual(new float[] { 10f }, y2);
                CollectionAssert.AreEqual(new float[] { 2f }, ave_ferr.State);
            }
        }

        [TestMethod]
        public void ReshapeTest() {
            float[] u = new float[12] { 7, 10, 2, 5, 11, 3, 8, 9, 4, 12, 1, 6, };
            float[] v = new float[12] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
            float[] err = (new float[12]).Select((_, idx) => u[idx] - v[idx]).ToArray();
            float[] err_p1 = (new float[12]).Select((_, idx) => (u[idx] + 1) - v[idx]).ToArray();

            Tensor intensor = (Shape.Map0D(3, 4), u);
            Tensor outtensor1 = (Shape.Vector(12), v);
            Tensor outtensor2 = (Shape.Map0D(4, 3), v);
            Tensor outtensor3 = (Shape.Map1D(2, 3, 2), v);

            {
                ParameterField fin = intensor;
                ParameterField g1 = outtensor1;
                VariableField g2 = outtensor2;
                VariableField g3 = outtensor3;

                Field ferr1 = Reshape(fin, g1.Shape) - g1;
                Field ferr2 = Reshape(fin, g2.Shape) - g2;
                Field ferr3 = Reshape(fin, g3.Shape) - g3;

                (Flow flow, List<ParameterField> parameters) = Flow.Optimize(ferr1, ferr2, ferr3);

                flow.Execute();

                Assert.AreEqual(28, flow.NodeCount);
                Assert.AreEqual(17, flow.VariableNodeCount);
                Assert.AreEqual(11, flow.FunctionNodeCount);
                Assert.AreEqual(4, flow.InTensorCount);
                Assert.AreEqual(2, flow.OutTensorCount);

                CollectionAssert.AreEqual(err.Select((c) => 3 * c).ToArray(), fin.GradState);
                CollectionAssert.AreEqual(err.Select((c) => -c).ToArray(), g1.GradState);
            }

            {
                ParameterField fin = intensor;
                ParameterField g1 = outtensor1;
                VariableField g2 = outtensor2;
                VariableField g3 = outtensor3;

                Field fp1 = fin + 1;
                Field ferr1 = Reshape(fp1, g1.Shape) - g1;
                Field ferr2 = Reshape(fp1, g2.Shape) - g2;
                Field ferr3 = Reshape(fp1, g3.Shape) - g3;

                (Flow flow, List<ParameterField> parameters) = Flow.Optimize(ferr1, ferr2, ferr3);

                flow.Execute();

                Assert.AreEqual(30, flow.NodeCount);
                Assert.AreEqual(18, flow.VariableNodeCount);
                Assert.AreEqual(12, flow.FunctionNodeCount);
                Assert.AreEqual(4, flow.InTensorCount);
                Assert.AreEqual(2, flow.OutTensorCount);

                CollectionAssert.AreEqual(err_p1.Select((c) => 3 * c).ToArray(), fin.GradState);
                CollectionAssert.AreEqual(err_p1.Select((c) => -c).ToArray(), g1.GradState);
            }

            {
                ParameterField fin = intensor;
                ParameterField g1 = outtensor1;
                VariableField g2 = outtensor2;
                VariableField g3 = outtensor3;

                Field ferr2 = Reshape(fin, g2.Shape) - g2;
                Field ferr1 = Reshape(fin, g1.Shape) - g1;
                Field ferr3 = Reshape(fin, g3.Shape) - g3;

                (Flow flow, List<ParameterField> parameters) = Flow.Optimize(ferr2, ferr1, ferr3);

                flow.Execute();

                Assert.AreEqual(28, flow.NodeCount);
                Assert.AreEqual(17, flow.VariableNodeCount);
                Assert.AreEqual(11, flow.FunctionNodeCount);
                Assert.AreEqual(4, flow.InTensorCount);
                Assert.AreEqual(2, flow.OutTensorCount);

                CollectionAssert.AreEqual(err.Select((c) => 3 * c).ToArray(), fin.GradState);
                CollectionAssert.AreEqual(err.Select((c) => -c).ToArray(), g1.GradState);
            }

            {
                ParameterField fin = intensor;
                ParameterField g1 = outtensor1;
                VariableField g2 = outtensor2;
                VariableField g3 = outtensor3;

                Field fp1 = fin + 1;
                Field ferr2 = Reshape(fp1, g2.Shape) - g2;
                Field ferr1 = Reshape(fp1, g1.Shape) - g1;
                Field ferr3 = Reshape(fp1, g3.Shape) - g3;

                (Flow flow, List<ParameterField> parameters) = Flow.Optimize(ferr2, ferr1, ferr3);

                flow.Execute();

                Assert.AreEqual(30, flow.NodeCount);
                Assert.AreEqual(18, flow.VariableNodeCount);
                Assert.AreEqual(12, flow.FunctionNodeCount);
                Assert.AreEqual(4, flow.InTensorCount);
                Assert.AreEqual(2, flow.OutTensorCount);

                CollectionAssert.AreEqual(err_p1.Select((c) => 3 * c).ToArray(), fin.GradState);
                CollectionAssert.AreEqual(err_p1.Select((c) => -c).ToArray(), g1.GradState);
            }

            {
                ParameterField fin = intensor;
                ParameterField g1 = outtensor1;
                VariableField g2 = outtensor2;
                VariableField g3 = outtensor3;

                Field ferr2 = Reshape(fin, g2.Shape) - g2;
                Field ferr3 = Reshape(fin, g3.Shape) - g3;
                Field ferr1 = Reshape(fin, g1.Shape) - g1;

                (Flow flow, List<ParameterField> parameters) = Flow.Optimize(ferr2, ferr3, ferr1);

                flow.Execute();

                Assert.AreEqual(28, flow.NodeCount);
                Assert.AreEqual(17, flow.VariableNodeCount);
                Assert.AreEqual(11, flow.FunctionNodeCount);
                Assert.AreEqual(4, flow.InTensorCount);
                Assert.AreEqual(2, flow.OutTensorCount);

                CollectionAssert.AreEqual(err.Select((c) => 3 * c).ToArray(), fin.GradState);
                CollectionAssert.AreEqual(err.Select((c) => -c).ToArray(), g1.GradState);
            }

            {
                ParameterField fin = intensor;
                ParameterField g1 = outtensor1;
                VariableField g2 = outtensor2;
                VariableField g3 = outtensor3;

                Field fp1 = fin + 1;
                Field ferr2 = Reshape(fp1, g2.Shape) - g2;
                Field ferr3 = Reshape(fp1, g3.Shape) - g3;
                Field ferr1 = Reshape(fp1, g1.Shape) - g1;

                (Flow flow, List<ParameterField> parameters) = Flow.Optimize(ferr2, ferr3, ferr1);

                flow.Execute();

                Assert.AreEqual(30, flow.NodeCount);
                Assert.AreEqual(18, flow.VariableNodeCount);
                Assert.AreEqual(12, flow.FunctionNodeCount);
                Assert.AreEqual(4, flow.InTensorCount);
                Assert.AreEqual(2, flow.OutTensorCount);

                CollectionAssert.AreEqual(err_p1.Select((c) => 3 * c).ToArray(), fin.GradState);
                CollectionAssert.AreEqual(err_p1.Select((c) => -c).ToArray(), g1.GradState);
            }
        }

        [TestMethod]
        public void DuplicateTest() {
            Tensor intensor = (Shape.Scalar, new float[] { 1.5f });
            Tensor outtensor = (Shape.Scalar, new float[] { 7f });

            /*square*/
            {
                ParameterField fi = intensor;
                VariableField fo = outtensor;

                Field fj = fi + 1;

                Field f3 = fj * fj;
                Field ferr = f3 - fo;
                (Flow flow, List<ParameterField> _) = Flow.Optimize(ferr);

                flow.Execute();

                float[] y = fi.GradState;

                CollectionAssert.AreEqual(new float[] { 2 * 2.5f * (2.5f * 2.5f - 7f) }, y, "x * x");
            }

            /*cube*/
            {
                ParameterField fi = intensor;
                VariableField fo = outtensor;

                Field fj = fi + 1;

                Field f3 = fj * fj * fj;
                Field ferr = f3 - fo;
                (Flow flow, List<ParameterField> _) = Flow.Optimize(ferr);

                flow.Execute();

                float[] y = fi.GradState;

                CollectionAssert.AreEqual(new float[] { 3 * 2.5f * 2.5f * (2.5f * 2.5f * 2.5f - 7f) }, y, "x * x * x");
            }

            /*quad*/
            {
                ParameterField fi = intensor;
                VariableField fo = outtensor;

                Field fj = fi + 1;

                Field f3 = fj * fj * fj * fj;
                Field ferr = f3 - fo;
                (Flow flow, List<ParameterField> _) = Flow.Optimize(ferr);

                flow.Execute();

                float[] y = fi.GradState;

                CollectionAssert.AreEqual(new float[] { 4 * 2.5f * 2.5f * 2.5f * (2.5f * 2.5f * 2.5f * 2.5f - 7f) }, y, "x * x * x * x");
            }

            /*squa_type1*/
            {
                ParameterField fi = intensor;
                VariableField fo = outtensor;

                Field fj = fi + 1;

                Field f3 = fj / fj * fj * fj;
                Field ferr = f3 - fo;
                (Flow flow, List<ParameterField> _) = Flow.Optimize(ferr);

                flow.Execute();

                float[] y = fi.GradState;

                CollectionAssert.AreEqual(new float[] { 2 * 2.5f * (2.5f * 2.5f - 7f) }, y, "x / x * x * x");
            }

            /*squa_type2*/
            {
                ParameterField fi = intensor;
                VariableField fo = outtensor;

                Field fj = fi + 1;

                Field f3 = fj * fj / fj * fj;
                Field ferr = f3 - fo;
                (Flow flow, List<ParameterField> _) = Flow.Optimize(ferr);

                flow.Execute();

                float[] y = fi.GradState;

                CollectionAssert.AreEqual(new float[] { 2 * 2.5f * (2.5f * 2.5f - 7f) }, y, "x * x / x * x");
            }

            /*squa_type3*/
            {
                ParameterField fi = intensor;
                VariableField fo = outtensor;

                Field fj = fi + 1;

                Field f3 = fj * fj * fj / fj;
                Field ferr = f3 - fo;
                (Flow flow, List<ParameterField> _) = Flow.Optimize(ferr);

                flow.Execute();

                float[] y = fi.GradState;

                CollectionAssert.AreEqual(new float[] { 2 * 2.5f * (2.5f * 2.5f - 7f) }, y, "x * x * x / x");
            }
        }

        [TestMethod]
        public void StoreTest() {
            Tensor in1tensor = (Shape.Scalar, new float[] { 1 });
            Tensor in2tensor = (Shape.Scalar, new float[] { 2 });
            Tensor outtensor = (Shape.Scalar, new float[] { -3 });

            {
                ParameterField f1 = new ParameterField(in1tensor);
                VariableField f2 = new VariableField(in2tensor);
                VariableField f4 = new VariableField(in2tensor);
                VariableField fo = new VariableField(outtensor);

                Field f3 = f1 + f2;
                Field ferr = f3 - fo;
                StoreField n3 = f3;
                StoreField n4 = f3 * f4;

                (Flow optimize_flow, List<ParameterField> parameters) = Flow.Optimize(ferr);

                optimize_flow.Execute();

                Assert.AreEqual(13, optimize_flow.NodeCount);
                Assert.AreEqual(10, optimize_flow.VariableNodeCount);
                Assert.AreEqual(3, optimize_flow.FunctionNodeCount);
                Assert.AreEqual(4, optimize_flow.InTensorCount);
                Assert.AreEqual(3, optimize_flow.OutTensorCount);

                CollectionAssert.AreEquivalent(new ParameterField[] { f1 }, parameters);
                Assert.AreEqual(in1tensor, f1.ValueTensor);
                Assert.AreEqual(in2tensor, f2.ValueTensor);
                Assert.AreEqual(in2tensor, f4.ValueTensor);
                Assert.AreEqual(6, f1.GradState[0]);
                Assert.AreEqual(3, n3.State[0]);
                Assert.AreEqual(6, n4.State[0]);
                Assert.AreEqual(null, f2.Grad);
                Assert.AreEqual(null, f4.Grad);
            }

            {
                ParameterField f1 = new ParameterField(in1tensor);
                VariableField f2 = new VariableField(in2tensor);
                VariableField f4 = new VariableField(in2tensor);
                VariableField fo = new VariableField(outtensor);

                Field f3 = f1 + f2;
                Field ferr = f3 - fo;
                StoreField n3 = f3;
                StoreField n4 = f3 + f1;

                (Flow optimize_flow, List<ParameterField> parameters) = Flow.Optimize(ferr);

                optimize_flow.Execute();

                Assert.AreEqual(12, optimize_flow.NodeCount);
                Assert.AreEqual(9, optimize_flow.VariableNodeCount);
                Assert.AreEqual(3, optimize_flow.FunctionNodeCount);
                Assert.AreEqual(3, optimize_flow.InTensorCount);
                Assert.AreEqual(3, optimize_flow.OutTensorCount);

                CollectionAssert.AreEquivalent(new ParameterField[] { f1 }, parameters);
                Assert.AreEqual(in1tensor, f1.ValueTensor);
                Assert.AreEqual(in2tensor, f2.ValueTensor);
                Assert.AreEqual(in2tensor, f4.ValueTensor);
                Assert.AreEqual(6, f1.GradState[0]);
                Assert.AreEqual(3, n3.State[0]);
                Assert.AreEqual(4, n4.State[0]);
                Assert.AreEqual(null, f2.Grad);
                Assert.AreEqual(null, f4.Grad);
            }
        }

        [TestMethod]
        public void BadFlowTest() {
            Tensor intensor = (Shape.Scalar, new float[] { -1.5f });
            Tensor outtensor = (Shape.Scalar, new float[] { 2 });

            {
                ParameterField f1 = new ParameterField(intensor);
                VariableField fo = new VariableField(outtensor);

                Field f2 = Abs(f1);
                Field ferr1 = f2 - fo;

                Field f3 = f2 * f2;
                Field ferr2 = f3 - fo * fo;

                Assert.ThrowsException<ArgumentException>(() => {
                    (Flow flow, List<ParameterField> parameters) = Flow.Optimize(ferr1, ferr1);
                });
            }

            {
                ParameterField f1 = new ParameterField(intensor);
                VariableField fo = new VariableField(outtensor);

                Field f2 = Abs(f1);

                Field f3 = f2 * f2;

                Assert.ThrowsException<ArgumentException>(() => {
                    (Flow flow, List<ParameterField> parameters) = Flow.Optimize(f2, f3);
                });
            }

            {
                ParameterField f1 = new ParameterField(intensor);
                VariableField fo = new VariableField(outtensor);

                Field f2 = Abs(f1);

                Field f3 = f2 * f2;

                Assert.ThrowsException<ArgumentException>(() => {
                    (Flow flow, List<ParameterField> parameters) = Flow.Optimize(f3, f2);
                });
            }
        }
    }
}
