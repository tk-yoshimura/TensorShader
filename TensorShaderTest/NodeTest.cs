using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;

namespace TensorShaderTest {
    internal class DummyFunction : Function {
        public DummyFunction(int inputs, int outputs)
            : base(inputs, outputs, allow_resubstitution: false) { }

        internal override (Tensor[], Operator) GenerateOperator(Tensor[] intensors, Tensor[] outtensors) {
            throw new NotImplementedException();
        }

        public override Shape[] OutputShapes(params Shape[] inshapes) {
            throw new NotImplementedException();
        }
    }

    [TestClass]
    public class NodeTest {
        [TestMethod]
        public void TreeTest() {
            // i1 -> f2 -> t3
            //            (t3, i4) -> f5 -> t6 -> f7 -> (t8, t9)
            //                                           t8 -> o10
            //                                           t9 -> o11
            //                              t6 -> o12
            InputNode i1 = new InputNode(Shape.Scalar);

            VariableNode t3 = new TemporaryNode(Shape.Scalar);
            FunctionNode f2 = new FunctionNode(new DummyFunction(1, 1), new VariableNode[] { i1 }, new VariableNode[] { t3 });

            InputNode i4 = new InputNode(Shape.Scalar);

            VariableNode t6 = new TemporaryNode(Shape.Scalar);
            FunctionNode f5 = new FunctionNode(new DummyFunction(2, 1), new VariableNode[] { t3, i4 }, new VariableNode[] { t6 });

            VariableNode t8 = new TemporaryNode(Shape.Scalar);
            VariableNode t9 = new TemporaryNode(Shape.Scalar);
            FunctionNode f7 = new FunctionNode(new DummyFunction(1, 2), new VariableNode[] { t6 }, new VariableNode[] { t8, t9 });

            VariableNode o10 = t8.Save();
            VariableNode o11 = t9.Save();
            VariableNode o12 = t6.Save();

            CollectionAssert.AreEqual(new Node[] { }, i1.InNodes.ToArray());
            CollectionAssert.AreEqual(new Node[] { f2 }, i1.OutNodes.ToArray());

            CollectionAssert.AreEqual(new Node[] { i1 }, f2.InNodes.ToArray());
            CollectionAssert.AreEqual(new Node[] { t3 }, f2.OutNodes.ToArray());

            CollectionAssert.AreEqual(new Node[] { f2 }, t3.InNodes.ToArray());
            CollectionAssert.AreEqual(new Node[] { f5 }, t3.OutNodes.ToArray());

            CollectionAssert.AreEqual(new Node[] { }, i4.InNodes.ToArray());
            CollectionAssert.AreEqual(new Node[] { f5 }, i4.OutNodes.ToArray());

            CollectionAssert.AreEqual(new Node[] { t3, i4 }, f5.InNodes.ToArray());
            CollectionAssert.AreEqual(new Node[] { t6 }, f5.OutNodes.ToArray());

            CollectionAssert.AreEqual(new Node[] { f5 }, t6.InNodes.ToArray());
            CollectionAssert.AreEqual(new Node[] { f7, o12 }, t6.OutNodes.ToArray());

            CollectionAssert.AreEqual(new Node[] { t6 }, f7.InNodes.ToArray());
            CollectionAssert.AreEqual(new Node[] { t8, t9 }, f7.OutNodes.ToArray());

            CollectionAssert.AreEqual(new Node[] { f7 }, t8.InNodes.ToArray());
            CollectionAssert.AreEqual(new Node[] { o10 }, t8.OutNodes.ToArray());

            CollectionAssert.AreEqual(new Node[] { f7 }, t9.InNodes.ToArray());
            CollectionAssert.AreEqual(new Node[] { o11 }, t9.OutNodes.ToArray());

            CollectionAssert.AreEqual(new Node[] { t8 }, o10.InNodes.ToArray());
            CollectionAssert.AreEqual(new Node[] { }, o10.OutNodes.ToArray());

            CollectionAssert.AreEqual(new Node[] { t9 }, o11.InNodes.ToArray());
            CollectionAssert.AreEqual(new Node[] { }, o11.OutNodes.ToArray());

            CollectionAssert.AreEqual(new Node[] { t6 }, o12.InNodes.ToArray());
            CollectionAssert.AreEqual(new Node[] { }, o12.OutNodes.ToArray());
        }

        [TestMethod]
        public void DirectTest() {
            // i1 -> f2 -> t3 -> o4
            InputNode i1 = new InputNode(Shape.Scalar);
            VariableNode t3 = new TemporaryNode(Shape.Scalar);
            FunctionNode f2 = new FunctionNode(new DummyFunction(1, 1), new VariableNode[] { i1 }, new VariableNode[] { t3 });
            VariableNode o4 = t3.Save();

            List<Node> nodes_sorted = Flow.ExecutionOrderSort(new List<InputNode> { i1 });

            CollectionAssert.AreEqual(new Node[] { i1, f2, t3, o4 }, nodes_sorted);
        }

        [TestMethod]
        public void LoopTest() {
            // i1 -> f2 -> t3 -> o4
            //             t3 -> f5 -> t3
            InputNode i1 = new InputNode(Shape.Scalar);
            VariableNode t3 = new TemporaryNode(Shape.Scalar);
            _ = new FunctionNode(new DummyFunction(1, 1), new VariableNode[] { i1 }, new VariableNode[] { t3 });
            _ = t3.Save();
            _ = new FunctionNode(new DummyFunction(1, 1), new VariableNode[] { t3 }, new VariableNode[] { t3 });

            Assert.ThrowsException<ArgumentException>(
                () => {
                    List<Node> nodes_sorted = Flow.ExecutionOrderSort(new List<InputNode> { i1 });
                }
            );
        }

        [TestMethod]
        public void In1Out2Test() {
            // i1 -> f2 -> t3 -> o4
            //             t3 -> f5 -> t6 -> o7
            InputNode i1 = new InputNode(Shape.Scalar);

            VariableNode t3 = new TemporaryNode(Shape.Scalar);
            FunctionNode f2 = new FunctionNode(new DummyFunction(1, 1), new VariableNode[] { i1 }, new VariableNode[] { t3 });

            VariableNode o4 = t3.Save();

            VariableNode t6 = new TemporaryNode(Shape.Scalar);
            FunctionNode f5 = new FunctionNode(new DummyFunction(1, 1), new VariableNode[] { t3 }, new VariableNode[] { t6 });

            VariableNode o7 = t6.Save();

            List<Node> nodes_sorted = Flow.ExecutionOrderSort(new List<InputNode> { i1 });

            CollectionAssert.AreEqual(new Node[] { i1, f2, t3, o4, f5, t6, o7 }, nodes_sorted);
        }

        [TestMethod]
        public void In2Out1Test() {
            // i1 -> f2 -> t3
            // i4 -> f5 -> t6
            //             (t3, t6) -> f7 -> t8 -> o9
            InputNode i1 = new InputNode(Shape.Scalar);

            VariableNode t3 = new TemporaryNode(Shape.Scalar);
            FunctionNode f2 = new FunctionNode(new DummyFunction(1, 1), new VariableNode[] { i1 }, new VariableNode[] { t3 });

            InputNode i4 = new InputNode(Shape.Scalar);

            VariableNode t6 = new TemporaryNode(Shape.Scalar);
            FunctionNode f5 = new FunctionNode(new DummyFunction(1, 1), new VariableNode[] { i4 }, new VariableNode[] { t6 });

            VariableNode t8 = new TemporaryNode(Shape.Scalar);
            FunctionNode f7 = new FunctionNode(new DummyFunction(2, 1), new VariableNode[] { t3, t6 }, new VariableNode[] { t8 });

            VariableNode o9 = t8.Save();

            List<Node> nodes_sorted = Flow.ExecutionOrderSort(new List<InputNode> { i1, i4 });

            CollectionAssert.AreEqual(new Node[] { i1, f2, t3, i4, f5, t6, f7, t8, o9 }, nodes_sorted);
        }

        [TestMethod]
        public void In2Out2Test() {
            // i1 -> f2 -> t3
            // i4 -> f5 -> t6
            //             (t3, t6) -> f7 -> (t8, t10)
            //                                t8 -> o9
            //                                t10 -> o11
            InputNode i1 = new InputNode(Shape.Scalar);

            VariableNode t3 = new TemporaryNode(Shape.Scalar);
            FunctionNode f2 = new FunctionNode(new DummyFunction(1, 1), new VariableNode[] { i1 }, new VariableNode[] { t3 });

            InputNode i4 = new InputNode(Shape.Scalar);

            VariableNode t6 = new TemporaryNode(Shape.Scalar);
            FunctionNode f5 = new FunctionNode(new DummyFunction(1, 1), new VariableNode[] { i4 }, new VariableNode[] { t6 });

            VariableNode t8 = new TemporaryNode(Shape.Scalar);
            VariableNode t10 = new TemporaryNode(Shape.Scalar);
            FunctionNode f7 = new FunctionNode(new DummyFunction(2, 2), new VariableNode[] { t3, t6 }, new VariableNode[] { t8, t10 });

            VariableNode o9 = t8.Save();
            VariableNode o11 = t10.Save();

            List<Node> nodes_sorted = Flow.ExecutionOrderSort(new List<InputNode> { i1, i4 });

            CollectionAssert.AreEqual(new Node[] { i1, f2, t3, i4, f5, t6, f7, t8, o9, t10, o11 }, nodes_sorted);
        }

        [TestMethod]
        public void In2Out2BridgeTest() {
            // i1 -> f2 -> t3
            // i4 -> f5 -> t6
            //             (t3, t6) -> f7 -> t8 -> f9 -> (t10, t12)
            //                                            t10 -> o11
            //                                            t12 -> o13
            InputNode i1 = new InputNode(Shape.Scalar);

            VariableNode t3 = new TemporaryNode(Shape.Scalar);
            FunctionNode f2 = new FunctionNode(new DummyFunction(1, 1), new VariableNode[] { i1 }, new VariableNode[] { t3 });

            InputNode i4 = new InputNode(Shape.Scalar);

            VariableNode t6 = new TemporaryNode(Shape.Scalar);
            FunctionNode f5 = new FunctionNode(new DummyFunction(1, 1), new VariableNode[] { i4 }, new VariableNode[] { t6 });

            VariableNode t8 = new TemporaryNode(Shape.Scalar);
            FunctionNode f7 = new FunctionNode(new DummyFunction(2, 1), new VariableNode[] { t3, t6 }, new VariableNode[] { t8 });

            VariableNode t10 = new TemporaryNode(Shape.Scalar);
            VariableNode t12 = new TemporaryNode(Shape.Scalar);
            FunctionNode f9 = new FunctionNode(new DummyFunction(1, 2), new VariableNode[] { t8 }, new VariableNode[] { t10, t12 });

            VariableNode o11 = t10.Save();
            VariableNode o13 = t12.Save();

            List<Node> nodes_sorted = Flow.ExecutionOrderSort(new List<InputNode> { i1, i4 });

            CollectionAssert.AreEqual(new Node[] { i1, f2, t3, i4, f5, t6, f7, t8, f9, t10, o11, t12, o13 }, nodes_sorted);
        }

        [TestMethod]
        public void RejoinTest() {
            // i1 -> f2 -> (t3, t6)
            //              t3 -> f4 -> t5
            //              t6 -> f7 -> t8
            //                          (t5, t8) -> f9 -> t10 -> o11
            InputNode i1 = new InputNode(Shape.Scalar);

            VariableNode t3 = new TemporaryNode(Shape.Scalar);
            VariableNode t6 = new TemporaryNode(Shape.Scalar);
            FunctionNode f2 = new FunctionNode(new DummyFunction(1, 2), new VariableNode[] { i1 }, new VariableNode[] { t3, t6 });

            VariableNode t5 = new TemporaryNode(Shape.Scalar);
            FunctionNode f4 = new FunctionNode(new DummyFunction(1, 1), new VariableNode[] { t3 }, new VariableNode[] { t5 });

            VariableNode t8 = new TemporaryNode(Shape.Scalar);
            FunctionNode f7 = new FunctionNode(new DummyFunction(1, 1), new VariableNode[] { t6 }, new VariableNode[] { t8 });

            VariableNode t10 = new TemporaryNode(Shape.Scalar);
            FunctionNode f9 = new FunctionNode(new DummyFunction(2, 1), new VariableNode[] { t5, t8 }, new VariableNode[] { t10 });

            VariableNode o11 = t10.Save();

            List<Node> nodes_sorted = Flow.ExecutionOrderSort(new List<InputNode> { i1 });

            CollectionAssert.AreEqual(new Node[] { i1, f2, t3, f4, t5, t6, f7, t8, f9, t10, o11 }, nodes_sorted);
        }

        [TestMethod]
        public void DoubleRejoinTest() {
            // i1 -> f2 -> (t3, t6)
            //              t3 -> f4 -> t5
            // i7 -> f8 -> (t9, t12)
            //              t9 -> f10 -> t11
            //              (t6, t12) -> f13 -> (t14, t17)
            //                                   ( t5, t14) -> f15 -> t16
            //                                   (t11, t17) -> f18 -> t19
            //                                                        (t16, t19) -> f20 -> t21 -> o22
            InputNode i1 = new InputNode(Shape.Scalar);

            VariableNode t3 = new TemporaryNode(Shape.Scalar);
            VariableNode t6 = new TemporaryNode(Shape.Scalar);
            FunctionNode f2 = new FunctionNode(new DummyFunction(1, 2), new VariableNode[] { i1 }, new VariableNode[] { t3, t6 });

            VariableNode t5 = new TemporaryNode(Shape.Scalar);
            FunctionNode f4 = new FunctionNode(new DummyFunction(1, 1), new VariableNode[] { t3 }, new VariableNode[] { t5 });

            InputNode i7 = new InputNode(Shape.Scalar);

            VariableNode t9 = new TemporaryNode(Shape.Scalar);
            VariableNode t12 = new TemporaryNode(Shape.Scalar);
            FunctionNode f8 = new FunctionNode(new DummyFunction(1, 2), new VariableNode[] { i7 }, new VariableNode[] { t9, t12 });

            VariableNode t11 = new TemporaryNode(Shape.Scalar);
            FunctionNode f10 = new FunctionNode(new DummyFunction(1, 1), new VariableNode[] { t9 }, new VariableNode[] { t11 });

            VariableNode t14 = new TemporaryNode(Shape.Scalar);
            VariableNode t17 = new TemporaryNode(Shape.Scalar);
            FunctionNode f13 = new FunctionNode(new DummyFunction(2, 2), new VariableNode[] { t6, t12 }, new VariableNode[] { t14, t17 });

            VariableNode t16 = new TemporaryNode(Shape.Scalar);
            FunctionNode f15 = new FunctionNode(new DummyFunction(2, 1), new VariableNode[] { t5, t14 }, new VariableNode[] { t16 });

            VariableNode t19 = new TemporaryNode(Shape.Scalar);
            FunctionNode f18 = new FunctionNode(new DummyFunction(2, 1), new VariableNode[] { t11, t17 }, new VariableNode[] { t19 });

            VariableNode t21 = new TemporaryNode(Shape.Scalar);
            FunctionNode f20 = new FunctionNode(new DummyFunction(2, 1), new VariableNode[] { t16, t19 }, new VariableNode[] { t21 });

            VariableNode o22 = t21.Save();

            List<Node> nodes_sorted = Flow.ExecutionOrderSort(new List<InputNode> { i1, i7 });

            CollectionAssert.AreEqual(new Node[] { i1, f2, t3, f4, t5, t6, i7, f8, t9, f10, t11, t12, f13, t14, f15, t16, t17, f18, t19, f20, t21, o22 }, nodes_sorted);
        }

        [TestMethod]
        public void ResidualConnectionTest() {
            // i1 -> f2 -> t3 -> f4 -> t5 -> f6 -> t7
            //                                    (t5, t7) -> f8 -> t9
            //                                                     (t3, t9) -> f10 -> t11
            //                                                                       (i1, t11) -> f12 -> t13 -> o14
            InputNode i1 = new InputNode(Shape.Scalar);

            VariableNode t3 = new TemporaryNode(Shape.Scalar);
            FunctionNode f2 = new FunctionNode(new DummyFunction(1, 1), new VariableNode[] { i1 }, new VariableNode[] { t3 });

            VariableNode t5 = new TemporaryNode(Shape.Scalar);
            FunctionNode f4 = new FunctionNode(new DummyFunction(1, 1), new VariableNode[] { t3 }, new VariableNode[] { t5 });

            VariableNode t7 = new TemporaryNode(Shape.Scalar);
            FunctionNode f6 = new FunctionNode(new DummyFunction(1, 1), new VariableNode[] { t5 }, new VariableNode[] { t7 });

            VariableNode t9 = new TemporaryNode(Shape.Scalar);
            FunctionNode f8 = new FunctionNode(new DummyFunction(2, 1), new VariableNode[] { t5, t7 }, new VariableNode[] { t9 });

            VariableNode t11 = new TemporaryNode(Shape.Scalar);
            FunctionNode f10 = new FunctionNode(new DummyFunction(2, 1), new VariableNode[] { t3, t9 }, new VariableNode[] { t11 });

            VariableNode t13 = new TemporaryNode(Shape.Scalar);
            FunctionNode f12 = new FunctionNode(new DummyFunction(2, 1), new VariableNode[] { i1, t11 }, new VariableNode[] { t13 });

            VariableNode o14 = t13.Save();

            List<Node> nodes_sorted = Flow.ExecutionOrderSort(new List<InputNode> { i1 });

            CollectionAssert.AreEqual(new Node[] { i1, f2, t3, f4, t5, f6, t7, f8, t9, f10, t11, f12, t13, o14 }, nodes_sorted);
        }
    }
}
