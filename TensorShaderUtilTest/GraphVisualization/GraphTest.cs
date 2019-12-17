using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using TensorShader.Layers;
using TensorShaderUtil.GraphVisualization;

namespace TensorShaderUtilTest.GraphVisualization {
    [TestClass]
    public class GraphTest {
        [TestMethod]
        public void BuildTest() {
            int inchannels = 4, outchannels = 6, inwidth = 13, inheight = 17, kwidth = 3, kheight = 5, batch = 7;

            VariableField x = new Tensor(Shape.Map2D(inchannels, inwidth, inheight, batch));

            Layer layer = new Convolution2D(inchannels, outchannels, kwidth, kheight, use_bias:true, pad_mode:PaddingMode.Edge, "conv");

            Field y = layer.Forward(x);

            (Graph.Node[] nodes, Graph.Edge[] edges) = Graph.Build(y);

            Assert.AreEqual(9, nodes.Length);
            Assert.AreEqual(8, edges.Length);

            foreach(var node in nodes) {
                Console.WriteLine(node.Name);
            }
            foreach(var edge in edges) {
                Console.WriteLine($"{edge.InNode.Name} -> {edge.OutNode.Name}");
            }

            DotWriter.Write("test.dot", nodes, edges);

            Graph.WriteDotFile("test2.dot", y);

            (Flow flow, Parameters parameters) = Flow.Optimize(y);
            flow.Execute();

            Assert.AreEqual(2, parameters.Count);
            Assert.AreEqual(inchannels, layer.InChannels);
            Assert.AreEqual(outchannels, layer.OutChannels);
            Assert.AreEqual(kwidth, layer.Width);
            Assert.AreEqual(kheight, layer.Height);
        }
    }
}
