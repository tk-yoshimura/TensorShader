using System.IO;

namespace TensorShaderUtil.GraphVisualization {
    /// <summary>Dotフォーマット出力</summary>
    public static class DotWriter {
        /// <summary>書き込み</summary>
        public static void Write(string filepath, Graph.Node[] nodes, Graph.Edge[] edges) {
            using (var sw = new StreamWriter(filepath)) {
                sw.WriteLine("digraph tensorshader_graph {");
                sw.Write("  graph [\n    charset = \"UTF-8\";\n    labelloc = \"t\",\n    labeljust = \"c\",\n    bgcolor = white,\n" +
                         "    fontcolor = black,\n    fontsize = 18,\n    style = \"filled\",\n" +
                         "    rankdir = TB,\n    margin = 0.2,\n    layout = dot\n  ];\n");
                sw.Write("  node [\n    style = \"solid,filled\",\n    fontsize = 14,\n    fontcolor = black,\n" +
                         "    color = black,\n    fillcolor = white\n  ];\n");
                sw.Write("  edge [\n    color = black\n  ];\n");

                foreach (var node in nodes) {
                    string shape;
                    switch (node.Type) {
                        case Graph.NodeType.Link:
                            shape = "octagon";
                            break;
                        default:
                            shape = "box";
                            break;
                    }

                    sw.Write($"  node{node.ID} [\n    label = \"{node.Name}\",\n    shape = {shape}\n  ];\n");
                }
                foreach (var edge in edges) {
                    sw.WriteLine($"  node{edge.InNode.ID} -> node{edge.OutNode.ID};");
                }
                sw.WriteLine("}");
            }
        }
    }
}
