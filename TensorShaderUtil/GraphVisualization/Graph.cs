using System.Collections.Generic;
using System.Linq;
using TensorShader;

namespace TensorShaderUtil.GraphVisualization {
    /// <summary>計算グラフ</summary>
    public static class Graph {
        /// <summary>ノード</summary>
        public class Node {
            /// <summary>ID</summary>
            public int ID{ set; get; }

            /// <summary>ノード名</summary>
            public string Name { set; get; }
        }

        /// <summary>エッジ</summary>
        public class Edge {
            /// <summary>入力ノード</summary>
            public Node InNode { set; get; }

            /// <summary>出力ノード</summary>
            public Node OutNode { set; get; }
        }

        /// <summary>出力フィールドから構築</summary>
        /// <param name="outputs">出力フィールドリスト</param>
        public static (Node[] nodes, Edge[] edges) Build(params Field[] outputs) {
            Dictionary<Field, Node> visited_field = new Dictionary<Field, Node>();
            Dictionary<Link, Node> visited_link = new Dictionary<Link, Node>();
            List<Edge> edges = new List<Edge>();

            int id = 0;

            Stack<Field> stack = new Stack<Field>(outputs);
            foreach(Field field in outputs) {
                visited_field.Add(field, new Node(){ ID = id++, Name = field.Name });
            }

            while(stack.Count > 0) {
                Field field_current = stack.Pop();

                Link link = field_current.OutLink;

                if (link == null) {
                    continue;
                }

                if (!visited_link.ContainsKey(link)) {
                    visited_link.Add(link, new Node(){ ID = id++, Name = link.Name });
                }

                edges.Add(new Edge() { InNode = visited_link[link], OutNode = visited_field[field_current] });

                foreach(Field field in link.InFields) {
                    if (!visited_field.ContainsKey(field)) {
                        stack.Push(field);
                        visited_field.Add(field, new Node(){ ID = id++, Name = field.Name });
                    }

                    edges.Add(new Edge() { InNode = visited_field[field], OutNode = visited_link[link] });
                }
            }

            var nodes = visited_field.Values.Concat(visited_link.Values);

            return (nodes.Reverse().ToArray(), ((IEnumerable<Edge>)edges).Reverse().ToArray());
        }

        /// <summary>Dotファイルに出力</summary>
        /// <param name="filepath">ファイルパス</param>
        /// <param name="outputs">出力フィールドリスト</param>
        public static void WriteDotFile(string filepath, params Field[] outputs) {
            (Node[] nodes, Edge[] edges) = Build(outputs);
            DotWriter.Write(filepath, nodes, edges);
        }
    }
}
