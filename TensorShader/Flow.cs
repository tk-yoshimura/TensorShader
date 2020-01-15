using System;
using System.Collections.Generic;
using System.Linq;
using TensorShaderCudaBackend;

namespace TensorShader {
    /// <summary>計算フロー</summary>
    public class Flow {
        private readonly List<Node> nodes;
        private readonly Dictionary<Node, Tensor> tensors;
        private readonly Dictionary<InputNode, Tensor> intensors;
        private readonly Dictionary<OutputNode, Tensor> outtensors;

        /// <summary>構成ノードリスト</summary>
        public IReadOnlyList<Node> Nodes => nodes.AsReadOnly();

        /// <summary>構成ノード数</summary>
        public int NodeCount => nodes.Count;

        /// <summary>変数構成ノード数</summary>
        public int VariableNodeCount => nodes.Count((node) => node is VariableNode);

        /// <summary>関数構成ノード数</summary>
        public int FunctionNodeCount => nodes.Count((node) => node is FunctionNode);

        /// <summary>保有テンソル数</summary>
        public int TensorCount => tensors.Values.Distinct().Count();

        /// <summary>保有バッファ数</summary>
        public int BufferCount => tensors.Select((tensor) => tensor.Value.Buffer).Distinct().Count();

        /// <summary>保有バッファ総要素数</summary>
        public ulong BufferSize {
            get {
                ulong size_sum = 0;

                foreach (ulong size in tensors.Select((tensor) => tensor.Value.Buffer).Distinct().Select((buffer) => buffer.Length)) {
                    size_sum += size;
                }

                return size_sum;
            }
        }

        /// <summary>入力テンソルテーブル</summary>
        public IReadOnlyDictionary<InputNode, Tensor> InTensors => intensors;

        /// <summary>入力テンソル数</summary>
        public int InTensorCount => intensors.Count;

        /// <summary>出力テンソルテーブル</summary>
        public IReadOnlyDictionary<OutputNode, Tensor> OutTensors => outtensors;

        /// <summary>出力テンソル数</summary>
        public int OutTensorCount => outtensors.Count;

        /// <summary>上限構成ノード数(2^30)</summary>
        public static int MaxFlowSize => 0x40000000;

        /// <summary>コンストラクタ</summary>
        /// <param name="nodes">ノードリスト</param>
        protected internal Flow(List<Node> nodes) {
            if (nodes == null) {
                throw new ArgumentException(nameof(nodes));
            }

            this.nodes = nodes;

            var lifespan = LifeSpan(this.nodes);

            this.tensors = AssignTensor(lifespan);

            AssignOperator();

            this.intensors = nodes.OfType<InputNode>()
                                   .ToDictionary((node) => node, (node) => node.Tensor);

            this.outtensors = nodes.OfType<OutputNode>()
                                   .ToDictionary((node) => node, (node) => node.Tensor);
        }

        /// <summary>入力ノードからフロー構築</summary>
        /// <param name="innodes">入力ノードリスト</param>
        public static Flow FromInputs(params InputNode[] innodes) {
            if (innodes.Length < 1 || innodes.IsDuplicated()) {
                throw new ArgumentException(nameof(innodes));
            }

            List<Node> nodes = ExecutionOrderSort(innodes);

            return new Flow(nodes);
        }

        /// <summary>出力ノードからフロー構築</summary>
        /// <param name="outnodes">出力ノードリスト</param>
        public static Flow FromOutputs(params OutputNode[] outnodes) {
            if (outnodes.Length < 1 || outnodes.IsDuplicated()) {
                throw new ArgumentException(nameof(outnodes));
            }

            (List<Node> tracenodes, List<InputNode> innodes) = BackTrace(outnodes);

            List<Node> nodes = ExecutionOrderSort(innodes, tracenodes);

            return new Flow(nodes);
        }

        /// <summary>実行順ソート</summary>
        /// <param name="innodes">入力ノードリスト</param>
        /// <param name="tracenodes">フロー関連ノード</param>
        /// <remarks>フロー関連ノードがnullであるなら入力ノードから到達できるノードをフロー関連ノードとする</remarks>
        internal static List<Node> ExecutionOrderSort(IEnumerable<InputNode> innodes, List<Node> tracenodes = null) {
            List<Tensor> input_tensors =
                innodes
                .Where((node) => node.Tensor != null)
                .Select((node) => node.Tensor)
                .ToList();

            Stack<Node> nodestack = new Stack<Node>(innodes.Reverse().Select((node) => node as Node));
            Dictionary<Node, List<Node>> innodes_table = new Dictionary<Node, List<Node>>();
            List<OutputNode> shared_tensor_output_nodes = new List<OutputNode>();

            List<Node> visited_nodes = new List<Node>();

            // 始端入力ノードから深さ優先探索
            while (nodestack.Count > 0) {
                // 探索ノードをポップ
                Node currentnode = nodestack.Pop();

                // 探索ノードの出力ノードを列挙
                var outnodes = currentnode.OutNodes.Reverse();

                if (visited_nodes.Contains(currentnode) || outnodes.Where((node) => node is InputNode).Count() > 0) {
                    throw new ArgumentException("Node list including circulation path.");
                }

                // 終端出力ノードを優先的に探索
                outnodes = outnodes.Where((node) => node is TemporaryNode || node is FunctionNode)
                    .Concat(
                        outnodes.Where((node) => node is OutputNode)
                    );

                foreach (var outnode in outnodes) {
                    // フローに関与するノードに含まれていない場合はスキップ
                    if (tracenodes != null && !tracenodes.Contains(outnode)) {
                        continue;
                    }

                    // 出力ノードの入力ノードリストから探索ノードを削除
                    if (!innodes_table.ContainsKey(outnode)) {
                        innodes_table.Add(outnode, new List<Node>(outnode.InNodes));
                    }

                    innodes_table[outnode].Remove(currentnode);

                    // 始端入力ノードと終端出力ノードがテンソルを共有している場合、後回し
                    // 始端入力ノードが受け取られる前に上書きされるのを防ぐ
                    if (outnode is OutputNode) {
                        Tensor outtensor = (outnode as OutputNode).Tensor;
                        if (outtensor != null && input_tensors.Contains(outtensor)) {
                            shared_tensor_output_nodes.Add(outnode as OutputNode);
                            continue;
                        }
                    }

                    // 出力ノードの入力ノードリストがすべて探索済みならば、出力ノードを探索対象とする
                    if (innodes_table[outnode].Count <= 0) {
                        nodestack.Push(outnode);
                    }
                }

                visited_nodes.Add(currentnode);
            }

            // 後回しにしていた終端出力ノードを探索済みノードに加える
            visited_nodes.AddRange(shared_tensor_output_nodes.OrderByDescending((node) => node.InNode is InputNode));

            // 関数ノードに到達不可な入力ノードがあるなら例外を送出
            if (innodes_table.Select((item) => item.Value.Count).Any((v) => v > 0)) {
                throw new ArgumentException("Node list including unreachable nodes.");
            }

            // 計算フローが大きすぎる場合は例外を送出
            if (visited_nodes.Count > MaxFlowSize) {
                throw new ArgumentException("Node list is too long.");
            }

            // 出力ノードのテンソルが重複しているなら例外を送出
            IEnumerable<Tensor> output_tensors = visited_nodes
                .OfType<OutputNode>()
                .Where((node) => node.Tensor != null)
                .Select((node) => node.Tensor);

            if (output_tensors.IsDuplicated()) {
                throw new ArgumentException("Node list including duplicate output tensor.");
            }

            // テンソルが重複している入力ノードが出力ノードとテンソルを共有しているなら例外を送出
            if (input_tensors.Duplicated().Intersect(output_tensors).Count() > 0) {
                throw new ArgumentException("Input nodes with duplicate tensors share tensors with output nodes.");
            }

            return visited_nodes;
        }

        /// <summary>出力ノードからバックトレース</summary>
        /// <param name="outnodes">出力ノード</param>
        internal static (List<Node> nodes, List<InputNode> innodes) BackTrace(params OutputNode[] outnodes) {
            Stack<Node> nodestack = new Stack<Node>(outnodes.Reverse().Select((node) => node as Node));
            List<Node> visited_nodes = new List<Node>();

            while (nodestack.Count > 0) {
                // 探索ノードをポップ
                Node currentnode = nodestack.Pop();

                // 探索済ノードに追加
                visited_nodes.Add(currentnode);

                // 入力ノードを探索
                foreach (Node innode in currentnode.InNodes) {
                    if (visited_nodes.Contains(innode) || nodestack.Contains(innode)) {
                        continue;
                    }
                    nodestack.Push(innode);
                }
            }

            List<InputNode> innodes = visited_nodes.OfType<InputNode>().ToList();

            return (visited_nodes, innodes);
        }

        /// <summary>ノードごとの生存期間</summary>
        protected internal static Dictionary<VariableNode, (int begin, int end)> LifeSpan(List<Node> nodes) {
            List<Tensor> input_tensors
                = nodes
                .OfType<InputNode>()
                .Select((node) => node.Tensor)
                .Where((tensor) => tensor != null)
                .ToList();

            (int begin, int end) lifespan(VariableNode node) {
                // 入力ノードの生存期間は無制限
                if (node is InputNode) {
                    return (0, int.MaxValue);
                }

                // 出力ノードの生存期間は入力ノードとテンソルを共有している限り無制限
                if (node is OutputNode outputnode) {
                    if (outputnode.Tensor == null || !input_tensors.Contains(outputnode.Tensor)) {
                        return (nodes.IndexOf(node), int.MaxValue);
                    }
                    else {
                        return (0, int.MaxValue);
                    }
                }

                // 生存期間の始点はノードが生成された時点
                int begin = node.InNodes.Select((innode) => nodes.IndexOf(innode)).Min();

                // 生存期間の終点はノードが最後に参照された時点
                int end = node.OutNodes
                    .Select((outnode) => nodes.IndexOf(outnode)
                                - ((outnode is VariableNode)
                                    || (outnode is FunctionNode && (outnode as FunctionNode).Function.AllowResubstitution) ? 1 : 0))
                    .Concat(new int[] { begin }).Max();

                return (begin, end);
            }

            return nodes.OfType<VariableNode>()
                        .ToDictionary((node) => node, (node) => lifespan(node));
        }

        /// <summary>テンソル対応付け</summary>
        protected internal static Dictionary<Node, Tensor> AssignTensor(Dictionary<VariableNode, (int begin, int end)> nodes_lifespan) {

            var buffers_list = new List<(int begin, int end, int length)>();
            var bufferid_table = new Dictionary<VariableNode, int>();
            var tensor_table = new Dictionary<Node, Tensor>();

            foreach (var node_lifespan in nodes_lifespan) {
                var node = node_lifespan.Key;
                var lifespan = node_lifespan.Value;

                //すでに割り当てられているならばスキップ
                if (node is TensorNode tensornode && tensornode.Tensor != null) {
                    tensor_table.Add(node, tensornode.Tensor);
                    continue;
                }

                //生存期間が重ならないバッファIDを検索
                for (int id = 0; id < buffers_list.Count; id++) {

                    //生存期間が重ならないならばノードにバッファIDを割り当て
                    if (buffers_list[id].end < lifespan.begin) {
                        int new_length = Math.Max(node.Shape.Length, buffers_list[id].length);
                        (int begin, int end, int length) new_span = (buffers_list[id].begin, lifespan.end, new_length);

                        bufferid_table.Add(node, id);
                        buffers_list[id] = new_span;
                        break;
                    }
                }

                //生存期間が重ならないバッファIDが存在しないならば新規バッファIDを生成
                if (!bufferid_table.ContainsKey(node)) {
                    bufferid_table.Add(node, buffers_list.Count);
                    buffers_list.Add((lifespan.begin, lifespan.end, node.Shape.Length));
                }
            }

            List<CudaArray<float>> buffers = buffers_list.Select((item) => (CudaArray<float>)(new float[item.length])).ToList();

            // テンソルテーブル確定
            foreach (var item in bufferid_table) {
                VariableNode node = item.Key;
                int buffer_id = item.Value;

                Tensor tensor = new TemporaryTensor(node.Shape, buffers[buffer_id]);
                tensor_table.Add(node, tensor);

                //テンソルノードならばテンソルを割り当て
                if (node is TensorNode tensornode) {
                    tensornode.AssignTensor(tensor);
                }
            }

            return tensor_table;
        }

        /// <summary>操作クラス対応付け</summary>
        protected internal void AssignOperator() {
            foreach (var funcnode in nodes.OfType<FunctionNode>()) {
                Tensor[] intensors = funcnode.InNodes.Select((innode) => tensors[innode]).ToArray();
                Tensor[] outtensors = funcnode.OutNodes.Select((innode) => tensors[innode]).ToArray();

                funcnode.AssignOperator(intensors, outtensors);
            }
        }

        /// <summary>到達可能なリンクおよびフィールドを列挙</summary>
        public static (List<Field> fields, List<Link> links) EnumerateReachableFields(bool forward, bool backward, params Field[] fields) {
            List<Link> reachable_links = new List<Link>();
            List<Field> reachable_fields = new List<Field>(fields.Distinct());
            Stack<Field> stack = new Stack<Field>(fields);

            while(stack.Count > 0) {
                Field field = stack.Pop();

                //探索フィールドを入力とするリンクを探索
                if (forward) {
                    foreach(Link link in field.InLinks) {
                        if (!reachable_links.Contains(link)) {
                            reachable_links.Add(link);
                        }

                        if (link.OutField != null && !reachable_fields.Contains(link.OutField)) {
                            stack.Push(link.OutField);
                            reachable_fields.Add(link.OutField);
                        }
                    }
                }
                //探索フィールドを出力とするリンクを探索
                if (backward) {
                    if(field.OutLink != null) {
                        if (!reachable_links.Contains(field.OutLink)) {
                            reachable_links.Add(field.OutLink);
                        }

                        foreach(Field push_field in field.OutLink.InFields) {
                            if (!reachable_fields.Contains(push_field)) {
                                stack.Push(push_field);
                                reachable_fields.Add(push_field);
                            }
                        }
                    }
                }
            }

            return (reachable_fields, reachable_links);
        }

        /// <summary>最適化計算グラフを構築</summary>
        /// <param name="error_fields">誤差フィールドのリスト</param>
        /// <returns>計算フローと最適化対象のパラメータ</returns>
        public static (Flow flow, Parameters parameters) Optimize(params Field[] error_fields) {
            if (error_fields.Length < 1) {
                throw new ArgumentException(nameof(error_fields));
            }

            if (error_fields.IsDuplicated()) {
                throw new ArgumentNullException("Error fields are duplicated.");
            }

            foreach (Field error_field in error_fields) {
                error_field.AddGrad(error_field.Value);
            }

            // 逆伝搬で到達可能なリンク・フィールドを探索、パラメータを列挙
            (List<Field> backward_reachable_fields, List<Link> backward_reachable_links) =
                EnumerateReachableFields(forward: false, backward: true, error_fields);
            List<ParameterField> parameters = backward_reachable_fields.OfType<ParameterField>().ToList();

            // 逆伝搬実行
            Stack<Field> backward_stack = new Stack<Field>(error_fields);
            Dictionary<Field, List<Link>> outlinks_table = new Dictionary<Field, List<Link>>();

            while (backward_stack.Count > 0) {
                Field field = backward_stack.Pop();

                if (field.OutLink != null) {
                    field.OutLink.Backward();

                    // 探索フィールドを出力したリンクの順伝搬時の入力フィールドを検索
                    foreach (Field push_field in field.OutLink.InFields) {
                        // 入力フィールドのリンクがすべて逆伝搬済みかチェック
                        if (!outlinks_table.ContainsKey(push_field)) {
                            outlinks_table.Add(
                                push_field,
                                push_field.InLinks.WhiteList(backward_reachable_links).ToList()
                            );
                        }

                        outlinks_table[push_field].Remove(field.OutLink);

                        if (outlinks_table[push_field].Count > 0) {
                            continue;
                        }

                        // 逆伝搬に必要な誤差変数がすべて準備できたら勾配を確定しスタックに追加
                        backward_stack.Push(push_field);
                    }
                }
            }

            // パラメータの勾配確定
            foreach (ParameterField parameter in parameters) {
                parameter.SaveGrad();
            }

            // 到達可能なリンク・フィールドを探索、入力ノードを列挙
            (List<Field> reachable_fields, _) =
                EnumerateReachableFields(forward: true, backward: true, error_fields);

            InputNode[] input_nodes = reachable_fields.Select((field) => field.Value)
                                                      .OfType<InputNode>()
                                                      .Distinct().ToArray();

            Flow flow = FromInputs(input_nodes);

            return (flow, parameters);
        }

        /// <summary>推論計算グラフを構築</summary>
        /// <returns>計算フロー</returns>
        public static Flow Inference(params StoreField[] fields) {
            Flow flow = FromOutputs(fields.Select((field) => field.OutputNode).ToArray());

            return flow;
        }

        /// <summary>計算フローを実行</summary>
        public void Execute() {
            foreach (var node in nodes) {
                if (node is InputNode inputnode) {
                    if (inputnode.Initializer != null) {
                        inputnode.Initializer.Execute();
                    }
                }
                else if (node is FunctionNode funcnode) {
                    Tensor[] intensors = funcnode.InNodes.Select((innode) => tensors[innode]).ToArray();
                    Tensor[] outtensors = funcnode.OutNodes.Select((innode) => tensors[innode]).ToArray();

                    funcnode.Execute(intensors, outtensors);
                }
                else if (node is OutputNode outputnode) {
                    VariableNode innode = outputnode.InNode;
                    Tensor intensor = tensors[innode];

                    if (intensor != outputnode.Tensor) {
                        intensor.CopyTo(outputnode.Tensor);
                    }
                }
            }
        }
    }

    /// <summary>列挙型拡張</summary>
    internal static class EnumerableExtend {
        /// <summary>引数に含まれる要素を返す</summary>
        /// <remarks>
        /// 積集合を使うと逆伝搬の際に重複したリンク/フィールドが除去されてしまうため必要
        /// </remarks>
        public static IEnumerable<TSource> WhiteList<TSource>(this IEnumerable<TSource> first, IEnumerable<TSource> second) {
            foreach (var item in first) {
                if (second.Contains(item)) {
                    yield return item;
                }
            }
        }

        /// <summary>重複した要素を返す</summary>
        public static IEnumerable<TSource> Duplicated<TSource>(this IEnumerable<TSource> source) {
            return source.GroupBy((item) => item).Where((group) => group.Count() > 1).Select((group) => group.Key);
        }

        /// <summary>重複要素があるか判定</summary>
        public static bool IsDuplicated<TSource>(this IEnumerable<TSource> source) {
            return source.Distinct().Count() != source.Count();
        }
    }
}
