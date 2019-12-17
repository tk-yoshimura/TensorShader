using System;
using System.Collections.Generic;
using System.Linq;

namespace TensorShader {
    /// <summary>計算ノード</summary>
    public abstract class Node {
        private readonly List<Node> innodes, outnodes;

        /// <summary>入力ノード</summary>
        internal IReadOnlyList<Node> InNodes => innodes.AsReadOnly();

        /// <summary>入力ノード数</summary>
        internal int InNodeCount => innodes.Count;

        /// <summary>出力ノード</summary>
        internal IReadOnlyList<Node> OutNodes => outnodes.AsReadOnly();

        /// <summary>出力ノード数</summary>
        internal int OutNodeCount => outnodes.Count;

        /// <summary>コンストラクタ</summary>
        /// <param name="innodes">入力ノード</param>
        protected Node(params Node[] innodes) {
            this.innodes = innodes.ToList();
            this.outnodes = new List<Node>();

            foreach (var innode in innodes) {
                innode.outnodes.Add(this);
            }
        }

        /// <summary>コンストラクタ</summary>
        /// <param name="innodes">入力ノード</param>
        /// <param name="outnodes">出力ノード</param>
        protected Node(Node[] innodes, Node[] outnodes) {
            this.innodes = new List<Node>(innodes);
            this.outnodes = new List<Node>(outnodes);

            foreach (var innode in innodes) {
                innode.outnodes.Add(this);
            }

            foreach (var outnode in outnodes) {
                outnode.innodes.Add(this);
            }
        }
    }

    /// <summary>変数ノード</summary>
    public abstract partial class VariableNode : Node {
        /// <summary>変数形状</summary>
        public Shape Shape { private set; get; }

        /// <summary>コンストラクタ</summary>
        /// <param name="shape">変数形状</param>
        /// <param name="innodes">入力ノード</param>
        protected VariableNode(Shape shape, params Node[] innodes)
            : base(innodes) {

            this.Shape = shape;
        }

        /// <summary>入力ノードに関数を適用し出力ノードを返す</summary>
        /// <param name="function">関数クラス</param>
        /// <param name="innodes">入力入力ノード</param>
        /// <remarks>出力ノードを生成し変数形状を確定するのみでテンソル演算は行わない</remarks>
        internal static VariableNode[] Apply(Function function, params VariableNode[] innodes) {
            var inshapes = innodes.Select((node) => node.Shape).ToArray();
            var outshapes = function.OutputShapes(inshapes);

            var outnodes = outshapes.Select((shape) => new TemporaryNode(shape)).ToArray();

            var funcnode = new FunctionNode(function, innodes, outnodes);

            return outnodes.OfType<VariableNode>().ToArray();
        }

        /// <summary>出力ノードに値を保存</summary>
        /// <param name="tensor">テンソル</param>
        /// <param name="name">ノード名</param>
        public OutputNode Save(Tensor tensor = null, string name = "") {
            if (this is OutputNode) {
                throw new InvalidOperationException("This node is terminated.");
            }

            return new OutputNode(this, tensor, name);
        }

        /// <summary>入力ノードの値を更新</summary>
        /// <param name="input_node">更新対象の入力ノード</param>
        /// <remarks>入力ノードのテンソルは割り当て済みである必要がある</remarks>
        public OutputNode Update(InputNode input_node) {
            if (input_node.Tensor == null) {
                throw new ArgumentException("Input node tensor needs to be assigned.");
            }

            return new OutputNode(this, input_node.Tensor, input_node.Name);
        }

        /// <summary>文字列化</summary>
        public override string ToString() {
            return Shape.ToString();
        }
    }

    /// <summary>固有のテンソルを有するノード</summary>
    public abstract class TensorNode : VariableNode {
        /// <summary>対応テンソル</summary>
        public Tensor Tensor { private set; get; } = null;

        /// <summary>コンストラクタ</summary>
        /// <param name="shape">形状</param>
        /// <param name="tensor">テンソル</param>
        /// <param name="innodes">入力ノード</param>
        internal TensorNode(Shape shape, Tensor tensor = null, params Node[] innodes)
            : base(shape, innodes) {
            if (tensor != null && tensor.Shape != shape) {
                throw new ArgumentException(nameof(tensor));
            }

            this.Tensor = tensor;
        }

        /// <summary>テンソル対応付け</summary>
        internal void AssignTensor(Tensor tensor) {
            if (tensor == null || tensor.Shape != Shape) {
                throw new ArgumentException(nameof(tensor));
            }

            if (this.Tensor != null) {
                throw new InvalidOperationException();
            }

            this.Tensor = tensor;
        }

        /// <summary>テンソルの状態</summary>
        public float[] State {
            set {
                if (Tensor == null) {
                    throw new InvalidOperationException("Tensor not assigned.");
                }

                Tensor.State = value;
            }
            get {
                if (Tensor == null) {
                    throw new InvalidOperationException("Tensor not assigned.");
                }

                return Tensor.State;
            }
        }
    }

    /// <summary>入力ノード</summary>
    /// <remarks>計算フローの始点として機能</remarks>
    public class InputNode : TensorNode {
        /// <summary>ノード名</summary>
        /// <remarks>計算フロー構築時に関与しない</remarks>
        public string Name { private set; get; } = string.Empty;

        /// <summary>初期化</summary>
        public Initializer Initializer { set; get; }

        /// <summary>コンストラクタ</summary>
        /// <param name="shape">形状</param>
        /// <param name="tensor">テンソル</param>
        /// <param name="name">ノード名</param>
        /// <remarks>tensor を null にした場合フロー構築時にテンソルが割り当てられる</remarks>
        public InputNode(Shape shape, Tensor tensor = null, string name = "")
            : base(shape, tensor) {
            this.Name = name;
        }

        /// <summary>コンストラクタ</summary>
        /// <param name="tensor">テンソル</param>
        /// <param name="name">ノード名</param>
        public InputNode(Tensor tensor, string name = "")
            : this(tensor.Shape, tensor, name) { }

        /// <summary>入力ノードへキャスト</summary>
        public static implicit operator InputNode(Tensor tensor) {
            return new InputNode(tensor);
        }
    }

    /// <summary>出力ノード</summary>
    /// <remarks>計算フローの終点として機能</remarks>
    public class OutputNode : TensorNode {
        /// <summary>ノード名</summary>
        /// <remarks>計算フロー構築時に関与しない</remarks>
        public string Name { private set; get; } = string.Empty;

        /// <summary>入力ノード</summary>
        internal VariableNode InNode => InNodes[0] as VariableNode;

        /// <summary>コンストラクタ</summary>
        /// <param name="node">入力ノード</param>
        /// <param name="tensor">テンソル</param>
        /// <param name="name">ノード名</param>
        /// <remarks>tensor を null にした場合フロー構築時にテンソルが割り当てられる</remarks>
        internal OutputNode(VariableNode node, Tensor tensor = null, string name = "")
            : base(node.Shape, tensor, node) {
            this.Name = name;
        }
    }

    /// <summary>一時ノード</summary>
    internal class TemporaryNode : VariableNode {
        /// <summary>一時ノード</summary>
        /// <param name="shape">形状</param>
        public TemporaryNode(Shape shape)
            : base(shape) { }
    }

    /// <summary>関数ノード</summary>
    internal class FunctionNode : Node {
        /// <summary>関数</summary>
        public Function Function { private set; get; }

        /// <summary>コンストラクタ</summary>
        /// <param name="function">関数</param>
        /// <param name="innodes">入力ノード</param>
        /// <param name="outnodes">出力ノード</param>
        public FunctionNode(Function function, VariableNode[] innodes, VariableNode[] outnodes)
            : base(innodes, outnodes) {
            if (function.Inputs != innodes.Length || function.Outputs != outnodes.Length) {
                throw new ArgumentException(nameof(function));
            }

            if (innodes.Any((node) => node is OutputNode)) {
                throw new ArgumentException(nameof(innodes));
            }

            if (outnodes.Any((node) => node is InputNode)) {
                throw new ArgumentException(nameof(outnodes));
            }

            this.Function = function;
        }

        /// <summary>関数を実行</summary>
        public void Execute(Tensor[] intensors, Tensor[] outtensors) {
            Function.Execute(intensors, outtensors);
        }

        /// <summary>操作クラス対応付け</summary>
        public void AssignOperator(Tensor[] intensors, Tensor[] outtensors) {
            Function.AssignOperator(intensors, outtensors);
        }

        /// <summary>文字列化</summary>
        public override string ToString() {
            return Function.ToString();
        }
    }
}
