using System;
using System.Collections.Generic;
using System.Linq;

namespace TensorShader {
    /// <summary>フィールド</summary>
    public partial class Field {
        private VariableNode grad;
        private readonly List<VariableNode> untreated_grads;

        /// <summary>コンストラクタ</summary>
        public Field() {
            this.untreated_grads = new List<VariableNode>();
            this.InLinks = new List<Link>();
        }

        /// <summary>値</summary>
        public VariableNode Value { protected set; get; }

        /// <summary>勾配</summary>
        public VariableNode Grad {
            get {
                AddUntreatedGrads();
                return grad;
            }
        }

        /// <summary>逆伝搬が有効か</summary>
        public bool EnableBackprop { set; get; }

        /// <summary>このフィールドを入力とするレイヤー</summary>
        public List<Link> InLinks { private set; get; }

        /// <summary>このフィールドを出力したレイヤー</summary>
        public Link OutLink { set; get; }

        /// <summary>始端ノードか否か</summary>
        public bool IsInitiate => OutLink == null;

        /// <summary>終端ノードか否か</summary>
        public bool IsTerminate => InLinks.Count <= 0;

        /// <summary>形状</summary>
        public Shape Shape => Value.Shape;

        /// <summary>フィールド名</summary>
        public virtual string Name => Shape.ToString();

        /// <summary>値</summary>
        public Tensor ValueTensor
            => (Value != null && (Value is TensorNode tensornode)) ? tensornode.Tensor : null;

        /// <summary>値を設定する</summary>
        public void AssignValue(VariableNode value) {
            if (Value != null) {
                throw new InvalidOperationException("Node is already assigned.");
            }
            if (value == null) {
                throw new ArgumentNullException(nameof(value));
            }

            this.Value = value;
        }

        /// <summary>勾配を追加する</summary>
        public void AddGrad(VariableNode grad) {
            if (grad == null) {
                throw new ArgumentNullException(nameof(grad));
            }

            if (grad.Shape != Shape) {
                throw new ArgumentException(ExceptionMessage.Shape(grad.Shape, Shape));
            }

            if (grad != null && grad is OutputNode) {
                throw new InvalidOperationException("Node is terminated.");
            }

            untreated_grads.Add(grad);
        }

        /// <summary>未追加の勾配を追加する</summary>
        private void AddUntreatedGrads() {
            if (untreated_grads.Count <= 0) {
                return;
            }

            if (grad != null && grad is OutputNode) {
                throw new InvalidOperationException("Node is terminated.");
            }

            if (grad == null) {
                if (untreated_grads.Count == 1) {
                    grad = untreated_grads.First();
                }
                else {
                    grad = VariableNode.Sum(untreated_grads.ToArray());
                }
            }
            else {
                if (untreated_grads.Count == 1) {
                    grad += untreated_grads.First();
                }
                else {
                    grad += VariableNode.Sum(untreated_grads.ToArray());
                }
            }

            untreated_grads.Clear();
        }

        /// <summary>勾配を確定させる</summary>
        internal void ConfirmGrad() {
            AddUntreatedGrads();

            if (grad == null) {
                return;
            }
            grad = new OutputNode(grad);
        }

        /// <summary>ストアフィールドに値を保存</summary>
        /// <param name="tensor">テンソル</param>
        /// <param name="name">ノード名</param>
        public StoreField Save(Tensor tensor = null, string name = "") {
            return Value.Save(tensor, name);
        }

        /// <summary>文字列化</summary>
        public override string ToString() {
            return Shape.ToString();
        }
    }

    /// <summary>変数フィールド</summary>
    public class VariableField : Field {
        /// <summary>フィールド名</summary>
        public override string Name => string.IsNullOrEmpty((Value as InputNode).Name) ? base.Name : (Value as InputNode).Name;

        /// <summary>コンストラクタ</summary>
        public VariableField(VariableNode node) {
            if (node == null) {
                throw new ArgumentNullException(nameof(node));
            }

            if (node is OutputNode) {
                throw new ArgumentException("Node is terminated.");
            }

            this.Value = node;
            this.EnableBackprop = false;
        }

        /// <summary>コンストラクタ</summary>
        public VariableField(Tensor tensor, string name = "")
            : this(new InputNode(tensor.Shape, tensor, name)) { }

        /// <summary>変数フィールドへキャスト</summary>
        public static implicit operator VariableField(VariableNode node) {
            return new VariableField(node);
        }

        /// <summary>変数フィールドへキャスト</summary>
        public static implicit operator VariableField(Tensor tensor) {
            return new InputNode(tensor);
        }
    }

    /// <summary>パラメータカテゴリ</summary>
    public enum ParameterCategory {
        /// <summary>未定義</summary>
        Undefined,
        /// <summary>カーネル</summary>
        Kernel,
        /// <summary>バイアス項</summary>
        Bias,
        /// <summary>スケール項</summary>
        Scale,
        /// <summary>勾配を更新に用いない</summary>
        UnnecessaryGrad
    }

    /// <summary>パラメータフィールド</summary>
    public class ParameterField : Field {
        /// <summary>更新則</summary>
        public List<Updater> Updaters { private set; get; }

        /// <summary>カテゴリ</summary>
        public ParameterCategory Category { private set; get; }

        /// <summary>パラメータ名</summary>
        public override string Name => string.IsNullOrEmpty((Value as InputNode).Name) ? base.Name : (Value as InputNode).Name;

        /// <summary>コンストラクタ</summary>
        public ParameterField(InputNode node, ParameterCategory category = ParameterCategory.Undefined) {
            this.Updaters = new List<Updater>();
            this.Category = category;
            this.Value = node;
            this.EnableBackprop = true;
        }

        /// <summary>コンストラクタ</summary>
        public ParameterField(Tensor tensor, string name = "", ParameterCategory category = ParameterCategory.Undefined)
            : this(new InputNode(tensor.Shape, tensor, name), category) { }

        /// <summary>勾配</summary>
        public Tensor GradTensor
            => (Grad != null && (Grad is TensorNode tensornode)) ? tensornode.Tensor : null;

        /// <summary>更新</summary>
        public void Update() {
            foreach (Updater updater in Updaters) {
                updater.Execute();
            }
        }

        /// <summary>更新則を追加</summary>
        public void AddUpdater(Updater updater) {
            if (Grad == null) {
                if (Name == string.Empty) {
                    throw new InvalidOperationException($"Parameter field for which grad can't be defined.");
                }
                else {
                    throw new InvalidOperationException($"Parameter field \"{Name}\" for which grad can't be defined.");
                }
            }

            if (Updaters.Select((item) => item.Name).Contains(updater.Name)) {
                throw new ArgumentException("The list already contains an updater with the same name.");
            }

            Updaters.Add(updater);
        }

        /// <summary>初期化</summary>
        public void Initialize(Func<Tensor, Initializer> initializer) {
            initializer(this.ValueTensor).Execute();
        }

        /// <summary>変数フィールドへキャスト</summary>
        public static implicit operator ParameterField(Tensor tensor) {
            return new ParameterField(new InputNode(tensor));
        }

        /// <summary>勾配を出力ノードに保存</summary>
        public void SaveGrad() {
            ConfirmGrad();
        }
    }

    /// <summary>ストアフィールド</summary>
    public class StoreField {
        /// <summary>テンソルの状態</summary>
        public float[] State => OutputNode.State;

        /// <summary>変数形状</summary>
        public Shape Shape => OutputNode.Shape;

        /// <summary>対応テンソル</summary>
        public Tensor Tensor => OutputNode.Tensor;

        /// <summary>ノード名</summary>
        public string Name => OutputNode.Name;

        /// <summary>出力ノード</summary>
        internal OutputNode OutputNode { get; }

        /// <summary>コンストラクタ</summary>
        public StoreField(OutputNode output_node) {
            this.OutputNode = output_node;
        }

        /// <summary>出力ノードからのキャスト</summary>
        public static implicit operator StoreField(OutputNode output_node) {
            return new StoreField(output_node);
        }

        /// <summary>文字列化</summary>
        public override string ToString() {
            return OutputNode.ToString();
        }
    }
}
