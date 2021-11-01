using System;
using System.Linq;

namespace TensorShader {
    /// <summary>関数クラス</summary>
    internal abstract class Function {
        private Tensor[] intensors = Enumerable.Empty<Tensor>().ToArray(), outtensors = Enumerable.Empty<Tensor>().ToArray();
        private Tensor[] tensors = null;
        private Operator ope = null;

        /// <summary>入力数</summary>
        public int Inputs { private set; get; }

        /// <summary>出力数</summary>
        public int Outputs { private set; get; }

        /// <summary>入出力テンソルの重複を受容するか</summary>
        /// <remarks>計算フロー構築時テンソルの共有に関与 受容するならばMaterialize()で同一テンソル指定時の処理を追加すること</remarks>
        public bool AllowResubstitution { private set; get; }

        /// <summary>関数名</summary>
        public virtual string Name => GetType().Name.Split('.').Last();

        /// <summary>コンストラクタ</summary>
        /// <param name="inputs">入力数</param>
        /// <param name="outputs">出力数</param>
        /// <param name="allow_resubstitution">入出力テンソルの重複を受容するか</param>
        public Function(int inputs, int outputs, bool allow_resubstitution) {
            if (inputs < 1 || outputs < 1) {
                throw new ArgumentException($"{nameof(inputs)}, {nameof(outputs)}");
            }

            this.Inputs = inputs;
            this.Outputs = outputs;
            this.AllowResubstitution = allow_resubstitution;
        }

        /// <summary>変数の有効性をチェック</summary>
        public void CheckArgumentsCount(Tensor[] intensors, Tensor[] outtensors) {
            if (intensors is null || intensors.Length != Inputs || intensors.Any((tensor) => tensor is null)) {
                throw new ArgumentException(null, nameof(intensors));
            }
            if (outtensors is null || outtensors.Length != Outputs || outtensors.Any((tensor) => tensor is null)) {
                throw new ArgumentException(null, nameof(outtensors));
            }
        }

        /// <summary>関数を実行</summary>
        /// <remarks>入出力テンソル変更時、初回呼び出しのみ操作クラス確定のオーバヘッドが発生</remarks>
        public void Execute(Tensor[] intensors, Tensor[] outtensors) {
            AssignOperator(intensors, outtensors);
            this.ope.Execute(this.tensors);
        }

        /// <summary>操作クラス対応付け</summary>
        public void AssignOperator(Tensor[] intensors, Tensor[] outtensors) {
            if (!this.intensors.SequenceEqual(intensors) || !this.outtensors.SequenceEqual(outtensors)) {
                CheckArgumentsCount(intensors, outtensors);

                try {
                    (this.tensors, this.ope) = GenerateOperator(intensors, outtensors);
                }
                catch (NotImplementedException) {
                    string inshapes = "input : " + string.Join(",", intensors.Select((tensors) => tensors.Shape.ToString()));
                    string outshapes = "output : " + string.Join(",", outtensors.Select((tensors) => tensors.Shape.ToString()));

                    throw new NotImplementedException($"Detect undefined function calls. {inshapes} {outshapes}");
                }

                this.intensors = (Tensor[])intensors.Clone();
                this.outtensors = (Tensor[])outtensors.Clone();
            }
        }

        /// <summary>操作クラスを生成する</summary>
        internal abstract (Tensor[] tensors, Operator ope) GenerateOperator(Tensor[] intensors, Tensor[] outtensors);

        /// <summary>出力テンソル形状を返す</summary>
        public abstract Shape[] OutputShapes(params Shape[] inshapes);

        /// <summary>入力テンソル形状をチェックする</summary>
        public virtual void CheckInputShapes(params Shape[] inshapes) {
            if (inshapes.Length != Inputs) {
                throw new ArgumentException(ExceptionMessage.ArgumentCount(nameof(inshapes), inshapes.Length, Inputs));
            }
        }

        /// <summary>文字列化</summary>
        public override string ToString() {
            return Name;
        }
    }
}
