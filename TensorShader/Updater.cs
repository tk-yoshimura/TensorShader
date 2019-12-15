using System.Collections.Generic;
using System.Linq;

namespace TensorShader {
    /// <summary>パラメータ更新則</summary>
    public abstract class Updater {
        /// <summary>値</summary>
        protected InputNode Value { private set; get; }

        /// <summary>勾配</summary>
        protected InputNode Grad { private set; get; }

        /// <summary>演算フロー</summary>
        protected Flow Flow { private set; get; }

        /// <summary>リンク名</summary>
        public virtual string Name => GetType().Name.Split('.').Last();

        /// <summary>コンストラクタ</summary>
        /// <param name="parameter">更新対象パラメータ</param>
        protected Updater(ParameterField parameter) {
            this.Value = new InputNode(parameter.ValueTensor);
            this.Grad = new InputNode(parameter.GradTensor);
        }

        /// <summary>更新フロー</summary>
        public abstract Flow UpdateFlow();

        /// <summary>更新</summary>
        public void Execute() {
            if (Flow == null) {
                Flow = UpdateFlow();
            }

            Flow.Execute();
        }

        /// <summary>内部状態</summary>
        /// <remarks>ファイル入出力時に使用</remarks>
        public virtual Dictionary<string, Tensor> States {
            get {
                return new Dictionary<string, Tensor>();
            }
        }

        /// <summary>初期化</summary>
        public virtual void Initialize() {
            return;
        }
    }
}
