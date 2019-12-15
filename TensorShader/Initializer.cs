using System;
using System.Linq;

namespace TensorShader {
    /// <summary>テンソル初期化</summary>
    public abstract class Initializer {
        /// <summary>値</summary>
        protected Tensor Tensor { private set; get; }

        /// <summary>リンク名</summary>
        public virtual string Name => GetType().Name.Split('.').Last();

        /// <summary>コンストラクタ</summary>
        public Initializer(Tensor tensor) {
            this.Tensor = tensor ?? throw new ArgumentNullException(nameof(tensor));
        }

        /// <summary>初期化</summary>
        public abstract void Execute();
    }
}
