using System;
using System.Linq;

namespace TensorShader {
    /// <summary>レイヤー</summary>
    public abstract class Layer {
        /// <summary>ラベル</summary>
        public string Label { private set; get; }

        /// <summary>入力チャネル数</summary>
        public virtual int InChannels => throw new NotImplementedException();

        /// <summary>出力チャネル数</summary>
        public virtual int OutChannels => throw new NotImplementedException();

        /// <summary>カーネルサイズ</summary>
        public virtual int Width => throw new NotImplementedException();

        /// <summary>カーネルサイズ</summary>
        public virtual int Height => throw new NotImplementedException();

        /// <summary>カーネルサイズ</summary>
        public virtual int Depth => throw new NotImplementedException();

        /// <summary>コンストラクタ</summary>
        public Layer(string label) {
            if (label == null || label == string.Empty) {
                throw new ArgumentException(nameof(label));
            }

            this.Label = label;
        }

        /// <summary>適用</summary>
        public abstract Field Forward(params Field[] fields);

        /// <summary>レイヤー名</summary>
        public virtual string Name => GetType().Name.Split('.').Last();

        /// <summary>文字列化</summary>
        public override string ToString() {
            return Name;
        }
    }
}
