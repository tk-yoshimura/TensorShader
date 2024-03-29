﻿using System;
using System.Linq;

namespace TensorShaderCudaBackend.Shaders.Elementwise {

    /// <summary>要素独立演算</summary>
    public abstract class Elementwise : Shader {

        /// <summary>配列数</summary>
        public int Arrays { private set; get; }

        /// <summary>関数名</summary>
        public string FuncName { private set; get; }

        /// <summary>識別子</summary>
        public override sealed string Signature => $"{GetType().Name.Split(',').Last()} {FuncName}";

        /// <summary>コンストラクタ</summary>
        /// <param name="arrays">配列数</param>
        /// <param name="name">関数名</param>
        public Elementwise(int arrays, string name) {
            if (arrays < 1) {
                throw new ArgumentException(null, nameof(arrays));
            }

            this.Arrays = arrays;
            this.FuncName = name;
        }

        /// <summary>実行</summary>
        public override sealed void Execute(Stream stream, params object[] args) {
            CheckArgument(args);

            uint length = (args.Last() as uint?).Value;

            Kernel.Execute(length, dynamic_shared_memory_bytes: 0, stream, args);
        }

        /// <summary>引数チェック</summary>
        protected override sealed void CheckArgument(params object[] args) {
            if (args is null || args.Length != Arrays + 1) {
                throw new ArgumentException(null, nameof(args));
            }

            if (args[Arrays] is not uint length) {
                throw new ArgumentException(nameof(length));
            }

            for (int i = 0; i < Arrays; i++) {
                if (args[i] is not CudaArray<float> arr || arr.Length < length) {
                    throw new ArgumentException($"{nameof(arr)}[{i}]");
                }
            }
        }
    }
}
