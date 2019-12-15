using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using TensorShaderCudaAPI;

namespace TensorShaderCudaBackend.Elementwise {

    /// <summary>単項演算</summary>
    public abstract class UnaryArithmetric : Shader{

        /// <summary>コンストラクタ</summary>
        /// <param name="func">関数</param>
        /// <param name="name">関数名</param>
        /// <remarks>func e.g. #y = f(#x);</remarks>
        public UnaryArithmetric(string func, string name) { 
            string code = $@"
            __global__ void {name}(float *x, float *y, int length) {{
	            int i = blockDim.x * blockIdx.x + threadIdx.x;
	            if (i >= length) {{
		            return;
	            }}
                {func.Replace("#x", "x[i]").Replace("#y", "y[i]")};
            }}";

            this.Kernel = new Kernel(code, name);
        }

        /// <summary>実行</summary>
        public override sealed void Execute(Stream stream, params object[] args) {
            CheckArgument(args);

            int length = (args[2] as int?).Value;

            Kernel.Execute((uint)length, shared_memory_bytes:0, stream, args);
        }

        /// <summary>引数チェック</summary>
        protected override sealed void CheckArgument(params object[] args) {
            if(args == null || args.Length != 3) {
                throw new ArgumentException(nameof(args));
            }

            if(!(args[2] is int length)) { 
                throw new ArgumentException($"{nameof(args)}[2]");
            }

            if(!(args[0] is GpuArray<float> x) || x.Length < (ulong)length) { 
                throw new ArgumentException($"{nameof(args)}[0]");
            }

            if(!(args[1] is GpuArray<float> y) || y.Length < (ulong)length) { 
                throw new ArgumentException($"{nameof(args)}[1]");
            }
        }
    }
}
