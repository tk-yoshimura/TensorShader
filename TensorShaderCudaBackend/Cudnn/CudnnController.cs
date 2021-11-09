using System;
using System.Diagnostics;
using System.Reflection;
using System.Runtime.InteropServices;

namespace TensorShaderCudaBackend.Cudnn {

    /// <summary>コントローラ</summary>
    public class CudnnController : IDisposable {
        private const string algorithm_not_found =
            "The convolution algorithm that can be excuted in the workspace limits can't be found.";

        private IntPtr handle;
        private IntPtr workspace;

        internal Int64 WorkspaceSize { get; private set; } = 0;
        internal Int64 WorkspaceLimit { get; private set; } = 0;

        /// <summary>実行ストリーム</summary>
        public Stream Stream { private set; get; }

        /// <summary>コンストラクタ</summary>
        /// <param name="stream">実行ストリーム</param>
        /// <param name="workspace_limit_bytes">
        /// ワークスペース上限
        /// デフォルト: 16 MBytes
        /// </param>
        public CudnnController(Stream stream, Int64 workspace_limit_bytes = 0x1000000) {
            const Int64 workspace_init_size = 4;

            if (workspace_limit_bytes < workspace_init_size) {
                throw new ArgumentOutOfRangeException(nameof(workspace_limit_bytes));
            }

            handle = API.Cudnn.Handle.Create();
            API.Cudnn.Handle.SetStream(handle, stream.Ptr);
            this.Stream = stream;

            WorkspaceLimit = workspace_limit_bytes;

            ReallocateWorkspace(workspace_init_size);
        }

        /// <summary>畳み込み順伝搬アルゴリズム列挙</summary>
        public ConvolutionFwdAlgoPerf[] EnumConvolutionForwardAlgorithm(
            TensorDescriptor x_desc,
            FilterDescriptor w_desc,
            ConvolutionDescriptor conv_desc,
            TensorDescriptor y_desc,
            Int64 workspace_limit_bytes = Int64.MaxValue) {

            ConvolutionFwdAlgoPerf[] prefs = API.Cudnn.Convolution.Forward.EnumAlgorithm(
                handle, x_desc.Ptr, w_desc.Ptr, conv_desc.Ptr, y_desc.Ptr,
                Enum.GetValues<ConvolutionFwdAlgo>().Length,
                workspace_limit_bytes
            );

            return prefs;
        }

        /// <summary>畳み込み順伝搬</summary>
        public void ConvolutionForward(
            CudaArray<float> x, TensorDescriptor x_desc,
            CudaArray<float> w, FilterDescriptor w_desc,
            ConvolutionDescriptor conv_desc,
            CudaArray<float> y, TensorDescriptor y_desc,
            ConvolutionFwdAlgo algo,
            float alpha = 1f, float beta = 0f) {

            Int64 workspace_size = API.Cudnn.Convolution.Forward.GetWorkspaceSize(
                handle, x_desc.Ptr, w_desc.Ptr, conv_desc.Ptr, y_desc.Ptr, algo
            );
            if (workspace_size > WorkspaceLimit) {
                ConvolutionFwdAlgoPerf[] prefs =
                    EnumConvolutionForwardAlgorithm(x_desc, w_desc, conv_desc, y_desc, WorkspaceLimit);

                if (prefs.Length < 1) {
                    throw new NotSupportedException(algorithm_not_found);
                }
#if DEBUG
                Trace.WriteLine($"[{typeof(CudnnController).Name}.{MethodBase.GetCurrentMethod().Name}] algo: {algo} -> {prefs[0].algo}");
#endif
                algo = prefs[0].algo;
                workspace_size = prefs[0].memory;
            }
            ReallocateWorkspace(workspace_size);

            float[] a = new float[] { alpha }, b = new float[] { beta };
            GCHandle pinned_a_handle = GCHandle.Alloc(a, GCHandleType.Pinned);
            GCHandle pinned_b_handle = GCHandle.Alloc(b, GCHandleType.Pinned);

            try {
                IntPtr a_ptr = pinned_a_handle.AddrOfPinnedObject();
                IntPtr b_ptr = pinned_b_handle.AddrOfPinnedObject();

                API.Cudnn.Convolution.Forward.Execute(
                    handle, a_ptr, x_desc.Ptr, x.Ptr, w_desc.Ptr, w.Ptr, conv_desc.Ptr,
                    algo, workspace, workspace_size, b_ptr, y_desc.Ptr, y.Ptr
                );
            }
            finally {
                pinned_a_handle.Free();
                pinned_b_handle.Free();
            }
        }

        /// <summary>畳み込み逆伝搬(特徴マップ)アルゴリズム列挙</summary>
        public ConvolutionBwdDataAlgoPerf[] EnumConvolutionBackwardDataAlgorithm(
            FilterDescriptor w_desc,
            TensorDescriptor dy_desc,
            ConvolutionDescriptor conv_desc,
            TensorDescriptor dx_desc,
            Int64 workspace_limit_bytes = Int64.MaxValue) {

            ConvolutionBwdDataAlgoPerf[] prefs = API.Cudnn.Convolution.BackwardData.EnumAlgorithm(
                handle, w_desc.Ptr, dy_desc.Ptr, conv_desc.Ptr, dx_desc.Ptr,
                Enum.GetValues<ConvolutionBwdDataAlgo>().Length,
                workspace_limit_bytes
            );

            return prefs;
        }

        /// <summary>畳み込み逆伝搬(特徴マップ)</summary>
        public void ConvolutionBackwardData(
            CudaArray<float> w, FilterDescriptor w_desc,
            CudaArray<float> dy, TensorDescriptor dy_desc,
            ConvolutionDescriptor conv_desc,
            CudaArray<float> dx, TensorDescriptor dx_desc,
            ConvolutionBwdDataAlgo algo,
            float alpha = 1f, float beta = 0f) {

            Int64 workspace_size = API.Cudnn.Convolution.BackwardData.GetWorkspaceSize(
                handle, w_desc.Ptr, dy_desc.Ptr, conv_desc.Ptr, dx_desc.Ptr, algo
            );
            if (workspace_size > WorkspaceLimit) {
                ConvolutionBwdDataAlgoPerf[] prefs =
                    EnumConvolutionBackwardDataAlgorithm(w_desc, dy_desc, conv_desc, dx_desc, WorkspaceLimit);

                if (prefs.Length < 1) {
                    throw new NotSupportedException(algorithm_not_found);
                }
#if DEBUG
                Trace.WriteLine($"[{typeof(CudnnController).Name}.{MethodBase.GetCurrentMethod().Name}] algo: {algo} -> {prefs[0].algo}");
#endif
                algo = prefs[0].algo;
                workspace_size = prefs[0].memory;
            }
            ReallocateWorkspace(workspace_size);

            float[] a = new float[] { alpha }, b = new float[] { beta };
            GCHandle pinned_a_handle = GCHandle.Alloc(a, GCHandleType.Pinned);
            GCHandle pinned_b_handle = GCHandle.Alloc(b, GCHandleType.Pinned);

            try {
                IntPtr a_ptr = pinned_a_handle.AddrOfPinnedObject();
                IntPtr b_ptr = pinned_b_handle.AddrOfPinnedObject();

                API.Cudnn.Convolution.BackwardData.Execute(
                    handle, a_ptr, w_desc.Ptr, w.Ptr, dy_desc.Ptr, dy.Ptr, conv_desc.Ptr,
                    algo, workspace, workspace_size, b_ptr, dx_desc.Ptr, dx.Ptr
                );
            }
            finally {
                pinned_a_handle.Free();
                pinned_b_handle.Free();
            }
        }

        /// <summary>畳み込み逆伝搬(フィルタ)アルゴリズム列挙</summary>
        public ConvolutionBwdFilterAlgoPerf[] EnumConvolutionBackwardFilterAlgorithm(
            TensorDescriptor x_desc,
            TensorDescriptor dy_desc,
            ConvolutionDescriptor conv_desc,
            FilterDescriptor dw_desc,
            Int64 workspace_limit_bytes = Int64.MaxValue) {

            ConvolutionBwdFilterAlgoPerf[] prefs = API.Cudnn.Convolution.BackwardFilter.EnumAlgorithm(
                handle, x_desc.Ptr, dy_desc.Ptr, conv_desc.Ptr, dw_desc.Ptr,
                Enum.GetValues<ConvolutionBwdFilterAlgo>().Length,
                workspace_limit_bytes
            );

            return prefs;
        }

        /// <summary>畳み込み逆伝搬(フィルタ)</summary>
        public void ConvolutionBackwardFilter(
            CudaArray<float> x, TensorDescriptor x_desc,
            CudaArray<float> dy, TensorDescriptor dy_desc,
            ConvolutionDescriptor conv_desc,
            CudaArray<float> dw, FilterDescriptor dw_desc,
            ConvolutionBwdFilterAlgo algo,
            float alpha = 1f, float beta = 0f) {

            Int64 workspace_size = API.Cudnn.Convolution.BackwardFilter.GetWorkspaceSize(
                handle, x_desc.Ptr, dy_desc.Ptr, conv_desc.Ptr, dw_desc.Ptr, algo
            );
            if (workspace_size > WorkspaceLimit) {
                ConvolutionBwdFilterAlgoPerf[] prefs =
                    EnumConvolutionBackwardFilterAlgorithm(x_desc, dy_desc, conv_desc, dw_desc, WorkspaceLimit);

                if (prefs.Length < 1) {
                    throw new NotSupportedException(algorithm_not_found);
                }
#if DEBUG
                Trace.WriteLine($"[{typeof(CudnnController).Name}.{MethodBase.GetCurrentMethod().Name}] algo: {algo} -> {prefs[0].algo}");
#endif
                algo = prefs[0].algo;
                workspace_size = prefs[0].memory;
            }
            ReallocateWorkspace(workspace_size);

            float[] a = new float[] { alpha }, b = new float[] { beta };
            GCHandle pinned_a_handle = GCHandle.Alloc(a, GCHandleType.Pinned);
            GCHandle pinned_b_handle = GCHandle.Alloc(b, GCHandleType.Pinned);

            try {
                IntPtr a_ptr = pinned_a_handle.AddrOfPinnedObject();
                IntPtr b_ptr = pinned_b_handle.AddrOfPinnedObject();

                API.Cudnn.Convolution.BackwardFilter.Execute(
                    handle, a_ptr, x_desc.Ptr, x.Ptr, dy_desc.Ptr, dy.Ptr, conv_desc.Ptr,
                    algo, workspace, workspace_size, b_ptr, dw_desc.Ptr, dw.Ptr
                );
            }
            finally {
                pinned_a_handle.Free();
                pinned_b_handle.Free();
            }
        }

        private void ReallocateWorkspace(Int64 new_workspace_size) {
            if (WorkspaceSize >= new_workspace_size) {
                return;
            }

            if (new_workspace_size > WorkspaceLimit) {
                throw new ArgumentOutOfRangeException(nameof(new_workspace_size));
            }

            if (workspace != IntPtr.Zero) {
                API.Cuda.Memory.Free(ref workspace);
            }

            workspace = API.Cuda.Memory.Allocate<byte>((ulong)new_workspace_size);
            WorkspaceSize = new_workspace_size;
        }

        /// <summary>破棄</summary>
        public void Dispose() {
            API.Cudnn.Handle.Destroy(ref handle);
            API.Cuda.Memory.Free(ref workspace);
            GC.SuppressFinalize(this);
        }

        /// <summary>ファイナライザ</summary>
        ~CudnnController() {
            Dispose();
        }
    }
}
