using System;
using System.Runtime.InteropServices;

namespace TensorShaderCudaBackend.Cudnn {

    /// <summary>コントローラ</summary>
    public class Controller: IDisposable {
        private IntPtr handle;
        private IntPtr workspace;
        private readonly Int64 workspace_size_limit;

        /// <summary>コンストラクタ</summary>
        /// <param name="stream">実行ストリーム</param>
        /// <param name="workspace_size_limit">
        /// ワークスペース最大サイズ[Bytes] 
        /// デフォルト: 16MBytes
        /// </param>
        public Controller(Stream stream, Int64 workspace_size_limit = 0x1000000) {
            if (workspace_size_limit <= 0) {
                throw new ArgumentOutOfRangeException(nameof(workspace_size_limit));
            }

            handle = API.Cudnn.Handle.Create();
            API.Cudnn.Handle.SetStream(handle, stream.Ptr);
            workspace = API.Cuda.Memory.Allocate<byte>((ulong)workspace_size_limit);
            this.workspace_size_limit = workspace_size_limit;
        }

        /// <summary>畳み込み順伝搬</summary>
        public void ConvolutionForward(
            CudaArray<float> x, TensorDescriptor x_desc, 
            CudaArray<float> w, FilterDescriptor w_desc,
            ConvolutionDescriptor conv_desc,
            CudaArray<float> y, TensorDescriptor y_desc,
            float alpha = 1f, float beta = 0f){

            API.Cudnn.ConvolutionFwdAlgo algo = API.Cudnn.Convolution.Forward.Algorithm(
                handle, x_desc.Ptr, w_desc.Ptr, conv_desc.Ptr, y_desc.Ptr, 
                API.Cudnn.ConvolutionFwdPreference.SpecifyWorkspaceLimit, workspace_size_limit);

            float[] a = new float[] { alpha }, b = new float[] { beta };
            GCHandle pinned_a_handle = GCHandle.Alloc(a, GCHandleType.Pinned);
            GCHandle pinned_b_handle = GCHandle.Alloc(b, GCHandleType.Pinned);

            try {
                IntPtr a_ptr = pinned_a_handle.AddrOfPinnedObject();
                IntPtr b_ptr = pinned_b_handle.AddrOfPinnedObject();

                API.Cudnn.Convolution.Forward.Execute(
                    handle, a_ptr, x_desc.Ptr, x.Ptr, w_desc.Ptr, w.Ptr, conv_desc.Ptr,
                    algo, workspace, workspace_size_limit, b_ptr, y_desc.Ptr, y.Ptr
                );
            }
            finally {
                pinned_a_handle.Free();
                pinned_b_handle.Free();
            }
        }

        /// <summary>畳み込み逆伝搬(特徴マップ)</summary>
        public void ConvolutionBackwardData(
            CudaArray<float> w, FilterDescriptor w_desc, 
            CudaArray<float> dy, TensorDescriptor dy_desc,
            ConvolutionDescriptor conv_desc,
            CudaArray<float> dx, TensorDescriptor dx_desc,
            float alpha = 1f, float beta = 0f){

            API.Cudnn.ConvolutionBwdDataAlgo algo = API.Cudnn.Convolution.BackwardData.Algorithm(
                handle, w_desc.Ptr, dy_desc.Ptr, conv_desc.Ptr, dx_desc.Ptr, 
                API.Cudnn.ConvolutionBwdDataPreference.SpecifyWorkspaceLimit, workspace_size_limit);

            float[] a = new float[] { alpha }, b = new float[] { beta };
            GCHandle pinned_a_handle = GCHandle.Alloc(a, GCHandleType.Pinned);
            GCHandle pinned_b_handle = GCHandle.Alloc(b, GCHandleType.Pinned);

            try {
                IntPtr a_ptr = pinned_a_handle.AddrOfPinnedObject();
                IntPtr b_ptr = pinned_b_handle.AddrOfPinnedObject();

                API.Cudnn.Convolution.BackwardData.Execute(
                    handle, a_ptr, w_desc.Ptr, w.Ptr, dy_desc.Ptr, dy.Ptr, conv_desc.Ptr,
                    algo, workspace, workspace_size_limit, b_ptr, dx_desc.Ptr, dx.Ptr
                );
            }
            finally {
                pinned_a_handle.Free();
                pinned_b_handle.Free();
            }
        }

        /// <summary>畳み込み逆伝搬(フィルタ)</summary>
        public void ConvolutionBackwardFilter(
            CudaArray<float> x, TensorDescriptor x_desc, 
            CudaArray<float> dy, TensorDescriptor dy_desc,
            ConvolutionDescriptor conv_desc,
            CudaArray<float> dw, FilterDescriptor dw_desc,
            float alpha = 1f, float beta = 0f){

            API.Cudnn.ConvolutionBwdFilterAlgo algo = API.Cudnn.Convolution.BackwardFilter.Algorithm(
                handle, x_desc.Ptr, dy_desc.Ptr, conv_desc.Ptr, dw_desc.Ptr, 
                API.Cudnn.ConvolutionBwdFilterPreference.SpecifyWorkspaceLimit, workspace_size_limit);

            float[] a = new float[] { alpha }, b = new float[] { beta };
            GCHandle pinned_a_handle = GCHandle.Alloc(a, GCHandleType.Pinned);
            GCHandle pinned_b_handle = GCHandle.Alloc(b, GCHandleType.Pinned);

            try {
                IntPtr a_ptr = pinned_a_handle.AddrOfPinnedObject();
                IntPtr b_ptr = pinned_b_handle.AddrOfPinnedObject();

                API.Cudnn.Convolution.BackwardFilter.Execute(
                    handle, a_ptr, x_desc.Ptr, x.Ptr, dy_desc.Ptr, dy.Ptr, conv_desc.Ptr,
                    algo, workspace, workspace_size_limit, b_ptr, dw_desc.Ptr, dw.Ptr
                );
            }
            finally {
                pinned_a_handle.Free();
                pinned_b_handle.Free();
            }
        }

        /// <summary>破棄</summary>
        public void Dispose() {
            API.Cudnn.Handle.Destroy(ref handle);
            API.Cuda.Memory.Free(ref workspace);
            GC.SuppressFinalize(this);
        }

        /// <summary>ファイナライザ</summary>
        ~Controller() {
            Dispose();
        }
    }
}
