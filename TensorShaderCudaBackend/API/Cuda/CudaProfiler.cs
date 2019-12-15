using System.IO;

namespace TensorShaderCudaBackend.API {

    public static partial class Cuda {

        /// <summary>プロファイラ</summary>
        public static class Profiler {

            /// <summary>初期化</summary>
            /// <remarks>
            /// configファイルはテキストファイルで作成し
            /// 以下の中から出力する情報を改行区切りで指定する
            /// gpustarttimestamp, gridsize3d, threadblocksize, dynsmemperblock, stasmemperblock, regperthread
            /// memtransfersize, memtransferdir, streamid, countermodeaggregate, active_warps, active_cycles
            /// </remarks>
            public static void Initialize(string config_filepath, string output_filepath) {
                string output_dirpath = Path.GetDirectoryName(output_filepath);

                if (!string.IsNullOrWhiteSpace(output_dirpath) && !Directory.Exists(output_dirpath)) {
                    Directory.CreateDirectory(output_dirpath);
                }

                ErrorCode result = NativeMethods.cudaProfilerInitialize(config_filepath, output_filepath, cudaOutputMode.cudaCSV);
                if (result != ErrorCode.Success) {
                    throw new CudaException(result);
                }
            }

            /// <summary>計測開始</summary>
            public static void Start() {
                ErrorCode result = NativeMethods.cudaProfilerStart();
                if (result != ErrorCode.Success) {
                    throw new CudaException(result);
                }
            }

            /// <summary>計測終了</summary>
            public static void Stop() {
                ErrorCode result = NativeMethods.cudaProfilerStop();
                if (result != ErrorCode.Success) {
                    throw new CudaException(result);
                }
            }
        }
    }
}
