using System;
using System.Collections.Generic;

namespace TensorShaderCudaBackend {

    /// <summary>Cuda配列リザーバー</summary>
    public static class CudaArrayReserver<T> where T : struct, IComparable {
        private static volatile Dictionary<(Stream stream, int device_id, int index), CudaArray<T>> array_table;

        /// <summary>Cuda配列リクエスト</summary>
        public static CudaArray<T> Request(Stream stream, int device_id, int index, ulong length) {
            if (array_table == null) {
                array_table = new Dictionary<(Stream stream, int device_id, int index), CudaArray<T>>();
            }

            (Stream stream, int device_id, int index) key = (stream, device_id, index);

            if (array_table.ContainsKey(key)) {
                if (array_table[key].Length < length) {
                    API.Cuda.Synchronize();

                    lock (array_table) {
                        array_table[key] = new CudaArray<T>(length);
                    }
                }
            }
            else {
                int current_device_id = API.Cuda.CurrectDeviceID;

                lock (array_table) {
                    if (device_id != current_device_id) {
                        API.Cuda.CurrectDeviceID = device_id;
                    }

                    array_table.Add(key, new CudaArray<T>(length));

                    if (device_id != current_device_id) {
                        API.Cuda.CurrectDeviceID = current_device_id;
                    }
                }
            }

            return array_table[key];
        }

        /// <summary>初期化</summary>
        public static void Clear() {
            API.Cuda.Synchronize();

            lock (array_table) {
                array_table.Clear();
            }
        }
    }
}
