using System;
using System.Runtime.InteropServices;

namespace TensorShaderCudaBackend.API {
    using size_t = Int64;

    public static partial class Cuda {

#pragma warning disable CS1591 // 公開されている型またはメンバーの XML コメントがありません

        /// <summary>デバイスプロパティ</summary>
        [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Ansi)]
        public struct DeviceProp {

            /// <summary>デバイス識別名</summary>
            [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 256)]
            public string Name;

            /// <summary>UUID</summary>
            [MarshalAs(UnmanagedType.ByValArray, SizeConst = 16)]
            public byte[] UUID;

            /// <summary>LUID</summary>
            [MarshalAs(UnmanagedType.ByValArray, SizeConst = 8)]
            public byte[] LUID;

            /// <summary>LUIDデバイスノードマスク</summary>
            public uint LuidDeviceNodeMask;

            /// <summary>グローバルメモリサイズ</summary>
            public size_t GlobalMemoryBytes;

            /// <summary>シェアードメモリサイズ</summary>
            public size_t SharedMemoryBytesPerBlock;

            /// <summary>32bitレジスタサイズ</summary>
            public int RegisterSizePerBlock;

            /// <summary>Warpサイズ</summary>
            public int WarpSize;

            /// <summary>メモリコピーに許可される最大ピッチ</summary>
            public size_t MemoryPitchBytes;

            /// <summary>ブロックごとのスレッド最大数</summary>
            public int MaxThreadsPerBlock;

            /// <summary>ブロックごとのスレッド最大次元数</summary>
            [MarshalAs(UnmanagedType.ByValArray, SizeConst = 3)]
            public int[] MaxThreadsDim;

            /// <summary>グリッド最大次元数</summary>
            [MarshalAs(UnmanagedType.ByValArray, SizeConst = 3)]
            public int[] MaxGridsDim;

            /// <summary>動作周波数[kHz]</summary>
            public int ClockRate;

            /// <summary>定数メモリサイズ</summary>
            public size_t ConstMemoryBytes;

            /// <summary>ComputeCapability メジャーバージョン</summary>
            public int Major;

            /// <summary>ComputeCapability マイナーバージョン</summary>
            public int Minor;

            /// <summary>テクスチャアラインメント</summary>
            public size_t TextureAlignment;

            /// <summary>テクスチャピッチアラインメント</summary>
            public size_t TexturePitchAlignment;

            /// <summary>cudaMemcpy()とカーネルの実行を同時に行えるか</summary>
            public int DeviceOverlap;

            /// <summary>マルチプロセッサ数</summary>
            public int MultiProcessorCount;

            /// <summary>カーネル実行時のTDR(Timeout Detection Recovery)があるか</summary>
            public int KernelExecTimeout;

            /// <summary>統合GPUであるか</summary>
            public int Integrated;

            /// <summary>ホストメモリをCUDAデバイスのアドレス空間にマップできるか</summary>
            public int CanMapHostMemory;

            /// <summary>計算モード</summary>
            public int ComputeMode;

            public int MaxTexture1D;

            public int MaxTexture1DMipmap;

            public int MaxTexture1DLinear;

            [MarshalAs(UnmanagedType.ByValArray, SizeConst = 2)]
            public int[] MaxTexture2D;

            [MarshalAs(UnmanagedType.ByValArray, SizeConst = 2)]
            public int[] MaxTexture2DMipmap;

            [MarshalAs(UnmanagedType.ByValArray, SizeConst = 3)]
            public int[] MaxTexture2DLinear;

            [MarshalAs(UnmanagedType.ByValArray, SizeConst = 2)]
            public int[] MaxTexture2DGather;

            [MarshalAs(UnmanagedType.ByValArray, SizeConst = 3)]
            public int[] MaxTexture3D;

            [MarshalAs(UnmanagedType.ByValArray, SizeConst = 3)]
            public int[] MaxTexture3DAlt;

            public int MaxTextureCubemap;

            [MarshalAs(UnmanagedType.ByValArray, SizeConst = 2)]
            public int[] MaxTexture1DLayered;

            [MarshalAs(UnmanagedType.ByValArray, SizeConst = 3)]
            public int[] MaxTexture2DLayered;

            [MarshalAs(UnmanagedType.ByValArray, SizeConst = 2)]
            public int[] MaxTextureCubemapLayered;

            public int MaxSurface1D;

            [MarshalAs(UnmanagedType.ByValArray, SizeConst = 2)]
            public int[] MaxSurface2D;

            [MarshalAs(UnmanagedType.ByValArray, SizeConst = 3)]
            public int[] MaxSurface3D;

            [MarshalAs(UnmanagedType.ByValArray, SizeConst = 2)]
            public int[] MaxSurface1DLayered;

            [MarshalAs(UnmanagedType.ByValArray, SizeConst = 3)]
            public int[] MaxSurface2DLayered;

            public int MaxSurfaceCubemap;

            [MarshalAs(UnmanagedType.ByValArray, SizeConst = 2)]
            public int[] MaxSurfaceCubemapLayered;

            public size_t SurfaceAlignment;

            /// <summary>同じコンテキストにおいて複数のカーネルの同時実行が可能か</summary>
            public int ConcurrentKernels;

            /// <summary>ECCをサポートするか</summary>
            public int ECCSupport;

            /// <summary>PCIバスID</summary>
            public int PciBusID;

            /// <summary>PCIデバイスID</summary>
            public int PciDeviceID;

            /// <summary>PCIドメインID</summary>
            public int PciDomainID;

            /// <summary>TCCドライバを使うか</summary>
            public int TccDriver;

            public int AsyncEngineCount;

            public int UnifiedAddressing;

            /// <summary>メモリ動作周波数 [kHz]</summary>
            public int MemoryClockRate;

            /// <summary>メモリバス帯域</summary>
            public int MemoryBusWidth;

            /// <summary>L2キャッシュサイズ</summary>
            public int L2CacheBytes;

            /// <summary>プロセッサごとの最大スレッド数</summary>
            public int MaxThreadsPerMultiProcessor;

            public int StreamPrioritiesSupported;

            public int GlobalL1CacheSupported;

            public int LocalL1CacheSupported;

            public size_t SharedMemoryPerMultiprocessor;

            public int RegisterPerMultiprocessor;

            public int ManagedMemory;

            public int IsMultiGpuBoard;

            public int MultiGpuBoardGroupID;

            public int HostNativeAtomicSupported;

            public int SingleToDoublePrecisionPerfRatio;

            public int PageableMemoryAccess;

            public int ConcurrentManagedAccess;

            public int ComputePreemptionSupported;

            public int CanUseHostPointerForRegisteredMemory;

            public int CooperativeLaunch;

            public int CooperativeMultiDeviceLaunch;

            public size_t SharedMemoryPerBlockOptin;

            public int PageableMemoryAccessUsesHostPageTables;

            public int DirectManagedMemoryAccessFromHost;
        }
#pragma warning restore CS1591 // 公開されている型またはメンバーの XML コメントがありません
    }
}
