using System.Reflection;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

// アセンブリに関する一般情報は以下の属性セットをとおして制御されます。
// 制御されます。アセンブリに関連付けられている情報を変更するには、
// これらの属性値を変更してください。

#if CUDA_10_0
[assembly: AssemblyTitle("TensorShaderCudaBackend CUDA10.0")]
#elif CUDA_10_1
[assembly: AssemblyTitle("TensorShaderCudaBackend CUDA10.1")]
#elif CUDA_10_2
[assembly: AssemblyTitle("TensorShaderCudaBackend CUDA10.2")]
#elif CUDA_10_3
[assembly: AssemblyTitle("TensorShaderCudaBackend CUDA10.3")]
#elif CUDA_10_4
[assembly: AssemblyTitle("TensorShaderCudaBackend CUDA10.4")]
#else
[assembly: AssemblyTitle("TensorShaderCudaBackend CUDA10.1")]
#endif

[assembly: AssemblyDescription("Cuda Kernel Implementations")]

#if DEBUG
[assembly: AssemblyConfiguration("Debug")]
#else
[assembly: AssemblyConfiguration("Release")]
#endif

[assembly: AssemblyCompany("T.Yoshimura")]
[assembly: AssemblyProduct("TensorShaderCudaBackend")]
[assembly: AssemblyCopyright("Copyright © T.Yoshimura 2019-2020")]
[assembly: AssemblyTrademark("")]
[assembly: AssemblyCulture("")]

// ComVisible を false に設定すると、このアセンブリ内の型は COM コンポーネントから
// 参照できなくなります。COM からこのアセンブリ内の型にアクセスする必要がある場合は、
// その型の ComVisible 属性を true に設定してください。
[assembly: ComVisible(false)]

// このプロジェクトが COM に公開される場合、次の GUID が typelib の ID になります
[assembly: Guid("d0343c7a-36a5-4f87-89d9-c0bff56721f7")]

// アセンブリのバージョン情報は、以下の 4 つの値で構成されています:
//
//      メジャー バージョン
//      マイナー バージョン
//      ビルド番号
//      リビジョン
//
// すべての値を指定するか、次を使用してビルド番号とリビジョン番号を既定に設定できます
// 既定値にすることができます:
// [assembly: AssemblyVersion("1.0.*")]
[assembly: AssemblyVersion("4.4.1.*")]

[assembly: InternalsVisibleTo("TensorShaderCudaBackendTest")]