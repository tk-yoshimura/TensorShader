using System.Reflection;
using System.Runtime.InteropServices;

// アセンブリに関する一般情報は以下の属性セットをとおして制御されます。
// 制御されます。アセンブリに関連付けられている情報を変更するには、
// これらの属性値を変更してください。
[assembly: AssemblyTitle("TensorShaderUtil")]
[assembly: AssemblyDescription("Deep Learning .NET libraryy, CUDA accelerated. Utility")]

#if DEBUG
[assembly: AssemblyConfiguration("Debug")]
#else
[assembly: AssemblyConfiguration("Release")]
#endif

[assembly: AssemblyCompany("")]
[assembly: AssemblyProduct("TensorShaderUtil")]
[assembly: AssemblyCopyright("Copyright © T.Yoshimura 2019-2020")]
[assembly: AssemblyTrademark("")]
[assembly: AssemblyCulture("")]
// ComVisible を false に設定すると、このアセンブリ内の型は COM コンポーネントから
// 参照できなくなります。COM からこのアセンブリ内の型にアクセスする必要がある場合は、
// その型の ComVisible 属性を true に設定してください。
[assembly: ComVisible(false)]
// このプロジェクトが COM に公開される場合、次の GUID が typelib の ID になります
[assembly: Guid("9291e2d8-e250-4886-8845-aaf42b3c070c")]
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
[assembly: AssemblyVersion("4.1.*")]
