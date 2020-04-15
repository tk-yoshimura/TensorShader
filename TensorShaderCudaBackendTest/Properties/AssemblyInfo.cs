using System.Reflection;
using System.Runtime.InteropServices;

[assembly: AssemblyTitle("TensorShaderCudaBackendTest")]
[assembly: AssemblyDescription("Cuda Kernel Implementations Test")]

#if DEBUG
[assembly: AssemblyConfiguration("Debug")]
#else
[assembly: AssemblyConfiguration("Release")]
#endif

[assembly: AssemblyCompany("T.Yoshimura")]
[assembly: AssemblyProduct("TensorShaderCudaBackendTest")]
[assembly: AssemblyCopyright("Copyright © T.Yoshimura 2019-2020")]
[assembly: AssemblyTrademark("")]
[assembly: AssemblyCulture("")]

[assembly: ComVisible(false)]

[assembly: Guid("6f14bf0a-748a-4f30-89b8-e42adc2e7480")]

// [assembly: AssemblyVersion("1.0.*")]
[assembly: AssemblyVersion("4.6.0.*")]
