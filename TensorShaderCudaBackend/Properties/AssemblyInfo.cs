using System.Reflection;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

[assembly: AssemblyTitle("TensorShaderCudaBackend")]
[assembly: AssemblyDescription("Cuda Kernel Implementations")]

#if DEBUG
[assembly: AssemblyConfiguration("Debug")]
#else
[assembly: AssemblyConfiguration("Release")]
#endif

[assembly: AssemblyCompany("T.Yoshimura")]
[assembly: AssemblyProduct("TensorShaderCudaBackend")]
[assembly: AssemblyCopyright("Copyright © T.Yoshimura 2019-2021")]
[assembly: AssemblyTrademark("")]
[assembly: AssemblyCulture("")]

[assembly: ComVisible(false)]

[assembly: Guid("d0343c7a-36a5-4f87-89d9-c0bff56721f7")]

[assembly: AssemblyVersion("5.5.1.*")]

[assembly: InternalsVisibleTo("TensorShaderCudaBackendTest")]