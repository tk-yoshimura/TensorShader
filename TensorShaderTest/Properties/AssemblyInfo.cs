using System.Reflection;
using System.Runtime.InteropServices;

[assembly: AssemblyTitle("TensorShaderTest")]
[assembly: AssemblyDescription("Deep Learning .NET library, For Regression. Test")]

#if DEBUG
[assembly: AssemblyConfiguration("Debug")]
#else
[assembly: AssemblyConfiguration("Release")]
#endif

[assembly: AssemblyCompany("T.Yoshimura")]
[assembly: AssemblyProduct("TensorShaderTest")]
[assembly: AssemblyCopyright("Copyright © T.Yoshimura 2019-2022")]
[assembly: AssemblyTrademark("")]
[assembly: AssemblyCulture("")]

[assembly: ComVisible(false)]

[assembly: Guid("66a05297-9440-42a4-855f-2fd162428b8b")]

[assembly: AssemblyVersion("5.6.0.*")]