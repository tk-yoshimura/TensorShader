using System.Reflection;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

[assembly: AssemblyTitle("TensorShader Deep Learning .NET library")]
[assembly: AssemblyDescription("Deep Learning .NET library, For Regression.")]

#if DEBUG
[assembly: AssemblyConfiguration("Debug")]
#else
[assembly: AssemblyConfiguration("Release")]
#endif

[assembly: AssemblyCompany("T.Yoshimura")]
[assembly: AssemblyProduct("TensorShader")]
[assembly: AssemblyCopyright("Copyright © T.Yoshimura 2019-2021")]
[assembly: AssemblyTrademark("")]
[assembly: AssemblyCulture("")]

[assembly: ComVisible(false)]

[assembly: Guid("e6d7e72c-eb7a-4697-9b1a-3e87ba76323a")]

[assembly: AssemblyVersion("5.3.0.*")]

[assembly: InternalsVisibleTo("TensorShaderTest")]
