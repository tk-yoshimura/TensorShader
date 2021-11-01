using System.Reflection;
using System.Runtime.InteropServices;

[assembly: AssemblyTitle("TensorShaderPreset")]
[assembly: AssemblyDescription("Deep Learning .NET library, For Regression. Preset")]

#if DEBUG
[assembly: AssemblyConfiguration("Debug")]
#else
[assembly: AssemblyConfiguration("Release")]
#endif

[assembly: AssemblyCompany("T.Yoshimura")]
[assembly: AssemblyProduct("TensorShaderPreset")]
[assembly: AssemblyCopyright("Copyright © T.Yoshimura 2019-2021")]
[assembly: AssemblyTrademark("")]
[assembly: AssemblyCulture("")]

[assembly: ComVisible(false)]

[assembly: Guid("5c4cedaf-3962-4d5b-b17f-35fa788f2016")]

[assembly: AssemblyVersion("5.5.0.*")]
