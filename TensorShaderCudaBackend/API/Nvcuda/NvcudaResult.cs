﻿namespace TensorShaderCudaBackend.API {
    public static partial class Nvcuda {
        internal enum ResultCode {
            Success                              = 0,
            ErrorInvalidValue                    = 1,
            ErrorOutOfMemory                     = 2,
            ErrorNotInitialized                  = 3,
            ErrorDeinitialized                   = 4,
            ErrorProfilerDisabled                = 5,
            ErrorProfilerNotInitialized          = 6,
            ErrorProfilerAlreadyStarted          = 7,
            ErrorProfilerAlreadyStopped          = 8,
            ErrorNoDevice                      = 100,
            ErrorInvalidDevice                 = 101,
            ErrorInvalidImage                  = 200,
            ErrorInvalidContext                = 201,
            ErrorContextAlreadyCurrent         = 202,
            ErrorMapFailed                     = 205,
            ErrorUnmapFailed                   = 206,
            ErrorArrayIsMapped                 = 207,
            ErrorAlreadyMapped                 = 208,
            ErrorNoBinaryForGpu                = 209,
            ErrorAlreadyAcquired               = 210,
            ErrorNotMapped                     = 211,
            ErrorNotMappedAsArray              = 212,
            ErrorNotMappedAsPointer            = 213,
            ErrorEccUncorrectable              = 214,
            ErrorUnsupportedLimit              = 215,
            ErrorContextAlreadyInUse           = 216,
            ErrorPeerAccessUnsupported         = 217,
            ErrorInvalidPtx                    = 218,
            ErrorInvalidGraphicsContext        = 219,
            ErrorNvlinkUncorrectable           = 220,
            Error_jitCompilerNotFound          = 221,
            ErrorInvalidSource                 = 300,
            ErrorFileNotFound                  = 301,
            ErrorSharedObjectSymbolNotFound    = 302,
            ErrorSharedObjectInitFailed        = 303,
            ErrorOperatingSystem               = 304,
            ErrorInvalidHandle                 = 400,
            ErrorIllegalState                  = 401,
            ErrorNotFound                      = 500,
            ErrorNotReady                      = 600,
            ErrorIllegalAddress                = 700,
            ErrorLaunchOutOfResources          = 701,
            ErrorLaunchTimeout                 = 702,
            ErrorLaunchIncompatibleTexturing   = 703,
            ErrorPeerAccessAlreadyEnabled      = 704,
            ErrorPeerAccessNotEnabled          = 705,
            ErrorPrimaryContextActive          = 708,
            ErrorContextIsDestroyed            = 709,
            ErrorAssert                        = 710,
            ErrorTooManyPeers                  = 711,
            ErrorHostMemoryAlreadyRegistered   = 712,
            ErrorHostMemoryNotRegistered       = 713,
            ErrorHardwareStack                 = 714,
            ErrorIllegalInstruction            = 715,
            ErrorMisalignedAddress             = 716,
            ErrorInvalidAddressSpace           = 717,
            ErrorInvalidPc                     = 718,
            ErrorLaunchFailed                  = 719,
            ErrorCooperativeLaunchTooLarge     = 720,
            ErrorNotPermitted                  = 800,
            ErrorNotSupported                  = 801,
            ErrorSystemNotReady                = 802,
            ErrorSystemDriverMismatch          = 803,
            ErrorCompatNotSupportedOnDevice    = 804,
            ErrorStreamCaptureUnsupported      = 900,
            ErrorStreamCaptureInvalidated      = 901,
            ErrorStreamCaptureMerge            = 902,
            ErrorStreamCaptureUnmatched        = 903,
            ErrorStreamCaptureUnjoined         = 904,
            ErrorStreamCaptureIsolation        = 905,
            ErrorStreamCaptureImplicit         = 906,
            ErrorCapturedEvent                 = 907,
            ErrorStreamCaptureWrongThread      = 908,
            ErrorUnknown                       = 999
        }
    }
}
