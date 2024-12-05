#include <assert.h>
#include <stdio.h>

#include "icla_types.h"

extern "C"
icla_bool_t   icla_bool_const  ( char lapack_char )
{
    switch( lapack_char ) {
        case 'N': case 'n': return IclaFalse;
        case 'Y': case 'y': return IclaTrue;
        default:
            fprintf( stderr, "Error in %s: unexpected value %c\n", __func__, lapack_char );
            return IclaFalse;
    }
}

extern "C"
icla_order_t  icla_order_const ( char lapack_char )
{
    switch( lapack_char ) {
        case 'R': case 'r': return IclaRowMajor;
        case 'C': case 'c': return IclaColMajor;
        default:
            fprintf( stderr, "Error in %s: unexpected value %c\n", __func__, lapack_char );
            return IclaRowMajor;
    }
}

extern "C"
icla_trans_t  icla_trans_const ( char lapack_char )
{
    switch( lapack_char ) {
        case 'N': case 'n': return IclaNoTrans;
        case 'T': case 't': return IclaTrans;
        case 'C': case 'c': return IclaConjTrans;
        default:
            fprintf( stderr, "Error in %s: unexpected value %c\n", __func__, lapack_char );
            return IclaNoTrans;
    }
}

extern "C"
icla_uplo_t   icla_uplo_const  ( char lapack_char )
{
    switch( lapack_char ) {
        case 'U': case 'u': return IclaUpper;
        case 'L': case 'l': return IclaLower;
        default:            return IclaFull;

    }
}

extern "C"
icla_diag_t   icla_diag_const  ( char lapack_char )
{
    switch( lapack_char ) {
        case 'N': case 'n': return IclaNonUnit;
        case 'U': case 'u': return IclaUnit;
        default:
            fprintf( stderr, "Error in %s: unexpected value %c\n", __func__, lapack_char );
            return IclaNonUnit;
    }
}

extern "C"
icla_side_t   icla_side_const  ( char lapack_char )
{
    switch( lapack_char ) {
        case 'L': case 'l': return IclaLeft;
        case 'R': case 'r': return IclaRight;
        case 'B': case 'b': return IclaBothSides;
        default:
            fprintf( stderr, "Error in %s: unexpected value %c\n", __func__, lapack_char );
            return IclaLeft;
    }
}

extern "C"
icla_norm_t   icla_norm_const  ( char lapack_char )
{
    switch( lapack_char ) {
        case 'O': case 'o': case '1': return IclaOneNorm;
        case '2':           return IclaTwoNorm;
        case 'F': case 'f': case 'E': case 'e': return IclaFrobeniusNorm;
        case 'I': case 'i': return IclaInfNorm;
        case 'M': case 'm': return IclaMaxNorm;

        default:
            fprintf( stderr, "Error in %s: unexpected value %c\n", __func__, lapack_char );
            return IclaOneNorm;
    }
}

extern "C"
icla_dist_t   icla_dist_const  ( char lapack_char )
{
    switch( lapack_char ) {
        case 'U': case 'u': return IclaDistUniform;
        case 'S': case 's': return IclaDistSymmetric;
        case 'N': case 'n': return IclaDistNormal;
        default:
            fprintf( stderr, "Error in %s: unexpected value %c\n", __func__, lapack_char );
            return IclaDistUniform;
    }
}

extern "C"
icla_sym_t    icla_sym_const   ( char lapack_char )
{
    switch( lapack_char ) {
        case 'H': case 'h': return IclaHermGeev;
        case 'P': case 'p': return IclaHermPoev;
        case 'N': case 'n': return IclaNonsymPosv;
        case 'S': case 's': return IclaSymPosv;
        default:
            fprintf( stderr, "Error in %s: unexpected value %c\n", __func__, lapack_char );
            return IclaHermGeev;
    }
}

extern "C"
icla_pack_t   icla_pack_const  ( char lapack_char )
{
    switch( lapack_char ) {
        case 'N': case 'n': return IclaNoPacking;
        case 'U': case 'u': return IclaPackSubdiag;
        case 'L': case 'l': return IclaPackSupdiag;
        case 'C': case 'c': return IclaPackColumn;
        case 'R': case 'r': return IclaPackRow;
        case 'B': case 'b': return IclaPackLowerBand;
        case 'Q': case 'q': return IclaPackUpeprBand;
        case 'Z': case 'z': return IclaPackAll;
        default:
            fprintf( stderr, "Error in %s: unexpected value %c\n", __func__, lapack_char );
            return IclaNoPacking;
    }
}

extern "C"
icla_vec_t    icla_vec_const   ( char lapack_char )
{
    switch( lapack_char ) {
        case 'N': case 'n': return IclaNoVec;
        case 'V': case 'v': return IclaVec;
        case 'I': case 'i': return IclaIVec;
        case 'A': case 'a': return IclaAllVec;
        case 'S': case 's': return IclaSomeVec;
        case 'O': case 'o': return IclaOverwriteVec;
        default:
            fprintf( stderr, "Error in %s: unexpected value %c\n", __func__, lapack_char );
            return IclaNoVec;
    }
}

extern "C"
icla_range_t  icla_range_const ( char lapack_char )
{
    switch( lapack_char ) {
        case 'A': case 'a': return IclaRangeAll;
        case 'V': case 'v': return IclaRangeV;
        case 'I': case 'i': return IclaRangeI;
        default:
            fprintf( stderr, "Error in %s: unexpected value %c\n", __func__, lapack_char );
            return IclaRangeAll;
    }
}

extern "C"
icla_vect_t icla_vect_const( char lapack_char )
{
    switch( lapack_char ) {
        case 'Q': case 'q': return IclaQ;
        case 'P': case 'p': return IclaP;
        default:
            fprintf( stderr, "Error in %s: unexpected value %c\n", __func__, lapack_char );
            return IclaQ;
    }
}

extern "C"
icla_direct_t icla_direct_const( char lapack_char )
{
    switch( lapack_char ) {
        case 'F': case 'f': return IclaForward;
        case 'B': case 'b': return IclaBackward;
        default:
            fprintf( stderr, "Error in %s: unexpected value %c\n", __func__, lapack_char );
            return IclaForward;
    }
}

extern "C"
icla_storev_t icla_storev_const( char lapack_char )
{
    switch( lapack_char ) {
        case 'C': case 'c': return IclaColumnwise;
        case 'R': case 'r': return IclaRowwise;
        default:
            fprintf( stderr, "Error in %s: unexpected value %c\n", __func__, lapack_char );
            return IclaColumnwise;
    }
}

const char *icla2lapack_constants[] =
{
    "No",

    "Yes",

    "", "", "", "", "", "", "", "", "",

    "", "", "", "", "", "", "", "", "", "",

    "", "", "", "", "", "", "", "", "", "",

    "", "", "", "", "", "", "", "", "", "",

    "", "", "", "", "", "", "", "", "", "",

    "", "", "", "", "", "", "", "", "", "",

    "", "", "", "", "", "", "", "", "", "",

    "", "", "", "", "", "", "", "", "", "",

    "", "", "", "", "", "", "", "", "", "",

    "", "", "", "", "", "", "", "", "", "",

    "Row",

    "Column",

    "", "", "", "", "", "", "", "",

    "No transpose",

    "Transpose",

    "Conjugate transpose",

    "", "", "", "", "", "", "",

    "Upper",

    "Lower",

    "General",

    "", "", "", "", "", "", "",

    "Non-unit",

    "Unit",

    "", "", "", "", "", "", "", "",

    "Left",

    "Right",

    "Both",

    "", "", "", "", "", "", "",

    "", "", "", "", "", "", "", "", "", "",

    "", "", "", "", "", "", "", "", "", "",

    "1 norm",

    "",

    "2 norm",

    "Frobenius norm",

    "Infinity norm",

    "",

    "Maximum norm",

    "",

    "", "",

    "", "", "", "", "", "", "", "", "", "",

    "", "", "", "", "", "", "", "", "", "",

    "Uniform",

    "Symmetric",

    "Normal",

    "", "", "", "", "", "", "",

    "", "", "", "", "", "", "", "", "", "",

    "", "", "", "", "", "", "", "", "", "",

    "", "", "", "", "", "", "", "", "", "",

    "Hermitian",

    "Positive ev Hermitian",

    "NonSymmetric pos sv",

    "Symmetric pos sv",

    "", "", "", "", "", "",

    "", "", "", "", "", "", "", "", "", "",

    "", "", "", "", "", "", "", "", "", "",

    "", "", "", "", "", "", "", "", "", "",

    "", "", "", "", "", "", "", "", "", "",

    "No Packing",

    "U zero out subdiag",

    "L zero out superdiag",

    "C",

    "R",

    "B",

    "Q",

    "Z",

    "", "",

    "No vectors",

    "Vectors needed",

    "I",

    "All",

    "Some",

    "Overwrite",

    "", "", "", "",

    "All",

    "V",

    "I",

    "", "", "", "", "", "", "",

    "",

    "Q",

    "P",

    "", "", "", "", "", "", "",

    "", "", "", "", "", "", "", "", "", "",

    "", "", "", "", "", "", "", "", "", "",

    "", "", "", "", "", "", "", "", "", "",

    "", "", "", "", "", "", "", "", "", "",

    "", "", "", "", "", "", "", "", "", "",

    "", "", "", "", "", "", "", "", "", "",

    "Forward",

    "Backward",

    "", "", "", "", "", "", "", "",

    "Columnwise",

    "Rowwise",

    "", "", "", "", "", "", "", ""

};

extern "C"
const char* lapack_const_str( int icla_const )
{
    assert( icla_const >= icla2lapack_Min );
    assert( icla_const <= icla2lapack_Max );
    return icla2lapack_constants[ icla_const ];
}

extern "C"
const char* lapack_bool_const( icla_bool_t icla_const )
{
    assert( icla_const >= IclaFalse );
    assert( icla_const <= IclaTrue  );
    return icla2lapack_constants[ icla_const ];
}

extern "C"
const char* lapack_order_const( icla_order_t icla_const )
{
    assert( icla_const >= IclaRowMajor );
    assert( icla_const <= IclaColMajor );
    return icla2lapack_constants[ icla_const ];
}

extern "C"
const char* lapack_trans_const( icla_trans_t icla_const )
{
    assert( icla_const >= IclaNoTrans   );
    assert( icla_const <= IclaConjTrans );
    return icla2lapack_constants[ icla_const ];
}

extern "C"
const char* lapack_uplo_const ( icla_uplo_t icla_const )
{
    assert( icla_const >= IclaUpper );
    assert( icla_const <= IclaFull  );
    return icla2lapack_constants[ icla_const ];
}

extern "C"
const char* lapack_diag_const ( icla_diag_t icla_const )
{
    assert( icla_const >= IclaNonUnit );
    assert( icla_const <= IclaUnit    );
    return icla2lapack_constants[ icla_const ];
}

extern "C"
const char* lapack_side_const ( icla_side_t icla_const )
{
    assert( icla_const >= IclaLeft  );
    assert( icla_const <= IclaBothSides );
    return icla2lapack_constants[ icla_const ];
}

extern "C"
const char* lapack_norm_const  ( icla_norm_t   icla_const )
{
    assert( icla_const >= IclaOneNorm     );
    assert( icla_const <= iclaRealMaxNorm );
    return icla2lapack_constants[ icla_const ];
}

extern "C"
const char* lapack_dist_const  ( icla_dist_t   icla_const )
{
    assert( icla_const >= IclaDistUniform );
    assert( icla_const <= IclaDistNormal );
    return icla2lapack_constants[ icla_const ];
}

extern "C"
const char* lapack_sym_const   ( icla_sym_t    icla_const )
{
    assert( icla_const >= IclaHermGeev );
    assert( icla_const <= IclaSymPosv  );
    return icla2lapack_constants[ icla_const ];
}

extern "C"
const char* lapack_pack_const  ( icla_pack_t   icla_const )
{
    assert( icla_const >= IclaNoPacking );
    assert( icla_const <= IclaPackAll   );
    return icla2lapack_constants[ icla_const ];
}

extern "C"
const char* lapack_vec_const   ( icla_vec_t    icla_const )
{
    assert( icla_const >= IclaNoVec );
    assert( icla_const <= IclaOverwriteVec );
    return icla2lapack_constants[ icla_const ];
}

extern "C"
const char* lapack_range_const ( icla_range_t  icla_const )
{
    assert( icla_const >= IclaRangeAll );
    assert( icla_const <= IclaRangeI   );
    return icla2lapack_constants[ icla_const ];
}

extern "C"
const char* lapack_vect_const( icla_vect_t icla_const )
{
    assert( icla_const >= IclaQ );
    assert( icla_const <= IclaP );
    return icla2lapack_constants[ icla_const ];
}

extern "C"
const char* lapack_direct_const( icla_direct_t icla_const )
{
    assert( icla_const >= IclaForward );
    assert( icla_const <= IclaBackward );
    return icla2lapack_constants[ icla_const ];
}

extern "C"
const char* lapack_storev_const( icla_storev_t icla_const )
{
    assert( icla_const >= IclaColumnwise );
    assert( icla_const <= IclaRowwise    );
    return icla2lapack_constants[ icla_const ];
}

#ifdef ICLA_HAVE_CUDA

const int icla2cublas_constants[] =
{
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0,

    0,

    0,

    0, 0, 0, 0, 0, 0, 0, 0,
    CUBLAS_OP_N,

    CUBLAS_OP_T,

    CUBLAS_OP_C,

    0, 0, 0, 0, 0, 0, 0,
    CUBLAS_FILL_MODE_UPPER,

    CUBLAS_FILL_MODE_LOWER,

    0, 0, 0, 0, 0, 0, 0, 0,
    CUBLAS_DIAG_NON_UNIT,

    CUBLAS_DIAG_UNIT,

    0, 0, 0, 0, 0, 0, 0, 0,
    CUBLAS_SIDE_LEFT,

    CUBLAS_SIDE_RIGHT,

    0, 0, 0, 0, 0, 0, 0, 0
};

extern "C"
cublasOperation_t    cublas_trans_const ( icla_trans_t icla_const )
{
    assert( icla_const >= IclaNoTrans   );
    assert( icla_const <= IclaConjTrans );
    return (cublasOperation_t)  icla2cublas_constants[ icla_const ];
}

extern "C"
cublasFillMode_t     cublas_uplo_const  ( icla_uplo_t icla_const )
{
    assert( icla_const >= IclaUpper );
    assert( icla_const <= IclaLower );
    return (cublasFillMode_t)   icla2cublas_constants[ icla_const ];
}

extern "C"
cublasDiagType_t     cublas_diag_const  ( icla_diag_t icla_const )
{
    assert( icla_const >= IclaNonUnit );
    assert( icla_const <= IclaUnit    );
    return (cublasDiagType_t)   icla2cublas_constants[ icla_const ];
}

extern "C"
cublasSideMode_t     cublas_side_const  ( icla_side_t icla_const )
{
    assert( icla_const >= IclaLeft  );
    assert( icla_const <= IclaRight );
    return (cublasSideMode_t)   icla2cublas_constants[ icla_const ];
}

#endif

#ifdef ICLA_HAVE_HIP

const int icla2hipblas_constants[] =
{
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0,

    0,

    0,

    0, 0, 0, 0, 0, 0, 0, 0,
    HIPBLAS_OP_N,

    HIPBLAS_OP_T,

    HIPBLAS_OP_C,

    0, 0, 0, 0, 0, 0, 0,
    HIPBLAS_FILL_MODE_UPPER,

    HIPBLAS_FILL_MODE_LOWER,

    0, 0, 0, 0, 0, 0, 0, 0,
    HIPBLAS_DIAG_NON_UNIT,

    HIPBLAS_DIAG_UNIT,

    0, 0, 0, 0, 0, 0, 0, 0,
    HIPBLAS_SIDE_LEFT,

    HIPBLAS_SIDE_RIGHT,

    0, 0, 0, 0, 0, 0, 0, 0
};

extern "C"
hipblasOperation_t    hipblas_trans_const ( icla_trans_t icla_const )
{
    assert( icla_const >= IclaNoTrans   );
    assert( icla_const <= IclaConjTrans );
    return (hipblasOperation_t)  icla2hipblas_constants[ icla_const ];
}

extern "C"
hipblasFillMode_t     hipblas_uplo_const  ( icla_uplo_t icla_const )
{
    assert( icla_const >= IclaUpper );
    assert( icla_const <= IclaLower );
    return (hipblasFillMode_t)   icla2hipblas_constants[ icla_const ];
}

extern "C"
hipblasDiagType_t     hipblas_diag_const  ( icla_diag_t icla_const )
{
    assert( icla_const >= IclaNonUnit );
    assert( icla_const <= IclaUnit    );
    return (hipblasDiagType_t)   icla2hipblas_constants[ icla_const ];
}

extern "C"
hipblasSideMode_t     hipblas_side_const  ( icla_side_t icla_const )
{
    assert( icla_const >= IclaLeft  );
    assert( icla_const <= IclaRight );
    return (hipblasSideMode_t)   icla2hipblas_constants[ icla_const ];
}

#endif

#ifdef HAVE_CBLAS

extern "C"
enum CBLAS_ORDER     cblas_order_const  ( icla_order_t icla_const )
{
    assert( icla_const >= IclaRowMajor );
    assert( icla_const <= IclaColMajor );
    assert( (int)IclaRowMajor == CblasRowMajor );
    return (enum CBLAS_ORDER)     icla_const;
}

extern "C"
enum CBLAS_TRANSPOSE cblas_trans_const  ( icla_trans_t icla_const )
{
    assert( icla_const >= IclaNoTrans   );
    assert( icla_const <= IclaConjTrans );
    assert( (int)IclaNoTrans == CblasNoTrans );
    return (enum CBLAS_TRANSPOSE) icla_const;
}

extern "C"
enum CBLAS_UPLO      cblas_uplo_const   ( icla_uplo_t icla_const )
{
    assert( icla_const >= IclaUpper );
    assert( icla_const <= IclaLower );
    assert( (int)IclaUpper == CblasUpper );
    return (enum CBLAS_UPLO)      icla_const;
}

extern "C"
enum CBLAS_DIAG      cblas_diag_const   ( icla_diag_t icla_const )
{
    assert( icla_const >= IclaNonUnit );
    assert( icla_const <= IclaUnit    );
    assert( (int)IclaUnit == CblasUnit );
    return (enum CBLAS_DIAG)      icla_const;
}

extern "C"
enum CBLAS_SIDE      cblas_side_const   ( icla_side_t icla_const )
{
    assert( icla_const >= IclaLeft  );
    assert( icla_const <= IclaRight );
    assert( (int)IclaLeft == CblasLeft );
    return (enum CBLAS_SIDE)      icla_const;
}

#endif

