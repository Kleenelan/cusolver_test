#include <assert.h>
#include <stdio.h>

#include "icla_types.h"

extern "C"
icla_bool_t   icla_bool_const  ( char lapack_char )
{
    switch( lapack_char ) {
        case 'N': case 'n': return iclaFalse;
        case 'Y': case 'y': return iclaTrue;
        default:
            fprintf( stderr, "Error in %s: unexpected value %c\n", __func__, lapack_char );
            return iclaFalse;
    }
}

extern "C"
icla_order_t  icla_order_const ( char lapack_char )
{
    switch( lapack_char ) {
        case 'R': case 'r': return iclaRowMajor;
        case 'C': case 'c': return iclaColMajor;
        default:
            fprintf( stderr, "Error in %s: unexpected value %c\n", __func__, lapack_char );
            return iclaRowMajor;
    }
}

extern "C"
icla_trans_t  icla_trans_const ( char lapack_char )
{
    switch( lapack_char ) {
        case 'N': case 'n': return iclaNoTrans;
        case 'T': case 't': return iclaTrans;
        case 'C': case 'c': return iclaConjTrans;
        default:
            fprintf( stderr, "Error in %s: unexpected value %c\n", __func__, lapack_char );
            return iclaNoTrans;
    }
}

extern "C"
icla_uplo_t   icla_uplo_const  ( char lapack_char )
{
    switch( lapack_char ) {
        case 'U': case 'u': return iclaUpper;
        case 'L': case 'l': return iclaLower;
        default:            return iclaFull;

    }
}

extern "C"
icla_diag_t   icla_diag_const  ( char lapack_char )
{
    switch( lapack_char ) {
        case 'N': case 'n': return iclaNonUnit;
        case 'U': case 'u': return iclaUnit;
        default:
            fprintf( stderr, "Error in %s: unexpected value %c\n", __func__, lapack_char );
            return iclaNonUnit;
    }
}

extern "C"
icla_side_t   icla_side_const  ( char lapack_char )
{
    switch( lapack_char ) {
        case 'L': case 'l': return iclaLeft;
        case 'R': case 'r': return iclaRight;
        case 'B': case 'b': return iclaBothSides;
        default:
            fprintf( stderr, "Error in %s: unexpected value %c\n", __func__, lapack_char );
            return iclaLeft;
    }
}

extern "C"
icla_norm_t   icla_norm_const  ( char lapack_char )
{
    switch( lapack_char ) {
        case 'O': case 'o': case '1': return iclaOneNorm;
        case '2':           return iclaTwoNorm;
        case 'F': case 'f': case 'E': case 'e': return iclaFrobeniusNorm;
        case 'I': case 'i': return iclaInfNorm;
        case 'M': case 'm': return iclaMaxNorm;

        default:
            fprintf( stderr, "Error in %s: unexpected value %c\n", __func__, lapack_char );
            return iclaOneNorm;
    }
}

extern "C"
icla_dist_t   icla_dist_const  ( char lapack_char )
{
    switch( lapack_char ) {
        case 'U': case 'u': return iclaDistUniform;
        case 'S': case 's': return iclaDistSymmetric;
        case 'N': case 'n': return iclaDistNormal;
        default:
            fprintf( stderr, "Error in %s: unexpected value %c\n", __func__, lapack_char );
            return iclaDistUniform;
    }
}

extern "C"
icla_sym_t    icla_sym_const   ( char lapack_char )
{
    switch( lapack_char ) {
        case 'H': case 'h': return iclaHermGeev;
        case 'P': case 'p': return iclaHermPoev;
        case 'N': case 'n': return iclaNonsymPosv;
        case 'S': case 's': return iclaSymPosv;
        default:
            fprintf( stderr, "Error in %s: unexpected value %c\n", __func__, lapack_char );
            return iclaHermGeev;
    }
}

extern "C"
icla_pack_t   icla_pack_const  ( char lapack_char )
{
    switch( lapack_char ) {
        case 'N': case 'n': return iclaNoPacking;
        case 'U': case 'u': return iclaPackSubdiag;
        case 'L': case 'l': return iclaPackSupdiag;
        case 'C': case 'c': return iclaPackColumn;
        case 'R': case 'r': return iclaPackRow;
        case 'B': case 'b': return iclaPackLowerBand;
        case 'Q': case 'q': return iclaPackUpeprBand;
        case 'Z': case 'z': return iclaPackAll;
        default:
            fprintf( stderr, "Error in %s: unexpected value %c\n", __func__, lapack_char );
            return iclaNoPacking;
    }
}

extern "C"
icla_vec_t    icla_vec_const   ( char lapack_char )
{
    switch( lapack_char ) {
        case 'N': case 'n': return iclaNoVec;
        case 'V': case 'v': return iclaVec;
        case 'I': case 'i': return iclaIVec;
        case 'A': case 'a': return iclaAllVec;
        case 'S': case 's': return iclaSomeVec;
        case 'O': case 'o': return iclaOverwriteVec;
        default:
            fprintf( stderr, "Error in %s: unexpected value %c\n", __func__, lapack_char );
            return iclaNoVec;
    }
}

extern "C"
icla_range_t  icla_range_const ( char lapack_char )
{
    switch( lapack_char ) {
        case 'A': case 'a': return iclaRangeAll;
        case 'V': case 'v': return iclaRangeV;
        case 'I': case 'i': return iclaRangeI;
        default:
            fprintf( stderr, "Error in %s: unexpected value %c\n", __func__, lapack_char );
            return iclaRangeAll;
    }
}

extern "C"
icla_vect_t icla_vect_const( char lapack_char )
{
    switch( lapack_char ) {
        case 'Q': case 'q': return iclaQ;
        case 'P': case 'p': return iclaP;
        default:
            fprintf( stderr, "Error in %s: unexpected value %c\n", __func__, lapack_char );
            return iclaQ;
    }
}

extern "C"
icla_direct_t icla_direct_const( char lapack_char )
{
    switch( lapack_char ) {
        case 'F': case 'f': return iclaForward;
        case 'B': case 'b': return iclaBackward;
        default:
            fprintf( stderr, "Error in %s: unexpected value %c\n", __func__, lapack_char );
            return iclaForward;
    }
}

extern "C"
icla_storev_t icla_storev_const( char lapack_char )
{
    switch( lapack_char ) {
        case 'C': case 'c': return iclaColumnwise;
        case 'R': case 'r': return iclaRowwise;
        default:
            fprintf( stderr, "Error in %s: unexpected value %c\n", __func__, lapack_char );
            return iclaColumnwise;
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
    assert( icla_const >= iclaFalse );
    assert( icla_const <= iclaTrue  );
    return icla2lapack_constants[ icla_const ];
}

extern "C"
const char* lapack_order_const( icla_order_t icla_const )
{
    assert( icla_const >= iclaRowMajor );
    assert( icla_const <= iclaColMajor );
    return icla2lapack_constants[ icla_const ];
}

extern "C"
const char* lapack_trans_const( icla_trans_t icla_const )
{
    assert( icla_const >= iclaNoTrans   );
    assert( icla_const <= iclaConjTrans );
    return icla2lapack_constants[ icla_const ];
}

extern "C"
const char* lapack_uplo_const ( icla_uplo_t icla_const )
{
    assert( icla_const >= iclaUpper );
    assert( icla_const <= iclaFull  );
    return icla2lapack_constants[ icla_const ];
}

extern "C"
const char* lapack_diag_const ( icla_diag_t icla_const )
{
    assert( icla_const >= iclaNonUnit );
    assert( icla_const <= iclaUnit    );
    return icla2lapack_constants[ icla_const ];
}

extern "C"
const char* lapack_side_const ( icla_side_t icla_const )
{
    assert( icla_const >= iclaLeft  );
    assert( icla_const <= iclaBothSides );
    return icla2lapack_constants[ icla_const ];
}

extern "C"
const char* lapack_norm_const  ( icla_norm_t   icla_const )
{
    assert( icla_const >= iclaOneNorm     );
    assert( icla_const <= iclaRealMaxNorm );
    return icla2lapack_constants[ icla_const ];
}

extern "C"
const char* lapack_dist_const  ( icla_dist_t   icla_const )
{
    assert( icla_const >= iclaDistUniform );
    assert( icla_const <= iclaDistNormal );
    return icla2lapack_constants[ icla_const ];
}

extern "C"
const char* lapack_sym_const   ( icla_sym_t    icla_const )
{
    assert( icla_const >= iclaHermGeev );
    assert( icla_const <= iclaSymPosv  );
    return icla2lapack_constants[ icla_const ];
}

extern "C"
const char* lapack_pack_const  ( icla_pack_t   icla_const )
{
    assert( icla_const >= iclaNoPacking );
    assert( icla_const <= iclaPackAll   );
    return icla2lapack_constants[ icla_const ];
}

extern "C"
const char* lapack_vec_const   ( icla_vec_t    icla_const )
{
    assert( icla_const >= iclaNoVec );
    assert( icla_const <= iclaOverwriteVec );
    return icla2lapack_constants[ icla_const ];
}

extern "C"
const char* lapack_range_const ( icla_range_t  icla_const )
{
    assert( icla_const >= iclaRangeAll );
    assert( icla_const <= iclaRangeI   );
    return icla2lapack_constants[ icla_const ];
}

extern "C"
const char* lapack_vect_const( icla_vect_t icla_const )
{
    assert( icla_const >= iclaQ );
    assert( icla_const <= iclaP );
    return icla2lapack_constants[ icla_const ];
}

extern "C"
const char* lapack_direct_const( icla_direct_t icla_const )
{
    assert( icla_const >= iclaForward );
    assert( icla_const <= iclaBackward );
    return icla2lapack_constants[ icla_const ];
}

extern "C"
const char* lapack_storev_const( icla_storev_t icla_const )
{
    assert( icla_const >= iclaColumnwise );
    assert( icla_const <= iclaRowwise    );
    return icla2lapack_constants[ icla_const ];
}

#ifdef ICLA_HAVE_clBLAS

const int icla2clblas_constants[] =
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

    clblasRowMajor,

    clblasColumnMajor,

    0, 0, 0, 0, 0, 0, 0, 0,
    clblasNoTrans,

    clblasTrans,

    clblasConjTrans,

    0, 0, 0, 0, 0, 0, 0,
    clblasUpper,

    clblasLower,

    0, 0, 0, 0, 0, 0, 0, 0,
    clblasNonUnit,

    clblasUnit,

    0, 0, 0, 0, 0, 0, 0, 0,
    clblasLeft,

    clblasRight,

    0, 0, 0, 0, 0, 0, 0, 0
};

extern "C"
clblasOrder       clblas_order_const( icla_order_t icla_const )
{
    assert( icla_const >= iclaRowMajor );
    assert( icla_const <= iclaColMajor );
    return (clblasOrder)     icla2clblas_constants[ icla_const ];
}

extern "C"
clblasTranspose   clblas_trans_const( icla_trans_t icla_const )
{
    assert( icla_const >= iclaNoTrans   );
    assert( icla_const <= iclaConjTrans );
    return (clblasTranspose) icla2clblas_constants[ icla_const ];
}

extern "C"
clblasUplo        clblas_uplo_const ( icla_uplo_t icla_const )
{
    assert( icla_const >= iclaUpper );
    assert( icla_const <= iclaLower );
    return (clblasUplo)      icla2clblas_constants[ icla_const ];
}

extern "C"
clblasDiag        clblas_diag_const ( icla_diag_t icla_const )
{
    assert( icla_const >= iclaNonUnit );
    assert( icla_const <= iclaUnit    );
    return (clblasDiag)      icla2clblas_constants[ icla_const ];
}

extern "C"
clblasSide        clblas_side_const ( icla_side_t icla_const )
{
    assert( icla_const >= iclaLeft  );
    assert( icla_const <= iclaRight );
    return (clblasSide)      icla2clblas_constants[ icla_const ];
}

#endif

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
    assert( icla_const >= iclaNoTrans   );
    assert( icla_const <= iclaConjTrans );
    return (cublasOperation_t)  icla2cublas_constants[ icla_const ];
}

extern "C"
cublasFillMode_t     cublas_uplo_const  ( icla_uplo_t icla_const )
{
    assert( icla_const >= iclaUpper );
    assert( icla_const <= iclaLower );
    return (cublasFillMode_t)   icla2cublas_constants[ icla_const ];
}

extern "C"
cublasDiagType_t     cublas_diag_const  ( icla_diag_t icla_const )
{
    assert( icla_const >= iclaNonUnit );
    assert( icla_const <= iclaUnit    );
    return (cublasDiagType_t)   icla2cublas_constants[ icla_const ];
}

extern "C"
cublasSideMode_t     cublas_side_const  ( icla_side_t icla_const )
{
    assert( icla_const >= iclaLeft  );
    assert( icla_const <= iclaRight );
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
    assert( icla_const >= iclaNoTrans   );
    assert( icla_const <= iclaConjTrans );
    return (hipblasOperation_t)  icla2hipblas_constants[ icla_const ];
}

extern "C"
hipblasFillMode_t     hipblas_uplo_const  ( icla_uplo_t icla_const )
{
    assert( icla_const >= iclaUpper );
    assert( icla_const <= iclaLower );
    return (hipblasFillMode_t)   icla2hipblas_constants[ icla_const ];
}

extern "C"
hipblasDiagType_t     hipblas_diag_const  ( icla_diag_t icla_const )
{
    assert( icla_const >= iclaNonUnit );
    assert( icla_const <= iclaUnit    );
    return (hipblasDiagType_t)   icla2hipblas_constants[ icla_const ];
}

extern "C"
hipblasSideMode_t     hipblas_side_const  ( icla_side_t icla_const )
{
    assert( icla_const >= iclaLeft  );
    assert( icla_const <= iclaRight );
    return (hipblasSideMode_t)   icla2hipblas_constants[ icla_const ];
}

#endif

#ifdef HAVE_CBLAS

extern "C"
enum CBLAS_ORDER     cblas_order_const  ( icla_order_t icla_const )
{
    assert( icla_const >= iclaRowMajor );
    assert( icla_const <= iclaColMajor );
    assert( (int)iclaRowMajor == CblasRowMajor );
    return (enum CBLAS_ORDER)     icla_const;
}

extern "C"
enum CBLAS_TRANSPOSE cblas_trans_const  ( icla_trans_t icla_const )
{
    assert( icla_const >= iclaNoTrans   );
    assert( icla_const <= iclaConjTrans );
    assert( (int)iclaNoTrans == CblasNoTrans );
    return (enum CBLAS_TRANSPOSE) icla_const;
}

extern "C"
enum CBLAS_UPLO      cblas_uplo_const   ( icla_uplo_t icla_const )
{
    assert( icla_const >= iclaUpper );
    assert( icla_const <= iclaLower );
    assert( (int)iclaUpper == CblasUpper );
    return (enum CBLAS_UPLO)      icla_const;
}

extern "C"
enum CBLAS_DIAG      cblas_diag_const   ( icla_diag_t icla_const )
{
    assert( icla_const >= iclaNonUnit );
    assert( icla_const <= iclaUnit    );
    assert( (int)iclaUnit == CblasUnit );
    return (enum CBLAS_DIAG)      icla_const;
}

extern "C"
enum CBLAS_SIDE      cblas_side_const   ( icla_side_t icla_const )
{
    assert( icla_const >= iclaLeft  );
    assert( icla_const <= iclaRight );
    assert( (int)iclaLeft == CblasLeft );
    return (enum CBLAS_SIDE)      icla_const;
}

#endif

