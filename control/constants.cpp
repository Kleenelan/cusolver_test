#include <assert.h>
#include <stdio.h>

#include "icla_types.h"

// =============================================================================
/// @addtogroup icla_const
/// Convert LAPACK character constants to ICLA constants.
/// This is a one-to-many mapping, requiring multiple translators
/// (e.g., "N" can be NoTrans or NonUnit or NoVec).
/// Matching is case-insensitive.
/// @{

// These functions and cases are in the same order as the constants are
// declared in icla_types.h

/******************************************************************************/
/// @retval iclaFalse if lapack_char = 'N'
/// @retval iclaTrue  if lapack_char = 'Y'
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

/******************************************************************************/
/// @retval iclaRowMajor if lapack_char = 'R'
/// @retval iclaColMajor if lapack_char = 'C'
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

/******************************************************************************/
/// @retval iclaNoTrans   if lapack_char = 'N'
/// @retval iclaTrans     if lapack_char = 'T'
/// @retval iclaConjTrans if lapack_char = 'C'
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

/******************************************************************************/
/// @retval iclaUpper if lapack_char = 'U'
/// @retval iclaLower if lapack_char = 'L'
/// @retval iclaFull  otherwise
extern "C"
icla_uplo_t   icla_uplo_const  ( char lapack_char )
{
    switch( lapack_char ) {
        case 'U': case 'u': return iclaUpper;
        case 'L': case 'l': return iclaLower;
        default:            return iclaFull;        // see laset
    }
}

/******************************************************************************/
/// @retval iclaNonUnit if lapack_char = 'N'
/// @retval iclaUnit    if lapack_char = 'U'
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

/******************************************************************************/
/// @retval iclaLeft      if lapack_char = 'L'
/// @retval iclaRight     if lapack_char = 'R'
/// @retval iclaBothSides if lapack_char = 'B'
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

/******************************************************************************/
/// @retval iclaOneNorm       if lapack_char = '1' or 'O'
/// @retval iclaTwoNorm       if lapack_char = '2'
/// @retval iclaFrobeniusNorm if lapack_char = 'F' or 'E'
/// @retval iclaInfNorm       if lapack_char = 'I'
/// @retval iclaMaxNorm       if lapack_char = 'M'
extern "C"
icla_norm_t   icla_norm_const  ( char lapack_char )
{
    switch( lapack_char ) {
        case 'O': case 'o': case '1': return iclaOneNorm;
        case '2':           return iclaTwoNorm;
        case 'F': case 'f': case 'E': case 'e': return iclaFrobeniusNorm;
        case 'I': case 'i': return iclaInfNorm;
        case 'M': case 'm': return iclaMaxNorm;
        // iclaRealOneNorm
        // iclaRealInfNorm
        // iclaRealMaxNorm
        default:
            fprintf( stderr, "Error in %s: unexpected value %c\n", __func__, lapack_char );
            return iclaOneNorm;
    }
}

/******************************************************************************/
/// @retval iclaDistUniform   if lapack_char = 'U'
/// @retval iclaDistSymmetric if lapack_char = 'S'
/// @retval iclaDistNormal    if lapack_char = 'N'
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

/******************************************************************************/
/// @retval iclaHermGeev   if lapack_char = 'H'
/// @retval iclaHermPoev   if lapack_char = 'P'
/// @retval iclaNonsymPosv if lapack_char = 'N'
/// @retval iclaSymPosv    if lapack_char = 'S'
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

/******************************************************************************/
/// @retval iclaNoPacking     if lapack_char = 'N'
/// @retval iclaPackSubdiag   if lapack_char = 'U'
/// @retval iclaPackSupdiag   if lapack_char = 'L'
/// @retval iclaPackColumn    if lapack_char = 'C'
/// @retval iclaPackRow       if lapack_char = 'R'
/// @retval iclaPackLowerBand if lapack_char = 'B'
/// @retval iclaPackUpeprBand if lapack_char = 'Q'
/// @retval iclaPackAll       if lapack_char = 'Z'
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

/******************************************************************************/
/// @retval iclaNoVec        if lapack_char = 'N'
/// @retval iclaVec          if lapack_char = 'V'
/// @retval iclaIVec         if lapack_char = 'I'
/// @retval iclaAllVec       if lapack_char = 'A'
/// @retval iclaSomeVec      if lapack_char = 'S'
/// @retval iclaOverwriteVec if lapack_char = 'O'
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

/******************************************************************************/
/// @retval iclaRangeAll if lapack_char = 'A'
/// @retval iclaRangeV   if lapack_char = 'V'
/// @retval iclaRangeI   if lapack_char = 'I'
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

/******************************************************************************/
/// @retval iclaQ if lapack_char = 'Q'
/// @retval iclaP if lapack_char = 'P'
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

/******************************************************************************/
/// @retval iclaForward  if lapack_char = 'F'
/// @retval iclaBackward if lapack_char = 'B'
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

/******************************************************************************/
/// @retval iclaColumnwise if lapack_char = 'C'
/// @retval iclaRowwise    if lapack_char = 'R'
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

// =============================================================================
/// @}
// end group icla_const


// =============================================================================
/// @addtogroup lapack_const
/// Convert ICLA constants to LAPACK constants.
/// Though LAPACK only cares about the first character,
/// the string is generally descriptive, such as "Upper".
/// @{

// The icla2lapack_constants table has an entry for each ICLA constant,
// enumerated on the right, with a corresponding LAPACK string.
// The lapack_*_const() functions return entries from this table.
// The lapacke_*_const() functions defined in icla_types.h
// return a single character (e.g., 'U' for "Upper").

const char *icla2lapack_constants[] =
{
    "No",                                    //  0: iclaFalse
    "Yes",                                   //  1: iclaTrue (zlatrs)
    "", "", "", "", "", "", "", "", "",      //  2-10
    "", "", "", "", "", "", "", "", "", "",  // 11-20
    "", "", "", "", "", "", "", "", "", "",  // 21-30
    "", "", "", "", "", "", "", "", "", "",  // 31-40
    "", "", "", "", "", "", "", "", "", "",  // 41-50
    "", "", "", "", "", "", "", "", "", "",  // 51-60
    "", "", "", "", "", "", "", "", "", "",  // 61-70
    "", "", "", "", "", "", "", "", "", "",  // 71-80
    "", "", "", "", "", "", "", "", "", "",  // 81-90
    "", "", "", "", "", "", "", "", "", "",  // 91-100
    "Row",                                   // 101: iclaRowMajor
    "Column",                                // 102: iclaColMajor
    "", "", "", "", "", "", "", "",          // 103-110
    "No transpose",                          // 111: iclaNoTrans
    "Transpose",                             // 112: iclaTrans
    "Conjugate transpose",                   // 113: iclaConjTrans
    "", "", "", "", "", "", "",              // 114-120
    "Upper",                                 // 121: iclaUpper
    "Lower",                                 // 122: iclaLower
    "General",                               // 123: iclaFull; see lascl for "G"
    "", "", "", "", "", "", "",              // 124-130
    "Non-unit",                              // 131: iclaNonUnit
    "Unit",                                  // 132: iclaUnit
    "", "", "", "", "", "", "", "",          // 133-140
    "Left",                                  // 141: iclaLeft
    "Right",                                 // 142: iclaRight
    "Both",                                  // 143: iclaBothSides (dtrevc)
    "", "", "", "", "", "", "",              // 144-150
    "", "", "", "", "", "", "", "", "", "",  // 151-160
    "", "", "", "", "", "", "", "", "", "",  // 161-170
    "1 norm",                                // 171: iclaOneNorm
    "",                                      // 172: iclaRealOneNorm
    "2 norm",                                // 173: iclaTwoNorm
    "Frobenius norm",                        // 174: iclaFrobeniusNorm
    "Infinity norm",                         // 175: iclaInfNorm
    "",                                      // 176: iclaRealInfNorm
    "Maximum norm",                          // 177: iclaMaxNorm
    "",                                      // 178: iclaRealMaxNorm
    "", "",                                  // 179-180
    "", "", "", "", "", "", "", "", "", "",  // 181-190
    "", "", "", "", "", "", "", "", "", "",  // 191-200
    "Uniform",                               // 201: iclaDistUniform
    "Symmetric",                             // 202: iclaDistSymmetric
    "Normal",                                // 203: iclaDistNormal
    "", "", "", "", "", "", "",              // 204-210
    "", "", "", "", "", "", "", "", "", "",  // 211-220
    "", "", "", "", "", "", "", "", "", "",  // 221-230
    "", "", "", "", "", "", "", "", "", "",  // 231-240
    "Hermitian",                             // 241 iclaHermGeev
    "Positive ev Hermitian",                 // 242 iclaHermPoev
    "NonSymmetric pos sv",                   // 243 iclaNonsymPosv
    "Symmetric pos sv",                      // 244 iclaSymPosv
    "", "", "", "", "", "",                  // 245-250
    "", "", "", "", "", "", "", "", "", "",  // 251-260
    "", "", "", "", "", "", "", "", "", "",  // 261-270
    "", "", "", "", "", "", "", "", "", "",  // 271-280
    "", "", "", "", "", "", "", "", "", "",  // 281-290
    "No Packing",                            // 291 iclaNoPacking
    "U zero out subdiag",                    // 292 iclaPackSubdiag
    "L zero out superdiag",                  // 293 iclaPackSupdiag
    "C",                                     // 294 iclaPackColumn
    "R",                                     // 295 iclaPackRow
    "B",                                     // 296 iclaPackLowerBand
    "Q",                                     // 297 iclaPackUpeprBand
    "Z",                                     // 298 iclaPackAll
    "", "",                                  // 299-300
    "No vectors",                            // 301 iclaNoVec
    "Vectors needed",                        // 302 iclaVec
    "I",                                     // 303 iclaIVec
    "All",                                   // 304 iclaAllVec
    "Some",                                  // 305 iclaSomeVec
    "Overwrite",                             // 306 iclaOverwriteVec
    "", "", "", "",                          // 307-310
    "All",                                   // 311 iclaRangeAll
    "V",                                     // 312 iclaRangeV
    "I",                                     // 313 iclaRangeI
    "", "", "", "", "", "", "",              // 314-320
    "",                                      // 321
    "Q",                                     // 322
    "P",                                     // 323
    "", "", "", "", "", "", "",              // 324-330
    "", "", "", "", "", "", "", "", "", "",  // 331-340
    "", "", "", "", "", "", "", "", "", "",  // 341-350
    "", "", "", "", "", "", "", "", "", "",  // 351-360
    "", "", "", "", "", "", "", "", "", "",  // 361-370
    "", "", "", "", "", "", "", "", "", "",  // 371-380
    "", "", "", "", "", "", "", "", "", "",  // 381-390
    "Forward",                               // 391: iclaForward
    "Backward",                              // 392: iclaBackward
    "", "", "", "", "", "", "", "",          // 393-400
    "Columnwise",                            // 401: iclaColumnwise
    "Rowwise",                               // 402: iclaRowwise
    "", "", "", "", "", "", "", ""           // 403-410
    // Remember to add a comma!
};

/******************************************************************************/
/// maps any ICLA constant to its corresponding LAPACK string
extern "C"
const char* lapack_const_str( int icla_const )
{
    assert( icla_const >= icla2lapack_Min );
    assert( icla_const <= icla2lapack_Max );
    return icla2lapack_constants[ icla_const ];
}

/******************************************************************************/
/// inverse of icla_bool_const()
extern "C"
const char* lapack_bool_const( icla_bool_t icla_const )
{
    assert( icla_const >= iclaFalse );
    assert( icla_const <= iclaTrue  );
    return icla2lapack_constants[ icla_const ];
}

/******************************************************************************/
/// inverse of icla_order_const()
extern "C"
const char* lapack_order_const( icla_order_t icla_const )
{
    assert( icla_const >= iclaRowMajor );
    assert( icla_const <= iclaColMajor );
    return icla2lapack_constants[ icla_const ];
}

/******************************************************************************/
/// inverse of icla_trans_const()
extern "C"
const char* lapack_trans_const( icla_trans_t icla_const )
{
    assert( icla_const >= iclaNoTrans   );
    assert( icla_const <= iclaConjTrans );
    return icla2lapack_constants[ icla_const ];
}

/******************************************************************************/
/// inverse of icla_uplo_const()
extern "C"
const char* lapack_uplo_const ( icla_uplo_t icla_const )
{
    assert( icla_const >= iclaUpper );
    assert( icla_const <= iclaFull  );
    return icla2lapack_constants[ icla_const ];
}

/******************************************************************************/
/// inverse of icla_diag_const()
extern "C"
const char* lapack_diag_const ( icla_diag_t icla_const )
{
    assert( icla_const >= iclaNonUnit );
    assert( icla_const <= iclaUnit    );
    return icla2lapack_constants[ icla_const ];
}

/******************************************************************************/
/// inverse of icla_side_const()
extern "C"
const char* lapack_side_const ( icla_side_t icla_const )
{
    assert( icla_const >= iclaLeft  );
    assert( icla_const <= iclaBothSides );
    return icla2lapack_constants[ icla_const ];
}

/******************************************************************************/
/// inverse of icla_norm_const()
extern "C"
const char* lapack_norm_const  ( icla_norm_t   icla_const )
{
    assert( icla_const >= iclaOneNorm     );
    assert( icla_const <= iclaRealMaxNorm );
    return icla2lapack_constants[ icla_const ];
}

/******************************************************************************/
/// inverse of icla_dist_const()
extern "C"
const char* lapack_dist_const  ( icla_dist_t   icla_const )
{
    assert( icla_const >= iclaDistUniform );
    assert( icla_const <= iclaDistNormal );
    return icla2lapack_constants[ icla_const ];
}

/******************************************************************************/
/// inverse of icla_sym_const()
extern "C"
const char* lapack_sym_const   ( icla_sym_t    icla_const )
{
    assert( icla_const >= iclaHermGeev );
    assert( icla_const <= iclaSymPosv  );
    return icla2lapack_constants[ icla_const ];
}

/******************************************************************************/
/// inverse of icla_pack_const()
extern "C"
const char* lapack_pack_const  ( icla_pack_t   icla_const )
{
    assert( icla_const >= iclaNoPacking );
    assert( icla_const <= iclaPackAll   );
    return icla2lapack_constants[ icla_const ];
}

/******************************************************************************/
/// inverse of icla_vec_const()
extern "C"
const char* lapack_vec_const   ( icla_vec_t    icla_const )
{
    assert( icla_const >= iclaNoVec );
    assert( icla_const <= iclaOverwriteVec );
    return icla2lapack_constants[ icla_const ];
}

/******************************************************************************/
/// inverse of icla_range_const()
extern "C"
const char* lapack_range_const ( icla_range_t  icla_const )
{
    assert( icla_const >= iclaRangeAll );
    assert( icla_const <= iclaRangeI   );
    return icla2lapack_constants[ icla_const ];
}

/******************************************************************************/
/// inverse of icla_vect_const()
extern "C"
const char* lapack_vect_const( icla_vect_t icla_const )
{
    assert( icla_const >= iclaQ );
    assert( icla_const <= iclaP );
    return icla2lapack_constants[ icla_const ];
}

/******************************************************************************/
/// inverse of icla_direct_const()
extern "C"
const char* lapack_direct_const( icla_direct_t icla_const )
{
    assert( icla_const >= iclaForward );
    assert( icla_const <= iclaBackward );
    return icla2lapack_constants[ icla_const ];
}

/******************************************************************************/
/// inverse of icla_storev_const()
extern "C"
const char* lapack_storev_const( icla_storev_t icla_const )
{
    assert( icla_const >= iclaColumnwise );
    assert( icla_const <= iclaRowwise    );
    return icla2lapack_constants[ icla_const ];
}

// =============================================================================
/// @}
// end group lapack_const


#ifdef ICLA_HAVE_clBLAS
// =============================================================================
/// @addtogroup clblas_const
/// Convert ICLA constants to AMD clBLAS constants.
/// Available if ICLA_HAVE_OPENCL was defined when ICLA was compiled.
/// TODO: we do not currently provide inverse converters (clBLAS => ICLA).
/// @{

// The icla2clblas_constants table has an entry for each ICLA constant,
// enumerated on the right, with a corresponding clBLAS constant.

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
    0,                      // 100
    clblasRowMajor,         // 101: iclaRowMajor
    clblasColumnMajor,      // 102: iclaColMajor
    0, 0, 0, 0, 0, 0, 0, 0,
    clblasNoTrans,          // 111: iclaNoTrans
    clblasTrans,            // 112: iclaTrans
    clblasConjTrans,        // 113: iclaConjTrans
    0, 0, 0, 0, 0, 0, 0,
    clblasUpper,            // 121: iclaUpper
    clblasLower,            // 122: iclaLower
    0, 0, 0, 0, 0, 0, 0, 0,
    clblasNonUnit,          // 131: iclaNonUnit
    clblasUnit,             // 132: iclaUnit
    0, 0, 0, 0, 0, 0, 0, 0,
    clblasLeft,             // 141: iclaLeft
    clblasRight,            // 142: iclaRight
    0, 0, 0, 0, 0, 0, 0, 0
};

/******************************************************************************/
/// @retval clblasRowMajor    if icla_const = iclaRowMajor
/// @retval clblasColumnMajor if icla_const = iclaColMajor
extern "C"
clblasOrder       clblas_order_const( icla_order_t icla_const )
{
    assert( icla_const >= iclaRowMajor );
    assert( icla_const <= iclaColMajor );
    return (clblasOrder)     icla2clblas_constants[ icla_const ];
}

/******************************************************************************/
/// @retval clblasNoTrans   if icla_const = iclaNoTrans
/// @retval clblasTrans     if icla_const = iclaTrans
/// @retval clblasConjTrans if icla_const = iclaConjTrans
extern "C"
clblasTranspose   clblas_trans_const( icla_trans_t icla_const )
{
    assert( icla_const >= iclaNoTrans   );
    assert( icla_const <= iclaConjTrans );
    return (clblasTranspose) icla2clblas_constants[ icla_const ];
}

/******************************************************************************/
/// @retval clblasUpper if icla_const = iclaUpper
/// @retval clblasLower if icla_const = iclaLower
extern "C"
clblasUplo        clblas_uplo_const ( icla_uplo_t icla_const )
{
    assert( icla_const >= iclaUpper );
    assert( icla_const <= iclaLower );
    return (clblasUplo)      icla2clblas_constants[ icla_const ];
}

/******************************************************************************/
/// @retval clblasNonUnit if icla_const = iclaNonUnit
/// @retval clblasUnit    if icla_const = iclaUnit
extern "C"
clblasDiag        clblas_diag_const ( icla_diag_t icla_const )
{
    assert( icla_const >= iclaNonUnit );
    assert( icla_const <= iclaUnit    );
    return (clblasDiag)      icla2clblas_constants[ icla_const ];
}

/******************************************************************************/
/// @retval clblasLeft  if icla_const = iclaLeft
/// @retval clblasRight if icla_const = iclaRight
extern "C"
clblasSide        clblas_side_const ( icla_side_t icla_const )
{
    assert( icla_const >= iclaLeft  );
    assert( icla_const <= iclaRight );
    return (clblasSide)      icla2clblas_constants[ icla_const ];
}

// =============================================================================
/// @}
// end group clblas_const
#endif  // ICLA_HAVE_OPENCL


#ifdef ICLA_HAVE_CUDA
// =============================================================================
/// @addtogroup cublas_const
/// Convert ICLA constants to NVIDIA cuBLAS constants.
/// Available if ICLA_HAVE_CUDA was defined when ICLA was compiled.
/// TODO: we do not currently provide inverse converters (cuBLAS => ICLA).
/// @{

// The icla2cublas_constants table has an entry for each ICLA constant,
// enumerated on the right, with a corresponding cuBLAS constant.

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
    0,                      // 100
    0,                      // 101: iclaRowMajor
    0,                      // 102: iclaColMajor
    0, 0, 0, 0, 0, 0, 0, 0,
    CUBLAS_OP_N,            // 111: iclaNoTrans
    CUBLAS_OP_T,            // 112: iclaTrans
    CUBLAS_OP_C,            // 113: iclaConjTrans
    0, 0, 0, 0, 0, 0, 0,
    CUBLAS_FILL_MODE_UPPER, // 121: iclaUpper
    CUBLAS_FILL_MODE_LOWER, // 122: iclaLower
    0, 0, 0, 0, 0, 0, 0, 0,
    CUBLAS_DIAG_NON_UNIT,   // 131: iclaNonUnit
    CUBLAS_DIAG_UNIT,       // 132: iclaUnit
    0, 0, 0, 0, 0, 0, 0, 0,
    CUBLAS_SIDE_LEFT,       // 141: iclaLeft
    CUBLAS_SIDE_RIGHT,      // 142: iclaRight
    0, 0, 0, 0, 0, 0, 0, 0
};

/******************************************************************************/
/// @retval CUBLAS_OP_N if icla_const = iclaNoTrans
/// @retval CUBLAS_OP_T if icla_const = iclaTrans
/// @retval CUBLAS_OP_C if icla_const = iclaConjTrans
extern "C"
cublasOperation_t    cublas_trans_const ( icla_trans_t icla_const )
{
    assert( icla_const >= iclaNoTrans   );
    assert( icla_const <= iclaConjTrans );
    return (cublasOperation_t)  icla2cublas_constants[ icla_const ];
}

/******************************************************************************/
/// @retval CUBLAS_FILL_MODE_UPPER if icla_const = iclaUpper
/// @retval CUBLAS_FILL_MODE_LOWER if icla_const = iclaLower
extern "C"
cublasFillMode_t     cublas_uplo_const  ( icla_uplo_t icla_const )
{
    assert( icla_const >= iclaUpper );
    assert( icla_const <= iclaLower );
    return (cublasFillMode_t)   icla2cublas_constants[ icla_const ];
}

/******************************************************************************/
/// @retval CUBLAS_DIAG_NONUNIT if icla_const = iclaNonUnit
/// @retval CUBLAS_DIAG_UNIT    if icla_const = iclaUnit
extern "C"
cublasDiagType_t     cublas_diag_const  ( icla_diag_t icla_const )
{
    assert( icla_const >= iclaNonUnit );
    assert( icla_const <= iclaUnit    );
    return (cublasDiagType_t)   icla2cublas_constants[ icla_const ];
}

/******************************************************************************/
/// @retval CUBLAS_SIDE_LEFT  if icla_const = iclaLeft
/// @retval CUBLAS_SIDE_RIGHT if icla_const = iclaRight
extern "C"
cublasSideMode_t     cublas_side_const  ( icla_side_t icla_const )
{
    assert( icla_const >= iclaLeft  );
    assert( icla_const <= iclaRight );
    return (cublasSideMode_t)   icla2cublas_constants[ icla_const ];
}

// =============================================================================
/// @}
#endif  // ICLA_HAVE_CUDA



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
    0,                      // 100
    0,                      // 101: iclaRowMajor
    0,                      // 102: iclaColMajor
    0, 0, 0, 0, 0, 0, 0, 0,
    HIPBLAS_OP_N,           // 111: iclaNoTrans
    HIPBLAS_OP_T,           // 112: iclaTrans
    HIPBLAS_OP_C,           // 113: iclaConjTrans
    0, 0, 0, 0, 0, 0, 0,
    HIPBLAS_FILL_MODE_UPPER,// 121: iclaUpper
    HIPBLAS_FILL_MODE_LOWER,// 122: iclaLower
    0, 0, 0, 0, 0, 0, 0, 0,
    HIPBLAS_DIAG_NON_UNIT,  // 131: iclaNonUnit
    HIPBLAS_DIAG_UNIT,      // 132: iclaUnit
    0, 0, 0, 0, 0, 0, 0, 0,
    HIPBLAS_SIDE_LEFT,      // 141: iclaLeft
    HIPBLAS_SIDE_RIGHT,     // 142: iclaRight
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


#endif // ICLA_HAVE_HIP

#ifdef HAVE_CBLAS
// =============================================================================
/// @addtogroup cblas_const
/// Convert ICLA constants to CBLAS constants.
/// Available if HAVE_CBLAS was defined when ICLA was compiled.
/// ICLA constants have the same value as CBLAS constants,
/// which these routines verify by asserts.
/// TODO: we do not currently provide inverse converters (CBLAS => ICLA),
/// though it is a trivial cast since the values are the same.
/// @{

/******************************************************************************/
/// @retval CblasRowMajor if icla_const = iclaRowMajor
/// @retval CblasColMajor if icla_const = iclaColMajor
extern "C"
enum CBLAS_ORDER     cblas_order_const  ( icla_order_t icla_const )
{
    assert( icla_const >= iclaRowMajor );
    assert( icla_const <= iclaColMajor );
    assert( (int)iclaRowMajor == CblasRowMajor );
    return (enum CBLAS_ORDER)     icla_const;
}

/******************************************************************************/
/// @retval CblasNoTrans   if icla_const = iclaNoTrans
/// @retval CblasTrans     if icla_const = iclaTrans
/// @retval CblasConjTrans if icla_const = iclaConjTrans
extern "C"
enum CBLAS_TRANSPOSE cblas_trans_const  ( icla_trans_t icla_const )
{
    assert( icla_const >= iclaNoTrans   );
    assert( icla_const <= iclaConjTrans );
    assert( (int)iclaNoTrans == CblasNoTrans );
    return (enum CBLAS_TRANSPOSE) icla_const;
}

/******************************************************************************/
/// @retval CblasUpper if icla_const = iclaUpper
/// @retval CblasLower if icla_const = iclaLower
extern "C"
enum CBLAS_UPLO      cblas_uplo_const   ( icla_uplo_t icla_const )
{
    assert( icla_const >= iclaUpper );
    assert( icla_const <= iclaLower );
    assert( (int)iclaUpper == CblasUpper );
    return (enum CBLAS_UPLO)      icla_const;
}

/******************************************************************************/
/// @retval CblasNonUnit if icla_const = iclaNonUnit
/// @retval CblasUnit    if icla_const = iclaUnit
extern "C"
enum CBLAS_DIAG      cblas_diag_const   ( icla_diag_t icla_const )
{
    assert( icla_const >= iclaNonUnit );
    assert( icla_const <= iclaUnit    );
    assert( (int)iclaUnit == CblasUnit );
    return (enum CBLAS_DIAG)      icla_const;
}

/******************************************************************************/
/// @retval CblasLeft  if icla_const = iclaLeft
/// @retval CblasRight if icla_const = iclaRight
extern "C"
enum CBLAS_SIDE      cblas_side_const   ( icla_side_t icla_const )
{
    assert( icla_const >= iclaLeft  );
    assert( icla_const <= iclaRight );
    assert( (int)iclaLeft == CblasLeft );
    return (enum CBLAS_SIDE)      icla_const;
}

// =============================================================================
/// @}
// end group cblas_const

#endif  // HAVE_CBLAS
