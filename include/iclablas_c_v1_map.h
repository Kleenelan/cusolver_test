/*
    -- ICLA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @generated from include/iclablas_z_v1_map.h, normal z -> c, Fri Nov 29 12:16:14 2024
*/

#ifndef ICLABLAS_C_V1_MAP_H
#define ICLABLAS_C_V1_MAP_H

#ifdef ICLA_NO_V1
#error "Since ICLA_NO_V1 is defined, icla.h is invalid; use icla_v2.h"
#endif

// =============================================================================
// map function names to old v1 routines

#define iclablas_ctranspose_inplace        iclablas_ctranspose_inplace_v1
#define iclablas_ctranspose_conj_inplace   iclablas_ctranspose_conj_inplace_v1
#define iclablas_ctranspose                iclablas_ctranspose_v1
#define iclablas_ctranspose_conj           iclablas_ctranspose_conj_v1
#define iclablas_cgetmatrix_transpose      iclablas_cgetmatrix_transpose_v1
#define iclablas_csetmatrix_transpose      iclablas_csetmatrix_transpose_v1
#define iclablas_cprbt                     iclablas_cprbt_v1
#define iclablas_cprbt_mv                  iclablas_cprbt_mv_v1
#define iclablas_cprbt_mtv                 iclablas_cprbt_mtv_v1
#define icla_cgetmatrix_1D_col_bcyclic     icla_cgetmatrix_1D_col_bcyclic_v1
#define icla_csetmatrix_1D_col_bcyclic     icla_csetmatrix_1D_col_bcyclic_v1
#define icla_cgetmatrix_1D_row_bcyclic     icla_cgetmatrix_1D_row_bcyclic_v1
#define icla_csetmatrix_1D_row_bcyclic     icla_csetmatrix_1D_row_bcyclic_v1
#define iclablas_cgeadd                    iclablas_cgeadd_v1
#define iclablas_cgeadd2                   iclablas_cgeadd2_v1
#define iclablas_clacpy                    iclablas_clacpy_v1
#define iclablas_clacpy_conj               iclablas_clacpy_conj_v1
#define iclablas_clacpy_sym_in             iclablas_clacpy_sym_in_v1
#define iclablas_clacpy_sym_out            iclablas_clacpy_sym_out_v1
#define iclablas_clange                    iclablas_clange_v1
#define iclablas_clanhe                    iclablas_clanhe_v1
#define iclablas_clansy                    iclablas_clansy_v1
#define iclablas_clarfg                    iclablas_clarfg_v1
#define iclablas_clascl                    iclablas_clascl_v1
#define iclablas_clascl_2x2                iclablas_clascl_2x2_v1
#define iclablas_clascl2                   iclablas_clascl2_v1
#define iclablas_clascl_diag               iclablas_clascl_diag_v1
#define iclablas_claset                    iclablas_claset_v1
#define iclablas_claset_band               iclablas_claset_band_v1
#define iclablas_claswp                    iclablas_claswp_v1
#define iclablas_claswp2                   iclablas_claswp2_v1
#define iclablas_claswp_sym                iclablas_claswp_sym_v1
#define iclablas_claswpx                   iclablas_claswpx_v1
#define iclablas_csymmetrize               iclablas_csymmetrize_v1
#define iclablas_csymmetrize_tiles         iclablas_csymmetrize_tiles_v1
#define iclablas_ctrtri_diag               iclablas_ctrtri_diag_v1
#define iclablas_scnrm2_adjust             iclablas_scnrm2_adjust_v1
#define iclablas_scnrm2_check              iclablas_scnrm2_check_v1
#define iclablas_scnrm2_cols               iclablas_scnrm2_cols_v1
#define iclablas_scnrm2_row_check_adjust   iclablas_scnrm2_row_check_adjust_v1
#define icla_clarfb_gpu                    icla_clarfb_gpu_v1
#define icla_clarfb_gpu_gemm               icla_clarfb_gpu_gemm_v1
#define icla_clarfbx_gpu                   icla_clarfbx_gpu_v1
#define icla_clarfg_gpu                    icla_clarfg_gpu_v1
#define icla_clarfgtx_gpu                  icla_clarfgtx_gpu_v1
#define icla_clarfgx_gpu                   icla_clarfgx_gpu_v1
#define icla_clarfx_gpu                    icla_clarfx_gpu_v1
#define iclablas_caxpycp                   iclablas_caxpycp_v1
#define iclablas_cswap                     iclablas_cswap_v1
#define iclablas_cswapblk                  iclablas_cswapblk_v1
#define iclablas_cswapdblk                 iclablas_cswapdblk_v1
#define iclablas_cgemv                     iclablas_cgemv_v1
#define iclablas_cgemv_conj                iclablas_cgemv_conj_v1
#define iclablas_chemv                     iclablas_chemv_v1
#define iclablas_csymv                     iclablas_csymv_v1
#define iclablas_cgemm                     iclablas_cgemm_v1
#define iclablas_cgemm_reduce              iclablas_cgemm_reduce_v1
#define iclablas_chemm                     iclablas_chemm_v1
#define iclablas_csymm                     iclablas_csymm_v1
#define iclablas_csyr2k                    iclablas_csyr2k_v1
#define iclablas_cher2k                    iclablas_cher2k_v1
#define iclablas_csyrk                     iclablas_csyrk_v1
#define iclablas_cherk                     iclablas_cherk_v1
#define iclablas_ctrsm                     iclablas_ctrsm_v1
#define iclablas_ctrsm_outofplace          iclablas_ctrsm_outofplace_v1
#define iclablas_ctrsm_work                iclablas_ctrsm_work_v1

#undef icla_csetvector
#undef icla_cgetvector
#undef icla_ccopyvector
#undef icla_csetmatrix
#undef icla_cgetmatrix
#undef icla_ccopymatrix

#define icla_csetvector                    icla_csetvector_v1
#define icla_cgetvector                    icla_cgetvector_v1
#define icla_ccopyvector                   icla_ccopyvector_v1
#define icla_csetmatrix                    icla_csetmatrix_v1
#define icla_cgetmatrix                    icla_cgetmatrix_v1
#define icla_ccopymatrix                   icla_ccopymatrix_v1

#define icla_icamax                        icla_icamax_v1
#define icla_icamin                        icla_icamin_v1
#define icla_scasum                        icla_scasum_v1
#define icla_caxpy                         icla_caxpy_v1
#define icla_ccopy                         icla_ccopy_v1
#define icla_cdotc                         icla_cdotc_v1
#define icla_cdotu                         icla_cdotu_v1
#define icla_scnrm2                        icla_scnrm2_v1
#define icla_crot                          icla_crot_v1
#define icla_csrot                         icla_csrot_v1
#define icla_crotm                         icla_crotm_v1
#define icla_crotmg                        icla_crotmg_v1
#define icla_cscal                         icla_cscal_v1
#define icla_csscal                        icla_csscal_v1
#define icla_cswap                         icla_cswap_v1
#define icla_cgemv                         icla_cgemv_v1
#define icla_cgerc                         icla_cgerc_v1
#define icla_cgeru                         icla_cgeru_v1
#define icla_chemv                         icla_chemv_v1
#define icla_cher                          icla_cher_v1
#define icla_cher2                         icla_cher2_v1
#define icla_ctrmv                         icla_ctrmv_v1
#define icla_ctrsv                         icla_ctrsv_v1
#define icla_cgemm                         icla_cgemm_v1
#define icla_csymm                         icla_csymm_v1
#define icla_chemm                         icla_chemm_v1
#define icla_csyr2k                        icla_csyr2k_v1
#define icla_cher2k                        icla_cher2k_v1
#define icla_csyrk                         icla_csyrk_v1
#define icla_cherk                         icla_cherk_v1
#define icla_ctrmm                         icla_ctrmm_v1
#define icla_ctrsm                         icla_ctrsm_v1

#endif // ICLABLAS_C_V1_MAP_H
