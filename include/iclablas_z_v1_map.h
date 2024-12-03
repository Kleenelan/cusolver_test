/*
    -- ICLA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> s d c
*/

#ifndef ICLABLAS_Z_V1_MAP_H
#define ICLABLAS_Z_V1_MAP_H

#ifdef ICLA_NO_V1
#error "Since ICLA_NO_V1 is defined, icla.h is invalid; use icla_v2.h"
#endif

// =============================================================================
// map function names to old v1 routines

#define iclablas_ztranspose_inplace        iclablas_ztranspose_inplace_v1
#define iclablas_ztranspose_conj_inplace   iclablas_ztranspose_conj_inplace_v1
#define iclablas_ztranspose                iclablas_ztranspose_v1
#define iclablas_ztranspose_conj           iclablas_ztranspose_conj_v1
#define iclablas_zgetmatrix_transpose      iclablas_zgetmatrix_transpose_v1
#define iclablas_zsetmatrix_transpose      iclablas_zsetmatrix_transpose_v1
#define iclablas_zprbt                     iclablas_zprbt_v1
#define iclablas_zprbt_mv                  iclablas_zprbt_mv_v1
#define iclablas_zprbt_mtv                 iclablas_zprbt_mtv_v1
#define icla_zgetmatrix_1D_col_bcyclic     icla_zgetmatrix_1D_col_bcyclic_v1
#define icla_zsetmatrix_1D_col_bcyclic     icla_zsetmatrix_1D_col_bcyclic_v1
#define icla_zgetmatrix_1D_row_bcyclic     icla_zgetmatrix_1D_row_bcyclic_v1
#define icla_zsetmatrix_1D_row_bcyclic     icla_zsetmatrix_1D_row_bcyclic_v1
#define iclablas_zgeadd                    iclablas_zgeadd_v1
#define iclablas_zgeadd2                   iclablas_zgeadd2_v1
#define iclablas_zlacpy                    iclablas_zlacpy_v1
#define iclablas_zlacpy_conj               iclablas_zlacpy_conj_v1
#define iclablas_zlacpy_sym_in             iclablas_zlacpy_sym_in_v1
#define iclablas_zlacpy_sym_out            iclablas_zlacpy_sym_out_v1
#define iclablas_zlange                    iclablas_zlange_v1
#define iclablas_zlanhe                    iclablas_zlanhe_v1
#define iclablas_zlansy                    iclablas_zlansy_v1
#define iclablas_zlarfg                    iclablas_zlarfg_v1
#define iclablas_zlascl                    iclablas_zlascl_v1
#define iclablas_zlascl_2x2                iclablas_zlascl_2x2_v1
#define iclablas_zlascl2                   iclablas_zlascl2_v1
#define iclablas_zlascl_diag               iclablas_zlascl_diag_v1
#define iclablas_zlaset                    iclablas_zlaset_v1
#define iclablas_zlaset_band               iclablas_zlaset_band_v1
#define iclablas_zlaswp                    iclablas_zlaswp_v1
#define iclablas_zlaswp2                   iclablas_zlaswp2_v1
#define iclablas_zlaswp_sym                iclablas_zlaswp_sym_v1
#define iclablas_zlaswpx                   iclablas_zlaswpx_v1
#define iclablas_zsymmetrize               iclablas_zsymmetrize_v1
#define iclablas_zsymmetrize_tiles         iclablas_zsymmetrize_tiles_v1
#define iclablas_ztrtri_diag               iclablas_ztrtri_diag_v1
#define iclablas_dznrm2_adjust             iclablas_dznrm2_adjust_v1
#define iclablas_dznrm2_check              iclablas_dznrm2_check_v1
#define iclablas_dznrm2_cols               iclablas_dznrm2_cols_v1
#define iclablas_dznrm2_row_check_adjust   iclablas_dznrm2_row_check_adjust_v1
#define icla_zlarfb_gpu                    icla_zlarfb_gpu_v1
#define icla_zlarfb_gpu_gemm               icla_zlarfb_gpu_gemm_v1
#define icla_zlarfbx_gpu                   icla_zlarfbx_gpu_v1
#define icla_zlarfg_gpu                    icla_zlarfg_gpu_v1
#define icla_zlarfgtx_gpu                  icla_zlarfgtx_gpu_v1
#define icla_zlarfgx_gpu                   icla_zlarfgx_gpu_v1
#define icla_zlarfx_gpu                    icla_zlarfx_gpu_v1
#define iclablas_zaxpycp                   iclablas_zaxpycp_v1
#define iclablas_zswap                     iclablas_zswap_v1
#define iclablas_zswapblk                  iclablas_zswapblk_v1
#define iclablas_zswapdblk                 iclablas_zswapdblk_v1
#define iclablas_zgemv                     iclablas_zgemv_v1
#define iclablas_zgemv_conj                iclablas_zgemv_conj_v1
#define iclablas_zhemv                     iclablas_zhemv_v1
#define iclablas_zsymv                     iclablas_zsymv_v1
#define iclablas_zgemm                     iclablas_zgemm_v1
#define iclablas_zgemm_reduce              iclablas_zgemm_reduce_v1
#define iclablas_zhemm                     iclablas_zhemm_v1
#define iclablas_zsymm                     iclablas_zsymm_v1
#define iclablas_zsyr2k                    iclablas_zsyr2k_v1
#define iclablas_zher2k                    iclablas_zher2k_v1
#define iclablas_zsyrk                     iclablas_zsyrk_v1
#define iclablas_zherk                     iclablas_zherk_v1
#define iclablas_ztrsm                     iclablas_ztrsm_v1
#define iclablas_ztrsm_outofplace          iclablas_ztrsm_outofplace_v1
#define iclablas_ztrsm_work                iclablas_ztrsm_work_v1

#undef icla_zsetvector
#undef icla_zgetvector
#undef icla_zcopyvector
#undef icla_zsetmatrix
#undef icla_zgetmatrix
#undef icla_zcopymatrix

#define icla_zsetvector                    icla_zsetvector_v1
#define icla_zgetvector                    icla_zgetvector_v1
#define icla_zcopyvector                   icla_zcopyvector_v1
#define icla_zsetmatrix                    icla_zsetmatrix_v1
#define icla_zgetmatrix                    icla_zgetmatrix_v1
#define icla_zcopymatrix                   icla_zcopymatrix_v1

#define icla_izamax                        icla_izamax_v1
#define icla_izamin                        icla_izamin_v1
#define icla_dzasum                        icla_dzasum_v1
#define icla_zaxpy                         icla_zaxpy_v1
#define icla_zcopy                         icla_zcopy_v1
#define icla_zdotc                         icla_zdotc_v1
#define icla_zdotu                         icla_zdotu_v1
#define icla_dznrm2                        icla_dznrm2_v1
#define icla_zrot                          icla_zrot_v1
#define icla_zdrot                         icla_zdrot_v1
#define icla_zrotm                         icla_zrotm_v1
#define icla_zrotmg                        icla_zrotmg_v1
#define icla_zscal                         icla_zscal_v1
#define icla_zdscal                        icla_zdscal_v1
#define icla_zswap                         icla_zswap_v1
#define icla_zgemv                         icla_zgemv_v1
#define icla_zgerc                         icla_zgerc_v1
#define icla_zgeru                         icla_zgeru_v1
#define icla_zhemv                         icla_zhemv_v1
#define icla_zher                          icla_zher_v1
#define icla_zher2                         icla_zher2_v1
#define icla_ztrmv                         icla_ztrmv_v1
#define icla_ztrsv                         icla_ztrsv_v1
#define icla_zgemm                         icla_zgemm_v1
#define icla_zsymm                         icla_zsymm_v1
#define icla_zhemm                         icla_zhemm_v1
#define icla_zsyr2k                        icla_zsyr2k_v1
#define icla_zher2k                        icla_zher2k_v1
#define icla_zsyrk                         icla_zsyrk_v1
#define icla_zherk                         icla_zherk_v1
#define icla_ztrmm                         icla_ztrmm_v1
#define icla_ztrsm                         icla_ztrsm_v1

#endif // ICLABLAS_Z_V1_MAP_H
