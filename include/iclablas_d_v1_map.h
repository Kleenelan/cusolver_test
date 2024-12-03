
#ifndef ICLABLAS_D_V1_MAP_H
#define ICLABLAS_D_V1_MAP_H

#ifdef ICLA_NO_V1
#error "Since ICLA_NO_V1 is defined, icla.h is invalid; use icla_v2.h"
#endif

#define iclablas_dtranspose_inplace        iclablas_dtranspose_inplace_v1
#define iclablas_dtranspose_inplace   iclablas_dtranspose_inplace_v1
#define iclablas_dtranspose                iclablas_dtranspose_v1
#define iclablas_dtranspose           iclablas_dtranspose_v1
#define iclablas_dgetmatrix_transpose      iclablas_dgetmatrix_transpose_v1
#define iclablas_dsetmatrix_transpose      iclablas_dsetmatrix_transpose_v1
#define iclablas_dprbt                     iclablas_dprbt_v1
#define iclablas_dprbt_mv                  iclablas_dprbt_mv_v1
#define iclablas_dprbt_mtv                 iclablas_dprbt_mtv_v1
#define icla_dgetmatrix_1D_col_bcyclic     icla_dgetmatrix_1D_col_bcyclic_v1
#define icla_dsetmatrix_1D_col_bcyclic     icla_dsetmatrix_1D_col_bcyclic_v1
#define icla_dgetmatrix_1D_row_bcyclic     icla_dgetmatrix_1D_row_bcyclic_v1
#define icla_dsetmatrix_1D_row_bcyclic     icla_dsetmatrix_1D_row_bcyclic_v1
#define iclablas_dgeadd                    iclablas_dgeadd_v1
#define iclablas_dgeadd2                   iclablas_dgeadd2_v1
#define iclablas_dlacpy                    iclablas_dlacpy_v1
#define iclablas_dlacpy_conj               iclablas_dlacpy_conj_v1
#define iclablas_dlacpy_sym_in             iclablas_dlacpy_sym_in_v1
#define iclablas_dlacpy_sym_out            iclablas_dlacpy_sym_out_v1
#define iclablas_dlange                    iclablas_dlange_v1
#define iclablas_dlansy                    iclablas_dlansy_v1
#define iclablas_dlansy                    iclablas_dlansy_v1
#define iclablas_dlarfg                    iclablas_dlarfg_v1
#define iclablas_dlascl                    iclablas_dlascl_v1
#define iclablas_dlascl_2x2                iclablas_dlascl_2x2_v1
#define iclablas_dlascl2                   iclablas_dlascl2_v1
#define iclablas_dlascl_diag               iclablas_dlascl_diag_v1
#define iclablas_dlaset                    iclablas_dlaset_v1
#define iclablas_dlaset_band               iclablas_dlaset_band_v1
#define iclablas_dlaswp                    iclablas_dlaswp_v1
#define iclablas_dlaswp2                   iclablas_dlaswp2_v1
#define iclablas_dlaswp_sym                iclablas_dlaswp_sym_v1
#define iclablas_dlaswpx                   iclablas_dlaswpx_v1
#define iclablas_dsymmetrize               iclablas_dsymmetrize_v1
#define iclablas_dsymmetrize_tiles         iclablas_dsymmetrize_tiles_v1
#define iclablas_dtrtri_diag               iclablas_dtrtri_diag_v1
#define iclablas_dnrm2_adjust             iclablas_dnrm2_adjust_v1
#define iclablas_dnrm2_check              iclablas_dnrm2_check_v1
#define iclablas_dnrm2_cols               iclablas_dnrm2_cols_v1
#define iclablas_dnrm2_row_check_adjust   iclablas_dnrm2_row_check_adjust_v1
#define icla_dlarfb_gpu                    icla_dlarfb_gpu_v1
#define icla_dlarfb_gpu_gemm               icla_dlarfb_gpu_gemm_v1
#define icla_dlarfbx_gpu                   icla_dlarfbx_gpu_v1
#define icla_dlarfg_gpu                    icla_dlarfg_gpu_v1
#define icla_dlarfgtx_gpu                  icla_dlarfgtx_gpu_v1
#define icla_dlarfgx_gpu                   icla_dlarfgx_gpu_v1
#define icla_dlarfx_gpu                    icla_dlarfx_gpu_v1
#define iclablas_daxpycp                   iclablas_daxpycp_v1
#define iclablas_dswap                     iclablas_dswap_v1
#define iclablas_dswapblk                  iclablas_dswapblk_v1
#define iclablas_dswapdblk                 iclablas_dswapdblk_v1
#define iclablas_dgemv                     iclablas_dgemv_v1
#define iclablas_dgemv_conj                iclablas_dgemv_conj_v1
#define iclablas_dsymv                     iclablas_dsymv_v1
#define iclablas_dsymv                     iclablas_dsymv_v1
#define iclablas_dgemm                     iclablas_dgemm_v1
#define iclablas_dgemm_reduce              iclablas_dgemm_reduce_v1
#define iclablas_dsymm                     iclablas_dsymm_v1
#define iclablas_dsymm                     iclablas_dsymm_v1
#define iclablas_dsyr2k                    iclablas_dsyr2k_v1
#define iclablas_dsyr2k                    iclablas_dsyr2k_v1
#define iclablas_dsyrk                     iclablas_dsyrk_v1
#define iclablas_dsyrk                     iclablas_dsyrk_v1
#define iclablas_dtrsm                     iclablas_dtrsm_v1
#define iclablas_dtrsm_outofplace          iclablas_dtrsm_outofplace_v1
#define iclablas_dtrsm_work                iclablas_dtrsm_work_v1

#undef icla_dsetvector
#undef icla_dgetvector
#undef icla_dcopyvector
#undef icla_dsetmatrix
#undef icla_dgetmatrix
#undef icla_dcopymatrix

#define icla_dsetvector                    icla_dsetvector_v1
#define icla_dgetvector                    icla_dgetvector_v1
#define icla_dcopyvector                   icla_dcopyvector_v1
#define icla_dsetmatrix                    icla_dsetmatrix_v1
#define icla_dgetmatrix                    icla_dgetmatrix_v1
#define icla_dcopymatrix                   icla_dcopymatrix_v1

#define icla_idamax                        icla_idamax_v1
#define icla_idamin                        icla_idamin_v1
#define icla_dasum                        icla_dasum_v1
#define icla_daxpy                         icla_daxpy_v1
#define icla_dcopy                         icla_dcopy_v1
#define icla_ddot                         icla_ddot_v1
#define icla_ddot                         icla_ddot_v1
#define icla_dnrm2                        icla_dnrm2_v1
#define icla_drot                          icla_drot_v1
#define icla_drot                         icla_drot_v1
#define icla_drotm                         icla_drotm_v1
#define icla_drotmg                        icla_drotmg_v1
#define icla_dscal                         icla_dscal_v1
#define icla_dscal                        icla_dscal_v1
#define icla_dswap                         icla_dswap_v1
#define icla_dgemv                         icla_dgemv_v1
#define icla_dger                         icla_dger_v1
#define icla_dger                         icla_dger_v1
#define icla_dsymv                         icla_dsymv_v1
#define icla_dsyr                          icla_dsyr_v1
#define icla_dsyr2                         icla_dsyr2_v1
#define icla_dtrmv                         icla_dtrmv_v1
#define icla_dtrsv                         icla_dtrsv_v1
#define icla_dgemm                         icla_dgemm_v1
#define icla_dsymm                         icla_dsymm_v1
#define icla_dsymm                         icla_dsymm_v1
#define icla_dsyr2k                        icla_dsyr2k_v1
#define icla_dsyr2k                        icla_dsyr2k_v1
#define icla_dsyrk                         icla_dsyrk_v1
#define icla_dsyrk                         icla_dsyrk_v1
#define icla_dtrmm                         icla_dtrmm_v1
#define icla_dtrsm                         icla_dtrsm_v1

#endif

