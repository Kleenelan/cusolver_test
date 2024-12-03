
#ifndef ICLABLAS_S_V1_MAP_H
#define ICLABLAS_S_V1_MAP_H

#ifdef ICLA_NO_V1
#error "Since ICLA_NO_V1 is defined, icla.h is invalid; use icla_v2.h"
#endif

#define iclablas_stranspose_inplace        iclablas_stranspose_inplace_v1
#define iclablas_stranspose_inplace   iclablas_stranspose_inplace_v1
#define iclablas_stranspose                iclablas_stranspose_v1
#define iclablas_stranspose           iclablas_stranspose_v1
#define iclablas_sgetmatrix_transpose      iclablas_sgetmatrix_transpose_v1
#define iclablas_ssetmatrix_transpose      iclablas_ssetmatrix_transpose_v1
#define iclablas_sprbt                     iclablas_sprbt_v1
#define iclablas_sprbt_mv                  iclablas_sprbt_mv_v1
#define iclablas_sprbt_mtv                 iclablas_sprbt_mtv_v1
#define icla_sgetmatrix_1D_col_bcyclic     icla_sgetmatrix_1D_col_bcyclic_v1
#define icla_ssetmatrix_1D_col_bcyclic     icla_ssetmatrix_1D_col_bcyclic_v1
#define icla_sgetmatrix_1D_row_bcyclic     icla_sgetmatrix_1D_row_bcyclic_v1
#define icla_ssetmatrix_1D_row_bcyclic     icla_ssetmatrix_1D_row_bcyclic_v1
#define iclablas_sgeadd                    iclablas_sgeadd_v1
#define iclablas_sgeadd2                   iclablas_sgeadd2_v1
#define iclablas_slacpy                    iclablas_slacpy_v1
#define iclablas_slacpy_conj               iclablas_slacpy_conj_v1
#define iclablas_slacpy_sym_in             iclablas_slacpy_sym_in_v1
#define iclablas_slacpy_sym_out            iclablas_slacpy_sym_out_v1
#define iclablas_slange                    iclablas_slange_v1
#define iclablas_slansy                    iclablas_slansy_v1
#define iclablas_slansy                    iclablas_slansy_v1
#define iclablas_slarfg                    iclablas_slarfg_v1
#define iclablas_slascl                    iclablas_slascl_v1
#define iclablas_slascl_2x2                iclablas_slascl_2x2_v1
#define iclablas_slascl2                   iclablas_slascl2_v1
#define iclablas_slascl_diag               iclablas_slascl_diag_v1
#define iclablas_slaset                    iclablas_slaset_v1
#define iclablas_slaset_band               iclablas_slaset_band_v1
#define iclablas_slaswp                    iclablas_slaswp_v1
#define iclablas_slaswp2                   iclablas_slaswp2_v1
#define iclablas_slaswp_sym                iclablas_slaswp_sym_v1
#define iclablas_slaswpx                   iclablas_slaswpx_v1
#define iclablas_ssymmetrize               iclablas_ssymmetrize_v1
#define iclablas_ssymmetrize_tiles         iclablas_ssymmetrize_tiles_v1
#define iclablas_strtri_diag               iclablas_strtri_diag_v1
#define iclablas_snrm2_adjust             iclablas_snrm2_adjust_v1
#define iclablas_snrm2_check              iclablas_snrm2_check_v1
#define iclablas_snrm2_cols               iclablas_snrm2_cols_v1
#define iclablas_snrm2_row_check_adjust   iclablas_snrm2_row_check_adjust_v1
#define icla_slarfb_gpu                    icla_slarfb_gpu_v1
#define icla_slarfb_gpu_gemm               icla_slarfb_gpu_gemm_v1
#define icla_slarfbx_gpu                   icla_slarfbx_gpu_v1
#define icla_slarfg_gpu                    icla_slarfg_gpu_v1
#define icla_slarfgtx_gpu                  icla_slarfgtx_gpu_v1
#define icla_slarfgx_gpu                   icla_slarfgx_gpu_v1
#define icla_slarfx_gpu                    icla_slarfx_gpu_v1
#define iclablas_saxpycp                   iclablas_saxpycp_v1
#define iclablas_sswap                     iclablas_sswap_v1
#define iclablas_sswapblk                  iclablas_sswapblk_v1
#define iclablas_sswapdblk                 iclablas_sswapdblk_v1
#define iclablas_sgemv                     iclablas_sgemv_v1
#define iclablas_sgemv_conj                iclablas_sgemv_conj_v1
#define iclablas_ssymv                     iclablas_ssymv_v1
#define iclablas_ssymv                     iclablas_ssymv_v1
#define iclablas_sgemm                     iclablas_sgemm_v1
#define iclablas_sgemm_reduce              iclablas_sgemm_reduce_v1
#define iclablas_ssymm                     iclablas_ssymm_v1
#define iclablas_ssymm                     iclablas_ssymm_v1
#define iclablas_ssyr2k                    iclablas_ssyr2k_v1
#define iclablas_ssyr2k                    iclablas_ssyr2k_v1
#define iclablas_ssyrk                     iclablas_ssyrk_v1
#define iclablas_ssyrk                     iclablas_ssyrk_v1
#define iclablas_strsm                     iclablas_strsm_v1
#define iclablas_strsm_outofplace          iclablas_strsm_outofplace_v1
#define iclablas_strsm_work                iclablas_strsm_work_v1

#undef icla_ssetvector
#undef icla_sgetvector
#undef icla_scopyvector
#undef icla_ssetmatrix
#undef icla_sgetmatrix
#undef icla_scopymatrix

#define icla_ssetvector                    icla_ssetvector_v1
#define icla_sgetvector                    icla_sgetvector_v1
#define icla_scopyvector                   icla_scopyvector_v1
#define icla_ssetmatrix                    icla_ssetmatrix_v1
#define icla_sgetmatrix                    icla_sgetmatrix_v1
#define icla_scopymatrix                   icla_scopymatrix_v1

#define icla_isamax                        icla_isamax_v1
#define icla_isamin                        icla_isamin_v1
#define icla_sasum                        icla_sasum_v1
#define icla_saxpy                         icla_saxpy_v1
#define icla_scopy                         icla_scopy_v1
#define icla_sdot                         icla_sdot_v1
#define icla_sdot                         icla_sdot_v1
#define icla_snrm2                        icla_snrm2_v1
#define icla_srot                          icla_srot_v1
#define icla_srot                         icla_srot_v1
#define icla_srotm                         icla_srotm_v1
#define icla_srotmg                        icla_srotmg_v1
#define icla_sscal                         icla_sscal_v1
#define icla_sscal                        icla_sscal_v1
#define icla_sswap                         icla_sswap_v1
#define icla_sgemv                         icla_sgemv_v1
#define icla_sger                         icla_sger_v1
#define icla_sger                         icla_sger_v1
#define icla_ssymv                         icla_ssymv_v1
#define icla_ssyr                          icla_ssyr_v1
#define icla_ssyr2                         icla_ssyr2_v1
#define icla_strmv                         icla_strmv_v1
#define icla_strsv                         icla_strsv_v1
#define icla_sgemm                         icla_sgemm_v1
#define icla_ssymm                         icla_ssymm_v1
#define icla_ssymm                         icla_ssymm_v1
#define icla_ssyr2k                        icla_ssyr2k_v1
#define icla_ssyr2k                        icla_ssyr2k_v1
#define icla_ssyrk                         icla_ssyrk_v1
#define icla_ssyrk                         icla_ssyrk_v1
#define icla_strmm                         icla_strmm_v1
#define icla_strsm                         icla_strsm_v1

#endif

