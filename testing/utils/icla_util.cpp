
#include <stdarg.h>
#include <string.h>
#include <assert.h>
#include <errno.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "icla_v2.h"
#include "testings.h"

#include "../control/icla_threadsetting.h"

#if   defined(ICLA_HAVE_CUDA)
    const char* g_platform_str = "cuBLAS";

#elif defined(ICLA_HAVE_HIP)
    const char* g_platform_str = "HIPBLAS";

#else
    #error "unknown platform"
#endif

extern "C"
void icla_assert( bool condition, const char* msg, ... )
{
    if ( ! condition ) {
        printf( "Assert failed: " );
        va_list va;
        va_start( va, msg );
        vprintf( msg, va );
        va_end( va );
        printf( "\n" );
        exit(1);
    }
}

extern "C"
void icla_assert_warn( bool condition, const char* msg, ... )
{
    if ( ! condition ) {
        printf( "Assert failed: " );
        va_list va;
        va_start( va, msg );
        vprintf( msg, va );
        va_end( va );
        printf( "\n" );
    }
}

const char *usage_short =
"%% Usage: %s [options] [-h|--help]\n\n";

const char *usage =
"Options are:\n"
"  -n m[,n[,k]      Adds problem sizes. All of -n, -N, --range are now synonymous.\n"
"  -N m[,n[,k]      m, n, k can each be a single size or an inclusive range start:end:step.\n"
"  --range m[,n[,k] If two ranges are given, the number of sizes is limited by the smaller range.\n"
"                   If only m,n are given, then k=n. If only m is given, then n=k=m.\n"
"                   Examples:  -N 100  -N 100,200,300  -N 100,200:1000:100,300  -N 100:1000:100\n"
"  Default test sizes are the range 1088 : 10304 : 1024, that is, 1K+64 : 10K+64 : 1K.\n"
"  For batched, default sizes are     32 :   512 :   32.\n"
"\n"
"  -c  --[no]check  Whether to check results. Some tests always check.\n"
"                   Also set with $ICLA_TESTINGS_CHECK.\n"
"  -c2 --check2     For getrf, check residual |Ax-b| instead of |PA-LU|.\n"
"  -l  --[no]lapack Whether to run lapack. Some tests always run lapack.\n"
"                   Also set with $ICLA_RUN_LAPACK.\n"
"      --[no]warmup Whether to warmup. Not yet implemented in most cases.\n"
"                   Also set with $ICLA_WARMUP.\n"
"  --dev x          GPU device to use, default 0.\n"
"  --align n        Round up LDDA on GPU to multiple of align, default 32.\n"
"  --verbose        Verbose output.\n"
"\n"
"The following options apply to only some routines.\n"
"  --batch x        number of matrices for the batched routines, default 1000.\n"
"  --kl x           number of sub-diagonals in a band matrix, default 1.\n"
"  --ku x           number of super diagonals in a band matrix, default 1.\n"
"  --cache x        cache size to flush, in MiB, default 2 MiB * NUM_THREADS.\n"
"  --nb x           Block size, default set automatically.\n"
"  --nrhs x         Number of right hand sides, default 1.\n"
"  --nqueue x       Number of device queues, default 1.\n"
"  --ngpu x         Number of GPUs, default 1. Also set with $ICLA_NUM_GPUS.\n"
"                   (Some testers take --ngpu -1 to run the multi-GPU code with 1 GPU.\n"
"  --nsub x         Number of submatrices, default 1.\n"
"  --niter x        Number of iterations to repeat each test, default 1.\n"
"  --nthread x      Number of CPU threads for some experimental codes, default 1.\n"
"                   (For most testers, set $OMP_NUM_THREADS or $MKL_NUM_THREADS\n"
"                    to control the number of CPU threads.)\n"
"  --offset x       Offset from beginning of matrix, default 0.\n"
"  --itype [123]    Generalized Hermitian-definite eigenproblem type, default 1.\n"
"  --svd-work x     SVD workspace size, one of:\n"
"         query*    queries LAPACK and ICLA\n"
"         doc       is what LAPACK and ICLA document as required\n"
"         doc_old   is what LAPACK <= 3.6 documents\n"
"         min       is minimum required, which may be smaller than doc\n"
"         min_old   is minimum required by LAPACK <= 3.6\n"
"         min_fast  is minimum to take fast path in gesvd\n"
"         min-1     is (minimum - 1), to test error return\n"
"         opt       is optimal\n"
"         opt_old   is optimal as computed by LAPACK <= 3.6\n"
"         opt_slow  is optimal for slow path in gesvd\n"
"         max       is maximum that will be used\n"
"\n"
"  --version x      version of routine, e.g., during development, default 1.\n"
"  --fraction [lower,]upper  computes eigen/singular values and vectors indexed in range [lower*N + 1, upper*N]. Default lower=0, upper=1.\n"
"  --irange   [lower,]upper  computes eigen/singular values and vectors indexed in range [lower, upper]. Default lower=1, upper=min(m,n).\n"
"  --vrange   lower,upper    computes eigen/singular values and vectors in range [lower, upper].\n"
"  --tolerance x    accuracy tolerance, multiplied by machine epsilon, default 30.\n"
"  --tol x          same.\n"
"  -L -U -F         uplo   = Lower*, Upper, or Full.\n"
"  -[NTC][NTC]      transA = NoTrans*, Trans, or ConjTrans (first letter) and\n"
"                   transB = NoTrans*, Trans, or ConjTrans (second letter).\n"
"  -[TC]            transA = Trans or ConjTrans. Default is NoTrans. Doesn't change transB.\n"
"  -S[LR]           side   = Left*, Right.\n"
"  -D[NU]           diag   = NonUnit*, Unit.\n"
"  --jobu [nsoa]    No*, Some, Overwrite, or All left singular vectors (U). gesdd uses this for jobz.\n"
"  --jobv [nsoa]    No*, Some, Overwrite, or All right singular vectors (V).\n"
"  -J[NV]           jobz   = No* or Vectors; compute eigenvectors (symmetric).\n"
"  -L[NV]           jobvl  = No* or Vectors; compute left  eigenvectors (non-symmetric).\n"
"  -R[NV]           jobvr  = No* or Vectors; compute right eigenvectors (non-symmetric).\n"
"\n"
"  --matrix name    name of test matrix type from icla_generate, default 'rand',\n"
"                   or 'rand_dominant' if SPD required (e.g., for posv).\n"
"  --cond   kA      where applicable, condition number for test matrix, default sqrt( 1/eps ); see icla_generate_matrix.\n"
"  --condD  kD      where applicable, condition number for scaling test matrix, default 1; see icla_generate_matrix.\n"
"\n"
"                   * default values\n";

icla_opts::icla_opts( icla_opts_t flag )
{
    int nt = icla_get_lapack_numthreads();

    this->batchcount = 300;
    this->device   = 0;
    this->cache    = 2*1024*1024 * nt;

    this->align    = 32;
    this->nb       = 0;

    this->nrhs     = 1;
    this->nqueue   = 1;
    this->ngpu     = icla_num_gpus();
    this->nsub     = 1;
    this->niter    = 1;
    this->nthread  = 1;
    this->offset   = 0;
    this->itype    = 1;
    this->version  = 1;
    this->verbose  = 0;

    this->kl = 1;
    this->ku = 1;

    this->fraction_lo = 0.;
    this->fraction_up = 1.;
    this->irange_lo = 0;
    this->irange_up = 0;
    this->vrange_lo = 0;
    this->vrange_up = 0;

    this->tolerance = 30.;
    this->check     = (getenv("ICLA_TESTINGS_CHECK") != NULL);
    this->icla     = true;
    this->lapack    = (getenv("ICLA_RUN_LAPACK")     != NULL);
    this->warmup    = (getenv("ICLA_WARMUP")         != NULL);

    this->uplo      = iclaLower;

    this->transA    = iclaNoTrans;

    this->transB    = iclaNoTrans;

    this->side      = iclaLeft;

    this->diag      = iclaNonUnit;

    this->jobz      = iclaNoVec;

    this->jobvr     = iclaNoVec;

    this->jobvl     = iclaNoVec;

    this->matrix    = "rand";
    this->cond      = 0;

    this->condD     = 1;

    this->iseed[0]  = 0;
    this->iseed[1]  = 0;
    this->iseed[2]  = 0;
    this->iseed[3]  = 1;

    if ( flag == iclaOptsBatched ) {

        this->default_nstart = 32;
        this->default_nstep  = 32;
        this->default_nend   = 512;
    }
    else {

        this->default_nstart = 1024 + 64;
        this->default_nstep  = 1024;
        this->default_nend   = 10304;
    }
}

bool scan_comma( char** handle )
{
    char* ptr = *handle;

    while( *ptr == ' ' ) {
        ptr += 1;
    }

    if ( *ptr == ',' ) {
        *handle = ptr + 1;
        return true;
    }
    else {
        return false;
    }
}

bool scan_range( char** handle, int* start, int* end, int* step )
{
    int bytes1, bytes3, cnt;
    char* ptr = *handle;
    cnt = sscanf( ptr, "%d%n:%d:%d%n", start, &bytes1, end, step, &bytes3 );
    if ( cnt == 3 ) {
        *handle += bytes3;
        return (*start >= 0 && *end >= 0 && (*step >= 0 ? *start <= *end : *start >= *end));
    }
    else if ( cnt == 1 ) {
        *handle += bytes1;
        *end  = *start;
        *step = 0;
        return (*start >= 0);
    }
    else {
        return false;
    }
}

void icla_opts::parse_opts( int argc, char** argv )
{
    printf( usage_short, argv[0] );

    icla_int_t ndevices;
    icla_device_t devices[ iclaMaxGPUs ];
    icla_getdevices( devices, iclaMaxGPUs, &ndevices );

    this->ntest = 0;
    for( int i = 1; i < argc; ++i ) {

        if ( (strcmp("-n",      argv[i]) == 0 ||
              strcmp("-N",      argv[i]) == 0 ||
              strcmp("--range", argv[i]) == 0) && i+1 < argc )
        {
            i++;
            int m_start, m_end, m_step;
            int n_start, n_end, n_step;
            int k_start, k_end, k_step;
            char* ptr = argv[i];
            bool valid = scan_range( &ptr, &m_start, &m_end, &m_step );
            if ( valid ) {
                if ( *ptr == '\0' ) {
                    n_start = k_start = m_start;
                    n_end   = k_end   = m_end;
                    n_step  = k_step  = m_step;
                }
                else {
                    valid = scan_comma( &ptr ) && scan_range( &ptr, &n_start, &n_end, &n_step );
                    if ( valid ) {
                        if ( *ptr == '\0' ) {
                            k_start = n_start;
                            k_end   = n_end;
                            k_step  = n_step;
                        }
                        else {
                            valid = scan_comma( &ptr ) && scan_range( &ptr, &k_start, &k_end, &k_step );
                            valid = (valid && *ptr == '\0');
                        }
                    }
                }
            }

            icla_assert( valid, "error: '%s %s' is not valid, expected (m|m_start:m_end:m_step)[,(n|n_start:n_end:n_step)[,(k|k_start:k_end:k_step)]]\n",
                          argv[i-1], argv[i] );

            if ( m_step == 0 && n_step == 0 && k_step == 0 ) {
                icla_assert( this->ntest < MAX_NTEST, "error: %s %s exceeded maximum number of tests (%d).\n",
                              argv[i-1], argv[i], MAX_NTEST );
                this->msize[ this->ntest ] = m_start;
                this->nsize[ this->ntest ] = n_start;
                this->ksize[ this->ntest ] = k_start;
                this->ntest++;
            }
            else {
                for( int m=m_start, n=n_start, k=k_start;
                     (m_step >= 0 ? m <= m_end : m >= m_end) &&
                     (n_step >= 0 ? n <= n_end : n >= n_end) &&
                     (k_step >= 0 ? k <= k_end : k >= k_end);
                     m += m_step, n += n_step, k += k_step )
                {
                    icla_assert( this->ntest < MAX_NTEST, "error: %s %s exceeded maximum number of tests (%d).\n",
                                  argv[i-1], argv[i], MAX_NTEST );
                    this->msize[ this->ntest ] = m;
                    this->nsize[ this->ntest ] = n;
                    this->ksize[ this->ntest ] = k;
                    this->ntest++;
                }
            }
        }

        else if ( strcmp("--dev", argv[i]) == 0 && i+1 < argc ) {
            this->device = atoi( argv[++i] );
            icla_assert( this->device >= 0 && this->device < ndevices,
                          "error: --dev %s is invalid; ensure dev in [0,%d].\n", argv[i], ndevices-1 );
        }
        else if ( strcmp("--align", argv[i]) == 0 && i+1 < argc ) {
            this->align = atoi( argv[++i] );
            icla_assert( this->align >= 1 && this->align <= 4096,
                          "error: --align %s is invalid; ensure align in [1,4096].\n", argv[i] );
        }
        else if ( strcmp("--cache", argv[i]) == 0 && i+1 < argc ) {
            double tmp = atof( argv[++i] );
            icla_assert( tmp >= 1 && tmp <= 1024,
                          "error: --cache %s is invalid; ensure cache in [1,1024].\n", argv[i] );
            this->cache = icla_int_t( tmp * 1024 * 1024 );
        }
        else if ( strcmp("--nrhs",    argv[i]) == 0 && i+1 < argc ) {
            this->nrhs = atoi( argv[++i] );
            icla_assert( this->nrhs >= 0,
                          "error: --nrhs %s is invalid; ensure nrhs >= 0.\n", argv[i] );
        }
        else if ( strcmp("--nb",      argv[i]) == 0 && i+1 < argc ) {
            this->nb = atoi( argv[++i] );
            icla_assert( this->nb > 0,
                          "error: --nb %s is invalid; ensure nb > 0.\n", argv[i] );
        }
        else if ( strcmp("--ngpu",    argv[i]) == 0 && i+1 < argc ) {
            this->ngpu = atoi( argv[++i] );
            icla_assert( this->ngpu <= iclaMaxGPUs,
                          "error: --ngpu %s exceeds iclaMaxGPUs, %d.\n", argv[i], iclaMaxGPUs );
            icla_assert( this->ngpu <= ndevices,
                          "error: --ngpu %s exceeds number of CUDA or OpenCL devices, %d.\n", argv[i], ndevices );

            icla_assert( this->ngpu > 0 || this->ngpu == -1,
                          "error: --ngpu %s is invalid; ensure ngpu != 0.\n", argv[i] );

            char env_num_gpus[20];

            #if defined( _WIN32 ) || defined( _WIN64 )
                snprintf( env_num_gpus, sizeof(env_num_gpus), "ICLA_NUM_GPUS=%lld", (long long) abs(this->ngpu) );
                putenv( env_num_gpus );
            #else
                snprintf( env_num_gpus, sizeof(env_num_gpus), "%lld", (long long) abs(this->ngpu) );
                setenv( "ICLA_NUM_GPUS", env_num_gpus, true );
            #endif
        }
        else if ( strcmp("--nsub", argv[i]) == 0 && i+1 < argc ) {
            this->nsub = atoi( argv[++i] );
            icla_assert( this->nsub > 0,
                          "error: --nsub %s is invalid; ensure nsub > 0.\n", argv[i] );
        }
        else if ( strcmp("--nqueue", argv[i]) == 0 && i+1 < argc ) {
            this->nqueue = atoi( argv[++i] );
            icla_assert( this->nqueue > 0,
                          "error: --nqueue %s is invalid; ensure nqueue > 0.\n", argv[i] );
        }
        else if ( strcmp("--niter",   argv[i]) == 0 && i+1 < argc ) {
            this->niter = atoi( argv[++i] );
            icla_assert( this->niter > 0,
                          "error: --niter %s is invalid; ensure niter > 0.\n", argv[i] );
        }
        else if ( strcmp("--nthread", argv[i]) == 0 && i+1 < argc ) {
            this->nthread = atoi( argv[++i] );
            icla_assert( this->nthread > 0,
                          "error: --nthread %s is invalid; ensure nthread > 0.\n", argv[i] );
        }
        else if ( strcmp("--offset", argv[i]) == 0 && i+1 < argc ) {
            this->offset = atoi( argv[++i] );
            icla_assert( this->offset >= 0,
                          "error: --offset %s is invalid; ensure offset >= 0.\n", argv[i] );
        }
        else if ( strcmp("--itype",   argv[i]) == 0 && i+1 < argc ) {
            this->itype = atoi( argv[++i] );
            icla_assert( this->itype >= 1 && this->itype <= 3,
                          "error: --itype %s is invalid; ensure itype in [1,2,3].\n", argv[i] );
        }
        else if ( strcmp("--version", argv[i]) == 0 && i+1 < argc ) {
            this->version = atoi( argv[++i] );
            icla_assert( this->version >= 1,
                          "error: --version %s is invalid; ensure version > 0.\n", argv[i] );
        }
        else if ( strcmp("--fraction", argv[i]) == 0 && i+1 < argc ) {
            int cnt = sscanf( argv[++i], "%lf,%lf", &this->fraction_lo, &this->fraction_up );
            printf( "fraction cnt %d, lo %.2e, up %.2e\n", cnt, this->fraction_lo, this->fraction_up );
            if (cnt == 1) {
                this->fraction_up = this->fraction_lo;
                this->fraction_lo = 0;
            }
            icla_assert( (cnt == 1 || cnt == 2) &&
                          this->fraction_lo >= 0 &&
                          this->fraction_lo <= this->fraction_up &&
                          this->fraction_up <= 1,
                          "error: --fraction %s is invalid; ensure 0 <= lower <= upper <= 1.\n", argv[i] );
        }
        else if ( strcmp("--irange", argv[i]) == 0 && i+1 < argc ) {
            int lo, hi;
            int cnt = sscanf( argv[++i], "%d,%d", &lo, &hi );
            this->irange_lo = lo;
            this->irange_up = hi;
            if (cnt == 1) {
                this->irange_up = this->irange_lo;
                this->irange_lo = 1;
            }
            icla_assert( (cnt == 1 || cnt == 2) &&
                          this->irange_lo >= 1 &&
                          this->irange_lo <= this->irange_up,
                          "error: --irange %s is invalid; ensure 1 <= lower <= upper.\n", argv[i] );
        }
        else if ( strcmp("--vrange", argv[i]) == 0 && i+1 < argc ) {
            int cnt = sscanf( argv[++i], "%lf,%lf", &this->vrange_lo, &this->vrange_up );
            icla_assert( cnt == 2 &&
                          this->vrange_lo <= this->vrange_up,
                          "error: --vrange %s is invalid; ensure lower <= upper.\n", argv[i] );
        }
        else if ( (strcmp("--tol",       argv[i]) == 0 ||
                   strcmp("--tolerance", argv[i]) == 0) && i+1 < argc ) {
            this->tolerance = atof( argv[++i] );
            icla_assert( this->tolerance >= 0 && this->tolerance <= 1000,
                          "error: --tolerance %s is invalid; ensure tolerance in [0,1000].\n", argv[i] );
        }
        else if ( strcmp("--batch", argv[i]) == 0 && i+1 < argc ) {
            this->batchcount = atoi( argv[++i] );
            icla_assert( this->batchcount > 0,
                          "error: --batch %s is invalid; ensure batch > 0.\n", argv[i] );
        }
        else if ( strcmp("--kl", argv[i]) == 0 && i+1 < argc ) {
            this->kl = atoi( argv[++i] );
            icla_assert( this->kl >= 0,
                          "error: --kl %s is invalid; ensure kl >= 0.\n", argv[i] );
        }
        else if ( strcmp("--ku", argv[i]) == 0 && i+1 < argc ) {
            this->ku = atoi( argv[++i] );
            icla_assert( this->ku >= 0,
                          "error: --ku %s is invalid; ensure ku >= 0.\n", argv[i] );
        }

        else if ( strcmp("-c",         argv[i]) == 0 ||
                  strcmp("--check",    argv[i]) == 0 ) { this->check  = 1; }
        else if ( strcmp("-c2",        argv[i]) == 0 ||
                  strcmp("--check2",   argv[i]) == 0 ) { this->check  = 2; }
        else if ( strcmp("--nocheck",  argv[i]) == 0 ) { this->check  = 0; }

        else if ( strcmp("-l",         argv[i]) == 0 ||
                  strcmp("--lapack",   argv[i]) == 0 ) { this->lapack = true;  }
        else if ( strcmp("--nolapack", argv[i]) == 0 ) { this->lapack = false; }

        else if ( strcmp("--icla",    argv[i]) == 0 ) { this->icla  = true;  }
        else if ( strcmp("--noicla",  argv[i]) == 0 ) { this->icla  = false; }

        else if ( strcmp("--warmup",   argv[i]) == 0 ) { this->warmup = true;  }
        else if ( strcmp("--nowarmup", argv[i]) == 0 ) { this->warmup = false; }

        else if ( strcmp("-v",         argv[i]) == 0 ||
                  strcmp("--verbose",  argv[i]) == 0 ) { this->verbose += 1;  }

        else if ( strcmp("-L",  argv[i]) == 0 ) { this->uplo = iclaLower; }
        else if ( strcmp("-U",  argv[i]) == 0 ) { this->uplo = iclaUpper; }
        else if ( strcmp("-F",  argv[i]) == 0 ) { this->uplo = iclaFull; }

        else if ( strcmp("-NN", argv[i]) == 0 ) { this->transA = iclaNoTrans;   this->transB = iclaNoTrans;   }
        else if ( strcmp("-NT", argv[i]) == 0 ) { this->transA = iclaNoTrans;   this->transB = iclaTrans;     }
        else if ( strcmp("-NC", argv[i]) == 0 ) { this->transA = iclaNoTrans;   this->transB = iclaConjTrans; }
        else if ( strcmp("-TN", argv[i]) == 0 ) { this->transA = iclaTrans;     this->transB = iclaNoTrans;   }
        else if ( strcmp("-TT", argv[i]) == 0 ) { this->transA = iclaTrans;     this->transB = iclaTrans;     }
        else if ( strcmp("-TC", argv[i]) == 0 ) { this->transA = iclaTrans;     this->transB = iclaConjTrans; }
        else if ( strcmp("-CN", argv[i]) == 0 ) { this->transA = iclaConjTrans; this->transB = iclaNoTrans;   }
        else if ( strcmp("-CT", argv[i]) == 0 ) { this->transA = iclaConjTrans; this->transB = iclaTrans;     }
        else if ( strcmp("-CC", argv[i]) == 0 ) { this->transA = iclaConjTrans; this->transB = iclaConjTrans; }
        else if ( strcmp("-T",  argv[i]) == 0 ) { this->transA = iclaTrans;     }
        else if ( strcmp("-C",  argv[i]) == 0 ) { this->transA = iclaConjTrans; }

        else if ( strcmp("-SL", argv[i]) == 0 ) { this->side  = iclaLeft;  }
        else if ( strcmp("-SR", argv[i]) == 0 ) { this->side  = iclaRight; }

        else if ( strcmp("-DN", argv[i]) == 0 ) { this->diag  = iclaNonUnit; }
        else if ( strcmp("-DU", argv[i]) == 0 ) { this->diag  = iclaUnit;    }

        else if ( strcmp("-JN", argv[i]) == 0 ) { this->jobz  = iclaNoVec; }
        else if ( strcmp("-JV", argv[i]) == 0 ) { this->jobz  = iclaVec;   }

        else if ( strcmp("-LN", argv[i]) == 0 ) { this->jobvl = iclaNoVec; }
        else if ( strcmp("-LV", argv[i]) == 0 ) { this->jobvl = iclaVec;   }

        else if ( strcmp("-RN", argv[i]) == 0 ) { this->jobvr = iclaNoVec; }
        else if ( strcmp("-RV", argv[i]) == 0 ) { this->jobvr = iclaVec;   }

        else if ( strcmp("--svd-work", argv[i]) == 0 && i+1 < argc ) {
            i += 1;
            char *token;
            char *arg = strdup( argv[i] );
            for (token = strtok( arg, ", " );
                 token != NULL;
                 token = strtok( NULL, ", " ))
            {
                if ( *token == '\0' ) {
 }
                else if ( strcmp( token, "all"       ) == 0 ) { this->svd_work.push_back( iclaSVD_all        ); }
                else if ( strcmp( token, "query"     ) == 0 ) { this->svd_work.push_back( iclaSVD_query      ); }
                else if ( strcmp( token, "doc"       ) == 0 ) { this->svd_work.push_back( iclaSVD_doc        ); }
                else if ( strcmp( token, "doc_old"   ) == 0 ) { this->svd_work.push_back( iclaSVD_doc_old    ); }
                else if ( strcmp( token, "min"       ) == 0 ) { this->svd_work.push_back( iclaSVD_min        ); }
                else if ( strcmp( token, "min-1"     ) == 0 ) { this->svd_work.push_back( iclaSVD_min_1      ); }
                else if ( strcmp( token, "min_old"   ) == 0 ) { this->svd_work.push_back( iclaSVD_min_old    ); }
                else if ( strcmp( token, "min_old-1" ) == 0 ) { this->svd_work.push_back( iclaSVD_min_old_1  ); }
                else if ( strcmp( token, "min_fast"  ) == 0 ) { this->svd_work.push_back( iclaSVD_min_fast   ); }
                else if ( strcmp( token, "min_fast-1") == 0 ) { this->svd_work.push_back( iclaSVD_min_fast_1 ); }
                else if ( strcmp( token, "opt"       ) == 0 ) { this->svd_work.push_back( iclaSVD_opt        ); }
                else if ( strcmp( token, "opt_old"   ) == 0 ) { this->svd_work.push_back( iclaSVD_opt_old    ); }
                else if ( strcmp( token, "opt_slow"  ) == 0 ) { this->svd_work.push_back( iclaSVD_opt_slow   ); }
                else if ( strcmp( token, "max"       ) == 0 ) { this->svd_work.push_back( iclaSVD_max        ); }
                else {
                    icla_assert( false, "error: --svd-work '%s' is invalid\n", argv[i] );
                }
            }
            free( arg );
        }

        else if ( strcmp("--jobu", argv[i]) == 0 && i+1 < argc ) {
            i += 1;
            const char* arg = argv[i];
            while( *arg != '\0' ) {
                this->jobu.push_back( icla_vec_const( *arg ));
                ++arg;
                if ( *arg == ',' )
                    ++arg;
            }
        }
        else if ( (strcmp("--jobv",  argv[i]) == 0 ||
                   strcmp("--jobvt", argv[i]) == 0) && i+1 < argc ) {
            i += 1;
            const char* arg = argv[i];
            while( *arg != '\0' ) {
                this->jobv.push_back( icla_vec_const( *arg ));
                ++arg;
                if ( *arg == ',' )
                    ++arg;
            }
        }

        else if ( strcmp("--matrix", argv[i]) == 0 && i+1 < argc) {
            i += 1;
            this->matrix = argv[i];
        }
        else if ( strcmp("--cond", argv[i]) == 0 && i+1 < argc) {
            i += 1;
            this->cond = atof( argv[i] );
            icla_assert( this->cond >= 1,
                          "error: --cond %s is invalid; ensure cond >= 1.\n", argv[i] );
        }
        else if ( strcmp("--condD", argv[i]) == 0 && i+1 < argc) {
            i += 1;
            this->condD = atof( argv[i] );
            icla_assert( this->condD >= 1,
                          "error: --condD %s is invalid; ensure condD >= 1.\n", argv[i] );
        }

        else if ( strcmp("-h",     argv[i]) == 0 ||
                  strcmp("--help", argv[i]) == 0 ) {
            fprintf( stderr, usage, argv[0], MAX_NTEST );
            exit(0);
        }
        else {
            fprintf( stderr, "error: unrecognized option %s\n", argv[i] );
            exit(1);
        }
    }

    if ( this->svd_work.size() == 0 ) {
        this->svd_work.push_back( iclaSVD_query );
    }
    if ( this->jobu.size() == 0 ) {
        this->jobu.push_back( iclaNoVec );
    }
    if ( this->jobv.size() == 0 ) {
        this->jobv.push_back( iclaNoVec );
    }

    if ( this->ntest == 0 ) {
        icla_int_t n2 = this->default_nstart;

        while( n2 <= this->default_nend && this->ntest < MAX_NTEST ) {
            this->msize[ this->ntest ] = n2;
            this->nsize[ this->ntest ] = n2;
            this->ksize[ this->ntest ] = n2;
            n2 += this->default_nstep;

            this->ntest++;
        }
    }
    assert( this->ntest <= MAX_NTEST );

    #if defined(ICLA_HAVE_CUDA) || defined(ICLA_HAVE_HIP)
    icla_setdevice( this->device );
    #endif

    icla_queue_create( devices[ this->device ], &this->queues2[ 0 ] );
    icla_queue_create( devices[ this->device ], &this->queues2[ 1 ] );
    this->queues2[ 2 ] = NULL;

    this->queue = this->queues2[ 0 ];

    #if defined(ICLA_HAVE_HIP)

        this->handle = icla_queue_get_hipblas_handle( this->queue );
    #elif defined(ICLA_HAVE_CUDA)

        this->handle = icla_queue_get_cublas_handle( this->queue );
    #else
        #error "unknown platform"
    #endif
}

void icla_opts::get_range(
    icla_int_t n, icla_range_t* range,
    double* vl, double* vu,
    icla_int_t* il, icla_int_t* iu )
{
    *range = iclaRangeAll;
    *il = -1;
    *iu = -1;
    *vl = ICLA_D_NAN;
    *vu = ICLA_D_NAN;
    if (this->fraction_lo != 0 || this->fraction_up != 1) {
        *range = iclaRangeI;
        *il = this->fraction_lo * n;
        *iu = this->fraction_up * n;
        printf( "fraction (%.2f, %.2f) => irange (%lld, %lld)\n",
                this->fraction_lo, this->fraction_up,
                (long long) *il, (long long) *iu );
    }
    else if (this->irange_lo != 0 || this->irange_up != 0) {
        *range = iclaRangeI;
        *il = min( this->irange_lo, n );
        *iu = min( this->irange_up, n );
        printf( "irange (%lld, %lld)\n", (long long) *il, (long long) *iu );
    }
    else if (this->vrange_lo != 0 || this->vrange_up != 0) {
        *range = iclaRangeV;
        *vl = this->vrange_lo;
        *vu = this->vrange_up;
        printf( "vrange (%.2e, %.2e)\n", *vl, *vu );
    }
}

void icla_opts::get_range(
    icla_int_t n, icla_range_t* range,
    float* vl, float* vu,
    icla_int_t* il, icla_int_t* iu )
{
    double dvl, dvu;
    this->get_range( n, range, &dvl, &dvu, il, iu );
    *vl = float(dvl);
    *vu = float(dvu);
}

void icla_opts::cleanup()
{
    this->queue = NULL;
    icla_queue_destroy( this->queues2[0] );
    icla_queue_destroy( this->queues2[1] );
    this->queues2[0] = NULL;
    this->queues2[1] = NULL;

    #if defined(ICLA_HAVE_CUDA) || defined(ICLA_HAVE_HIP)
    this->handle = NULL;
    #endif
}

#ifdef HAVE_PAPI
#include <papi.h>
#include <string.h>

#endif

int gPAPI_flops_set = -1;

extern "C"
void flops_init()
{
    #ifdef HAVE_PAPI
    int err = PAPI_library_init( PAPI_VER_CURRENT );
    if ( err != PAPI_VER_CURRENT ) {
        fprintf( stderr, "Error: PAPI couldn't initialize: %s (%d)\n",
                 PAPI_strerror(err), err );
    }

    err = PAPI_create_eventset( &gPAPI_flops_set );
    if ( err != PAPI_OK ) {
        fprintf( stderr, "Error: PAPI_create_eventset failed\n" );
    }

    err = PAPI_assign_eventset_component( gPAPI_flops_set, 0 );
    if ( err != PAPI_OK ) {
        fprintf( stderr, "Error: PAPI_assign_eventset_component failed: %s (%d)\n",
                 PAPI_strerror(err), err );
    }

    PAPI_option_t opt;
    memset( &opt, 0, sizeof(PAPI_option_t) );
    opt.inherit.inherit  = PAPI_INHERIT_ALL;
    opt.inherit.eventset = gPAPI_flops_set;
    err = PAPI_set_opt( PAPI_INHERIT, &opt );
    if ( err != PAPI_OK ) {
        fprintf( stderr, "Error: PAPI_set_opt failed: %s (%d)\n",
                 PAPI_strerror(err), err );
    }

    err = PAPI_add_event( gPAPI_flops_set, PAPI_FP_OPS );
    if ( err != PAPI_OK ) {
        fprintf( stderr, "Error: PAPI_add_event failed: %s (%d)\n",
                 PAPI_strerror(err), err );
    }

    err = PAPI_start( gPAPI_flops_set );
    if ( err != PAPI_OK ) {
        fprintf( stderr, "Error: PAPI_start failed: %s (%d)\n",
                 PAPI_strerror(err), err );
    }
    #endif

}

void icla_flush_cache( size_t cache_size )
{
    unsigned char* buf = (unsigned char*) malloc( 2 * cache_size );

    int nthread = 1;
    #pragma omp parallel
    #pragma omp master
    {
        #ifdef _OPENMP
        nthread = omp_get_num_threads();
        #endif
    }

    size_t per_core = 2 * cache_size / nthread;

    #pragma omp parallel
    {
        int tid = 0;
        #ifdef _OPENMP
        tid = omp_get_thread_num();
        #endif
        for (size_t i = tid * per_core; i < (tid + 1) * per_core; ++i) {
            buf[i] = i % 256;
        }
    }

    free( buf );
}
