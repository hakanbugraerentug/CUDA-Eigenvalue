#pragma once

#include <armadillo>
#include <math.h>
#include "consts.h"
#include "mkl.h"

using namespace arma;
using namespace arma::newarp;
/*
int mkl_extreme_eigs_dense(Mat<dtype>& Aorg, MKL_INT k0,
			   Col<dtype>& eigs) { //k0 maks/min kaç tane eigenvalue bulmak istediğimin sayısı
  MKL_INT n = N, lda = LDA, ldz = LDZ, il, iu, m, info;
  double abstol, vl, vu;
  // Local arrays
  MKL_INT ifail[N];
  double w[N];
  MKL_Complex16 z[LDZ*N];
  MKL_Complex16 a[LDA*N] = {
    { 6.51,  0.00}, {-5.92, -9.53}, {-2.46, -2.91}, { 8.84, -3.21},
    { 0.00,  0.00}, {-1.73,  0.00}, { 6.50, -2.09}, { 1.32, -8.81},
    { 0.00,  0.00}, { 0.00,  0.00}, { 6.90,  0.00}, {-0.59, -2.47},
    { 0.00,  0.00}, { 0.00,  0.00}, { 0.00,  0.00}, {-2.85,  0.00}
  };
  // Executable statements
  printf( "LAPACKE_zheevx (column-major, high-level) Example Program Results\n" );
  // Negative abstol means using the default value
  abstol = -1.0;
  // Set VL, VU to compute eigenvalues in half-open (VL,VU] interval
  vl = 0.0;
  vu = 100;
  // Solve eigenproblem
  info = LAPACKE_zheevx( LAPACK_COL_MAJOR, 'V', 'V', 'L', n, a, lda,
			 vl, vu, il, iu, abstol, &m, w, z, ldz, ifail );
  // Check for convergence
  if( info > 0 ) {
    printf( "The algorithm failed to compute eigenvalues.\n" );
    exit( 1 );
  }
  // Print the number of eigenvalues found
  printf( "\n The total number of eigenvalues found:%2i\n", m );
  // Print eigenvalues

  for(int i = 0; i < m; i++) {
    printf("%d: %f\n", i, w[i]);
  }
}*/ /* End of LAPACKE_zheevx Example */
/*
int mkl_extreme_eigs(Mat<dtype>& Aorg, MKL_INT k0,
		     Col<dtype>& eigs) { //k0 maks/min kaç tane eigenvalue bulmak istediğimin sayısı

  // Matrix A of size N in CSR format
  //MKL_INT N = Aorg.n_rows;               //number of rows in matrix A

  //MKL_INT* ia = new MKL_INT[N + 1]; // ia bir rowda kaç tane non-zero değer olduğunu tutuyo (1 fazlası olarak ama)
  //MKL_INT* ja = new MKL_INT[N * N]; // ja columdaki indexlerini tutuyo non-zero olanlarınkini
  //dtype* a = new dtype[N * N]; // non-zero elementlerin değerlerini tutuyo

  //int nnz = 1;
  //for(int i = 0; i < N; i++) {
    //ia[i] = nnz;
    //for(int j = 0; j < N; j++) {
      //if(Aorg(i,j) != 0) {
	//ja[nnz - 1] = j + 1;
	//a[nnz - 1] = Aorg(i,j);
	//nnz++;
      //}
    //}
  //}
  //ia[N] = nnz;

    // Matrix A of size N in CSR format */
  /*  MKL_INT N = 4;               // number of rows in matrix A
    MKL_INT M = 4;
    MKL_INT nnz = 8;

    MKL_INT ia[5] = {1,3,5,7,9};
    MKL_INT ja[8] = {1,2,1,2,3,4,3,4};
    double   a[8] = {6.0,2.0,2.0,3.0,2.0,-1.0,-1.0,2.0};


  // mkl_sparse_d_ev input parameters */
  //char         which = 'S'; /* Which eigenvalues to calculate. ('L' - largest (algebraic) eigenvalues, 'S' - smallest (algebraic) eigenvalues) */
  //MKL_INT      pm[128];     /* This array is used to pass various parameters to Extended Eigensolver Extensions routines. */

  /* mkl_sparse_d_ev output parameters */
  //MKL_INT      k;           /* Number of eigenvalues found (might be less than k0). */
  //double       E[16];   /* Eigenvalues */
  //double       X[16];   /* Eigenvectors */
  //double       res[16]; /* Residual */

  /* Local variables */
  //MKL_INT      info;               /* Errors */
  //MKL_INT      compute_vectors = 0;/* Flag to compute eigenvecors */
  //MKL_INT      tol = 7;            /* Tolerance */
  //double       Y[16];               /* Y=(X')*X-I */
  //double       sparsity;           /* Sparsity of randomly generated matrix */
  //MKL_INT      i, j;
  //double       smax, t;

  /* Input variables for DGEMM */
  //char         DGEMMC = 'T';       /* Character for GEMM routine, transposed case */
  //char         DGEMMN = 'N';       /* Character for GEMM routine, non-transposed case */
  //double       one  = 1.0;         /* alpha parameter for GEMM */
  //double       zero = 0.0;         /* beta  parameter for GEMM */
  //MKL_INT      ldx  = N;           /* Leading dimension for source arrays in GEMM */
  //MKL_INT      ldy;                /* Leading dimension for destination array in GEMM */

  /* Sparse BLAS IE variables */
  //sparse_status_t status;
  //sparse_matrix_t A = NULL; /* Handle containing sparse matrix in internal data structure */
  //struct matrix_descr descr; /* Structure specifying sparse matrix properties */

  /* Create handle for matrix A stored in CSR format */
  //descr.type = SPARSE_MATRIX_TYPE_GENERAL; /* Full matrix is stored */
  //status = mkl_sparse_d_create_csr ( &A, SPARSE_INDEX_BASE_ONE, N, N, ia, ia+1, ja, a );

  /* Step 2. Call mkl_sparse_ee_init to define default input values */
  //mkl_sparse_ee_init(pm);

  //pm[1] = tol; /* Set tolerance */
  //pm[6] = compute_vectors;

  //cout << "Start" << endl;
  /* Step 3. Solve the standard Ax = ex eigenvalue problem. */
  //info = mkl_sparse_d_ev(&which, pm, A, descr, k0, &k, E, X, res);

  //delete [] a;
  //delete [] ia;
  //delete [] ja;

  //printf("mkl_sparse_d_ev output info %d \n",info);
  //if ( info != 0 )
    //{
      //printf("Routine mkl_sparse_d_ev returns code of ERROR: %i", (int)info);
      //return 1;
    //}

  //printf("*************************************************\n");
  //printf("************** REPORT ***************************\n");
  //printf("*************************************************\n");
  //printf("#mode found/subspace %d %d \n", k, k0);
  //printf("Index/Exact Eigenvalues/Estimated Eigenvalues/Residuals\n");
  //for (i=0; i<k; i++)
    //{
      //printf("   %d  %.15e %.15e \n", i, E[i], res[i]);
    //}
  //return 1;
//}

void alloc4D(dtype**** &ptr, int n1, int n2, int n3, int n4) {
  ptr = new dtype***[n1];
  for(int i = 0; i < n1; i++) {
    ptr[i] = new dtype**[n2];
    for(int j = 0; j < n2; j++) {
      ptr[i][j] = new dtype*[n3];
      for(int k = 0; k < n3; k++) {
	ptr[i][j][k] = new dtype[n4];
      }
    }
  }
}

void free4D(dtype**** &ptr, int n1, int n2, int n3, int n4) {
  for(int i = 0; i < n1; i++) {
    for(int j = 0; j < n2; j++) {
      for(int k = 0; k < n3; k++) {
        delete [] ptr[i][j][k];
      }
      delete [] ptr[i][j];
    }
    delete [] ptr[i];
  }
  delete [] ptr;
}

void alloc5D(dtype***** &ptr, int n1, int n2, int n3, int n4, int n5) {
  ptr = new dtype****[n1];
  for(int i = 0; i < n1; i++) {
    alloc4D(ptr[i], n2, n3, n4, n5);
  }
}

void free5D(dtype***** &ptr, int n1, int n2, int n3, int n4, int n5) {
  for(int i = 0; i < n1; i++) {
    free4D(ptr[i], n2, n3, n4, n5);
  }
  delete [] ptr;
}

void alloc6D(dtype****** &ptr, int n1, int n2, int n3, int n4, int n5, int n6) {
  ptr = new dtype*****[n1];
  for(int i = 0; i < n1; i++) {
    alloc5D(ptr[i], n2, n3, n4, n5, n6);
  }
}

void free6D(dtype****** &ptr, int n1, int n2, int n3, int n4, int n5, int n6) {
  for(int i = 0; i < n1; i++) {
    free5D(ptr[i], n2, n3, n4, n5, n6);
  }
  delete [] ptr;
}


//void alloc4D(dtype**** &ptr, int n1, int n2, int n3, int n4) {
//}

Col<dtype> cheb_di(dtype start, dtype end, int N) {
  Col<dtype> v_di(N, fill::zeros);
  for(int i = 1; i <= N; i += 2) {
    v_di(i-1) = (end - start) / (1.0 - ((i-1) * (i-1)));
  }
  return v_di;
}

Col<dtype> cheb_int(dtype start, dtype end, int N) {
  dtype scale = 2 / (end - start);
  Col<dtype> vecint = zeros<Col<dtype>>(N);
  for(int k = 0; k < N; k += 2) {
    vecint(k) = 2 / ((1 - (k * k)) * scale);
  }
  return vecint;
}

void cheb(int N,
	  Mat<dtype> &IT, Mat<dtype> &FT) {
  for(int i = 1; i <= N; i++) {
    for(int j = 1; j <= N; j++) {
      IT(i-1,j-1) = cos((j-1) * pi * (N - i)/(N - 1));
    }
  }
  //  IT.print();
  FT = inv(IT); /////////////////////////
  //Mat<dtype> I = IT * FT;
  //I.print();
}

void derivative(dtype start, dtype end, int N,
		Mat<dtype>& D) {
  dtype scale = (end - start) / 2;
  int odd = (N % 2 == 1);

  int DN = N / 2;

  D.zeros(); //sıfırla fill ediyo
  for(int i = 0; i < DN; i++) {
    D(0, 2*i + 1) = 2*i + 1;
  }

  for(int i = 2; i <= N; i++) {
    if(i % 2 == 0) {
      for(int j = 1; j < DN + odd; j++) { //SOR burada odd gerekli mi, bu caseler birbirinin ayni mi yoksa
	      D(i-1, 2*j) = 4 * j;
      }
    }
    else {
      for(int j = 1; j < DN; j++) {
	      D(i-1, 2*j + 1) = 2 * (2*j + 1);
      }
    }
  }
  /////////////////
  D = trimatu(D) / scale; //trimatu --> creates new matrix and copies the triangular upper part and set others to 0
}

Col<dtype> lobat(int N) {
  Col<dtype> x(N);
  int nm1 = N - 1;
  for(int i = 0; i < N; i++) {
    x(i) = sin((pi * ((-1 * nm1) + (2.0 * i))) / (2.0 * nm1));
  }
  return x;
}

void slobat(dtype start, dtype end, int N,
	    Col<dtype>& s) { //This function calculates G/L points spanning -1 to 1, given n the number of sampling points required
  Col<dtype> lbt = lobat(N);
  dtype alpha = (end - start) / 2;
  dtype beta = (end + start) / 2;
  Col<dtype> z(N);
  z.ones(); //birle fill ediyo
  s  = alpha * lbt + beta * z;
}

void inner_product_helper(dtype start, dtype end, int N,
			  Mat<dtype>& V) {
  int np2 = 2 * N;
  Mat<dtype> IT2(np2, np2);
  Mat<dtype> FT2(np2, np2);
  cheb(np2, IT2, FT2);

  Mat<dtype> IT(N, N);
  Mat<dtype> FT(N, N);
  cheb(N, IT, FT);

  Col<dtype> v_di = cheb_di(start, end, np2);
  Mat<dtype> v_d2N = diagmat(trans(v_di) * FT2); ////////////

  Mat<dtype> merged = join_cols(eye<Mat<dtype>>(N,N), zeros<Mat<dtype>>(N,N));
  Mat<dtype> S = IT2 * merged * FT;

  V = trans(S) * v_d2N * S; ///////////////
}
