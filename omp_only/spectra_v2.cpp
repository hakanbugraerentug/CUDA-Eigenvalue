#include <iostream>
#include <math.h>
#include <chrono>
#include <limits>
#include <fstream>

#include "omp.h"
#include "consts.h"
#include "helpers.h"
#include "mkl.h"

using namespace std;

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////MODELING PART STARTS//////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////

class FGM;

struct Space {
public:
  dtype start;
  dtype end;
  int no_points;

  Mat<dtype> IT; //Inverse transformation matrix
  Mat<dtype> FT; //forward transformation matrix so FT = inv(IT)
  Mat<dtype> D;
  Col<dtype> s;
  Mat<dtype> V;
  Mat<dtype> Q1;

  Space(dtype start, dtype end, int no_points)
    : start(start), end(end), no_points(no_points), IT(no_points, no_points), FT(no_points, no_points), D(no_points, no_points), s(no_points) {
    discretize();
  }

  void discretize() {
    cheb(no_points, IT, FT);
    DBG(cout << "IT\n"; IT.print(); cout << endl;);
    DBG(cout << "FT\n"; FT.print(); cout << endl;);
    derivative(start, end, no_points, D);
    DBG(cout << "D\n";D.print(); cout << endl;);
    slobat(start, end, no_points, s);
    DBG(cout << "s\n"; s.print(); cout << endl;);
    inner_product_helper(start, end, no_points, V);
    DBG(cout << "V\n"; V.print(); cout << endl;);
    Q1 = IT * D * FT;
    DBG(cout << "Q1\n"; Q1.print(); cout << endl;);
  }
};

class Shape { //Space(0, z_dim, z_sample) neden böyle de Space(-z_dim/2, z_dim/2, z_sample) değil??????
public:
  Shape(dtype x_dim, dtype y_dim, dtype z_dim,
	int x_sample, int y_sample, int z_sample,
	dtype xcurve = 0, dtype ycurve = 0) :
    dim{x_dim, y_dim, z_dim}, curve{xcurve, ycurve},
    is_curved(~(xcurve == 0 && ycurve == 0)),
    spaces{Space(-x_dim/2, x_dim/2, x_sample), Space(-y_dim/2, y_dim/2, y_sample), Space(0, z_dim, z_sample)},
    xyz(x_sample * y_sample * z_sample),
    VD(xyz, xyz),
    QDx(xyz, xyz, fill::zeros), QDy(xyz, xyz, fill::zeros), QDz(xyz, xyz, fill::zeros) {
      vector_map_nojac();
  }

  void vector_map_nojac() {
    int npx = spaces[0].no_points;
    int npy = spaces[1].no_points;
    int npz = spaces[2].no_points;

    int xyz = npx * npy * npz;
    Mat<dtype> VDx(xyz, xyz, fill::zeros);
    Mat<dtype> VDy(xyz, xyz, fill::zeros);
    Mat<dtype> VDz(xyz, xyz, fill::zeros);

    for(int i = 1; i <= npx; i++) {
      for(int j = 1; j <= npy; j++) {
	      for(int k = 1; k <= npz; k++) {
	        int I = ((i-1) * npy * npz) + ((j-1) * npz) + k;

          for(int l = 1; l <= npx; l++) {
            int J = ((l-1) * npy * npz) + ((j-1) * npz) + k;
            VDx(J-1, I-1) += spaces[0].V(l-1, i-1);
            QDx(J-1, I-1) += spaces[0].Q1(l-1, i-1);
          }

          for(int l = 1; l <= npy; l++) {
            int J = ((i-1) * npy * npz) + ((l-1) * npz) + k;
            VDy(J-1, I-1) += spaces[1].V(l-1, j-1);
            QDy(J-1, I-1) += spaces[1].Q1(l-1, j-1);
          }

          for(int l = 1; l <= npz; l++) {
            int J = ((i-1) * npy * npz) + ((j-1) * npz) + l;
            VDz(J-1, I-1) += spaces[2].V(l-1, k-1);
            QDz(J-1, I-1) += spaces[2].Q1(l-1, k-1);
          }
	      }
      }
    }

    VD = VDx * VDy * VDz;
  }

  const dtype dim[3];
  const bool is_curved;
  const dtype curve[2];

  Space spaces[3];

  const int xyz;
  Mat<dtype> VD;
  Mat<dtype> QDx;
  Mat<dtype> QDy;
  Mat<dtype> QDz;
};

class Material {
public:
  Material(dtype _mod_elasticity,
	   dtype _poisson_ratio,
	   dtype _density)
    : mod_elasticity(_mod_elasticity),
      poisson_ratio(_poisson_ratio),
      density(_density) {}

  //member variables
  const dtype mod_elasticity;
  const dtype poisson_ratio;
  const dtype density;
};

//Functionally graded material
class FGM {
public:
  FGM(Shape& _shape,
      Material& first, Material& second,
      dtype _ctrl_y, dtype _ctrl_z) :
    shape(_shape),
    ctrl_y(_ctrl_y), ctrl_z(_ctrl_z),
    mats{first, second},
    np{_shape.spaces[0].no_points, _shape.spaces[1].no_points, _shape.spaces[2].no_points},
    nxyz(np[0] * np[1] * np[2]),
    mu(np[0], np[1], np[2], fill::zeros),
    lame(np[0], np[1], np[2], fill::zeros),
    rho(np[0], np[1], np[2], fill::zeros),
    VD_mu(nxyz, nxyz, fill::zeros),
    VD_lame(nxyz, nxyz, fill::zeros),
    VD_rho(nxyz, nxyz, fill::zeros),
    M(3 * nxyz, 3 * nxyz, fill::zeros),
    K(3 * nxyz, 3 * nxyz, fill::zeros)
  {
    FG_var_MT();
    inner_product();

    double start, end;
    TIME("system-mat", start, end,
	  system_matrices()
	 );
  }

  //////////////////////////////////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////MODELING PART ENDS//////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////

  void compute(const int noeigs, const int ncv, int& nconv, double& small_eig,
	       const double shift = 0.01, const int max_iter = -1, const double tol = -1) {

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////COMPONENT SYNTHESIS STARTS//////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////

    Mat<dtype> BC_3D_I = boundary_condition_3d(0, 0);
    Mat<dtype> BC_3D_II = boundary_condition_3d(0, 1);

    Mat<dtype> BC_1 = beta_matrix_3d(BC_3D_I, 0);
    Mat<dtype> BC_2 = beta_matrix_3d(BC_3D_II, 0);
    Mat<dtype> BC = join_cols(BC_1, BC_2);

    Mat<dtype> U, V;
    Col<dtype> s;


    double start, end;
    TIME("SVD", start, end, svd(U, s, V, BC););


    TIME("Mul and Inv", start, end,

      Mat<dtype> P = V.submat(0, BC.n_rows, V.n_rows-1, BC.n_cols - 1);

      Mat<dtype> K_phy = P.t() * K * P;
      Mat<dtype> M_phy = P.t() * M * P;

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////COMPONENT SYNTHESIS ENDS//////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////EIGENVALUE STARTS//////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////

      Mat<dtype> a0 = M_phy.i() * K_phy;
	 );

    cx_vec eigval;
    cx_mat eigvec;

    /*
    start = omp_get_wtime();
    eig_gen(eigval, eigvec, a0, "nobalance");
    end = omp_get_wtime();
    cout << "Computing all (" << eigvec.n_rows << ") eigs took " << end - start << " seconds\n";
    sort(eigval, "descend").print();
    */
    /*
    SpMat<dtype> S(a0);
    S.print();
    */
    Mat<dtype> X = a0 + shift * eye(size(a0));
    DenseGenMatProd<dtype> op(X);
    //ncv_Parameter that controls the convergence speed of the algorithm. Typically a larger ncv_ means faster convergence,
    //but it may also result in greater memory use and more matrix operations in each iteration. This parameter must satisfy
    //nev+2≤ncv≤n, and is advised to take ncv≥2⋅nev+1.
    #pragma omp critical
    {
    GenEigsSolver<dtype, EigsSelect::SMALLEST_MAGN, DenseGenMatProd<dtype> > eigs(op, noeigs, ncv);
    eigs.init();

    if(max_iter == -1) {
      TIME("Eigen", start, end, nconv = eigs.compute());
    } else if (tol == -1) {
      TIME("Eigen", start, end, nconv = eigs.compute(max_iter));
    } else {
      TIME("Eigen", start, end, nconv = eigs.compute(max_iter, tol));
    }

    if(nconv > 0) {
      arma::cx_vec evalues = sort(eigs.eigenvalues() - shift);
      small_eig = evalues(0).real();
    }
    }

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////EIGENVALUE ENDS//////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////

   /*
    Col<dtype> eigvecreal;
    TIME("sparse eig: ", start, end, mkl_extreme_eigs_dense(a0, noeigs, eigvecreal););
    */
  }

  Mat<dtype> beta_matrix_3d(Mat<dtype>& BC_3D, int xyz) {
    Mat<dtype> BC(3 * nxyz / np[xyz], 3 * nxyz, fill::zeros);
    int ids[3];
    for(int dim = 0; dim < 3; dim++) {
      for(int i = 0; i < np[0]; i++) {
        ids[0] = i;
        for(int j = 0; j < np[1]; j++) {
          ids[1] = j;
          for(int k = 0; k < np[2]; k++) {
            ids[2] = k;

            int idx = dim * (nxyz / np[xyz]);
            if(xyz == 0) idx += j * np[2] + k;
            else if(xyz == 1) idx += i * np[2] + k;
            else if(xyz == 2) idx += i * np[1] + j;
            int idy = dim * nxyz + i * np[1] * np[2] + j * np[2] + k;

            BC(idx, idy) = BC_3D(dim, ids[xyz]);

          }
        }
      }
    }
    return BC;
  }

  void FG_var_MT() {
    Col<dtype>& x = shape.spaces[0].s;
    Col<dtype>& y = shape.spaces[1].s;
    Col<dtype>& z = shape.spaces[2].s;

    dtype K_m = (mats[0].mod_elasticity / 3) / (1 - 2 * mats[0].poisson_ratio);
    dtype G_m = (mats[0].mod_elasticity / 2) / (1 + mats[0].poisson_ratio);

    dtype K_c = (mats[1].mod_elasticity / 3) / (1 - 2 * mats[1].poisson_ratio);
    dtype G_c = (mats[1].mod_elasticity / 2) / (1 + mats[1].poisson_ratio);

    dtype V_min = 0;
    dtype V_max = 1;

    //for matlab conversion - can be removed later
    dtype c = shape.dim[2];
    dtype b = shape.dim[1];
    dtype p = ctrl_z;
    dtype q = ctrl_y;
    dtype rho_m = mats[0].density;
    dtype rho_c = mats[1].density;

    for(int j = 0; j < np[1]; j++) {
      for(int k = 0; k < np[2]; k++) {
        //bu satirda funtion pointer olabilir
        dtype vcijk = V_min + (V_max-V_min) * pow((z(k)/c), p) * pow((0.5+y(j)/b), q);
        dtype vmijk  = 1 - vcijk;
        dtype rhotemp = (rho_c * vcijk) + (rho_m * vmijk);
        dtype K = K_m + (K_c - K_m) * vcijk / (1 + (1 - vcijk) * (3 * (K_c - K_m) / (3*K_m + 4*G_m)));
        dtype f1 = G_m*(9*K_m+8*G_m)/(6*(K_m+2*G_m));
        dtype G = G_m + (G_c-G_m) * vcijk/(1 + (1- vcijk)*( (G_c-G_m)/(G_m+f1)));
        dtype eijk = 9*K*G/(3*K+G);
        dtype poisijk = (3*K-2*G)/(2*(3*K+G));
        dtype mutemp = eijk / (2 * (1 + poisijk));
        dtype lametemp = (2 * mutemp * poisijk) / (1 - 2 * poisijk);

        for(int i = 0; i < np[0]; i++) {
          rho(i,j,k) = rhotemp;
          mu(i,j,k) = mutemp;
          lame(i,j,k) = lametemp;
        }
      }
    }
  }

  void tensor3(Col<dtype>& v_d3Nt, Mat<dtype>& Sst, int n, Cube<dtype>& X) {
#pragma omp parallel for collapse(3) //for now did not effect any calculated time output I will check it further
    for(int i = 0; i < n; i++) {
      for(int j = 0; j < n; j++) {
        for(int k = 0; k < n; k++) {
          dtype xijk = 0;
          #pragma omp parallel for reduction(+:xijk) //for now did not effect any calculated time output I will check it further
          for(int l = 0; l < 3 * n; l++) {
            xijk += v_d3Nt(l) * Sst(l,i) *  Sst(l,j) *  Sst(l,k);
          }
          //#pragma omp atomic
          X(i, j, k) = xijk;
        }
      }
    }
  }

  void inner_helper(Cube<dtype>& Axyz, Cube<dtype>& Xadl, Cube<dtype>& Ybem, Cube<dtype>& Zcfn,
		    Mat<dtype>& VD) {
    dtype**** Xadmn;
    alloc4D(Xadmn, np[0], np[0], np[1], np[2]);
#pragma omp parallel for collapse(4) //for now did not effect any calculated time output I will check it further
    for(int i = 0; i < np[0]; i++) {
      for(int j = 0; j < np[0]; j++) {
        for(int k = 0; k < np[1]; k++) {
          for(int l = 0; l < np[2]; l++) {
            dtype sum = 0;
            #pragma omp parallel for reduction(+:sum) //for now did not effect any calculated time output I will check it further
            for(int m = 0; m < np[0]; m++) {
              sum += Xadl(i, j, m) * Axyz(m, k, l);
            }
            //#pragma omp atomic
            Xadmn[i][j][k][l] = sum;
          }
        }
      }
    }

    dtype***** Gamma_adnbe;
    alloc5D(Gamma_adnbe, np[0], np[0], np[2], np[1], np[1]);
#pragma omp parallel for collapse(5) //for now did not effect any calculated time output I will check it further
    for(int i = 0; i < np[0]; i++) {
      for(int j = 0; j < np[0]; j++) {
        for(int k = 0; k < np[2]; k++) {
          for(int l = 0; l < np[1]; l++) {
            for(int m = 0; m < np[1]; m++) {
              dtype sum = 0;
              #pragma omp parallel for reduction(+:sum) //for now did not effect any calculated time output I will check it further
              for(int o = 0; o < np[1]; o++) {
                sum += Xadmn[i][j][o][k] * Ybem(o, l, m);
              }
              //#pragma omp atomic
              Gamma_adnbe[i][j][k][l][m] = sum;
            }
          }
        }
      }
    }

#pragma omp parallel for collapse(6) //for now did not effect any calculated time output I will check it further
    for(int i = 0; i < np[0]; i++) {
      for(int j = 0; j < np[0]; j++) {
        for(int k = 0; k < np[1]; k++) {
          for(int l = 0; l < np[1]; l++) {
            for(int m = 0; m < np[2]; m++) {
              for(int o = 0; o < np[2]; o++) {
                dtype sum = 0;
                #pragma omp parallel for reduction(+:sum) //for now did not effect any calculated time output I will check it further
                for(int v = 0; v < np[2]; v++) {
                  sum += Gamma_adnbe[i][j][v][k][l] * Zcfn(v, m, o);
                }
                int row = (i)*np[1]*np[2]+(k)*np[2] + m;
                int col = (j)*np[1]*np[2]+(l)*np[2] + o;

                //#pragma omp atomic
                VD(row, col) += sum;
              }
            }
          }
        }
      }
    }

    free4D(Xadmn, np[0], np[0], np[1], np[2]);
    free5D(Gamma_adnbe, np[0], np[0], np[2], np[1], np[1]);
  }

  void inner_product() {
    Mat<dtype> IFT[3][3][2];
    for(int i = 0; i < 3; i++) { //xyz loop
      for(int j = 0; j < 3; j++) { //123 loop
        int sz = (j + 1) * np[i];
        IFT[i][j][0] = zeros<Mat<dtype> >(sz, sz);
        IFT[i][j][1] = zeros<Mat<dtype> >(sz, sz);
        cheb(sz, IFT[i][j][0], IFT[i][j][1]);
      }
    }

    Col<dtype> v_d3N[3];
    for(int i = 0; i < 3; i++) {
      Col<dtype> temp = cheb_int(shape.spaces[i].start, shape.spaces[i].end, 3 * np[i]);
      v_d3N[i] = (temp.t() * IFT[i][2][1]).t();
    }


    Mat<dtype> Ss[3];
    for(int i = 0; i < 3; i++) {
      Mat<dtype> I(np[i], np[i], fill::eye);
      Mat<dtype> Z(np[i], np[i], fill::zeros);
      Ss[i] = IFT[i][2][0] * join_cols(I, Z, Z) * IFT[i][0][1];
    }

    Cube<dtype> Xadl(np[0], np[0], np[0], fill::zeros);
    tensor3(v_d3N[0], Ss[0], np[0], Xadl);
    Cube<dtype> Ybem(np[1], np[1], np[1], fill::zeros);
    tensor3(v_d3N[1], Ss[1], np[1], Ybem);
    Cube<dtype> Zcfn(np[2], np[2], np[2], fill::zeros);
    tensor3(v_d3N[2], Ss[2], np[2], Zcfn);

    inner_helper(mu, Xadl, Ybem, Zcfn, VD_mu);
    /*
    dtype sum = 0, mine = 10000000, maxe = -10000000;
    for(int i = 0 ; i < 9; i++) {
      for(int j = 0 ; j < 9; j++) {
	for(int k = 0 ; k < 9; k++) {
	  dtype elem = Xadl(i,j,k);
	  sum += elem;
	  if(mine > elem) mine = elem;
	  if(maxe < elem) maxe = elem;
	}
      }
    }
    cout << sum << " " << mine << " " << maxe << endl;*/
    inner_helper(rho, Xadl, Ybem, Zcfn, VD_rho);
    inner_helper(lame, Xadl, Ybem, Zcfn, VD_lame);
  }

  void system_matrices() {
    int ub = 0;
    int ue = nxyz-1;
    int vb = nxyz;
    int ve = 2 * nxyz - 1;
    int wb = 2 * nxyz;
    int we = 3 * nxyz - 1;

    M(span(ub, ue), span(ub, ue)) = VD_rho;
    M(span(vb, ve), span(vb, ve)) = VD_rho;
    M(span(wb, we), span(wb, we)) = VD_rho;

    // i) First {ep}=[B]*{q}
    Mat<dtype> epx(nxyz, 3 * nxyz, fill::zeros);
    epx(span(0, nxyz - 1), span(ub, ue)) = shape.QDx;
    Mat<dtype> epy(nxyz, 3 * nxyz);
    epy(span(0, nxyz - 1), span(vb, ve)) = shape.QDy;
    Mat<dtype> epz(nxyz, 3 * nxyz);
    epz(span(0, nxyz - 1), span(wb, we)) = shape.QDz;

    Mat<dtype> gammaxy(nxyz, 3 * nxyz);
    gammaxy(span(0, nxyz - 1), span(ub, ue)) = shape.QDy;
    gammaxy(span(0, nxyz - 1) ,span(vb, ve)) = shape.QDx;
    Mat<dtype> gammayz(nxyz, 3 * nxyz);
    gammayz(span(0, nxyz - 1), span(vb, ve)) = shape.QDz;
    gammayz(span(0, nxyz - 1), span(wb, we)) = shape.QDy;
    Mat<dtype> gammaxz(nxyz, 3 * nxyz);
    gammaxz(span(0, nxyz - 1), span(ub, ue)) = shape.QDz;
    gammaxz(span(0, nxyz - 1), span(wb, we)) = shape.QDx;

    // ii) Second the stress (sigma) term

    Mat<dtype> epxt = epx.t();
    Mat<dtype> epyt = epy.t();
    Mat<dtype> epzt = epz.t();

    Mat<dtype> xlame = epxt * VD_lame;
    Mat<dtype> ylame = epyt * VD_lame;
    Mat<dtype> zlame = epzt * VD_lame;
    Mat<dtype> epxyz = epx + epy + epz;

    K = (xlame + ylame + zlame) * epxyz +
      2 * epxt * VD_mu * epx +
      2 * epyt * VD_mu * epy +
      2 * epzt * VD_mu * epz  +
      gammaxy.t() * VD_mu * gammaxy +
      gammaxz.t() * VD_mu * gammaxz +
      gammayz.t() * VD_mu * gammayz;
  }


  Mat<dtype> boundary_condition_3d(int xyz, int ol) {
    dtype bc[3] = {1,1,1};
    Row<dtype> e = zeros<Row<dtype> >(np[xyz]);
    if(ol == 0) {
      e(0) = 1.0;
    }
    else {
      e(np[xyz] - 1) = 1.0;
    }
    Mat<dtype> BC =  join_cols(bc[0] * e, bc[1] * e, bc[2] * e);
    return BC;
  }

  int np[3]; //not to do this everytime
  int nxyz; //not to do this everytime

  Shape& shape;
  Material mats[2];
  const dtype ctrl_y;
  const dtype ctrl_z;

  Cube<dtype> mu;
  Cube<dtype> lame;
  Cube<dtype> rho;

  Mat<dtype> VD_mu;
  Mat<dtype> VD_lame;
  Mat<dtype> VD_rho;

  Mat<dtype> M;
  Mat<dtype> K;
};

ostream& operator<<(ostream& os, const Space& spc) {
  os << spc.start << "\t" << spc.end << "\t" << spc.no_points;
  return os;
}

ostream& operator<<(ostream& os, const Material& mat) {
  os << mat.mod_elasticity << "\t" << mat.poisson_ratio << "\t" << mat.density;
  return os;
}

ostream& operator<<(ostream& os, const Shape& shp) {
  os << "\tDims  : " << shp.dim[0] << "\t" << shp.dim[1] << "\t" << shp.dim[2] << "\n"
     << "\tCurved: " << shp.curve[0] << "\t" << shp.curve[1] << "\n"
     << "\t\tX-space: " << shp.spaces[0] << "\n"
     << "\t\tY-space: " << shp.spaces[1] << "\n"
     << "\t\tZ-space: " << shp.spaces[2] << "\n";
  return os;
}

ostream& operator<<(ostream& os, const FGM& fgm) {
  os  << "Shape -------------------------------------------\n"
     << fgm.shape
     << "Materials ---------------------------------------\n"
     << "\tMat 1: " << fgm.mats[0] << "\n"
     << "\tMat 2: " << fgm.mats[1] << "\n"
     << "Parameters --------------------------------------\n"
     << "\tCtrl : " << fgm.ctrl_y << "\t" << fgm.ctrl_z << "\n";
  cout << "-------------------------------------------------\n";
  return os;
}

//Fiber induced composite
class FIC {};
//Laminated composite
class LCO {};

int main(int argc, char** argv) {
  mkl_set_threading_layer(MKL_THREADING_GNU);
  mkl_set_dynamic(0);
  omp_set_dynamic(0);
  //omp_set_nested(1);//1);

  //omp_set_num_threads(2);
  //mkl_set_num_threads(8);
  //mkl_set_num_threads_local(8);

  //omp_set_max_active_levels(2);//2);

  ///////////////////////////////////
  //For creating and writing on a file that the found results which include ctrl_y ctrlz local_mineig compute_time
  //and at the end smallest_mineig and total_time.
  //You also need to comment out other lines under normal couts which are writing for result file, resfile.

  ofstream resfile;
  resfile.open ("result_openmp_only.txt");

  ///////////////////////////////////


  int nthreads = atoi(argv[1]);

  double min_ctrl_y = 0.1, max_ctrl_y = 0.4, min_ctrl_z = 0.175, max_ctrl_z = 0.4;
  double interval = 0.025;

  vector<pair<dtype, dtype> > problems;
  for(double cy = min_ctrl_y; round((max_ctrl_y - cy)/interval) >= 0; cy += interval) {
    for(double cz = min_ctrl_z; round((max_ctrl_z - cz)/interval) >= 0; cz += interval) {
      problems.push_back(make_pair(cy, cz));
    }
  }

  omp_set_num_threads(nthreads);
  mkl_set_num_threads(1);

  Material first(1, 0.3, 1);
  Material second(200.0/70.0, 0.3, 5700.0/2702.0);
  Shape shape(2, 1, 0.3, 9, 9, 9);

  double smallest_mineig =  std::numeric_limits<double>::max();
  double best_y = -1, best_z = -1;
  double ostart = omp_get_wtime();
  #pragma omp parallel for
  for(int i = 0; i < problems.size(); i++) {
      //mkl_set_num_threads_local(10);
      FGM fgm(shape, first, second, problems[i].first, problems[i].second);

      int nconv;
      double mineig;
      double start = omp_get_wtime();
      fgm.compute(3, 10, nconv, mineig, 0, 10000, 0.0001);
      double end = omp_get_wtime();

      #pragma omp critical
      {
	if(nconv > 0) {
	  cout << fgm.ctrl_y << " " << fgm.ctrl_z << " " << mineig << " " << end - start << endl;
      resfile << fgm.ctrl_y << "\t" << fgm.ctrl_z << "\t" << mineig << "\t" << end - start << endl;
      if(mineig < smallest_mineig) {
	    smallest_mineig = mineig;
	    best_y = fgm.ctrl_y;
	    best_z = fgm.ctrl_z;
	  }
	}
  else {
      cout << "No eigen: " << fgm.ctrl_y << " " << fgm.ctrl_z << " " << end - start << endl;
      resfile << "No eigen: " << fgm.ctrl_y << "\t" << fgm.ctrl_z << "\t" << end - start << endl;
	}
	}
    }

  cout << "Smallest eig: " << smallest_mineig << " - (" << best_y << ", " << best_z << ")" << endl;
  resfile << "Smallest eig: " << smallest_mineig << " - (" << best_y << ", " << best_z << ")" << endl;
  double oend = omp_get_wtime();
  cout << "Total time: " << oend - ostart << endl;
  resfile << "Total time: " << oend - ostart << endl;

  resfile.close();

}
