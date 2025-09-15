// main MP algorithm, with C compatibility
#include <iostream>
#include <unordered_map>
#include <vector>
#include <tuple>
#include <cmath>
#include "Graph.h"
#include "SISN_functions.hpp"

Graph G;
std::unordered_map<int, std::unordered_map<int, Eigen::VectorXd>> message;
std::unordered_map<int, std::unordered_map<int, Eigen::VectorXd>> edgeP;
std::vector<Eigen::VectorXd> marg;
std::vector<std::pair<int,int>> edgelist;


extern "C" {


  void reset_messages(double *data, int size) {
    Eigen::VectorXd x(size);
    for (int i=0; i<size; i++) x[i] = data[i];
    marg.resize(G.number_of_nodes());
    for (int i : G.nodes()) marg[i] = Eigen::VectorXd::Constant(size,0.);
    for (int i : G.nodes()) for (int j : G.neighbors(i)) {
      message[i][j] = x;
      marg[i] += x;
    }
  };

  int initialize(int num_nodes, int num_edges, int *edges, int N) {

    std::unordered_map<int, std::unordered_map<int, Eigen::VectorXd>> new_message;
    Graph new_G;

    for (int i=0; i<num_nodes; ++i) new_G.add_node(i);

    for (int v=0; v<num_edges; ++v) {
      int i = edges[2*v+1];
      int j = edges[2*v];
      new_G.add_edge(i,j);
      new_message[i][j] = Eigen::VectorXd::LinSpaced(N,0.1,0.2);
      new_message[j][i] = Eigen::VectorXd::LinSpaced(N,0.1,0.2);
    }
    message = new_message;
    G = new_G;
    edgelist = G.edges();

    marg.resize(G.number_of_nodes());
    for (int i : G.nodes()) {
      marg[i] = Eigen::VectorXd::Constant(N,0.0);
      for (int j : G.neighbors(i)) marg[i] += message[i][j];
    }

    return 0;
  }

  void init_edgeP(double *data, int dim) {
    Eigen::VectorXd P(dim);
    for (int i=0; i<dim; ++i) P[i] = data[i];
    for (int i : G.nodes()) for (int j : G.neighbors(i)) {
      edgeP[i][j] = P;
    }
  }


  int k_regular_graph(double k, double beta, double gamma, double tol,
		  int max_its, int N, double *output) {
    Eigen::VectorXd phi = Eigen::VectorXd::LinSpaced(N,0.9,1.0);
    int s;
    double delta;
    for (s=0; s<max_its; ++s) {
      Eigen::VectorXd P = Psi_inv(beta, gamma, (k-1)*phi, (k-1)*phi);
      auto phi_new = Ph(beta, P, N);
      if ((phi_new-phi).norm()<tol) {
        phi = phi_new;
	break;
      }
      phi = phi_new;
    }
    //auto M = Marginal(beta, gamma, k*phi);
    Eigen::VectorXd P = Psi_inv(beta, gamma, (k-1)*phi, (k-1)*phi);
    auto M = Marginal2(P, N);
    for (int i=0; i<M.size(); ++i) output[i] = M[i];
    return s;
  }

  int random_graph(double q, double k, double beta, double gamma, double tol,
		  int max_its, int N, double *output) {
    Eigen::VectorXd phi = Eigen::VectorXd::LinSpaced(N,0.1,0.2);
    int s;
    double delta;
    for (s=0; s<max_its; ++s) {
      Eigen::VectorXd P = Psi_inv(beta, gamma, q*phi, q*phi);
      auto phi_new = Ph(beta, P, N);
      if ((phi_new-phi).norm()<tol) {
        phi = phi_new;
	break;
      }
      phi = phi_new;
    }
    auto M = Marginal(beta, gamma, k*phi);
    for (int i=0; i<M.size(); ++i) output[i] = M[i];
    return s;
  }

  int k_regular_phi(double k, double beta, double gamma, double tol,
		  int max_its, int N, double *output, double *Q) {
    Eigen::VectorXd phi = Eigen::VectorXd::LinSpaced(N,0.1,0.2);
    int s;
    for (s=0; s<max_its; ++s) {
      Eigen::VectorXd P = Psi_inv(beta, gamma, (k-1)*phi, (k-1)*phi);
      auto phi_new = Ph(beta, P, N);
      if ((phi_new-phi).norm()<tol) {
        phi = phi_new;
	break;
      }
      phi = phi_new;
    }
    for (int i=0; i<N; ++i) output[i] = phi[i];
    auto M = QM(beta, gamma, (k-1)*phi, (k-1)*phi, false);
    int x = 0;
    for (int i=0; i<(N+1)*(N+1); ++i) for (int j=0; j<(N+1)*(N+1); ++j) {
      Q[x++] = M(i,j);
    }
    return s;
  }

  void k_regular_graph_Euler(double k, double beta, double gamma,
		  double h, double T, int N, double *output) {

    int dim = (N+1)*(N+1);
    Eigen::VectorXd P(dim);
    for (int i=0; i<dim; ++i) P[i] = output[i];
    Eigen::VectorXd phi = Ph(beta, P, N);
    double t=0.;
    while (t<T) {
      P += h*Psi_Euler(beta, gamma, P, (k-1)*phi, (k-1)*phi);
      phi = Ph(beta, P, N);
      t += h;
    }
    for (int i=0; i<dim; ++i) output[i] = P[i];
    auto M = Marginal(beta, gamma, k*phi);
    for (int i=0; i<N+1; ++i) output[dim+i] = M[i];
  }


  int star_graph(int k, double beta, double gamma, double tol,
		  int max_its, int N, double *output) {
    Eigen::VectorXd phi = Eigen::VectorXd::LinSpaced(N,0.01,0.02);
    int s;
    double delta;
    for (s=0; s<max_its; ++s) {
      Eigen::VectorXd P = Psi_inv(beta, gamma, (k-1)*phi, 0*phi);
      auto phi_new = Ph(beta, P, N);
      if ((phi_new-phi).norm()<tol) {
        phi = phi_new;
	break;
      }
      phi = phi_new;
    }
    Eigen::VectorXd P = Psi_inv(beta, gamma, (k-1)*phi, 0*phi);
    auto x = Ph(beta, P, N);
    auto y = Th(beta, P, N);
    auto M1 = Marginal(beta, gamma, k*x);
    auto M2 = Marginal(beta, gamma, y);
    auto M3 = (M1 + k*M2)/(k+1);
    for (int i=0; i<M3.size(); ++i) output[i] = M3[i];
    return s;
  }



  int full_algorithm(double beta, double gamma, double tol,
		  int max_its, int N, double *output, double *m_output) {
    int n = G.number_of_nodes();
    int m = G.number_of_edges();
    double delta = 2*m*tol;
    int num_its = 0;

    while ((delta > m*tol) && (num_its++<max_its)) {
      delta = 0.;

      #pragma omp parallel for reduction(+: delta)
      for (int u=0; u<m; ++u) {
	  auto[i, j] = edgelist[u];
	  Eigen::VectorXd m_j_i = message.at(j).at(i);
	  Eigen::VectorXd m_i_j = message.at(i).at(j);
	  Eigen::VectorXd phi = marg.at(j) - m_j_i;
          Eigen::VectorXd mu  = marg.at(i) - m_i_j;
	  Eigen::VectorXd P = Psi_inv(beta, gamma, phi, mu);
	  Eigen::VectorXd p = Th(beta, P, N);
	  Eigen::VectorXd q = Ph(beta, P, N);
	  marg.at(i) += p - m_i_j;
          marg.at(j) += q - m_j_i;
          message.at(i).at(j) = p;
          message.at(j).at(i) = q;
	  delta += (p-m_i_j).norm() + (q-m_j_i).norm();
      }

      #pragma omp parallel for
      for (int i=0; i<n; ++i) {
	marg.at(i) = Eigen::VectorXd::Constant(N,0);
	for (int j : G.neighbors(i)) marg.at(i) += message.at(i).at(j);
      }
    }

//    #pragma omp parallel for
//    for (int u=0; u<m; ++u) {
//          auto[i, j] = edgelist[u];
//          Eigen::VectorXd m_j_i = message.at(j).at(i);
//          Eigen::VectorXd m_i_j = message.at(i).at(j);
//          Eigen::VectorXd phi = marg.at(j) - m_j_i;
//          Eigen::VectorXd mu  = marg.at(i) - m_i_j;
//
//          Eigen::VectorXd Pi = Psi_inv(beta, gamma, phi, mu);
//          Eigen::VectorXd Mi = Marginal2(Pi, N);
//
//          Eigen::VectorXd Pj = Psi_inv(beta, gamma, mu, phi);
//          Eigen::VectorXd Mj = Marginal2(Pj, N);
//
//          int dim = Mi.size();
//          for (int l=0; l<dim; ++l) {
//            output[i*dim + l] += Mi[l] / G.degree(i);
//            output[j*dim + l] += Mj[l] / G.degree(j);
//          }
//    }


    #pragma omp parallel for
    for (int i=0; i<n; ++i) {
      Eigen::VectorXd M = Marginal(beta, gamma, marg.at(i));
      int dim = M.size();
      for (int j=0; j<dim; ++j) {
        output[i*dim + j] = M[j];
      }
    }

    #pragma omp parallel for
    for (int i=0; i<n; ++i) {
      auto M = marg.at(i);
      int dim = M.size();
      for (int j=0; j<dim; ++j) {
        m_output[i*dim + j] = M[j];
      }
    }

    return num_its;
  }

}

