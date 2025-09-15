#pragma once
#include <iostream>
#include <unordered_map>
#include <vector>
#include <tuple>
#include <cmath>
#include <Eigen/Dense>


#define X(Xi,Xj) ((Xj) + (Xi)*(N+1))
Eigen::MatrixXd QM(
		double beta,
		double gamma,
		const Eigen::VectorXd &phi,
		const Eigen::VectorXd &theta,
		bool pad )
{
  int N = phi.size();
  int dim = (N+1)*(N+1);
  Eigen::MatrixXd Q;
  if (pad) Q = Eigen::MatrixXd::Constant(dim, dim+1, 0.);
  else Q = Eigen::MatrixXd::Constant(dim, dim, 0.);
  const int I = 0;
  Q( X(I,I), X(I,N) ) = 1.;
  Q( X(I,I), X(N,I) ) = 1.;
  for (int n=1; n<N+1; ++n) {
    if (n>1) {
      Q( X(n,I), X(n-1,I) ) = gamma;
      Q( X(I,n), X(I,n-1) ) = gamma;
    }
    Q( X(n,I), X(I,I) ) = beta + phi[n-1];
    Q( X(I,n), X(I,I) ) = beta + theta[n-1];
    Q( X(n,I), X(n,N) ) = 1.;
    Q( X(I,n), X(N,n) ) = 1.;
  }
  for (int n=1; n<N+1; ++n) {
    for (int m=1; m<N+1; ++m) {
      if (n>1) Q( X(n,m), X(n-1,m) ) = gamma;
      if (m>1) Q( X(n,m), X(n,m-1) ) = gamma;
      Q( X(n,m), X(I,m) ) = phi[n-1];
      Q( X(n,m), X(n,I) ) = theta[m-1];
    }
  }
  auto q = Q.rowwise().sum();
  for (int i=0; i<dim; ++i) Q(i,i) = -q[i];
  if (pad) for (int i=0; i<dim; ++i) Q(i,dim) = 1.;
  return Q;
}


Eigen::VectorXd Psi_inv(
		double beta,
		double gamma,
		const Eigen::VectorXd &phi,
		const Eigen::VectorXd &theta)
{
  auto Q = QM(beta, gamma, phi, theta, true);
  Eigen::VectorXd d(Q.cols());
  d.setZero();
  d(d.size()-1) = 1.;
  Eigen::VectorXd P = Q.transpose().colPivHouseholderQr().solve(d);
  for (int i=0; i<P.size(); ++i) if (P[i]<1e-16) P[i]=1e-16;
  return P;
}


Eigen::VectorXd Ph(
		double beta,
		const Eigen::VectorXd &P,
		int N)
{
  Eigen::VectorXd ans(N);
  for (int n=1; n<N+1; ++n) {
    double Z=0;
    for (int m=0; m<N+1; ++m) Z += P( X(n,m) );
    ans(n-1) = beta * P( X(n, 0) ) / Z;
  }
  return ans;
}



Eigen::VectorXd Th(
		double beta,
		const Eigen::VectorXd &P,
		int N)
{
  Eigen::VectorXd ans(N);
  for (int n=1; n<N+1; ++n) {
    double Z=0;
    for (int m=0; m<N+1; ++m) Z += P( X(m,n) );
    ans(n-1) = beta * P( X(0, n) ) / Z;
  }
  return ans;
}

Eigen::VectorXd Psi_Euler(
		double beta,
		double gamma,
		const Eigen::VectorXd &P,
		const Eigen::VectorXd &phi,
		const Eigen::VectorXd &theta)
{
  auto Q = QM(beta, gamma, phi, theta, false);
  return (Q.transpose()*P);
}

Eigen::VectorXd Marginal(
		double beta,
		double gamma,
		const Eigen::VectorXd &phi)
{
  int N = phi.size();
  const int I = 0;
  Eigen::MatrixXd Q;
  Q = Eigen::MatrixXd::Constant(N+1, N+2, 0.);
  Q( I, N ) = 1.;
  for (int n=1; n<N+1; ++n) {
    if (n>1) Q(n,n-1) = gamma;
    Q(n,I) = phi[n-1];
  }
  auto q = Q.rowwise().sum();
  for (int i=0; i<N+1; ++i) Q(i,i) = -q[i];
  for (int i=0; i<N+1; ++i) Q(i,N+1) = 1.;
  Eigen::VectorXd d(Q.cols());
  d.setZero();
  d(d.size()-1) = 1.;
  Eigen::VectorXd P = Q.transpose().colPivHouseholderQr().solve(d);
  return P;
}

Eigen::VectorXd Marginal2(
		const Eigen::VectorXd &P,
		int N)
{
  Eigen::VectorXd ans(N+1);
  for (int n=0; n<N+1; ++n) {
    for (int m=0; m<N+1; ++m) ans(n) += P( X(m,n) );
  }
  return ans;
}

#undef X

