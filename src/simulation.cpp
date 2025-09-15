// main MP algorithm, with C compatibility
#include <iostream>
#include <random>


extern "C" {

  double SIS(int num_nodes, int num_edges, int *edges, double beta,
	   bool *x, double t0, double tf) {
    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<double> unif(0.,1.);
    std::uniform_int_distribution<int> random_node(0,num_nodes-1);
    std::uniform_int_distribution<int> random_edge(0,num_edges-1);
    std::exponential_distribution<double> dt(num_nodes + beta*num_edges);
    double q = (num_nodes / (num_nodes + beta*num_edges));
    while (t0 < tf) {
      t0 += dt(rng);
      if (unif(rng) < q) x[random_node(rng)] = 0;
      else {
        int u = random_edge(rng), i = edges[2*u], j = edges[2*u+1];
	x[i] = x[j] = (x[i] | x[j]);
      }
    }
    return t0;
  }

  double SIS_log(int num_nodes, int num_edges, int *edges, double beta,
	   bool *x, double t0, double tf, double *times) {
    double t = t0;
    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<double> unif(0.,1.);
    std::uniform_int_distribution<int> random_node(0,num_nodes-1);
    std::uniform_int_distribution<int> random_edge(0,num_edges-1);
    std::exponential_distribution<double> dt(num_nodes + beta*num_edges);
    double q = (num_nodes / (num_nodes + beta*num_edges));
    while (true) {
      t += dt(rng);
      if (t > tf) return tf;
      if (unif(rng) < q) {
	int i = random_node(rng);
	if (x[i]) {
	  x[i] = 0;
	  times[i] = t;
	}
      }
      else {
        int u = random_edge(rng), i = edges[2*u], j = edges[2*u+1];
	if (x[i] and !x[j]) {
	  x[j] = 1;
	  times[j] = t;
	}
	else if (!x[i] and x[j]) {
	  x[i] = 1;
	  times[i] = t;
	}
      }
    }
  }

  double SISv(int num_nodes, int num_edges, int *edges, double beta,
	   bool *x, double t0, double tf, bool *v) {
    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<double> unif(0.,1.);
    std::uniform_int_distribution<int> random_node(0,num_nodes-1);
    std::uniform_int_distribution<int> random_edge(0,num_edges-1);
    std::exponential_distribution<double> dt(num_nodes + beta*num_edges);
    double q = (num_nodes / (num_nodes + beta*num_edges));
    for (int i=0; i<num_nodes; ++i) v[i] = x[i];
    while (t0 < tf) {
      t0 += dt(rng);
      if (unif(rng) < q) x[random_node(rng)] = 0;
      else {
        int u = random_edge(rng), i = edges[2*u], j = edges[2*u+1];
	x[i] = x[j] = (x[i] | x[j]);
	if (x[i]) v[i] = true;
	if (x[j]) v[j] = true;
      }
    }
    return t0;
  }

}

