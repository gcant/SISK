import ctypes as ct
import numpy as np
import networkx as nx
import scipy.sparse
import scipy.sparse.linalg
import scipy.optimize
import scipy.linalg

# C arrays of ints/doubles using numpy
array_bool = np.ctypeslib.ndpointer(dtype=ct.c_bool, ndim=1, flags='CONTIGUOUS')
array_int = np.ctypeslib.ndpointer(dtype=ct.c_int,ndim=1, flags='CONTIGUOUS')
array_double = np.ctypeslib.ndpointer(dtype=np.double,ndim=1, flags='CONTIGUOUS')

lib = ct.cdll.LoadLibrary("./out/SISN_mp.so")

lib.reset_messages.argtypes = [ array_double, ct.c_int ]

lib.initialize.argtypes = [ ct.c_int, ct.c_int, array_int, ct.c_int ]
lib.initialize.restype = ct.c_int 

lib.full_algorithm.argtypes = [ ct.c_double, ct.c_double, ct.c_double,
        ct.c_int, ct.c_int, array_double, array_double]
lib.full_algorithm.restype = ct.c_int 

lib.k_regular_graph.argtypes = [ ct.c_double, ct.c_double, ct.c_double, ct.c_double,
        ct.c_int, ct.c_int, array_double]
lib.k_regular_graph.restype = ct.c_int

lib.random_graph.argtypes = [ ct.c_double, ct.c_double, ct.c_double, ct.c_double, ct.c_double,
        ct.c_int, ct.c_int, array_double]
lib.random_graph.restype = ct.c_int

lib.k_regular_phi.argtypes = [ ct.c_double, ct.c_double, ct.c_double, ct.c_double,
        ct.c_int, ct.c_int, array_double, array_double]
lib.k_regular_graph.restype = ct.c_int

lib.k_regular_graph_Euler.argtypes = [ ct.c_double, ct.c_double, ct.c_double,
        ct.c_double,  ct.c_double, ct.c_int, array_double ]

lib.star_graph.argtypes = [ ct.c_int, ct.c_double, ct.c_double, ct.c_double,
        ct.c_int, ct.c_int, array_double]
lib.star_graph.restype = ct.c_int


lib_sim = ct.cdll.LoadLibrary("./out/simulation.so")
lib_sim.SIS.argtypes = [ ct.c_int, ct.c_int, array_int, ct.c_double,
                    array_bool, ct.c_double, ct.c_double ]
lib_sim.SIS.restype = ct.c_double
lib_sim.SIS_log.argtypes = [ ct.c_int, ct.c_int, array_int, ct.c_double,
                    array_bool, ct.c_double, ct.c_double, array_double ]
lib_sim.SIS_log.restype = ct.c_double
lib_sim.SISv.argtypes = [ ct.c_int, ct.c_int, array_int, ct.c_double,
                    array_bool, ct.c_double, ct.c_double, array_bool ]
lib_sim.SISv.restype = ct.c_double

#lib_sim.SIS(10, 10, np.ones(20, dtype=ct.c_int), 0.5, np.ones(10, dtype=ct.c_bool), 0., 1.)


def rg(q, k, beta, gamma, N, tol=1e-12, max_its=500):
    output = np.zeros(N+1)
    s = lib.random_graph( ct.c_double(q), ct.c_double(k), ct.c_double(beta), ct.c_double(gamma),
            ct.c_double(tol), ct.c_int(max_its), ct.c_int(N), output )
    return output, s

def k_regular(k, beta, gamma, N, tol=1e-12, max_its=500):
    output = np.zeros(N+1)
    s = lib.k_regular_graph( ct.c_double(k), ct.c_double(beta), ct.c_double(gamma),
            ct.c_double(tol), ct.c_int(max_its), ct.c_int(N), output )
    return output, s

def k_regular_phi(k, beta, gamma, N, tol=1e-12, max_its=500):
    output = np.zeros(N)
    Q = np.zeros((N+1)**4)
    s = lib.k_regular_phi( ct.c_double(k), ct.c_double(beta), ct.c_double(gamma),
            ct.c_double(tol), ct.c_int(max_its), ct.c_int(N), output, Q )
    return output, s, Q.reshape((N+1)**2,(N+1)**2)

def star(k, beta, gamma, N, tol=1e-12, max_its=500):
    output = np.zeros(N+1)
    s = lib.star_graph( ct.c_int(k), ct.c_double(beta), ct.c_double(gamma),
            ct.c_double(tol), ct.c_int(max_its), ct.c_int(N), output )
    return output, s

def initMessages(G, N):
    n = ct.c_int(G.number_of_nodes())
    m = ct.c_int(G.number_of_edges())
    edges = np.array(list(G.edges())).flatten().astype(ct.c_int)
    return lib.initialize(n, m, edges, ct.c_int(N))

def runMP(G,beta,gamma,N,tol=1e-12,max_its=500,reset=True,init=True):
    if init:
        initMessages(G,N)
    if reset:
        lib.reset_messages(np.arange(1.,N+1)*0.1, N)
    output = np.zeros(G.number_of_nodes()*(N+1))
    m_output = np.zeros(G.number_of_nodes()*(N))
    s = lib.full_algorithm( ct.c_double(beta), ct.c_double(gamma),
            ct.c_double(tol), ct.c_int(max_its),
            ct.c_int(N), output, m_output)
    marginal = np.array(output).reshape( G.number_of_nodes(), N+1 )
    messages = np.array(m_output).reshape( G.number_of_nodes(), N )
    return marginal,s, messages


def initial_edgeP(pI, N):
    p = np.arange(N+1) * 0.01
    z = np.sum(p[1:])
    p[0] = pI*z / (1.-pI)
    p /= (p[0]+z)
    return p

def k_regular_Euler(k, beta, gamma, N, T, h):
    p0 = initial_edgeP(0.99999,N)
    output = np.zeros((N+1)*(N+1) + (N+1))
    for n in range(N+1):
        for m in range(N+1):
            output[ n*(N+1) + m ] = p0[n]*p0[m]
    lib.k_regular_graph_Euler( ct.c_double(k), ct.c_double(beta), ct.c_double(gamma),
            ct.c_double(h), ct.c_double(T), ct.c_int(N), output )
    P = output[:(N+1)*(N+1)]
    I = np.sum([P[m] for m in range(N+1)])
    marg = output[(N+1)*(N+1):]
    return P, marg, I

def k_regular_Euler_logged(k, beta, gamma, N, T, h, num_pts):
    #p0 = initial_edgeP(0.99999,N)
    p0 = initial_edgeP(0.001,N)
    output = np.zeros((N+1)*(N+1) + (N+1))
    for n in range(N+1):
        for m in range(N+1):
            output[ n*(N+1) + m ] = p0[n]*p0[m]
    ans = [np.sum([output[m] for m in range(N+1)])]
    for x in range(num_pts):
        lib.k_regular_graph_Euler( ct.c_double(k), ct.c_double(beta), ct.c_double(gamma),
            ct.c_double(h), ct.c_double(T), ct.c_int(N), output )
        ans.append(np.sum([output[m] for m in range(N+1)]))
    return output, np.array(ans)

def phi_to_absorb(phi, gamma, T):
    K = len(phi)
    Q = np.zeros((K+1, K+1))
    Q[0, K] = 1.0 
    for x in range(1, K):
        Q[x+1, x] = gamma
    for x in range(0, K):
        Q[x+1, 0] = phi[x]
    Q = Q - np.diag(np.sum(Q, axis=1))
    Q[0] = 0.0
    return np.array([scipy.linalg.expm(t*Q)[-1][0] for t in T])

