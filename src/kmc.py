import jax.numpy as jnp
import jax.numpy.linalg as jla
from jax import jvp, jacfwd, jacrev, random, grad

from jax.config import config
config.update("jax_enable_x64", True)


def sym2(mat):
    return mat + mat.T


def Teval(T, vlist):
    """Evaluating a tensor by contracting with a list of vectors, contracting from the left
    """
    m = len(vlist)
    for i in range(m):
        if i == 0:
            ret = jnp.tensordot(T, vlist[i], axes=((-1), 0))
        else:
            ret = jnp.tensordot(ret, vlist[i], axes=((-1), 0))
    return ret
 

class KMC(object):
    """
    """
    
    def __init__(self, n, c, params):
        """Initialize Kim-McCann function
        c is a function with signature
        c(x, y, params)
        x, y could be recovered by
        KMC.split(q)        

        """
        self.c = c
        self.n = n
        self.params = params
        self.bddc = jacfwd(jacrev(c, argnums=0), argnums=1)
        self.zr = jnp.zeros(n)
        
    def split(self, q):
        n = self.n
        return q[:n], q[n:]

    def q(self, x, y):
        return jnp.block([x, y])

    def setParams(self, params):
        self.params = params

    def kmc(self, q, Omg1, Omg2):
        """The metric. return a number
        """
        n = self.n        
        x, y = self.split(q)
        C = self.bddc(x, y, self.params)
        return -0.5*jnp.sum(Omg1[:n]*(C@Omg2[n:])) \
            - 0.5*jnp.sum(Omg2[:n]*(C@Omg1[n:]))

    def gkmc(self, q, Omg):
        n = self.n
        x, y = self.split(q)        
        C = self.bddc(x, y, self.params)
        return jnp.concatenate([-0.5*C@Omg[n:], - 0.5*C.T@Omg[:n]])

    def igkmc(self, q, Omg):
        n = self.n        
        x, y = self.split(q)                
        iC = jla.inv(self.bddc(x, y, self.params))
        return jnp.concatenate([-2*iC.T@Omg[n:], - 2*iC@Omg[:n]])

    def igkmc_mat(self, q):
        n = self.n        
        x, y = self.split(q)                
        
        znn = jnp.zeros((n, n))
        iC = jla.inv(self.bddc(x, y, self.params))
        return jnp.block([[znn, -2*iC.T],
                          [- 2*iC, znn]])

    def Gamma(self, q, Omg1, Omg2):
        n = self.n
        x, y = self.split(q)                
        
        C, Cx = jvp(lambda x: self.bddc(x, y, self.params), (x,), (Omg1[:n],))
        Cy = jvp(lambda y: self.bddc(x, y, self.params), (y,), (Omg1[n:],))[1]
        return jnp.concatenate([jla.solve(C.T, Cx.T@Omg2[:n]),
                                jla.solve(C, Cy@Omg2[n:])])

    def conjGamma(self, q, Xi, Omg):
        """ Conjugate Gamma with Xi fixed, in Euclidean metric
        """
        n = self.n
        x, y = self.split(q)        
        C = self.bddc(x, y, self.params)
        bddc_xxi = jvp(lambda x: self.bddc(x, y, self.params), (x,), (Xi[:n],))[1]
        bddcT_ybxi = jvp(lambda y: self.bddc(x, y, self.params).T, (y,), (Xi[n:],))[1]
        return jnp.concatenate(
            [- bddc_xxi@jla.solve(C, Omg[:n]),
             - bddcT_ybxi@jla.solve(C.T, Omg[n:])])    

    def Curvature(self, q, Omg1, Omg2, Omg3):
        n = self.n
        x, y = self.split(q)                
        
        DG1 = jvp(lambda q: self.Gamma(q, Omg2, Omg3), (q, ), (Omg1, ))[1]
    
        DG2 = jvp(lambda q: self.Gamma(q, Omg1, Omg3), (q, ), (Omg2, ))[1]
        GG1 = self.Gamma(q, Omg1, self.Gamma(q, Omg2, Omg3))
        GG2 = self.Gamma(q, Omg2, self.Gamma(q, Omg1, Omg3))
        
        return DG1 - DG2 + GG1 - GG2

    def Curv4(self, q, Omg1, Omg2, Omg3, Omg4):
        return self.kmc(q, self.Curvature(q, Omg1, Omg2, Omg3), Omg4)
    

class KMCSphere(object):
    """
    """
    
    def __init__(self, kmcA):
        """Initialize with an ambient kmcA
        """
        self.kmcA = kmcA
        self.n = kmcA.n
        self.c = kmcA.c
        self.bddc = kmcA.bddc
        self.params = kmcA.params

    def randQpoint(self, sk):
        n = self.n
        q = random.normal(sk, (2*n,)) 
        x = q[:n]/jla.norm(q[:n])
        y = q[n:]/jla.norm(q[n:])
        return jnp.concatenate([x, y])

    def randvec(self, sk, q):
        n = self.n        
        Omg = random.normal(sk, (2*n,))
        omgx = Omg[:n] - q[:n]*jnp.sum(q[:n]*Omg[:n])
        omgy = Omg[n:] - q[n:]*jnp.sum(q[n:]*Omg[n:])
        return jnp.concatenate([omgx, omgy])

    def split(self, q):
        return self.kmcA.split(q)
    
    def q(self, x, y):
        return jnp.block([x, y])

    def setParams(self, params):
        self.params = params
        self.kmcA.params = params        

    def kmc(self, q, Omg1, Omg2):
        """The metric. return a number
        """
        return self.kmcA.kmc(q, Omg1, Omg2)

    def gkmc(self, q, Omg):
        return self.kmcA.gkmc(q, Omg)

    def igkmc(self, q, Omg):
        return self.kmcA.igkmc(q, Omg)

    def igkmc_mat(self, q):
        return self.kmcA.igkmc_mat(q)

    def proj(self, q, Omg):
        n = self.n
        x, y = self.split(q)
        iC = jla.inv(self.bddc(x, y, self.params))
        yiCx = jnp.sum(y*(iC@x))
        return jnp.concatenate([Omg[:n] - iC.T@y*jnp.sum(x*Omg[:n])/yiCx,
                                Omg[n:] - iC@x*jnp.sum(y*Omg[n:])/yiCx])

    def DprojTan(self, q, Xi, Eta):
        n = self.n
        x, y = self.split(q)
        
        iC = jla.inv(self.bddc(x, y, self.params))
        yiCx = jnp.sum(y*(iC@x))
        return jnp.concatenate(
            [- iC.T@y*jnp.sum(Xi[:n]*Eta[:n])/yiCx,
             - iC@x*jnp.sum(Xi[n:]*Eta[n:])/yiCx])

    def Gamma(self, q, Xi1, Xi2):
        n = self.n
        x, y = self.split(q)
        
        C, Cx = jvp(lambda x: self.bddc(x, y, self.params), (x,), (Xi1[:n],))
        Cy = jvp(lambda y: self.bddc(x, y, self.params), (y,), (Xi1[n:],))[1]
        iC = jla.inv(C)
        yiCx = jnp.sum(y*(iC@x))
        return jnp.concatenate(
            [iC.T@Cx.T@Xi2[:n] - iC.T@y*(
                jnp.sum(x*(iC.T@Cx.T@Xi2[:n])) - jnp.sum(Xi1[:n]*Xi2[:n])) / yiCx,
             iC@Cy@Xi2[n:] - iC@x*(jnp.sum(y*(iC@Cy@Xi2[n:])) - jnp.sum(Xi1[n:]*Xi2[n:])) / yiCx
             ])

    def Two(self, q, Xi1, Xi2):
        """second fundamental form
        """
        n = self.n
        x, y = self.split(q)
        
        C, Cx = jvp(lambda x: self.bddc(x, y, self.params), (x,), (Xi1[:n],))
        Cy = jvp(lambda y: self.bddc(x, y, self.params), (q[n:],), (Xi1[n:],))[1]
        iC = jla.inv(C)
        yiCx = jnp.sum(y*(iC@x))
        
        # return Dpj + pLC
        return jnp.concatenate(
            [iC.T@y*(
                jnp.sum(x*(iC.T@Cx.T@Xi2[:n])) - jnp.sum(Xi1[:n]*Xi2[:n])) / yiCx,
             iC@x*(jnp.sum(y*(iC@Cy@Xi2[n:])) - jnp.sum(Xi1[n:]*Xi2[n:])) / yiCx
             ])

    def Curvature(self, q, Xi1, Xi2, Xi3):
        DG1 = jvp(lambda q: self.Gamma(q, Xi2, Xi3), (q, ), (Xi1, ))[1]    
        DG2 = jvp(lambda q: self.Gamma(q, Xi1, Xi3), (q, ), (Xi2, ))[1]
        GG1 = self.Gamma(q, Xi1, self.Gamma(q, Xi2, Xi3))
        GG2 = self.Gamma(q, Xi2, self.Gamma(q, Xi1, Xi3))
        
        return DG1 - DG2 + GG1 - GG2

    def Curv4(self, q, Xi1, Xi2, Xi3, Xi4):
        return self.kmc(q, self.Curvature(q, Xi1, Xi2, Xi3), Xi4)

    
class KMCAntenna(object):
    def __init__(self, Lambda):
        """Initialize with an ambient kmcA
        """
        self.n = Lambda.shape[0]        
        self.Lambda = Lambda
        self.iLambda = jla.inv(Lambda)

    def c(self, q):
        x, y = self.split(q)
        xmy = (1-jnp.sum(x*(self.Lambda@y)))        
        return -0.5*jnp.log(xmy)

    def bddc(self, q):
        x, y = self.split(q)        
        xmy = (1-jnp.sum(x*(self.Lambda@y)))
        return 0.5/(xmy*xmy)*self.Lambda@y[:, None]@x[None, :]@self.Lambda \
            + 0.5/xmy*self.Lambda

    def ibDDc(self, x, y):
        xmy = (1-jnp.sum(x*(self.Lambda@y)))        
        return 2*xmy*(self.iLambda - y[:, None]@x[None, :])

    def kmc(self, q, Omg1, Omg2):
        x, y = self.split(q)
        n = self.n
        Lbd = self.Lambda
        xmy = (1-jnp.sum(x*(Lbd@y)))        
        return - 0.25/xmy*(
            jnp.sum(Omg1[:n]*(Lbd@Omg2[n:])) + jnp.sum(Omg1[n:]*(Lbd@Omg2[:n]))
            + 1/xmy*jnp.sum(Omg1[:n]*(Lbd@y))*jnp.sum((Lbd@x)*Omg2[n:])
            + 1/xmy*jnp.sum(Omg1[n:]*(Lbd@x))*jnp.sum((Lbd@y)*Omg2[:n]))

    def projM(self, x, omg):
        return omg - x*jnp.sum(x*omg)
        
    def proj(self, q, Omg):
        x, y = self.split(q)
        n = self.n
        iLbd = self.iLambda
        
        xILbdy = (1-jnp.sum(q[:n]*(iLbd@q[n:])))
        
        return jnp.concatenate([
            Omg[:n] + 1/xILbdy*(iLbd@q[n:] - q[:n])*jnp.sum(q[:n]*Omg[:n]),
            Omg[n:] + 1/xILbdy*(iLbd@q[:n] - q[n:])*jnp.sum(q[n:]*Omg[n:])])

    def DprojTan(self, q, Xi, Eta):
        x, y = self.split(q)
        n = self.n
        
        iLbd = self.iLambda        
        xILbdy = (1-jnp.sum(q[:n]*(iLbd@q[n:])))
        
        return jnp.concatenate([
            1/xILbdy*(iLbd@q[n:] - q[:n])*jnp.sum(Xi[:n]*Eta[:n]),
            1/xILbdy*(iLbd@q[:n] - q[n:])*jnp.sum(Xi[n:]*Eta[n:])])
    
    def Gamma(self, q, Xi, Eta):
        x, y = self.split(q)
        n = self.n
        iLbd = self.iLambda
        Lbd = self.Lambda
        
        xmy = (1-jnp.sum(x*(Lbd@y)))
        xILbdy = (1-jnp.sum(q[:n]*(iLbd@q[n:])))

        return jnp.block(
            [1/(xmy)*jnp.sum((Lbd@y)*Eta[:n])*Xi[:n]
             + 1/(xmy)*jnp.sum((Lbd@y)*Xi[:n])*Eta[:n]
             - (iLbd@q[n:] - q[:n])*jnp.sum(Xi[:n]*Eta[:n])/xILbdy,
             1/(xmy)*jnp.sum((Lbd@x)*Eta[n:])*Xi[n:]
             + 1/(xmy)*jnp.sum((Lbd@x)*Xi[n:])*Eta[n:]
             - (iLbd@q[:n] - q[n:])*jnp.sum(Xi[n:]*Eta[n:])/xILbdy])

    def Two(self, q, Xi, Eta):
        x, y = self.split(q)
        n = self.n
        iLbd = self.iLambda
        Lbd = self.Lambda
        
        xmy = (1-jnp.sum(x*(Lbd@y)))

        # iC = 2*xmy*(iLbd - y[:, None]@x[None, :])
        xILbdy = (1-jnp.sum(q[:n]*(iLbd@q[n:])))
        S1 = 1/xILbdy*(iLbd@q[n:] - q[:n])*jnp.sum(Xi[:n]*Eta[:n])
        S2 = 1/xILbdy*(iLbd@q[:n] - q[n:])*jnp.sum(Xi[n:]*Eta[n:])
        
        return jnp.block([S1, S2])
    
    def CrossSec(self, q, xi, bxi):
        x, y = self.split(q)
        n = self.n
        iLbd = self.iLambda
        Lbd = self.Lambda
        
        x, y = q[:n], q[n:]        
        xmy = (1-jnp.sum(x*(Lbd@y)))
        xILbdy = (1-jnp.sum(x*(iLbd@y)))
        zr = jnp.zeros(n)

        return - 8*jnp.square(self.kmc(q, jnp.block([xi, zr]), jnp.block([zr, bxi]))) \
            + 1/(4*xmy*xILbdy)*jnp.sum(xi*xi)*jnp.sum(bxi*bxi)
        
    def randQpoint(self, sk):
        n = self.n
        q = random.normal(sk, (2*n,)) 
        x = q[:n]/jla.norm(q[:n])
        y = q[n:]/jla.norm(q[n:])
        return jnp.concatenate([x, y])

    def randvec(self, sk, q):
        n = self.n        
        Omg = random.normal(sk, (2*n,))
        omgx = Omg[:n] - q[:n]*jnp.sum(q[:n]*Omg[:n])
        omgy = Omg[n:] - q[n:]*jnp.sum(q[n:]*Omg[n:])
        return jnp.concatenate([omgx, omgy])

    def split(self, q):
        n = self.n
        return q[:n], q[n:]

    def q(self, x, y):
        return jnp.block([x, y])

    def setLambda(self, Lambda):
        self.Lambda = Lambda

    def Curvature(self, q, Xi1, Xi2, Xi3):
        n = self.n
        x, y = self.split(q)                
        
        DG1 = jvp(lambda q: self.Gamma(q, Xi2, Xi3), (q, ), (Xi1, ))[1]
    
        DG2 = jvp(lambda q: self.Gamma(q, Xi1, Xi3), (q, ), (Xi2, ))[1]
        GG1 = self.Gamma(q, Xi1, self.Gamma(q, Xi2, Xi3))
        GG2 = self.Gamma(q, Xi2, self.Gamma(q, Xi1, Xi3))
        
        return DG1 - DG2 + GG1 - GG2

    def Curv4(self, q, Xi1, Xi2, Xi3, Xi4):
        return self.kmc(q, self.Curvature(q, Xi1, Xi2, Xi3), Xi4)
    

class KMCDivergence(object):
    def __init__(self, kmcA, yf):
        self.kmcA = kmcA
        self.yf = yf
        self.n = kmcA.n
        self.jF = jacfwd(self.yf)

    def kmc(self, q, Omg1, Omg2):
        return self.kmcA.kmc(q, Omg1, Omg2)

    def randQpoint(self, sk):
        x = random.normal(sk, (self.n,))
        return jnp.concatenate([x, self.yf(x)])

    def randvec(self, sk, q):
        x = q[:self.n]
        xi = random.normal(sk, (self.n,))
        return jnp.concatenate([xi, self.jF(x)@xi])        

    def proj(self, q, Omg, al):
        n = self.n
        x, y = self.kmcA.split(q)
        jFx = self.jF(x)
        return  0.5*jnp.concatenate(
            [(1+al)*Omg[:n] + (1-al)*jla.solve(jFx, Omg[n:]),
             (1+al)*jFx@Omg[:n] + (1-al)*Omg[n:]])

    def projT(self, q, Omg, al):
        n = self.n
        x, y = self.kmcA.split(q)
        
        jFx = self.jF(x)
        return  0.5*jnp.concatenate(
            [(1+al)*Omg[:n] + (1+al)*jFx.T@Omg[n:],
             + (1-al)*jla.solve(jFx.T, Omg[:n]) + (1-al)*Omg[n:]])

    def DprojT(self, q, Xi, OForm, al):
        """This only works for form,
        ie OForm[n:] = jFx.T@OForm[:n]
        """
        n = self.n
        x, y = self.kmcA.split(q)
        
        jFx = self.jF(x)
        ijFx = jla.inv(jFx)
        hFTxi = jvp(lambda x: self.jF(x).T, (x,), (Xi[:n],))[1]
        return  0.5*jnp.concatenate(
            [(1+al)*hFTxi@OForm[n:],
             - (1-al)*ijFx.T@hFTxi@ijFx.T@OForm[:n]])
    
    def Gamma(self, q, Xi, Eta, al):
        n = self.n
        x = q[:n]
        jFx = self.jF(x)
        iDFx = jla.inv(jFx)
        K = self.kmcA.Gamma(q, Xi, Eta)
        hFxi = jvp(lambda q: self.jF(q), (x,), (Xi[:n],))[1]
    
        return 0.5*jnp.concatenate(
            [(1+al)*K[:n] + (1-al)*jla.solve(jFx, K[n:])
             + (1-al)*iDFx@hFxi@Eta[:n],
             (1+al)*jFx@K[:n] + (1-al)*K[n:]
             - (1+al)*hFxi@Eta[:n]])

    def Curvature(self, q, Xi1, Xi2, Xi3, al):
        DG1 = jvp(lambda q: self.Gamma(q, Xi2, Xi3, al), (q, ), (Xi1, ))[1]    
        DG2 = jvp(lambda q: self.Gamma(q, Xi1, Xi3, al), (q, ), (Xi2, ))[1]
        GG1 = self.Gamma(q, Xi1, self.Gamma(q, Xi2, Xi3, al), al)
        GG2 = self.Gamma(q, Xi2, self.Gamma(q, Xi1, Xi3, al), al)
        
        return DG1 - DG2 + GG1 - GG2

    def Curv4(self, q, Xi1, Xi2, Xi3, Xi4, al):
        return self.kmc(q, self.Curvature(q, Xi1, Xi2, Xi3, al), Xi4)

    def Two(self, q, Xi, Eta, al):
        n = self.n
        x = q[:n]
        K = self.kmcA.Gamma(q, Xi, Eta)

        jFx = self.jF(x)
        iDFx = jla.inv(jFx)
        hFxi = jvp(lambda q: self.jF(q), (x,), (Xi[:n],))[1]

        return 0.5*jnp.concatenate(
            [(1-al)*K[:n] - (1-al)*jla.solve(jFx, K[n:])
             - (1-al)*iDFx@hFxi@Eta[:n],
             - (1+al)*jFx@K[:n] + (1+al)*K[n:]
             + (1+al)*hFxi@Eta[:n]
             ])
    
    def conjTwo(self, q, Xi1, Form2, al):
        """ the conjugate second fundamental form
        Second input is a form. Output is a form complement
        to the cotangent bundle
        """
        # K is the conjugate Ambient Christoffel function
        K = self.kmcA.conjGamma(q, Xi1, Form2)
        return self.DprojT(q, Xi1, Form2, al) + K - self.projT(q, K, al)
    

        
