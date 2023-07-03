import jax.numpy as jnp
import jax.numpy.linalg as jla
from jax import jvp, random
from jax.config import config
config.update("jax_enable_x64", True)


def grand(key, n, k):
    key, sk = random.split(key)
    return random.normal(sk, (n, k)), key


def asym(mat):
    return 0.5*(mat - mat.T)


def sym2(mat):
    return mat + mat.T


def sym(a):
    return 0.5*sym2(a)


def Lyapunov(A, B):
    # solve AU + UA = B
    # A, B, U are symmetric
    yei, yv = jla.eigh(A)
    return yv@((yv.T@B@yv)/(yei[:, None] + yei[None, :]))@yv.T


def vcat(x, y):
    return jnp.concatenate([x, y], axis=0)


def splitzero(Omg):
    n = Omg.shape[0] // 2
    k = Omg.shape[1]
    zr = jnp.zeros((n, k))        
    Omgx = jnp.concatenate([Omg[:n, :], zr], axis=0)
    Omgy = jnp.concatenate([zr, Omg[n:, :]], axis=0)
    return Omgx, Omgy


class Semidefinite(object):
    def __init__(self, al, n, k):
        self.n = n
        self.k = k
        self.al = al

    def Vproj(self, q, A):
        n, k = q.shape[0] // 2, q.shape[1]
        x = q[:n, :]
        y = q[n:, :]
        S2 = x.T@y@y.T@x
        s, v = jla.eigh(S2)
        sh = jnp.sqrt(s)    
        Sig = v@(sh[:, None]*v.T)
        iSig = v@((1/sh)[:, None]*v.T)

        U = iSig@x.T@y
        return vcat(2*x@Lyapunov(Sig, asym(U@y.T@A[:n, :])),
                    2*y@U.T@Lyapunov(Sig, asym(x.T@A[n:, :]@U.T))@U)

    def Hproj(self, q, B):
        n = q.shape[0] // 2
        x = q[:n, :]
        y = q[n:, :]
        S2 = x.T@y@y.T@x
        s, v = jla.eigh(S2)
        sh = jnp.sqrt(s)    
        Sig = v@(sh[:, None]*v.T)
        iSig = v@((1/sh)[:, None]*v.T)

        U = iSig@x.T@y        
        return vcat(B[:n, :] - 2*x@Lyapunov(Sig, asym(U@y.T@B[:n, :])), 
                    B[n:, :] - 2*y@U.T@Lyapunov(Sig, asym(x.T@B[n:, :]@U.T))@U)

    def grandVec(self, key, q):
        n, k, = self.n, self.k
        Omg, key = grand(key, 2*n, k)
        return self.Hproj(q, Omg), key

    def grandVertical(self, key, q):
        n, k, = self.n, self.k        
        tmp, key = grand(key, 2*k, k)
        return vcat(q[:n, :]@asym(tmp[:k, :]), q[n:, :]@asym(tmp[k:, :])), key

    def randKMPoint(self, sk):
        n, k = self.n, self.k
        
        qtmp = random.normal(sk, (2*n, k))
        while jnp.abs(jla.det(qtmp[:n, :].T@qtmp[n:, :])) < 1e-5:
            qtmp = random.normal(sk, (2*n, k))
            
        return qtmp    

    def grandKMPoint(self, key):
        key, sk = random.split(key)
        return self.randKMPoint(sk), key

    def grandStfPoint(self, key):
        n, k = self.n, self.k
        key, sk = random.split(key)
        tmp = random.normal(sk, (2*n, k))
        x, _ = jla.qr(tmp[:n, :])
        y, _ = jla.qr(tmp[n:, :])        
        return vcat(x, y), key
    
    def inner(self, q, Xi1, Xi2):
        n, al = self.n, self.al
        x = q[:n, :]
        y = q[n:, :]

        S2 = x.T@y@y.T@x
        s, v = jla.eigh(S2)
        sh = jnp.sqrt(s)    
        Sig = v@(sh[:, None]*v.T)
        iSig = v@((1/sh)[:, None]*v.T)
        KK = al + jnp.trace(Sig)
        U = iSig@x.T@y
        return - 0.5/KK**2*(
            jnp.trace(Xi1[:n, :].T@y@U.T)*jnp.trace(x.T@Xi2[n:, :]@U.T)
            + jnp.trace(Xi2[:n, :].T@y@U.T)*jnp.trace(x.T@Xi1[n:, :]@U.T)) \
            + 0.5/KK*jnp.trace(Xi1[:n, :].T@Xi2[n:, :]@U.T) \
            + 0.5/KK*jnp.trace(Xi2[:n, :].T@Xi1[n:, :]@U.T)

    def g(self, q, Xi):
        n, al = self.n, self.al
        x = q[:n, :]
        y = q[n:, :]

        S2 = x.T@y@y.T@x
        s, v = jla.eigh(S2)
        sh = jnp.sqrt(s)    
        Sig = v@(sh[:, None]*v.T)
        iSig = v@((1/sh)[:, None]*v.T)
        KK = al + jnp.trace(Sig)
        U = iSig@x.T@y

        retx = - 0.5/KK**2*jnp.trace(x.T@Xi[n:, :]@U.T)*y@U.T \
            + 0.5/KK*Xi[n:, :]@U.T

        rety = - 0.5/KK**2*jnp.trace(y.T@Xi[:n, :]@U)*x@U \
            + 0.5/KK*Xi[:n, :]@U
        return vcat(retx, rety)

    def ginv(self, q, sXi):
        n, al = self.n, self.al
        x = q[:n, :]
        y = q[n:, :]

        S2 = x.T@y@y.T@x
        s, v = jla.eigh(S2)
        sh = jnp.sqrt(s)    
        Sig = v@(sh[:, None]*v.T)
        iSig = v@((1/sh)[:, None]*v.T)
        KK = al + jnp.trace(Sig)
        U = iSig@x.T@y
        
        retx =  2*KK*sXi[n:, :]@U.T + 2*KK/al*jnp.trace(y.T@sXi[n:, :])*x
        rety = 2*KK*sXi[:n, :]@U + 2*KK/al*jnp.trace(x.T@sXi[:n, :])*y
        return vcat(retx, rety)

    def DHprojTan(self, q, Xi, Eta):
        """Both Xi and Eta are horizontal
        This is good for simplified evaluation. If we need to take
        derivatives use DHproj
        """
        n = q.shape[0] // 2
        x = q[:n, :]
        y = q[n:, :]
        S2 = x.T@y@y.T@x
        s, v = jla.eigh(S2)
        sh = jnp.sqrt(s)    
        Sig = v@(sh[:, None]*v.T)
        iSig = v@((1/sh)[:, None]*v.T)
        U = iSig@x.T@y

        return vcat(- 2*x@Lyapunov(Sig, asym(U@Xi[n:, :].T@Eta[:n, :])),
                    - 2*y@U.T@Lyapunov(Sig, asym(Xi[:n, :].T@Eta[n:, :]@U.T))@U)

    def AONeill(self, q, Xi, Eta):
        n = q.shape[0] // 2
        x = q[:n, :]
        y = q[n:, :]
        S2 = x.T@y@y.T@x
        s, v = jla.eigh(S2)
        sh = jnp.sqrt(s)    
        Sig = v@(sh[:, None]*v.T)
        iSig = v@((1/sh)[:, None]*v.T)
        U = iSig@x.T@y
        
        return vcat(x@Lyapunov(Sig, asym(U@(Eta[n:, :].T@Xi[:n, :] - Xi[n:, :].T@Eta[:n, :]))),
                    y@U.T@Lyapunov(Sig, asym((Eta[:n, :].T@Xi[n:, :] - Xi[:n, :].T@Eta[n:, :])@U.T))@U)
    
    def adjAONeill(self, q, Xi, Ver):
        n = q.shape[0] // 2
        x = q[:n, :]
        y = q[n:, :]
        S2 = x.T@y@y.T@x
        s, v = jla.eigh(S2)
        sh = jnp.sqrt(s)    
        Sig = v@(sh[:, None]*v.T)
        iSig = v@((1/sh)[:, None]*v.T)

        U = iSig@x.T@y

        Xix, Xiy = Xi[:n, :], Xi[n:, :]
        Verx, Very = Ver[:n, :], Ver[n:, :]

        DSx = Xix.T@y@U.T
        DSy = x.T@Xiy@U.T
        Lx = Lyapunov(Sig, asym(U@y.T@Verx))
        Ly = U.T@Lyapunov(Sig, asym(x.T@Very@U.T))@U

        DLx = Lyapunov(Sig, asym(U@Xiy.T@Ver[:n, :]) - (DSx+DSy)@Lx - Lx@(DSx+DSy))
        DLy = U.T@Lyapunov(Sig, asym(Xix.T@Very@U.T  - (DSx+DSy)@U@Ly@U.T - U@Ly@U.T@(DSx+DSy)))@U    
        DHprojV = vcat(- 2*Xix@Lx - 2*x@DLx, - 2*Xiy@Ly - 2*y@DLy)

        return DHprojV - self.Hproj(q, self.GammaAmb(q, Xi, Ver))

    def Dg(self, q, Omg1, Omg2):
        n, al = self.n, self.al
        x = q[:n, :]
        y = q[n:, :]

        S2 = x.T@y@y.T@x
        s, v = jla.eigh(S2)
        sh = jnp.sqrt(s)    
        Sig = v@(sh[:, None]*v.T)
        iSig = v@((1/sh)[:, None]*v.T)
        KK = al + jnp.trace(Sig)
        U = iSig@x.T@y

        Omgx1 = Omg1[:n, :]
        Omgy1 = Omg1[n:, :]

        Omgx2 = Omg2[:n, :]
        Omgy2 = Omg2[n:, :]

        DSx1 = Omgx1.T@y@U.T - 2*Sig@Lyapunov(Sig, asym(Omgx1.T@y@U.T))
        DSy1 = x.T@Omgy1@U.T - 2*Sig@Lyapunov(Sig, asym(x.T@Omgy1@U.T))

        DSx2 = Omgx2.T@y@U.T - 2*Sig@Lyapunov(Sig, asym(Omgx2.T@y@U.T))
        DSy2 = x.T@Omgy2@U.T - 2*Sig@Lyapunov(Sig, asym(x.T@Omgy2@U.T))

        DUx1 = 2*Lyapunov(Sig, asym(Omgx1.T@y@U.T))@U
        DUy1 = 2*Lyapunov(Sig, asym(x.T@Omgy1@U.T))@U

        retx = 1/KK**3*jnp.trace(DSx1 + DSy1)*jnp.trace(DSy2)*y@U.T \
            - 0.5/KK**2*jnp.trace(Omgx1.T@Omgy2@U.T)*y@U.T \
            + 0.5/KK**2*jnp.trace(x.T@Omgy2@U.T@(DUx1 + DUy1)@U.T)*y@U.T \
            - 0.5/KK**2*jnp.trace(DSy2)*Omgy1@U.T \
            + 0.5/KK**2*jnp.trace(DSy2)*y@U.T@(DUx1 + DUy1)@U.T \
            - 0.5/KK**2*jnp.trace(DSx1 + DSy1)*Omgy2@U.T \
            - 0.5/KK*Omgy2@U.T@(DUx1 + DUy1)@U.T

        rety = 1/KK**3*jnp.trace(DSx1 + DSy1)*jnp.trace(DSx2)*x@U \
            - 0.5/KK**2*jnp.trace(Omgy1.T@Omgx2@U)*x@U \
            - 0.5/KK**2*jnp.trace(y.T@Omgx2@(DUx1+DUy1))*x@U \
            - 0.5/KK**2*jnp.trace(DSx2)*Omgx1@U \
            - 0.5/KK**2*jnp.trace(DSx2)*x@(DUx1 + DUy1) \
            - 0.5/KK**2*jnp.trace(DSx1 + DSy1)*Omgx2@U \
            + 0.5/KK*Omgx2@(DUx1 + DUy1)

        return vcat(retx, rety)

    def Xg(self, q, Omg1, Omg2):
        n, al = self.n, self.al
        x = q[:n, :]
        y = q[n:, :]

        S2 = x.T@y@y.T@x
        s, v = jla.eigh(S2)
        sh = jnp.sqrt(s)    
        Sig = v@(sh[:, None]*v.T)
        iSig = v@((1/sh)[:, None]*v.T)
        KK = al + jnp.trace(Sig)
        U = iSig@x.T@y

        Omgx1 = Omg1[:n, :]
        Omgy1 = Omg1[n:, :]

        Omgx2 = Omg2[:n, :]
        Omgy2 = Omg2[n:, :]

        DSx1 = Omgx1.T@y@U.T - 2*Sig@Lyapunov(Sig, asym(Omgx1.T@y@U.T))
        DSy1 = x.T@Omgy1@U.T - 2*Sig@Lyapunov(Sig, asym(x.T@Omgy1@U.T))

        DSx2 = Omgx2.T@y@U.T - 2*Sig@Lyapunov(Sig, asym(Omgx2.T@y@U.T))
        DSy2 = x.T@Omgy2@U.T - 2*Sig@Lyapunov(Sig, asym(x.T@Omgy2@U.T))

        DUx1 = 2*Lyapunov(Sig, asym(Omgx1.T@y@U.T))@U
        DUy1 = 2*Lyapunov(Sig, asym(x.T@Omgy1@U.T))@U

        DUx2 = 2*Lyapunov(Sig, asym(Omgx2.T@y@U.T))@U
        DUy2 = 2*Lyapunov(Sig, asym(x.T@Omgy2@U.T))@U

        retx = 1/KK**3*jnp.trace(DSy2)*jnp.trace(DSx1)*y@U.T \
            - 0.5/KK**2*jnp.trace(DSx1)*Omgy2@U.T \
            + 0.5/KK**2*jnp.trace(DSx1)*y@U.T@DUy2@U.T \
            + 0.5/KK**2*jnp.trace(DSy2)*y@U.T@DUx1@U.T \
            - 0.5/KK**2*y@U.T*jnp.trace(Omgy2@U.T@Omgx1.T) \
            - 1/KK*y@U.T@Lyapunov(Sig, asym(Omgx1.T@Omgy2@U.T)) \
            \
            + 1/KK**3*jnp.trace(DSx2)*jnp.trace(DSy1)*y@U.T \
            + 0.5/KK**2*y@U.T@DUx2@U.T*jnp.trace(DSy1) \
            - 0.5/KK**2*jnp.trace(DSx2)*Omgy1@U.T \
            + 0.5/KK**2*jnp.trace(DSx2)*y@U.T@DUy1@U.T \
            - 0.5/KK**2*y@U.T*jnp.trace(Omgx2@U@Omgy1.T) \
            + 1/KK*y@U.T@Lyapunov(Sig, asym(U@Omgy1.T@Omgx2))

        rety = 1/KK**3*jnp.trace(DSy2)*jnp.trace(DSx1)*x@U \
            - 0.5/KK**2*jnp.trace(DSx1)*x@DUy2 \
            \
            - 0.5/KK**2*jnp.trace(DSy2)*Omgx1@U \
            - 0.5/KK**2*jnp.trace(DSy2)*x@DUx1 \
            - 0.5/KK**2*jnp.trace(Omgy2@U.T@Omgx1.T)*x@U \
            + 1/KK*x@Lyapunov(Sig, asym(Omgx1.T@Omgy2@U.T))@U \
            \
            + 1/KK**3*jnp.trace(DSx2)*jnp.trace(DSy1)*x@U \
            - 0.5/KK**2*jnp.trace(DSy1)*Omgx2@U \
            - 0.5/KK**2*jnp.trace(DSy1)*x@DUx2 \
            - 0.5/KK**2*jnp.trace(DSx2)*x@DUy1 \
            - 0.5/KK**2*jnp.trace(Omgx2@U@Omgy1.T)*x@U \
            - 1/KK*x@Lyapunov(Sig, asym(U@Omgy1.T@Omgx2))@U

        return vcat(retx, rety)    

    def GammaAmbNumeric(self, q, Xi, Eta):
        return 0.5*self.ginv(q, self.Dg(q, Xi, Eta)
                             + self.Dg(q, Eta, Xi)
                             - self.Xg(q, Xi, Eta))

    def GammaAmb(self, q, Omg1, Omg2):
        """ both are not horizontal
        """
        # Kr = Dg(FRE, q, Omg1, Omg2) + Dg(FRE, q, Omg2, Omg1) - Xg(FRE, q, Omg1, Omg2)
        # Omg2 is horizontal, Omg1 is not
        n, al = self.n, self.al
        x = q[:n, :]
        y = q[n:, :]

        S2 = x.T@y@y.T@x
        s, v = jla.eigh(S2)
        sh = jnp.sqrt(s)    
        Sig = v@(sh[:, None]*v.T)
        iSig = v@((1/sh)[:, None]*v.T)
        KK = al + jnp.trace(Sig)
        U = iSig@x.T@y

        retx = - 1/KK*jnp.trace(Omg2[:n, :].T@y@U.T)*Omg1[:n, :] \
            - 1/KK*jnp.trace(Omg1[:n, :].T@y@U.T)*Omg2[:n, :] \
            + x@Lyapunov(Sig, asym(U@(Omg1[n:, :].T@Omg2[:n, :] + Omg2[n:, :].T@Omg1[:n, :]))) \
            + Omg1[:n, :]@Lyapunov(Sig, asym(x.T@Omg2[n:, :]@U.T)) \
            + Omg1[:n, :]@Lyapunov(Sig, asym(Omg2[:n, :].T@y@U.T)) \
            + Omg2[:n, :]@Lyapunov(Sig, asym(x.T@Omg1[n:, :]@U.T)) \
            + Omg2[:n, :]@Lyapunov(Sig, asym(Omg1[:n, :].T@y@U.T)) \
            + 1/KK*(jnp.trace(x.T@Omg2[n:, :]@U.T)
                    - jnp.trace(Omg2[:n, :].T@y@U.T))*x@Lyapunov(Sig, asym(Omg1[:n, :].T@y@U.T)) \
            + 1/KK*(jnp.trace(x.T@Omg1[n:, :]@U.T)
                    - jnp.trace(Omg1[:n, :].T@y@U.T))*x@Lyapunov(Sig, asym(Omg2[:n, :].T@y@U.T))            

        rety = - 1/KK*jnp.trace(Omg2[n:, :].T@x@U)*Omg1[n:, :] \
            - 1/KK*jnp.trace(Omg1[n:, :].T@x@U)*Omg2[n:, :] \
            - y@U.T@Lyapunov(Sig, asym(U@(Omg2[n:, :].T@Omg1[:n, :] + 
                                          Omg1[n:, :].T@Omg2[:n, :])))@U \
            + Omg1[n:, :]@U.T@Lyapunov(Sig, asym(U@(y.T@Omg2[:n, :] + Omg2[n:, :].T@x)))@U \
            + Omg2[n:, :]@U.T@Lyapunov(Sig, asym(U@(y.T@Omg1[:n, :]+Omg1[n:, :].T@x)))@U \
            + 1/KK*jnp.trace(Omg2[:n, :].T@y@U.T
                             - x.T@Omg2[n:, :]@U.T)*y@U.T@Lyapunov(Sig, asym(U@Omg1[n:, :].T@x))@U \
            + 1/KK*jnp.trace(Omg1[:n, :].T@y@U.T
                             - x.T@Omg1[n:, :]@U.T)*y@U.T@Lyapunov(Sig, asym(U@Omg2[n:, :].T@x))@U

        return vcat(retx, rety)

    def CurvSD31(self, q, Xi, Eta, Phi):
        D1 = jvp(lambda q: self.GammaH(q, Eta, Phi), (q,), (Xi,))[1]
        D2 = jvp(lambda q: self.GammaH(q, Xi, Phi), (q,), (Eta,))[1]
        G1 = self.GammaH(q, Xi, self.GammaH(q, Eta, Phi))
        G2 = self.GammaH(q, Eta, self.GammaH(q, Xi, Phi))
        return D1 - D2 + G1 - G2 - 2*self.adjAONeill(q, Phi, self.AONeill(q, Xi, Eta))
    
    def Curv4(self, q, Xi, Eta, Phi, Zeta):
        D1 = jvp(lambda q: self.GammaH(q, Eta, Phi), (q,), (Xi,))[1]
        D2 = jvp(lambda q: self.GammaH(q, Xi, Phi), (q,), (Eta,))[1]
        G1 = self.GammaH(q, Xi, self.GammaH(q, Eta, Phi))
        G2 = self.GammaH(q, Eta, self.GammaH(q, Xi, Phi))
        return self.inner(q, D1 - D2 + G1 - G2, Zeta) - 2*self.inner(
            q,
            self.AONeill(q, Xi, Eta),
            self.AONeill(q, Phi, Zeta))

    def CurvAmb(self, q, B1, B2, B3):
        D1 = jvp(lambda q: self.GammaAmb(q, B2, B3), (q,), (B1,))[1]
        D2 = jvp(lambda q: self.GammaAmb(q, B1, B3), (q,), (B2,))[1]
        G1 = self.GammaAmb(q, B1, self.GammaAmb(q, B2, B3))
        G2 = self.GammaAmb(q, B2, self.GammaAmb(q, B1, B3))
        return D1 - D2 + G1 - G2

    def DHproj(self, q, Omg, Xi):
        """ Xi is Horizontal
        Omg is not
        """
        n = q.shape[0] // 2
        x = q[:n, :]
        y = q[n:, :]
        S2 = x.T@y@y.T@x
        s, v = jla.eigh(S2)
        sh = jnp.sqrt(s)    
        Sig = v@(sh[:, None]*v.T)
        iSig = v@((1/sh)[:, None]*v.T)
        Xix, Xiy = Xi[:n, :], Xi[n:, :]
        Omgx, Omgy = Omg[:n, :], Omg[n:, :]
        U = iSig@x.T@y    

        DUx = 2*Lyapunov(Sig, asym(Omgx.T@y@U.T))@U
        DUy = 2*Lyapunov(Sig, asym(x.T@Omgy@U.T))@U    

        return vcat (- 2*x@Lyapunov(Sig, asym((DUx + DUy)@y.T@Xix  + U@Omgy.T@Xix)),
                     - 2*y@U.T@Lyapunov(Sig, asym(Omgx.T@Xiy@U.T + x.T@Xiy@(DUx+DUy).T))@U)
    
    def GammaH(self, q, Omg, Eta):
        """ We should use this when we take derivatives
        Eta is horizontal, Omg is not
        """
        return -self.DHproj(q, Omg, Eta) + self.Hproj(q, self.GammaAmb(q, Omg, Eta))

    def GammaHEval(self, q, Xi, Eta):
        """ this usually cannot be used if we take derivatives.
        Just use this if we need to evaluate at two horizontal vectors
        """
        return -self.DHprojTan(q, Xi, Eta) + self.Hproj(q, self.GammaAmb(q, Xi, Eta))
    

    def crossCurvAmb4(self, q, Xi):
        n, al = self.n, self.al
        x = q[:n, :]
        y = q[n:, :]

        S2 = x.T@y@y.T@x
        s, v = jla.eigh(S2)
        sh = jnp.sqrt(s)    
        Sig = v@(sh[:, None]*v.T)
        iSig = v@((1/sh)[:, None]*v.T)
        KK = al + jnp.trace(Sig)
        U = iSig@x.T@y
        return - 1/KK**4*jnp.trace(Xi[:n, :].T@y@U.T)*jnp.trace(Xi[:n, :].T@y@U.T)*jnp.trace(x.T@Xi[n:, :]@U.T)*jnp.trace(x.T@Xi[n:, :]@U.T) \
            + 2/KK**3*jnp.trace(Xi[:n, :].T@y@U.T)*jnp.trace(Xi[:n, :].T@Xi[n:, :]@U.T)*jnp.trace(x.T@Xi[n:, :]@U.T) \
            - 1/KK**2*jnp.trace(Xi[:n, :].T@Xi[n:, :]@U.T)*jnp.trace(Xi[:n, :].T@Xi[n:, :]@U.T) \
            - 0.5/KK*jnp.trace(Xi[:n, :].T@Xi[n:, :]@U.T@Lyapunov(Sig, asym(Xi[:n, :].T@Xi[n:, :]@U.T)))

    def crossCurv4(self, q, Xi):
        n, al = self.n, self.al
        x = q[:n, :]
        y = q[n:, :]

        S2 = x.T@y@y.T@x
        s, v = jla.eigh(S2)
        sh = jnp.sqrt(s)    
        Sig = v@(sh[:, None]*v.T)
        iSig = v@((1/sh)[:, None]*v.T)
        KK = al + jnp.trace(Sig)
        U = iSig@x.T@y
        retAmb = - self.inner(q, Xi, Xi)**2 \
            - 0.5/KK*jnp.trace(asym(Xi[:n, :].T@Xi[n:, :]@U.T)@Lyapunov(Sig, asym(Xi[:n, :].T@Xi[n:, :]@U.T)))

        ONeill = 1/KK*jnp.trace(
            Lyapunov(Sig, asym(U@(Xi[n:, :].T@Xi[:n, :])))@
            Sig@Lyapunov(Sig, asym(Xi[:n, :].T@Xi[n:, :]@U.T)))

        retAmb = - self.inner(q, Xi, Xi)**2 + ONeill
        return retAmb + 3*ONeill

    def CurvAmb4(self, q, B1, B2, B3, B4):
        return self.inner(q, self.CurvAmb(q, B1, B2, B3), B4)

    def c(self, q):
        n, k, al = self.n, self.k, self.al
        s = jla.svd(q[:n, :].T@q[n:, :], compute_uv=False)
        return -jnp.log(al + jnp.sum(s))

    def c2(self, q):
        n, k, al = self.n, self.k, self.al
        
        s, v = jla.eigh(q[:n, :].T@q[n:, :]@q[n:, :].T@q[:n, :])
        return -jnp.log(al + jnp.trace(v.T@(jnp.sqrt(s)[:, None]*v)))

    def Derc2(self, q, Xi):
        n, k, al = self.n, self.k, self.al
        
        x = q[:n, :]
        y = q[n:, :]

        s, v = jla.eigh(q[:n, :].T@q[n:, :]@q[n:, :].T@q[:n, :])
        sh = jnp.sqrt(s)
        Sig = v@(sh[:, None]*v.T)
        iSig = v@((1/sh)[:, None]*v.T)
        KK = al + jnp.trace(Sig)
        return - 1/KK*jnp.trace(
            iSig@Xi[:n, :].T@q[n:, :]@q[n:, :].T@q[:n, :]
            + iSig@q[:n, :].T@Xi[n:, :]@q[n:, :].T@q[:n, :])

    def Gradc(self, q):
        n, k, al = self.n, self.k, self.al
        
        s2 = q[:n, :].T@q[n:, :]@q[n:, :].T@q[:n, :]
        s, v = jla.eigh(s2)  
        sh = jnp.sqrt(s)
        Sig = v@(sh[:, None]*v.T)
        iSig = v@((1/sh)[:, None]*v.T)
        KK = al + jnp.trace(Sig)        
        return -1/KK * jnp.concatenate(
            [q[n:, :]@q[n:, :].T@q[:n, :]@iSig,
             q[:n, :]@iSig@q[:n, :].T@q[n:, :]])

    def Gradxc(self, q):
        n, k, al = self.n, self.k, self.al
        
        s2 = q[:n, :].T@q[n:, :]@q[n:, :].T@q[:n, :]
        s, v = jla.eigh(s2)
        sh = jnp.sqrt(s)
        
        Sig = v@(sh[:, None]*v.T)
        iSig = v@((1/sh)[:, None]*v.T)
        KK = al + jnp.trace(Sig)        
        
        return -1/KK * q[n:, :]@q[n:, :].T@q[:n, :]@iSig
    
    def bDDc(self, q, bxi):
        n, k, al = self.n, self.k, self.al
        
        x = q[:n, :]
        y = q[n:, :]
        s2 = q[:n, :].T@q[n:, :]@q[n:, :].T@q[:n, :]
        s, v = jla.eigh(s2)  
        sh = jnp.sqrt(s)
        Sig = v@(sh[:, None]*v.T)        
        iSig = v@((1/sh)[:, None]*v.T)
        
        Fy = sym2(x.T@bxi@y.T@x)
        DS = v@((v.T@Fy@v)/(sh[:, None] + sh[None, :]))@v.T
        KK = al + jnp.trace(Sig)

        return 1/KK**2*jnp.trace(iSig@x.T@bxi@y.T@x) * y@y.T@x@iSig \
            - 1/KK * (bxi@y.T@x@Sig + y@bxi.T@x@Sig
                      - y@y.T@x@iSig@x.T@bxi@y.T@x
                      - y@y.T@x@iSig@x.T@y@bxi.T@x
                      + y@y.T@x@DS)@jla.inv(s2)
    
    def DbcDc(self, q, xi):
        n, k, al = self.n, self.k, self.al
        
        x = q[:n, :]
        y = q[n:, :]
        s2 = y.T@x@x.T@y
        s, v = jla.eigh(s2)  
        sh = jnp.sqrt(s)
        Sig = v@(sh[:, None]*v.T)        
        iSig = v@((1/sh)[:, None]*v.T)
        
        Fx = sym2(y.T@xi@x.T@y)
        DS = v@((v.T@Fx@v)/(sh[:, None] + sh[None, :]))@v.T
        KK = al + jnp.trace(Sig)

        return 1/KK**2*jnp.trace(iSig@y.T@xi@x.T@y) * x@x.T@y@iSig \
            - 1/KK * (xi@x.T@y@Sig + x@xi.T@y@Sig
                      - x@x.T@y@iSig@y.T@xi@x.T@y
                      - x@x.T@y@iSig@y.T@x@xi.T@y
                      + x@x.T@y@DS)@jla.inv(s2)

    def projStief(self, qg, A):
        # normal satisfies iSig@x.T@Vx, iSig@y.T@vy are symmetric
        #
        n, al = self.n, self.al
        x = qg[:n, :]
        y = qg[n:, :]

        S2 = x.T@y@y.T@x
        s, v = jla.eigh(S2)
        sh = jnp.sqrt(s)    
        Sig = v@(sh[:, None]*v.T)
        iSig = v@((1/sh)[:, None]*v.T)
        KK = al + jnp.trace(Sig)
        U = iSig@x.T@y

        retx = y@U.T@Lyapunov(Sig, sym2(x.T@A[:n, :])) \
            + 1/(2*(al+jnp.trace(iSig)))*jnp.trace(iSig@sym2(x.T@A[:n, :]))*(x - y@U.T@iSig)

        rety = x@Lyapunov(Sig, U@sym2(y.T@A[n:, :])@U.T)@U \
            + 1/(2*(al+jnp.trace(iSig)))*jnp.trace(U.T@iSig@U@sym2(y.T@A[n:, :]))*(y - x@iSig@U)

        return A  - vcat(retx, rety)
    
    def grandStfVec(self, key, qg):
        n, k = self.n, self.k
        
        x = qg[:n, :]
        y = qg[n:, :]
        
        key, sk = random.split(key)
        tmp = random.normal(sk, (2*n, k))
        return vcat(tmp[:n, :] - x@sym(x.T@tmp[:n, :]),
                    tmp[n:, :] - y@sym(y.T@tmp[n:, :])), key

    def projGrass(self, qg, A):
        n, al = self.n, self.al
        x = qg[:n, :]
        y = qg[n:, :]

        S2 = x.T@y@y.T@x
        s, v = jla.eigh(S2)
        sh = jnp.sqrt(s)    
        Sig = v@(sh[:, None]*v.T)
        iSig = v@((1/sh)[:, None]*v.T)
        U = iSig@x.T@y

        return vcat(
            A[:n, :]
            - 2*x@Lyapunov(Sig, asym(U@y.T@A[:n, :]))
            - y@U.T@Lyapunov(Sig, sym2(x.T@A[:n, :])) - 1/(2*(al+jnp.trace(iSig)))*jnp.trace(iSig@sym2(x.T@A[:n, :]))*(x - y@U.T@iSig),
            A[n:, :]
            - 2*y@U.T@Lyapunov(Sig, asym(x.T@A[n:, :]@U.T))@U 
            - x@Lyapunov(Sig, U@sym2(y.T@A[n:, :])@U.T)@U - 1/(2*(al+jnp.trace(iSig)))*jnp.trace(U.T@iSig@U@sym2(y.T@A[n:, :]))*(y - x@iSig@U)
            )

    def DprojStf(self, qg, stXi, stEta):
        """ stXi, stEta are tangent
        """
        n, al = self.n, self.al
        x = qg[:n, :]
        y = qg[n:, :]

        S2 = x.T@y@y.T@x
        s, v = jla.eigh(S2)
        sh = jnp.sqrt(s)    
        Sig = v@(sh[:, None]*v.T)
        iSig = v@((1/sh)[:, None]*v.T)
        U = iSig@x.T@y

        retx = y@U.T@Lyapunov(Sig, sym2(stXi[:n, :].T@stEta[:n, :])) \
            + 1/(2*(al+jnp.trace(iSig)))*jnp.trace(iSig@sym2(stXi[:n, :].T@stEta[:n, :]))*(x - y@U.T@iSig)
        
        rety = x@Lyapunov(Sig, U@sym2(stXi[n:, :].T@stEta[n:, :])@U.T)@U \
            + 1/(2*(al+jnp.trace(iSig)))*jnp.trace(U.T@iSig@U@sym2(stXi[n:, :].T@stEta[n:, :]))*(y - x@iSig@U)

        return - vcat(retx, rety)

    def GammaStief(self, qg, stXi, stEta):
        return - self.DprojStf(qg, stXi, stEta) \
            + self.projStief(qg, self.GammaAmb(qg, stXi, stEta))

    def TwoStief(self, qg, stXi1, stXi2):
        GA = self.GammaAmb(qg, stXi1, stXi2)
        return self.DprojStf(qg, stXi1, stXi2) + GA - self.projStief(qg, GA)

    def CurvStief(self, qg, stXi1, stXi2, stXi3):
        D1 = jvp(lambda qg: self.GammaStief(qg, stXi2, stXi3), (qg,), (stXi1,))[1]
        D2 = jvp(lambda qg: self.GammaStief(qg, stXi1, stXi3), (qg,), (stXi2,))[1]
        G1 = self.GammaStief(qg, stXi1, self.GammaStief(qg, stXi2, stXi3))
        G2 = self.GammaStief(qg, stXi2, self.GammaStief(qg, stXi1, stXi3))
        return D1 - D2 + G1 - G2        
        
    def CurvStief4(self, qg, stXi1, stXi2, stXi3, stXi4):
        return self.inner(qg, self.CurvStief(qg, stXi1, stXi2, stXi3), stXi4)

    def DprojGrass(self, qg, stXi, grEta):
        """ grEta is Grassmann (pair)
        stXi is only Stiefel (pair)
        """
        n, al = self.n, self.al
        x = qg[:n, :]
        y = qg[n:, :]

        S2 = x.T@y@y.T@x
        s, v = jla.eigh(S2)
        sh = jnp.sqrt(s)    
        Sig = v@(sh[:, None]*v.T)
        iSig = v@((1/sh)[:, None]*v.T)
        U = iSig@x.T@y
        
        DU = 2*Lyapunov(Sig, asym(stXi[:n, :].T@y@U.T))@U + \
            2*Lyapunov(Sig, asym(x.T@stXi[n:, :]@U.T))@U

        return vcat(
            - 2*x@Lyapunov(Sig, asym(DU@y.T@grEta[:n, :] + U@stXi[n:, :].T@grEta[:n, :]))
            - y@U.T@Lyapunov(Sig, sym2(stXi[:n, :].T@grEta[:n, :]))
            - 1/(2*(al+jnp.trace(iSig)))*jnp.trace(iSig@sym2(stXi[:n, :].T@grEta[:n, :]))*(x - y@U.T@iSig),
            - 2*y@U.T@Lyapunov(Sig, asym(stXi[:n, :].T@grEta[n:, :]@U.T + x.T@grEta[n:, :]@DU.T))@U 
            - x@Lyapunov(Sig, U@sym2(stXi[n:, :].T@grEta[n:, :])@U.T)@U
            - 1/(2*(al+jnp.trace(iSig)))*jnp.trace(U.T@iSig@U@sym2(stXi[n:, :].T@grEta[n:, :]))*(y - x@iSig@U)
            )

    def GammaGrass(self, qg, grXi, grEta):
        return -self.DprojGrass(qg, grXi, grEta) \
            + self.projGrass(qg, self.GammaAmb(qg, grXi, grEta))

    def AGrass(self, qg, grXi1, grXi2):
        return 0.5*(self.DprojGrass(qg, grXi1, grXi2)
                    -  self.DprojGrass(qg, grXi2, grXi1))

    def adjAGrass(self, qg, grXi, grVer):
        n, al = self.n, self.al
        x = qg[:n, :]
        y = qg[n:, :]

        S2 = x.T@y@y.T@x
        s, v = jla.eigh(S2)
        sh = jnp.sqrt(s)    
        Sig = v@(sh[:, None]*v.T)
        iSig = v@((1/sh)[:, None]*v.T)
        U = iSig@x.T@y

        Xix, Xiy = grXi[:n, :], grXi[n:, :]
        Verx, Very = grVer[:n, :], grVer[n:, :]
        DS = Xix.T@y@U.T + x.T@Xiy@U.T
        tDS = jnp.trace(DS)

        iTI = 1/(al+jnp.trace(iSig))
        DiTI = - (-jnp.trace(iSig@iSig@DS))/(al+jnp.trace(iSig))**2
        LVerx = Lyapunov(Sig, asym(U@y.T@Verx))
        LStfx = Lyapunov(Sig, sym2(x.T@Verx))
        LVery = Lyapunov(Sig, asym(x.T@Very@U.T))
        LStfy = Lyapunov(Sig, U@sym2(y.T@Very)@U.T)
        DprojGrassV = vcat(
            - 2*Xix@LVerx
            - 2*x@Lyapunov(Sig, asym(U@Xiy.T@Verx) - DS@LVerx - LVerx@DS)
            - Xiy@U.T@LStfx - y@U.T@Lyapunov(Sig, sym2(Xix.T@Verx) - DS@LStfx - LStfx@DS)
            - 0.5*DiTI*jnp.trace(iSig@sym2(x.T@Verx))*(x - y@U.T@iSig)
            - 0.5*iTI*jnp.trace(-iSig@DS@iSig@sym2(x.T@Verx) + iSig@sym2(Xix.T@Verx))*(x - y@U.T@iSig)
            - 0.5*iTI*jnp.trace(iSig@sym2(x.T@Verx))*(Xix - Xiy@U.T@iSig + y@U.T@iSig@DS@iSig),        
            - 2*Xiy@U.T@LVery@U
            - 2*y@U.T@Lyapunov(Sig, asym(Xix.T@Very@U.T) - DS@LVery - LVery@DS)@U
            - Xix@LStfy@U - x@Lyapunov(Sig, U@sym2(Xiy.T@Very)@U.T - DS@LStfy - LStfy@DS)@U
            - 0.5*DiTI*jnp.trace(U.T@iSig@U@sym2(y.T@Very))*(y - x@iSig@U)
            - 0.5*iTI*jnp.trace(-U.T@iSig@DS@iSig@U@sym2(y.T@Very)+U.T@iSig@U@sym2(Xiy.T@Very))*(y - x@iSig@U)
            - 0.5*iTI*jnp.trace(U.T@iSig@U@sym2(y.T@Very))*(Xiy - Xix@iSig@U + x@iSig@DS@iSig@U)                                                  
        )

        return DprojGrassV - self.projGrass(qg, self.GammaAmb(qg, grXi, grVer))

    def CurvGrass31(self, qg, grXi1, grXi2, grXi3):
        D1 = jvp(lambda qg: self.GammaGrass(qg, grXi2, grXi3), (qg,), (grXi1,))[1]
        D2 = jvp(lambda qg: self.GammaGrass(qg, grXi1, grXi3), (qg,), (grXi2,))[1]
        G1 = self.GammaGrass(qg, grXi1, self.GammaGrass(qg, grXi2, grXi3))
        G2 = self.GammaGrass(qg, grXi2, self.GammaGrass(qg, grXi1, grXi3))
        return D1 - D2 + G1 - G2 - 2*self.adjAGrass(qg, grXi3, self.AONeill(qg, grXi1, grXi2))

    def CurvGrass4(self, qg, grXi1, grXi2, grXi3, grXi4):
        D1 = jvp(lambda qg: self.GammaGrass(qg, grXi2, grXi3), (qg,), (grXi1,))[1]
        D2 = jvp(lambda qg: self.GammaGrass(qg, grXi1, grXi3), (qg,), (grXi2,))[1]
        G1 = self.GammaGrass(qg, grXi1, self.GammaGrass(qg, grXi2, grXi3))
        G2 = self.GammaGrass(qg, grXi2, self.GammaGrass(qg, grXi1, grXi3))
        return self.inner(qg, D1 - D2 + G1 - G2, grXi4) - 2*self.inner(
            qg,
            self.AGrass(qg, grXi1, grXi2),
            self.AGrass(qg, grXi3, grXi4))
    
    def TwoAmbGrass(self, qg, grXi1, grXi2):
        """ The bundle second fundamental form
        Between the trivial bundle and the horizontal bundle (II^* in the paper).
        """
        GA = self.GammaAmb(qg, grXi1, grXi2)
        return self.DprojGrass(qg, grXi1, grXi2) + GA - self.projGrass(qg, GA)

    def crossCurvGrassNumeric(self, qg, grEta):
        grEtax, grEtay = splitzero(grEta)
        return self.CurvGrass4(qg, grEtax, grEtay, grEtay, grEtax)

    def crossCurvGrass(self, qg, grEta):
        grEtax, grEtay = splitzero(grEta)
        n, al = self.n, self.al
        x = qg[:n, :]
        y = qg[n:, :]

        S2 = x.T@y@y.T@x
        s, v = jla.eigh(S2)
        sh = jnp.sqrt(s)    
        Sig = v@(sh[:, None]*v.T)
        iSig = v@((1/sh)[:, None]*v.T)
        KK = al + jnp.trace(Sig)
        U = iSig@x.T@y
        
        ret = - self.inner(qg, grEta, grEta)**2 \
            + 4/KK*jnp.trace(
                Lyapunov(Sig, asym(U@(grEta[n:, :].T@grEta[:n, :])))@
                Sig@Lyapunov(Sig, asym(grEta[:n, :].T@grEta[n:, :]@U.T))) \
            + 2/KK*jnp.trace(Lyapunov(Sig, grEtax[:n, :].T@grEtax[:n, :])@Sig@Lyapunov(Sig, U@grEtay[n:, :].T@grEtay[n:, :]@U.T)) \
            - 0.5/KK/(al+jnp.trace(iSig))*jnp.trace(iSig@grEtax[:n, :].T@grEtax[:n, :])*jnp.trace(iSig@U@grEtay[n:, :].T@grEtay[n:, :]@U.T)                
        
        return ret
    
