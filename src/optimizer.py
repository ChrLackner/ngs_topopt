
import logging
import numpy as np
import ngsolve as ngs
import matplotlib.pyplot as plt
from .sensitivity import *
from .functions import Constraints

logger = logging.getLogger("ngs_topopt.optimizer")

def norm(a):
    return ngs.sqrt(ngs.InnerProduct(a,a)) + 1e-8

class TimeSteppingOptimizer():
    def __init__(self, forward_problem,objective_function, constraints = None):
        self.obj_func = objective_function
        self.fw_problem = forward_problem
        self.gfdesign = forward_problem.gfdesign
        if constraints is None:
            self.constraints = Constraints(self.gfdesign)
        else:
            self.constraints = constraints
        self.sensitivity = AdjointSensitivity(forward_problem = forward_problem,
                                              objective_function = objective_function)
        self.lamd = np.ones(len(self.constraints))
        self.obj_values = []
        self.time = []
        self.constr_vals = []
        self.sig = 5
        self.alpha = 0.2
        self.beta = 1.
        self.gamma = 0.1
        self.kappa = 0.00125
        u,v = self.gfdesign.space.TnT()
        self._mass = ngs.BilinearForm(self.gfdesign.space)
        self._mass += ngs.SymbolicBFI(u*v)
        self._inverse = None
        self._dt = None


    def Optimize(self, Redraw = {"blocking" : False}, dt = 0.1, tend = None):
        if tend is None:
            tend = 50*dt
        t = 0
        self._mass.Assemble()
        while t < tend:
            t += dt
            logger.info('\n*******************time step ' +str(t) +' ****************************\n')
            self.fw_problem.Solve()
            if Redraw:
                ngs.Redraw(**Redraw)
            self.UpdateDesign(dt)
            self.time.append(t)
            self.obj_values.append(self.obj_func.GetValue())
            self.constr_vals.append(self.constraints.values)
            logger.info("objective = " + str(self.obj_values[-1]))
            if len(self.constr_vals[0]):
                logger.info("error constraints = "+  str(self.constr_vals[-1]))

    def UpdateDesign(self,dt):
        self.constraints.Update()
        dobjdphi = self.sensitivity.Compute()
        logger.debug("dfdphi = " + str(sum(dobjdphi)))
        logger.debug("constr values = " + str(self.constraints.values))
        m = len(self.constraints)
        for i in range(m):
            if self.lamd[i] > 0 or self.constraints.values[i] > 0:
                self.lamd[i] += dt * self.sig * self.constraints.values[i]

        lam = self.lamd * np.exp(self.beta * np.array(self.constraints.values))
        for i in range(m):
            lam[i] = max(lam[i],0)

        logger.debug("constraints dvalues = " + str([sum(vec) for vec in self.constraints.dvalues]))
        logger.debug("lamd = " + str(self.lamd))
        logger.debug("lam = " + str(lam))
        dLdphi_hat = self.gfdesign.vec.CreateVector()
        dLdphi_hat.data = 1./norm(dobjdphi) * dobjdphi
        logger.debug("sum dldphi = " + str(sum(dLdphi_hat)))
        for i in range(m):
            dLdphi_hat.data += float(lam[i])/norm(self.constraints.dvalues[i]) * self.constraints.dvalues[i]
        dLdphi_hat *= -self.alpha * dt
        dLdphi_hat.data += self._mass.mat * self.gfdesign.vec
        logger.debug("gfdesign = " + str(sum(self.gfdesign.vec)))
        logger.debug("dLdphi_hat = " + str(sum(dLdphi_hat)))
        logger.debug("sum dldphi = " + str(sum(dLdphi_hat)))
        self.gfdesign.vec.data = self._GetInverse(dt) * dLdphi_hat
        logger.debug("gfdesign = " + str(sum(self.gfdesign.vec)))

    def PlotResults(self):
        import math
        fig = plt.subplots()
        nconstr = len(self.constr_vals[0])
        n = math.ceil(math.sqrt(nconstr+1))
        m = math.ceil((nconstr+1)/n)
        ax = plt.subplot2grid((m,n),(0,0))
        ax.plot(self.time,self.obj_values)
        plt.title(self.obj_func.name)
        for i in range(nconstr):
            axc = plt.subplot2grid((m,n),(int((1+i)/n),(1+i)%n))
            axc.plot(self.time,[val[i] for val in self.constr_vals])
            plt.title(self.constraints.names[i])
        plt.show()

    def _GetInverse(self,dt):
        if not self._inverse or self._dt != dt:
            logger.debug("Rebuild design inverse")
            D = self.gfdesign.space
            phi,psi = D.TnT()
            a = ngs.BilinearForm(D)
            a += ngs.SymbolicBFI(dt * self.kappa * ngs.grad(phi) * ngs.grad(psi) + phi * psi)
            a.Assemble()
            self._inverse = a.mat.Inverse(D.FreeDofs(),inverse="sparsecholesky")
            self._dt = dt
        return self._inverse
