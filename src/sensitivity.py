
import ngsolve as ngs
import numpy as np
import logging

logger = logging.getLogger("ngs_topopt.sensitivity")

class AdjointSensitivity():
    def __init__(self, forward_problem, objective_function):
        self.objective_function = objective_function
        self.forward_problem = forward_problem
        self.gfdesign = forward_problem.gfdesign
        self.gf = forward_problem.gf
        self.dfdphi = self.gfdesign.vec.CreateVector()
        self.dobjdu = self.gf.vec.CreateVector()
        self.lam = ngs.GridFunction(self.gf.space,"lam")
        self.updated = False

    def Update(self):
        logger.debug("Sensitivity initialization")
        phi = self.gfdesign.space.TrialFunction()
        if self.forward_problem.bf_dependent:
            self.bfdesign = ngs.BilinearForm(self.gfdesign.space)
            for integ in self.forward_problem.GetIntegratorsDesignEnergy():
                self.bfdesign += ngs.SymbolicEnergy(**integ(self.gf,self.lam,phi))
        if self.forward_problem.lf_dependent:
            D,V = self.gfdesign.space, self.gf.space
            self.bf_rhs = ngs.BilinearForm(D,V)
            phi, v = D.TrialFunction(), V.TestFunction()
            for integ in self.forward_problem.lf_integrators:
                self.bf_rhs += ngs.SymbolicBFI(**integ(phi,v))
        if self.objective_function.direct_dependency:
            self.tmp = self.gfdesign.vec.CreateVector()
            D = self.gfdesign.space
            phi = D.TrialFunction()
            self.bf_directdependency = ngs.BilinearForm(self.gfdesign.space)
            for integ in self.objective_function.integrators:
                self.bf_directdependency += ngs.SymbolicEnergy(**integ(self.gf,phi))
        self.updated = True
        

    def Compute(self):
        if not self.updated:
            self.Update()
        self.objective_function.GetDValue(self.dobjdu)
        np.conjugate(self.dobjdu.FV().NumPy(), out=self.dobjdu.FV().NumPy())
        self.lam.vec.data = self.forward_problem.GetInverse() * (-self.dobjdu)
        logger.debug("dfdu = " + str(sum(self.dobjdu.FV().NumPy())))
        logger.debug("ad lam = " + str(sum(self.lam.vec.FV().NumPy())))
        if self.forward_problem.bf_dependent:
            self.bfdesign.Apply(self.gfdesign.vec,self.dfdphi)
        else:
            self.dfdphi[:] = 0
        if self.forward_problem.lf_dependent:
            self.bf_rhs.AssembleLinearization(self.gfdesign.vec)
            self.bf_rhs.mat.MultTransAdd(-1,self.lam.vec,self.dfdphi)
        if self.objective_function.direct_dependency:
            self.bf_directdependency.Apply(self.gfdesign.vec,self.tmp)
            self.dfdphi.data += self.tmp
        logger.debug("dfdphi = " + str(sum(self.dfdphi)))
        return self.dfdphi

