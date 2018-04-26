
import logging
import ngsolve as ngs
from .utils import UsesArgument

logger = logging.getLogger("ngs_topopt.problem")

class ForwardProblem():

    def __init__(self, nonhomogeneousBND = None):
        self.nhBND = False
        self.inv = None
        self.solver = None
        self.updated = False
        self.initialdesign = (ngs.CoefficientFunction(0),None)
        self.bf_integrators = []
        self.lf_integrators = []
        self.bf_integrators_design = None

    def setDesignSpace(self, space, name_gfdesign="phi"):
        self.D = space
        self.gfdesign = ngs.GridFunction(self.D,name_gfdesign)
        self.updated = False
        return self

    def setInitialDesign(self, cf,region=None):
        self.initialdesign = (cf,region)
        return self

    def setDesign(self, cf, region=None):
        if region is None:
            self.gfdesign.Set(cf)
        else:
            self.gfdesign.Set(cf,definedon=region)

    def setStateSpace(self, space, name_gfstate="u"):
        self.V = space
        self.gf = ngs.GridFunction(self.V,name_gfstate)
        self.updated = False
        return self

    def setNonhomogeneousBND(self, cf, region):
        self.nhBND = (cf,region)
        self.updated = False
        return self

    def addBilinearFormIntegrators(self, *integrators):
        self.bf_integrators += integrators
        self.updated = False
        return self

    def addLinearFormIntegrators(self, *integrators):
        self.lf_integrators += integrators
        self.updated = False
        return self

    def setDesignIntegrators(self, integrators):
        if type(integrators) is list:
            self.bf_integrators_design = integrators
        else:
            self.bf_integrators_design = [integrators]
        self.updated = False
        return self

    def SetSolver(self,solver = ngs.CGSolver,bf = None, preconditioner={ "type" : "direct" },
                  solver_args = {}):
        if not bf:
            bf = self.bf
        self.prec = ngs.Preconditioner(bf, **preconditioner)
        self.solver_args = solver_args
        self.solver = solver
        return self

    def Update(self):
        logger.debug("Forward problem initialization")
        if self.initialdesign[1] is None:
            self.gfdesign.Set(self.initialdesign[0])
        else:
            self.gfdesign.Set(self.initialdesign[0], definedon=self.initialdesign[1])
        self.bf = ngs.BilinearForm(self.V)
        self.lf = ngs.LinearForm(self.V)
        self.bf_dependent = False
        self.lf_dependent = False
        u,v = self.V.TnT()
        for integ in self.bf_integrators:
            self.bf += ngs.SymbolicBFI(**integ(u,v,self.gfdesign))
            if UsesArgument(integ,-1):
                self.bf_dependent = True
        logger.debug("BilinearForm dependent on design? " + str(self.bf_dependent))
        for integ in self.lf_integrators:
            self.lf += ngs.SymbolicLFI(**integ(v,self.gfdesign))
            if UsesArgument(integ,-1):
                self.lf_dependent = True
        logger.debug("Linearform dependent on design? " + str(self.lf_dependent))
        self.updated = True

    def Solve(self, reassemble=True):
        if not self.updated:
            self.Update()
        inv = self.GetInverse(reassemble=reassemble)
        self.lf.Assemble()
        if self.nhBND:
            if self.nhBND[1] is None:
                self.gf.Set(self.nhBND[0])
            else:
                self.gf.Set(self.nhBND[0], definedon = self.nhBND[1])
            self.lf.vec.data -= self.bf.mat * self.gf.vec
            self.gf.vec.data += self.inv * self.lf.vec
        else:
            self.gf.vec.data = self.inv * self.lf.vec
        return self.gf

    def GetInverse(self, reassemble=False):
        if reassemble or self.inv is None:
            logger.debug("Rebuild inverse of forward problem.")
            if self.solver is None:
                self.SetSolver()
            self.bf.Assemble()
            self.inv = self.solver(self.bf.mat,self.prec.mat,**self.solver_args)
        return self.inv

    def GetIntegratorsDesignEnergy(self):
        if self.bf_integrators_design:
            return self.bf_integrators_design
        else:
            return self.bf_integrators

