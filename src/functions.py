
import logging
import ngsolve as ngs
from .utils import UsesArgument

logger = logging.getLogger("ngs_topopt.functions")

class ObjectiveFunction():
    def __init__(self, forward_problem, integrators, dfdu = None, name = "Objective"):
        self.name = name
        self.gf = forward_problem.gf
        self.gfdesign = forward_problem.gfdesign
        self.V = forward_problem.V
        self.D = forward_problem.D
        self.bf =  ngs.BilinearForm(self.V)
        u,v = self.V.TnT()
        if dfdu is not None:
            integ = dfdu(u,v,self.gfdesign)
            try:
                self.dfdu_linear = integ.pop("linear")
            except KeyError:
                self.dfdu_linear = False
            self.bfdfdu = ngs.BilinearForm(self.V, check_unused=False)
            self.bfdfdu += ngs.SymbolicBFI(**integ)
            if self.dfdu_linear:
                self.bfdfdu.Assemble()
        else:
            self.bfdfdu = None
        if type(integrators) is not list:
            self.integrators = [integrators]
        else:
            self.integrators = integrators
        if self.V.is_complex:
            self.energyargs = [integ(self.gf,self.gfdesign) for integ in self.integrators]
            for arg in self.energyargs:
                arg["cf"] = arg.pop("form")
                arg["mesh"] = self.V.mesh
        else:
            self.energyargs = None
        self.direct_dependency = False
        for integ in self.integrators:
            self.bf += ngs.SymbolicEnergy(**integ(u,self.gfdesign))
            if UsesArgument(integ,-1):
                self.direct_dependency = True
        logger.debug("Objective function direct dependency? " + str(self.direct_dependency))

    def GetValue(self):
        if self.energyargs is None:
            return self.bf.Energy(self.gf.vec)
        else:
            energy = 0
            for arg in self.energyargs:
                energy += ngs.Integrate(**arg)
            return energy
    def GetDValue(self,vec):
        if self.bfdfdu is None:
            self.bf.Apply(self.gf.vec,vec)
        else:
            if self.dfdu_linear:
                vec.data = self.bfdfdu.mat * self.gf.vec
            else:
                self.bfdfdu.Apply(self.gf.vec,vec)

class Constraints():
    def __init__(self,gfdesign):
        self.space = gfdesign.space
        self.gfdesign = gfdesign
        self.bfs = []
        self.maxvalues = []
        self.values = []
        self.dvalues = []
        self.names = []

    def AddMax(self, integrator, maxvalue, name = None):
        if name is None:
            self.names.append("Constraint " + str(len(self.names)+1))
        else:
            self.names.append(name)
        bf = ngs.BilinearForm(self.space)
        phi = self.space.TrialFunction()
        bf += ngs.SymbolicEnergy(**integrator(phi))
        self.maxvalues.append(maxvalue)
        self.bfs.append(bf)
        self.dvalues.append(self.gfdesign.vec.CreateVector())

    def AddMin(self,integrator, minvalue, name = None):
        if name is None:
            self.names.append("Constraint " + str(len(self.names)+1))
        else:
            self.names.append(name)
        bf = ngs.BilinearForm(self.space)
        phi = self.space.TrialFunction()
        ig = integrator(phi)
        form = ig.pop("form")
        bf += ngs.SymbolicEnergy((-1) * form, **ig)
        self.maxvalues.append((-1) * minvalue)
        self.bfs.append(bf)
        self.dvalues.append(self.gfdesign.vec.CreateVector())

    def Update(self):
        self.values = []
        for i,(maxval, bf) in enumerate(zip(self.maxvalues,self.bfs)):
            self.values.append(bf.Energy(self.gfdesign.vec) - maxval)
            bf.Apply(self.gfdesign.vec,self.dvalues[i])

    def __len__(self):
        return len(self.bfs)
