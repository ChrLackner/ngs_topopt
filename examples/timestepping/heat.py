
from netgen.geom2d import SplineGeometry,MakeRectangle
from ngsolve import *
import numpy as np
import ngs_topopt as to


def CreateMesh():
    geo = SplineGeometry()
    MakeRectangle(geo,(0,0),(7,4),bc="outer")
    MakeRectangle(geo,(1,1),(2,3),leftdomain=2,rightdomain=1)
    MakeRectangle(geo,(5,1.5),(6,2.5),leftdomain=3,rightdomain=1)
    MakeRectangle(geo,(2.1,1.8),(4,2.2),leftdomain=4,rightdomain=1)
    nmesh = geo.GenerateMesh(maxh=0.05)
    nmesh.SetMaterial(1,"air")
    nmesh.SetMaterial(2,"source")
    nmesh.SetMaterial(3,"drain")
    nmesh.SetMaterial(4,"box")
    return Mesh(nmesh)

mesh = CreateMesh()

D = H1(mesh,order=2,definedon="source")
V = H1(mesh,order=3,dirichlet="outer")

initD = CoefficientFunction(10*sin(3*np.pi*x)*sin(3*np.pi*x)*sin(3*np.pi*y)*sin(3*np.pi*y))

alpha_val = { "air" : 1,
              "source" : 1,
              "drain" : 1e3,
              "box" : 1e-7}

alpha = CoefficientFunction([alpha_val[mat] for mat in mesh.GetMaterials()])
bf_integrand = lambda _u,_v,_phi: { "form" : alpha * grad(_u) * grad(_v) }
lf_integrand = lambda _v, _phi: { "form" : _phi * _v, "definedon" : mesh.Materials("source") }

fwproblem = to.ForwardProblem()
fwproblem.setDesignSpace(D) \
         .setStateSpace(V) \
         .setInitialDesign(initD) \
         .addBilinearFormIntegrators(bf_integrand) \
         .addLinearFormIntegrators(lf_integrand)

Draw(fwproblem.gfdesign)
Draw(fwproblem.gf)
Draw(CoefficientFunction([fwproblem.gfdesign if mat == "source" else fwproblem.gf for mat in mesh.GetMaterials()]),mesh,"sol")

vol_source = Integrate(CoefficientFunction(1),mesh,definedon=mesh.Materials("source"))
constr = to.Constraints(fwproblem.gfdesign)
constr.AddMax(integrator = lambda _phi: { "form" : 1./vol_source * sqrt(_phi * _phi),
                                          "definedon" : mesh.Materials("source") },
              maxvalue = 2)

objective = to.ObjectiveFunction(forward_problem = fwproblem,
                                 integrators = [lambda _u, _phi:
                                                { "form" : -_u,
                                                  "definedon" : mesh.Materials("drain") }])

optimizer = to.TimeSteppingOptimizer(forward_problem = fwproblem,
                                     objective_function = objective,
                                     constraints = constr)
with TaskManager():
    optimizer.Optimize(tend = 10)

if __name__ == "__main__":
    optimizer.PlotResults()
