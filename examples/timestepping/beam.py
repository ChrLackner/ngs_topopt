
import netgen.geom2d as geom
from ngsolve import *
import ngs_topopt as to
import numpy as np

# from ngs_topopt.log import ch
# import logging
# ch.setLevel(logging.DEBUG)

def CreateMesh():
    geo = geom.SplineGeometry()
    pnts = [(0,0),(2,0),(2,0.45),(2,0.55),(2,1),(0,1),(0,0.55),(0,0.45)]
    gpts = [geo.AppendPoint(*p) for p in pnts]
    segs = [(0,1,"free"),(1,2,"free"),(2,3,"load"),(3,4,"free"),(4,5,"free"), (5,6,"fixed"),(6,7,"fixed"),(7,0,"fixed")]
    for p1,p2,bc in segs:
        geo.Append(["line",gpts[p1],gpts[p2]],bc=bc)
    return Mesh(geo.GenerateMesh(maxh=0.02))

mesh = CreateMesh()

# Parameters
d = 0.01
h = 1.0

cft = CoefficientFunction((0,-1))

epsilon = lambda _u: 0.5 * (grad(_u) + grad(_u).trans)


D = H1(mesh,order=1)
V = H1(mesh,order=1,dirichlet="fixed", dim=mesh.dim)

# self defined mypow, because NGSolve one doesn't know how to be deriven (but multiplication is ok)
def mypow(cf,i):
    result = CoefficientFunction(1)
    for j in range(i):
        result *= cf
    return result

H = lambda _phi: 0.5 + 15.0/16.0 * _phi/h - 5.0/8.0 * mypow(_phi/h,3) + 3.0/16.0 * mypow(_phi/h,5)

def rho(_phi):
    return IfPos(h+_phi,
                 IfPos(h-_phi,
                       CoefficientFunction((1-d)*H(_phi)+d),
                       CoefficientFunction(1)),
                 CoefficientFunction(d))


def sigma(u,_phi):
    return lam(_phi)*Trace(epsilon(u))*I + 2.0*mu(_phi)*epsilon(u)

p = 3
E0 = 10
E = lambda _phi: E0 * mypow(rho(_phi),p)
nu = 0.3
mu  = lambda _phi: E(_phi)/2/(1+nu)
lam = lambda _phi: E(_phi)*nu/((1+nu)*(1-2*nu))
I = Id(mesh.dim)


bf_integrand = lambda _u,_v,_phi: { "form" : 0.5 * InnerProduct(sigma(_u,_phi),epsilon(_v)) }
lf_integrand = lambda _v, _phi: { "form" : InnerProduct(cft,_v),
                                  "definedon" : mesh.Boundaries("load") }


forward_problem = to.ForwardProblem()
forward_problem.setDesignSpace(D) \
               .setStateSpace(V) \
               .setInitialDesign(CoefficientFunction(sin(3*np.pi*x)*sin(3*np.pi*y))) \
               .addBilinearFormIntegrators(bf_integrand) \
               .addLinearFormIntegrators(lf_integrand)


gfphi = forward_problem.gfdesign
Draw(forward_problem.gf)
Draw(gfphi)
Draw(rho(gfphi),mesh,"rho")
SetVisualization(deformation=True)
import ngsolve.internal as inter
inter.visoptions.scaledeform2 = 0.05

volume = Integrate(CoefficientFunction(1),mesh)
constraints = to.Constraints(gfphi)
constraints.AddMax(integrator = lambda _phi: { "form" : 1./volume * rho(_phi) },
                   maxvalue = 0.5,
                   name="Volume Constraint")

objective = to.ObjectiveFunction(forward_problem = forward_problem,
                                 integrators = lambda _u, _phi:
                                 { "form" : 0.5 * InnerProduct(sigma(_u,_phi),epsilon(_u)) },
                                 name = "Stiffness")
# alternative objective function:
# objective = to.ObjectiveFunction(forward_problem = forward_problem,
#                                  integrators = lambda _u, _phi:
#                                  {"form" : InnerProduct(cft,_u),
#                                   "definedon" : mesh.Boundaries("load") },
#                                  name = "Stiffness")

optimizer = to.TimeSteppingOptimizer(forward_problem = forward_problem,
                                     objective_function = objective,
                                     constraints = constraints)
with TaskManager():
    optimizer.Optimize(dt=0.02,tend=6,Redraw={"blocking" : True})

if __name__ == "__main__":
    optimizer.PlotResults()
