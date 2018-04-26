
import netgen.geom2d as geom
from ngsolve import *
import ngs_topopt as to
import numpy as np

geo = geom.SplineGeometry()

ngsglobals.msg_level=0

p = []
p.append(geo.AppendPoint(0,0))
p.append(geo.AppendPoint(1,-1))
p.append(geo.AppendPoint(2,0))
p.append(geo.AppendPoint(1,1))

for i in range(4):
    geo.Append(["line", p[i], p[(i+1)%4]],bc=i+1)

mesh = Mesh(geo.GenerateMesh(maxh=0.05))

Draw(mesh)

Vu = H1(mesh, order=3)
Vp = H1(mesh, order=3)

def setFirstDof():
    for el in Vp.Elements(BND):
        Vp.FreeDofs().Clear(el.dofs[0])
        return

setFirstDof()

r = 0.1

# S = exp(-100*((x-0.1)*(x-0.1) + y*y))
S = exp(-100*((x-0.1)*(x-0.1) + y*y)) - 0.5 * exp(-100 *((x-1.2)*(x-1.2) + (y-0.3)*(y-0.3))) - \
    0.5 * exp(-100 *((x-1.2)*(x-1.2) + (y+0.3)*(y+0.3)))
av = Integrate(S,mesh)
dom_size = Integrate(CoefficientFunction(1),mesh)
print("av = ", av)
S = S - av/dom_size
print("av = ", Integrate(S,mesh))
bf_integrand = lambda _p,_v,_u: { "form" : (_u*_u + r) *grad(_p) * grad(_v)}
lf_integrand = lambda _v, _u: { "form" : S * _v }

fwproblem = to.ForwardProblem()
fwproblem.setDesignSpace(Vu,name_gfdesign="u") \
         .setStateSpace(Vp, name_gfstate="p") \
         .setInitialDesign(CoefficientFunction(1)) \
         .addBilinearFormIntegrators(bf_integrand) \
         .addLinearFormIntegrators(lf_integrand)


D = 1e-2
gamma = 0.5
c = 1e3

objective = to.ObjectiveFunction(forward_problem = fwproblem, integrators = lambda _p,_u: {"form" : D*D/2 * grad(_u) * grad(_u) + c*c/2 * _u * _u * grad(_p)*grad(_p) + c*c*r/2 * grad(_p)*grad(_p) + 1/2/gamma * IfPos(_u,_u,-_u)})

# constraints = to.Constraints(fwproblem.gfdesign)
# constraints.AddMax(integrator= lambda _u: { "form" : 100*IfPos(-_u+0.2,-_u+0.2,0) },
#                    maxvalue=0,
#                    name = "u negative")


optimizer = to.TimeSteppingOptimizer(forward_problem = fwproblem,
                                     objective_function = objective)
#                                      constraints=constraints)

Draw(fwproblem.gfdesign)
Draw(fwproblem.gf)
Draw(grad(fwproblem.gf), mesh,"grad p")
with TaskManager():
    optimizer.Optimize(dt = 1e-2, tend= 1.5) #, Redraw={"blocking" : True})
Redraw()
optimizer.PlotResults()
