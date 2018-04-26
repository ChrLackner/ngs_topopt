
import netgen.geom2d as geom
import numpy as np
from ngsolve import *
import ngs_topopt as to


def CreateMesh():
    geo = geom.SplineGeometry()
    geom.MakeCircle(geo,(0,0),0.4,bc='outer')
    geom.MakeCircle(geo,(0,0),0.3,leftdomain=2,rightdomain=1)
    geom.MakeCircle(geo,(0,0.2),0.015,leftdomain=0,rightdomain=2,bc='source')
    geom.MakeCircle(geo,(0,-0.2),0.015,leftdomain=3,rightdomain=2)
    geom.MakeRectangle(geo,(-0.25,-0.05),(0.25,0.05),leftdomain=4,rightdomain=2)
    meshsize_box = 0.005
    geo.SetDomainMaxH(4,meshsize_box)
    nmesh = geo.GenerateMesh(maxh=0.01)
    nmesh.SetMaterial(1,'pml')
    nmesh.SetMaterial(2,'air')
    nmesh.SetMaterial(3,'drain')
    nmesh.SetMaterial(4,'box')
    mesh = Mesh(nmesh)
    mesh.Curve(5)
    return mesh

mesh = CreateMesh()

radpml = pml.Radial(origin=(0,0),rad=0.3,alpha=0.3J)
mesh.SetPML(radpml,definedon='pml')

D = H1(mesh,order=2,definedon='box')
V = H1(mesh,order=3,complex=True,dirichlet='source|outer')


def mypow(cf,i):
    result = CoefficientFunction(1)
    for j in range(i):
        result *= cf
    return result

d = 0.01
h = 1
H = lambda _phi: 0.5 + 15.0/16.0 * _phi/h - 5.0/8.0 * mypow(_phi/h,3) + 3.0/16.0 * mypow(_phi/h,5)
def rho(_phi):
    return IfPos(h+_phi,
                 IfPos(h-_phi,
                       CoefficientFunction((1-d)*H(_phi)+d),
                       CoefficientFunction(1)),
                 CoefficientFunction(d))


epsr_vals = lambda _phi: {'box' : 1+1.2*rho(_phi),
                          'air' : 1,
                          'pml' : 1,
                          'drain' : 1}
eps0 = 8.85e-12
def eps(_phi):
    return CoefficientFunction([eps0*epsr_vals(_phi)[mat] for mat in mesh.GetMaterials()])

k = 2*3.1415*5e9

mu0 = 1.257e-6
mu = CoefficientFunction(mu0)

bf_integrand = lambda _u,_v,_phi: { "form" : 1./mu * grad(_u)*grad(_v) - eps(_phi)*k*k* _u * _v }
bf_integrand_real = lambda _u,_v,_phi: { "form" : 1./mu * (grad(_u)*grad(_v)).real - eps(_phi)*k*k* (_u * _v).real }

E0 = CoefficientFunction(1)

fwproblem = to.ForwardProblem()
fwproblem.setDesignSpace(D) \
         .setStateSpace(V) \
         .setInitialDesign(sin(3*np.pi*10*(x-0.25))*sin(3*10*np.pi*(y-0.05)),mesh.Materials("box")) \
         .addBilinearFormIntegrators(bf_integrand) \
         .setDesignIntegrators(bf_integrand_real) \
         .setNonhomogeneousBND(cf=E0,region=mesh.Boundaries("source"))


cfrho = CoefficientFunction([rho(fwproblem.gfdesign) if mat=='box' else 0 for mat in mesh.GetMaterials()])

Draw(fwproblem.gf)
Draw(fwproblem.gfdesign)
Draw(cfrho,mesh,'rho')
Draw(CoefficientFunction([cfrho if mat=='box' else fwproblem.gf for mat in mesh.GetMaterials()]),mesh,'rho_and_E')


scaling = 1./Integrate(CoefficientFunction(1),mesh,definedon=mesh.Materials("drain"))
objective = to.ObjectiveFunction(forward_problem = fwproblem,
                                 integrators = lambda _u,_phi: {"form" : -scaling * (_u * Conj(_u)).real,
                                                                "definedon" : mesh.Materials("drain")},
                                 dfdu = lambda _u,_v,_phi: { "form" : -2*scaling * _u * _v,
                                                             "definedon" : mesh.Materials("drain"),
                                                             "linear" : True })

optimizer = to.TimeSteppingOptimizer(forward_problem = fwproblem,
                                     objective_function = objective)

with TaskManager():
    optimizer.Optimize(dt = 1e-3)

if __name__ == "__main__":
    optimizer.PlotResults()
