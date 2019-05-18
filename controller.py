import math

from scipy.interpolate import splrep, splev
from scipy.integrate import odeint

import numpy as np
from numpy.linalg import norm, inv

from pydrake.all import VectorSystem, Linearize
from pydrake.all import MathematicalProgram, Solve, Variables, Polynomial, MosekSolver, Expression


def rotmat2rpy(R):
    # assumes we're using an X-Y-Z convention to construct R
    #   rpy = [atan2(R(3,2),R(3,3)); ...
    #       atan2(-R(3,1),sqrt(R(3,2)^2 + R(3,3)^2)); ...
    #       atan2(R(2,1),R(1,1)) ];    
    return np.array([np.arctan2(R[2,1],  R[2,2]),
                     np.arctan2(-R[2,0], np.sqrt(R[2,1]**2+R[2,2]**2)),
                     np.arctan2(R[1,0],  R[0,0])
                     ]) # column vector...

def angularvel2rpydotMatrix(rpy):
    # p = rpy(2); y = rpy(3);
    # sy = sin(y); cy = cos(y); sp = sin(p); cp = cos(p); tp = sp / cp;
    # Phi = ...
    #   [cy/cp, sy/cp, 0; ...
    #   -sy,       cy, 0; ...
    #    cy*tp, tp*sy, 1];
    p, y = rpy[1], rpy[2]
    sy, cy, sp, cp = np.sin(y), np.cos(y), np.sin(p), np.cos(p)
    tp = sp / cp
    return np.array([[cy/cp, sy/cp, 0],
                     [-sy,      cy, 0],
                     [cy*tp, tp*sy, 1]
                    ])

def angularvel2rpydot(rpy, omega):
    return angularvel2rpydotMatrix(rpy).dot(omega)

def find_nearest(a, a0):
    "Element in nd array `a` closest to the scalar value `a0`"
    idx = np.abs(a[a<=a0] - a0).argmin()
    return idx

class CubicSpline():
    def __init__(self, ts, outs):
        self.n_out = outs.shape[1]
        self.splines = []
        for idx in range(self.n_out):
            spline = splrep(ts, outs[:, idx])
            self.splines.append(spline)

    def eval(self, t, der):
        out = np.array([
            splev(t, self.splines[idx], der=der) for idx in range(self.n_out)
        ])
        return out

class QuadrotorController(VectorSystem):
    traj_freq = 200
    eps = 1e-8

    def __init__(self, plant, y_traj, duration):
        n_in = 12
        n_out = 4
        VectorSystem.__init__(self, 12, 4)
        self.plant = plant
        # Find x_traj, u_traj @ traj_freq
        t_traj = np.linspace(0, duration, np.ceil(duration*QuadrotorController.traj_freq))
        x_traj, u_traj = np.zeros((len(t_traj),n_in)), np.zeros((len(t_traj), n_out))
        for idx, t in enumerate(t_traj):
            x_traj[idx, :], u_traj[idx, :] = self.get_x_and_u(plant, y_traj, t)
        # Create spline for x_traj, u_traj
        x_spline, u_spline = CubicSpline(t_traj, x_traj), CubicSpline(t_traj,   u_traj)
        # Give the spline to TVLQR
        K_traj = self.tvlqr(plant, x_spline, u_spline, t_traj)
        K_spline = CubicSpline(t_traj, np.reshape(K_traj, (K_traj.shape[0], -1)))
        # Save the TVLQR; use it in DoCalcVectorOutput
        #     TVLQR gives K(t), x0(t), u0(t)
        #     Want to return u^*=u0(T)-K(T)(x(t)-x0(T)) where T is that which is closest to t
        self.x_spline = x_spline
        self.u_spline = u_spline
        self.K_spline = K_spline

    def DoCalcVectorOutput(self, context, u, x, y):
        # u is ouput from plant (the quadrotor state)
        # x is the state of this controller? useless...
        # y is input into plant (the quadrotor control input)
        t = context.get_time()

        # nearest_idx = find_nearest(self.t_traj, t)
        # print("delta", self.t_traj[nearest_idx]-t, t)
        # y[:] = self.u_traj[find_nearest(self.t_traj, t)]

        # y[:] = np.array([1.23,1.23,1.23,1.23])

        # u^*=u0(T)-K(T)(x(t)-x0(T))
        u0 = self.u_spline.eval(t, 0)
        x0 = self.x_spline.eval(t,0 )
        x = u
        K = np.reshape(self.K_spline.eval(t, 0), (4,12))
        y[:] = u0-K.dot(x-x0)

    def get_x_and_u(self, plant, y_traj, t):
        # Set up constants
        g = np.array([0,0,-plant.g()])
        m = 0.5
        L = 0.175
        I = np.array([
            [0.0023, 0, 0],
            [0, 0.0023, 0],
            [0, 0, 0.0040]
        ])
        kF = 1.0
        kM = 0.0245
        # Set up derived constants...
        zW = g/norm(g)
        y = np.append(y_traj.eval(t, derivative_order=0), 0)
        yd = np.append(y_traj.eval(t, derivative_order=1), 0)
        ydd = np.append(y_traj.eval(t, derivative_order=2), 0)
        yddd = np.append(y_traj.eval(t, derivative_order=3), 0)
        ydddd = np.append(y_traj.eval(t, derivative_order=4), 0)
        a = ydd[:3]-g
        ad = yddd[:3]
        add = ydddd[:3]
        u1 = m*norm(a)
        u1d = m*ad.dot(a)/norm(a)
        u1dd = m*((add.dot(a)+ad.dot(ad))*norm(a)-(ad.dot(a))*(ad.dot(a))/norm(a))/norm(a)**2

        if norm(a) < QuadrotorController.eps:
            zB = np.array([0,0,1])
            zBd = np.array([0,0,0])
        else:
            zB = a/norm(a)
            zBd = (ad*norm(a)-a*(ad.dot(a))/norm(a))/norm(a)**2
        xC = np.array([np.cos(y[-1]), np.sin([-1]), 0])
        xCd = np.array([-np.sin(y[-1])*yd[-1], np.cos(y[-1])*yd[-1], 0])
        yB = np.cross(zB,xC);
        yBd = np.cross(zBd,xC)+np.cross(zB,xCd)
        yBd = (yBd*norm(yB)-yB*((yBd.dot(yB))/norm(yB)))/norm(yB)**2
        yB = yB/norm(yB)
        xB = np.cross(yB,zB)
        xBd = np.cross(yBd,zB)+np.cross(yB,zBd)

        R = np.vstack((xB,yB,zB)).T
        x = np.vstack((y[:3], rotmat2rpy(R), yd[:3], np.zeros(3)))

        h_omega = (1/u1)*(m*ad-u1d*zB)
        omegaBW = np.array([ -h_omega.dot(yB), h_omega.dot(xB), yd[-1]*zW.dot(zB) ])

        x[-1,:] = angularvel2rpydot(x[1,:], omegaBW)

        h_omegad = (m*add*u1-m*ad*u1d-(u1dd*zB+u1d*zBd)*u1+u1d*zB*u1d)/u1**2;
        omegadotBW = np.array([ -h_omegad.dot(yB)-h_omega.dot(yBd), h_omegad.dot(xB)+h_omega.dot(xBd), ydd[-1]*zW.dot(zB) ])

        mellinger_u = np.hstack((u1, I.dot(omegadotBW) + np.cross(omegaBW,I.dot(omegaBW))))

        omega_squared_to_u = np.array([[kF,    kF,   kF,   kF],
                                       [0,     kF*L, 0,    -kF*L],
                                       [-kF*L, 0,    kF*L, 0],
                                       [kM,    -kM,  kM,   -kM]])
        omega_squared = inv(omega_squared_to_u).dot(mellinger_u)

        return x.flatten(), omega_squared


    def tvlqr(self, plant, x_traj, u_traj, t_traj):
        # Give the spline to TVLQR
        #     ts from x_traj, u_traj
        #     using Linearize (see predefined_traj), compute A(t), B(t)
        #     using ode solver, solve diff ricati eq
        #     hmmm maybe we don't even need the spline :oo to get S(t)
        #     then compute K(t)
        L = len(t_traj)
        n_in = 12
        n_out = 4
        Q = np.identity(n_in)
        R = np.identity(n_out)
        Qf = 10*np.identity(n_in)
        def get_A_B(t):
            context = plant.CreateDefaultContext()
            context.SetContinuousState(x_traj.eval(t, 0))
            context.FixInputPort(0, u_traj.eval(t, 0))
            linear_system = Linearize(plant, context, equilibrium_check_tolerance=np.inf)
            return linear_system.A(), linear_system.B()
        def dynamics(S, t):
            S = np.reshape(S, (n_in, n_in))
            A, B = get_A_B(t)
            Sd = -(S.dot(A)+A.T.dot(S) -S.T.dot(B).dot(inv(R)).dot(B.T).dot(S)+Q)
            return Sd.flatten()
        # Begin the integration
        S_traj = odeint(dynamics, Qf.flatten(), t_traj[::-1])[::-1] # solve backwards and flip order
        S_traj = np.reshape(S_traj, (L, n_in, n_in))
        B_traj = np.array([get_A_B(t)[1] for t in t_traj])
        K_traj = np.matmul(inv(R), np.matmul(np.swapaxes(B_traj,1,2),S_traj))
        return K_traj

class PPTrajectory():
    def __init__(self, sample_times, num_vars, degree, continuity_degree):
        self.sample_times = sample_times
        self.n = num_vars
        self.degree = degree

        self.prog = MathematicalProgram()
        self.coeffs = []
        for i in range(len(sample_times)):
            self.coeffs.append(self.prog.NewContinuousVariables(
                num_vars, degree+1, "C"))
        self.result = None

        # Add continuity constraints
        for s in range(len(sample_times)-1):
            trel = sample_times[s+1]-sample_times[s]
            coeffs = self.coeffs[s]
            for var in range(self.n):
                for deg in range(continuity_degree+1):
                    # Don't use eval here, because I want left and right
                    # values of the same time
                    left_val = 0
                    for d in range(deg, self.degree+1):
                        left_val += coeffs[var, d]*np.power(trel, d-deg) * \
                               math.factorial(d)/math.factorial(d-deg)
                    right_val = self.coeffs[s+1][var, deg]*math.factorial(deg)
                    self.prog.AddLinearConstraint(left_val == right_val)

        # Add cost to minimize highest order terms
        for s in range(len(sample_times)-1):
            self.prog.AddQuadraticCost(np.eye(num_vars),
                                       np.zeros((num_vars, 1)),
                                       self.coeffs[s][:, -1])

    def eval(self, t, derivative_order=0):
        if derivative_order > self.degree:
            return 0

        s = 0
        while s < len(self.sample_times)-1 and t >= self.sample_times[s+1]:
            s += 1
        trel = t - self.sample_times[s]

        if self.result is None:
            coeffs = self.coeffs[s]
        else:
            coeffs = self.result.GetSolution(self.coeffs[s])

        deg = derivative_order
        val = 0*coeffs[:, 0]
        for var in range(self.n):
            for d in range(deg, self.degree+1):
                val[var] += coeffs[var, d]*np.power(trel, d-deg) * \
                       math.factorial(d)/math.factorial(d-deg)

        return val

    def add_constraint(self, t, derivative_order, lb, ub=None):
        '''
        Adds a constraint of the form d^deg lb <= x(t) / dt^deg <= ub
        '''
        if ub is None:
            ub = lb

        assert(derivative_order <= self.degree)
        val = self.eval(t, derivative_order)
        self.prog.AddLinearConstraint(val, lb, ub)

    def generate(self):
        self.result = Solve(self.prog)
        assert(self.result.is_success())

def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    theta = theta*math.pi/180.0
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])


class FullPPTrajectory():
    # clearly continuity degree is degree+1
    # why do we need dt?
    def __init__(self, start, goal, degree, n_segments, duration, regions, H): #sample_times, num_vars, continuity_degree):
        R = len(regions)
        N = n_segments
        M = 100
        prog = MathematicalProgram()
        
        # Set H
        if H is None:
            H = prog.NewBinaryVariables(R, N)
            for n_idx in range(N):
                prog.AddLinearConstraint(np.sum(H[:,n_idx])==1)

        poly_xs, poly_ys, poly_zs = [], [], []
        vars_ = np.zeros((N,)).astype(object)
        for n_idx in range(N):
            # for each segment
            t = prog.NewIndeterminates(1, "t"+str(n_idx))
            vars_[n_idx] = t[0]
            poly_x = prog.NewFreePolynomial(Variables(t), degree, "c_x_"+str(n_idx))
            poly_y = prog.NewFreePolynomial(Variables(t), degree, "c_y_"+str(n_idx))
            poly_z = prog.NewFreePolynomial(Variables(t), degree, "c_z_"+str(n_idx))
            for var_ in poly_x.decision_variables():
                prog.AddLinearConstraint(var_<=M)
                prog.AddLinearConstraint(var_>=-M)
            for var_ in poly_y.decision_variables():
                prog.AddLinearConstraint(var_<=M)
                prog.AddLinearConstraint(var_>=-M)
            for var_ in poly_z.decision_variables():
                prog.AddLinearConstraint(var_<=M)
                prog.AddLinearConstraint(var_>=-M)
            poly_xs.append(poly_x)
            poly_ys.append(poly_y)
            poly_zs.append(poly_z)
            phi = np.array([poly_x, poly_y, poly_z])
            for r_idx, region in enumerate(regions):
                # if r_idx == 0:
                #     break
                A = region[0]
                b = region[1]
                b = b + (1-H[r_idx, n_idx])*M
                b = [Polynomial(this_b) for this_b in b]
                q = b-A.dot(phi)
                sigma = []
                for q_idx in range(len(q)):
                    sigma_1 = prog.NewFreePolynomial(Variables(t), degree-1)
                    prog.AddSosConstraint(sigma_1)
                    sigma_2 = prog.NewFreePolynomial(Variables(t), degree-1)
                    prog.AddSosConstraint(sigma_2)
                    sigma.append(Polynomial(t[0])*sigma_1+(1-Polynomial(t[0]))*sigma_2)
                    # for var_ in sigma[q_idx].decision_variables():
                    #     prog.AddLinearConstraint(var_<=M)
                    #     prog.AddLinearConstraint(var_>=-M)
                    q_coeffs = q[q_idx].monomial_to_coefficient_map()
                    sigma_coeffs = sigma[q_idx].monomial_to_coefficient_map()
                    both_coeffs = set(q_coeffs.keys()) & set(sigma_coeffs.keys())
                    for coeff in both_coeffs:
                        # import pdb; pdb.set_trace()
                        prog.AddConstraint(q_coeffs[coeff] == sigma_coeffs[coeff])
                # res = Solve(prog)
                # print("x: " + str(res.GetSolution(poly_xs[0].ToExpression())))
                # print("y: " + str(res.GetSolution(poly_ys[0].ToExpression())))
                # print("z: " + str(res.GetSolution(poly_zs[0].ToExpression())))
                # import pdb; pdb.set_trace()
                # for this_q in q:
                #     prog.AddSosConstraint(this_q)
                # import pdb; pdb.set_trace()

        # cost = 0
        print("Constraint: x0(0)=x0")
        prog.AddConstraint(poly_xs[0].ToExpression().Substitute(vars_[0],0.0)==start[0])
        prog.AddConstraint(poly_ys[0].ToExpression().Substitute(vars_[0],0.0)==start[1])
        prog.AddConstraint(poly_zs[0].ToExpression().Substitute(vars_[0],0.0)==start[2])
        for idx, poly_x, poly_y, poly_z in zip(range(N), poly_xs, poly_ys, poly_zs):
            if idx < N-1:
                print("Constraint: x"+str(idx)+"(1)=x"+str(idx+1)+"(0)")
                next_poly_x, next_poly_y, next_poly_z = poly_xs[idx+1], poly_ys[idx+1], poly_zs[idx+1]
                prog.AddConstraint(poly_x.ToExpression().Substitute(vars_[idx],1.0)==next_poly_x.ToExpression().Substitute(vars_[idx+1],0.0))
                prog.AddConstraint(poly_y.ToExpression().Substitute(vars_[idx],1.0)==next_poly_y.ToExpression().Substitute(vars_[idx+1],0.0))
                prog.AddConstraint(poly_z.ToExpression().Substitute(vars_[idx],1.0)==next_poly_z.ToExpression().Substitute(vars_[idx+1],0.0))
            else:
                print("Constraint: x"+str(idx)+"(1)=xf")
                prog.AddConstraint(poly_x.ToExpression().Substitute(vars_[idx],1.0)==goal[0])
                prog.AddConstraint(poly_y.ToExpression().Substitute(vars_[idx],1.0)==goal[1])
                prog.AddConstraint(poly_z.ToExpression().Substitute(vars_[idx],1.0)==goal[2])

            # for 

        cost = Expression()
        for var_, polys in zip(vars_, [poly_xs, poly_ys, poly_zs]):
            for poly in polys:
                for _ in range(2):#range(2):
                    poly = poly.Differentiate(var_)
                poly = poly.ToExpression().Substitute({var_: 1.0})
                cost += poly**2
                # for dv in poly.decision_variables():
                #     cost += (Expression(dv))**2
        prog.AddCost(cost)

        res = Solve(prog)

        print("x: " + str(res.GetSolution(poly_xs[0].ToExpression())))
        print("y: " + str(res.GetSolution(poly_ys[0].ToExpression())))
        print("z: " + str(res.GetSolution(poly_zs[0].ToExpression())))

        self.poly_xs = [res.GetSolution(poly_x.ToExpression()) for poly_x in poly_xs]
        self.poly_ys = [res.GetSolution(poly_y.ToExpression()) for poly_y in poly_ys]
        self.poly_zs = [res.GetSolution(poly_z.ToExpression()) for poly_z in poly_zs]
        self.vars_ = vars_
        self.degree = degree

        # for name, poly in zip(['x','y','z'])
        # d_vars = poly_xs[0].decision_variables()
        # for d_var in d_vars:
        #     print(res.GetSolution(d_var))
        # import pdb; pdb.set_trace()

        # self.coeffs = []
        # for i in range(len(sample_times)):
        #     self.coeffs.append(self.prog.NewContinuousVariables(
        #         num_vars, degree+1, "C"))
        # self.result = None

        # # Add continuity constraints
        # for s in range(len(sample_times)-1):
        #     trel = sample_times[s+1]-sample_times[s]
        #     coeffs = self.coeffs[s]
        #     for var in range(self.n):
        #         for deg in range(continuity_degree+1):
        #             # Don't use eval here, because I want left and right
        #             # values of the same time
        #             left_val = 0
        #             for d in range(deg, self.degree+1):
        #                 left_val += coeffs[var, d]*np.power(trel, d-deg) * \
        #                        math.factorial(d)/math.factorial(d-deg)
        #             right_val = self.coeffs[s+1][var, deg]*math.factorial(deg)
        #             self.prog.AddLinearConstraint(left_val == right_val)

        # # Add cost to minimize highest order terms
        # for s in range(len(sample_times)-1):
        #     self.prog.AddQuadraticCost(np.eye(num_vars),
        #                                np.zeros((num_vars, 1)),
        #                                self.coeffs[s][:, -1])

    def eval(self, t, derivative_order=0):
        if derivative_order > self.degree:
            return 0
        if len(self.vars_) < t:
            idx = int(np.floor(t))
            t -= idx
        else:
            idx = len(self.vars_)-1
            t = 1.0
        polys = []
        var_ = self.vars_[idx]
        for poly in [self.poly_xs[idx], self.poly_ys[idx], self.poly_zs[idx]]:
            for _ in range(derivative_order):
                poly = poly.Differentiate(var_)
            polys.append(poly)
        return np.array([
            polys[0].Evaluate({var_: t}),
            polys[1].Evaluate({var_: t}),
            polys[2].Evaluate({var_: t})
        ])
