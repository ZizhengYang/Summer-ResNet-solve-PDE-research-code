def gen_analytical_solu(delta_x=1/20, delta_t=1/20, xmin=0, tmin=0, xmax=2 * np.pi, tmax=2 * np.pi, analytical_eq=PDE_analytical_solu):
    x = arange(xmin, xmax, delta_x)
    t = arange(tmin, tmax, delta_t)
    X,T = meshgrid(x, t) # grid of point
    solu = analytical_eq(X, T) # evaluation of the function on the grid
    return x, t, solu

def gen_discrete_average_solu(delta_x=1/20, delta_t=1/20, xmin=0, tmin=0, xmax=2 * np.pi, tmax=2 * np.pi, analytical_eq=PDE_analytical_solu):
    x = arange(xmin, xmax, delta_x)
    t = arange(tmin, tmax, delta_t)
    X,T = meshgrid(x, t) # grid of point
    Z = analytical_eq(X, T) # evaluation of the function on the grid
    solu = []
    for zz in Z:
        solu_t = []
        for j in range(len(zz)-1):
            value = (1/2) * (zz[j] + zz[j+1])
            solu_t.append(value)
        solu.append(solu_t)
    return x, t, solu

def gen_cell_average_solu(delta_x=1/20, delta_t=1/20, xmin=0, tmin=0, xmax=2 * np.pi, tmax=2 * np.pi, analytical_eq=PDE_analytical_solu):
    x = arange(xmin, xmax, delta_x)
    t = arange(tmin, tmax, delta_t)
    solu = []
    for ti in range(len(t)):
        solu_t = []
        for j in range(len(x)-1):
            value = integrate.quad(lambda x: analytical_eq(x, t[ti]), x[j], x[j+1])
            value = value[0] * (1/delta_x)
            solu_t.append(value)
        solu.append(solu_t)
    return x, t, solu