from numpy import linspace, maximum, minimum

from pyit2fls import IT2FS, trapezoid_mf



def evaluate_up(x, upmftype, params1, downmftype, params2):
    if upmftype == trapezoid_mf:
        x1 = trapezoid_mf(x, params1)
        print(x1)
    else: return False
    if downmftype == trapezoid_mf:
        x2 = trapezoid_mf(x, params2)
        print(x2)
    else: return False
    return (x1 + x2)/2

def main():
    myIT2FS = IT2FS(linspace(0., 255., 256), trapezoid_mf, [0, 100., 200., 225., 1.], trapezoid_mf, [50., 90., 140., 250., 0.7])
    myIT2FS.plot()
    print(evaluate_up(250, trapezoid_mf, [0. , 100. , 200. ,255. , 1.] ,trapezoid_mf, [50. , 90. , 140. ,250. , 0.7 ]))
if __name__ == '__main__':
    main()