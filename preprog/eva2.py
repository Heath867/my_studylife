from numpy import linspace, maximum, minimum

from pyit2fls import IT2FS, trapezoid_mf



def evaluate_up(x, upmftype, params1, downmftype, params2):
    if upmftype == trapezoid_mf:
        x1 = trapezoid_mf(x, params1)
        # print(x1)
    else: return False
    if downmftype == trapezoid_mf:
        x2 = trapezoid_mf(x, params2)
        # print(x2)
    else: return False
    return (x1 + x2)/2

def eval(A,x):
    # A = [50,120,60,155,210,220,90,190]
    B = sorted(A)
    p1 = []
    p2 = []
    for i in range(4) :
        if i %2 == 0 :
            p1.append(B[i])
            p1.append(B[7-i])
        else:
            p2.append(B[i])
            p2.append(B[7-i])

    # print(B)
    # print("p1:{}".format(p1),"p2:{}".format(p2))
    p1 = sorted(p1)
    p2 = sorted(p2)
    p1.append(1)
    p2.append(0.7)
    # print(p1)
    # print(p2)
    # myIT2FS = IT2FS(linspace(0., 255., 256), trapezoid_mf, p1, trapezoid_mf, p2)
    # myIT2FS.plot()
    return evaluate_up(x, trapezoid_mf, p1 ,trapezoid_mf, p2)

def main():
    A = [50, 120, 60, 155, 210, 220, 90, 190]
    print(eval(A,177))

if __name__ == '__main__':
    main()