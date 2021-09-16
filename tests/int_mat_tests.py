import numpy as np
import pfapack.pfaffian as pf


def int_rand_mat(n):
    A=np.matrix(list(map(np.vectorize(round), np.random.rand(n,n)*10)))
    return A - A.T

def float_rand_mat(n):
    A = np.random.rand(n,n)
    return A - A.T


def complex_rand_mat(n):
    A = np.random.rand(n,n) + np.random.rand(n,n)*1j
    return A - A.T


def int_rand_mat2(n):
    A=np.matrix(list(map(np.around, np.random.rand(n,n)*10)))
    return A - A.T

nmax = 200
dn = 20
nmin = 40
print("Testing integer matrices...")
for n in np.arange(nmin, nmax, dn):
    print("n=%i" % (n))
    A = int_rand_mat(n)
    H = pf.pfaffian(A, method="H")
    P = pf.pfaffian(A, method="P")
    da = np.linalg.det(A)
    #pritn("pfaff(A)=%i" % (da))
    print("Method H: ", np.log10(H ** 2) - np.log10(da))
    print("Method P: ", np.log10(P ** 2) - np.log10(da))
    print("\n")

print("Testing real matrices...")
for n in np.arange(nmin, nmax, dn):
    print("n=%i \n" % (n))
    A = float_rand_mat(n)
    H = pf.pfaffian(A, method="H")
    P = pf.pfaffian(A, method="P")
    da = np.linalg.det(A)
    # pritn("pfaff(A)=%i" % (da))
    print("Method H: ", np.log10(H ** 2) - np.log10(da))
    print("Method P: ", np.log10(P ** 2) - np.log10(da))
    print("\n")


print("Testing complex matrices...")
for n in np.arange(nmin, nmax, dn):
    print("n=%i \n"%(n))
    A = complex_rand_mat(n)
    H = pf.pfaffian(A, method="H")
    P = pf.pfaffian(A, method="P")
    da = np.linalg.det(A)
    # pritn("pfaff(A)=%i" % (da))
    print("Method H: ", np.log10(H ** 2) - np.log10(da))
    print("Method P: ", np.log10(P ** 2) - np.log10(da))
    print("\n")

