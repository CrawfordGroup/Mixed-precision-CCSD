"""
Script to compute the electronic correlation energy using
coupled-cluster theory through single and double excitations,
from a RHF reference wavefunction.
References:
- Algorithms from Daniel Crawford's programming website:
http://github.com/CrawfordGroup/ProgrammingProjects
- DPD Formulation of CC Equations: [Stanton:1991:4334]
"""

__authors__   =  "Daniel G. A. Smith"
__credits__   =  ["Daniel G. A. Smith", "Lori A. Burns"]

__copyright__ = "(c) 2014-2018, The Psi4NumPy Developers"
__license__   = "BSD-3-Clause"
__date__      = "2014-07-29"

import time
import numpy as np
np.set_printoptions(precision=8, linewidth=200, suppress=True)
import psi4
from opt_einsum import contract
from numpy.linalg import norm

# Set memory
psi4.set_memory('5 GB')
psi4.core.set_output_file('output.dat', False)

numpy_memory = 60

### Water molecule

#mol = psi4.geometry(
"""
O
H 1 1.1
H 1 1.1 2 104
symmetry c1
"""
#)

# The coordinates are from the SI of [DOI: 10.1021/acs.jctc.8b00321]

### Water cluster
## Number of water molecule
# 2

#mol = psi4.geometry(
"""
O -1.5167088799 -0.0875022822  0.0744338901
H -0.5688047242  0.0676402012 -0.0936613229
H -1.9654552961  0.5753254158 -0.4692384530
O  1.3898685804  0.0960995460 -0.0761488482
H  1.5926924704 -0.8335878302 -0.2679884752
H  1.5164596797  0.1745974125  0.8831816344
symmetry c1
"""
#)

# 3

#mol = psi4.geometry(
"""
O -1.4765014766 -0.6332885087  0.0898827346
H -1.9838499390 -0.7470663319 -0.7281466521
H -1.1474120728  0.2937933586  0.0499611499
O  1.3009060824 -0.9725326095  0.1123300788
H  1.6231813194 -1.4263451978 -0.6812347553
H  0.3383752926 -1.1745452783  0.1460364674
O  0.2066022512  1.5796259613 -0.1194070925
H  0.8505520801  0.8353634930 -0.0913819530
H  0.3749571446  2.0763034073  0.6962042911
symmetry c1
"""
#)

# 4

mol = psi4.geometry(
"""
O -1.9292712102 -0.0645569159  0.1117679206
H -2.4914119096  0.0073852098 -0.6743533999
H -1.3279797920  0.7200450587  0.0572351830
O  0.0556775311 -1.9834041949  0.1251954700
H -0.7452932853 -1.4024421984  0.1534116762
H -0.0578054438 -2.5263517257 -0.6694026790
O -0.0284527051  1.9145766189 -0.1269877741
H  0.0826356714  2.4643236809  0.6632720821
H  2.4967666426 -0.0808647657  0.7228603336
O  1.9591510378 -0.0026698519 -0.0796304864
H  0.7727195176  1.3336367194 -0.1473249006
H  1.3542775597 -0.7856500486 -0.0462427112
symmetry c1
"""
)

# 5

#mol = psi4.geometry(
"""
O -0.2876645445 -1.7012437583  0.2029164243
H -0.5715171913 -2.6278936014  0.2177698265
H -1.1211952210 -1.1567502352  0.1410434902
O  1.4494811981 -0.1832356135 -1.3664308071
H  0.8495877249 -0.8784465376 -1.0148018507
H  1.7885868060 -0.5222367118 -2.2083896644
O -2.3074119608  0.0600951514 -0.0549593403
H -2.8203525262  0.2892204097  0.7353401875
H  0.2626613561  1.4965603247  0.5922725167
O -0.2424726136  1.8126064878 -0.1909073151
H -1.6640497911  0.8139612893 -0.1634430169
H  0.2878907915  1.4335334111 -0.9188689235
O  1.3813546468  0.2415962603  1.5136093974
H  0.8356415445 -0.5611886184  1.3988328403
H  1.9779314005  0.1892600736  0.7452520171
symmetry c1
"""
#)

# 6

#mol = psi4.geometry(
"""
O -1.0056893157 -0.1043674637  1.7352314342
H -1.0664276447 -0.3762387797  2.6634016204
H -1.5071027077 -0.7913788396  1.2185432381
O  1.1470564813  0.3117796882 -0.0477367962
H  0.6103606068  0.0433421321  0.7275681882
H  2.0791424333  0.1211528401  0.1794662548
O -2.2063795158 -1.8574899369  0.0516378330
H -3.1585603577 -1.7418528352 -0.0901912407
H -1.2703988826  0.5556485786 -1.4822595530
O -0.9686544539 -0.3213451857 -1.8092615790
H -1.7829183163 -1.4200976013 -0.7381133801
H -0.0506196244 -0.3301023162 -1.4666019324
O -1.1414038621  2.0193691143 -0.2156011398
H -1.3619510638  1.5449391020  0.6091355621
H -0.1726333256  1.9018225581 -0.2491925658
O  3.9130618957 -0.1477904028  0.5773094099
H  4.3921274685  0.6778964502  0.4038875571
H  4.3084563274 -0.7931325208 -0.0300166098
symmetry c1
"""
#)

# 7

#mol = psi4.geometry(
"""
O -0.2578815727  0.8589987101  1.0903178485
H  0.5724321179  0.9957545630  1.6134194631
H -0.3379575048 -0.1213199214  1.0257127358
O  0.7053437993  1.0245432710 -1.4943936709 
H  0.3698578944  1.1189598601 -0.5656693305
H  1.0352513822  1.9002918475 -1.7483588311
O -0.3458650799 -1.7904919129  0.2521395252
H -0.8067013063 -2.5793300144  0.5772833215
H -2.2204485019  0.0514045678 -1.2281883065
O -1.6470360380 -0.5567512801 -1.7481190861
H -0.8635215557 -1.4793338112 -0.5471909111
H -0.9032735713  0.0137100383 -2.0286082118
O -2.8270840011  1.2361544476  0.0947724694
H -2.0207774964  1.2912094835  0.6512402118
H -3.0198162448  2.1508831339 -0.1596488039
O  2.3155522081  0.8259479989  1.9468048976
H  2.6243054144  0.3877453957  2.7538641108
H  2.4860312413  0.1745463055  1.2179450536
O  2.3555214451 -0.9273649757 -0.1787945084
H  1.5622606691 -1.4797809021 -0.0212223033
H  2.0415014395 -0.3067984167 -0.8674027785
symmetry c1
"""
#)


psi4.set_options({'basis': '3-21g',
                  'scf_type': 'pk',
                  'mp2_type': 'conv',
                  'freeze_core': 'false',
                  'e_convergence': 1e-10,
                  'd_convergence': 1e-10})

# CCSD Settings
E_conv = 1.e-7
maxiter = 20
print_amps = False
compare_psi4 = True

# First compute RHF energy using Psi4
scf_e, wfn = psi4.energy('SCF', return_wfn=True)

# Grab data from
C = wfn.Ca()
ndocc = wfn.doccpi()[0]
nmo = wfn.nmo()
SCF_E = wfn.energy()
eps = np.asarray(wfn.epsilon_a())

# Compute size of SO-ERI tensor in GB
ERI_Size = (nmo ** 4) * 128e-9
print('\nSize of the SO ERI tensor will be %4.2f GB.' % ERI_Size)
memory_footprint = ERI_Size * 5.2
if memory_footprint > numpy_memory:
    psi4.clean()
    raise Exception("Estimated memory utilization (%4.2f GB) exceeds numpy_memory \
                    limit of %4.2f GB." % (memory_footprint, numpy_memory))

# Integral generation from Psi4's MintsHelper
t = time.time()
mints = psi4.core.MintsHelper(wfn.basisset())
H = np.asarray(mints.ao_kinetic()) + np.asarray(mints.ao_potential())

print('\nTotal time taken for ERI integrals: %.3f seconds.\n' % (time.time() - t))

# Make spin-orbital MO antisymmetrized integrals
print('Starting AO -> spin-orbital MO transformation...')
t = time.time()
MO = np.asarray(mints.mo_spin_eri(C, C))
MO64 = np.asarray(mints.mo_spin_eri(C, C))

# Update nocc and nvirt
nso = nmo * 2
nocc = ndocc * 2
nvirt = nso - nocc

print("nso, ", nso)
print("nocc, ", nocc)
# Make slices
o = slice(0, nocc)
v = slice(nocc, MO.shape[0])

#Extend eigenvalues
eps = np.repeat(eps, 2)
Eocc = eps[o]
Evirt = eps[v]

print('..finished transformation in %.3f seconds.\n' % (time.time() - t))

# DPD approach to CCSD equations from [Stanton:1991:4334]

# occ orbitals i, j, k, l, m, n
# virt orbitals a, b, c, d, e, f
# all oribitals p, q, r, s, t, u, v


#Bulid Eqn 9: tilde{\Tau})
def build_tilde_tau(t1, t2):
    """Builds [Stanton:1991:4334] Eqn. 9"""
    ttau = t2.copy()
    tmp = 0.5 * contract('ia,jb->ijab', t1, t1)
    ttau += tmp
    ttau -= tmp.swapaxes(2, 3)
    return ttau


#Build Eqn 10: \Tau)
def build_tau(t1, t2):
    """Builds [Stanton:1991:4334] Eqn. 10"""
    ttau = t2.copy()
    tmp = contract('ia,jb->ijab', t1, t1)
    ttau += tmp
    ttau -= tmp.swapaxes(2, 3)
    return ttau


#Build Eqn 3:
def build_Fae(t1, t2):
    """Builds [Stanton:1991:4334] Eqn. 3"""
    Fae = F[v, v].copy()
    Fae[np.diag_indices_from(Fae)] = 0

    Fae -= 0.5 * contract('me,ma->ae', F[o, v], t1)
    Fae += contract('mf,mafe->ae', t1, MO[o, v, v, v])

    tmp_tau = build_tilde_tau(t1, t2)
    Fae -= 0.5 * contract('mnaf,mnef->ae', tmp_tau, MO[o, o, v, v])
    return Fae


#Build Eqn 4:
def build_Fmi(t1, t2):
    """Builds [Stanton:1991:4334] Eqn. 4"""
    Fmi = F[o, o].copy()
    Fmi[np.diag_indices_from(Fmi)] = 0

    Fmi += 0.5 * contract('ie,me->mi', t1, F[o, v])
    Fmi += contract('ne,mnie->mi', t1, MO[o, o, o, v])

    tmp_tau = build_tilde_tau(t1, t2)
    Fmi += 0.5 * contract('inef,mnef->mi', tmp_tau, MO[o, o, v, v])
    return Fmi


#Build Eqn 5:
def build_Fme(t1, t2):
    """Builds [Stanton:1991:4334] Eqn. 5"""
    Fme = F[o, v].copy()
    Fme += contract('nf,mnef->me', t1, MO[o, o, v, v])
    return Fme


#Build Eqn 6:
def build_Wmnij(t1, t2):
    """Builds [Stanton:1991:4334] Eqn. 6"""
    Wmnij = MO[o, o, o, o].copy()

    Pij = contract('je,mnie->mnij', t1, MO[o, o, o, v])
    Wmnij += Pij
    Wmnij -= Pij.swapaxes(2, 3)

    tmp_tau = build_tau(t1, t2)
    Wmnij += 0.25 * contract('ijef,mnef->mnij', tmp_tau, MO[o, o, v, v])
    return Wmnij


#Build Eqn 7:
def build_Wabef(t1, t2):
    """Builds [Stanton:1991:4334] Eqn. 7"""
    # Rate limiting step written using tensordot, ~10x faster
    # The commented out lines are consistent with the paper

    Wabef = MO[v, v, v, v].copy()

    Pab = contract('baef->abef', np.tensordot(t1, MO[v, o, v, v], axes=(0, 1)))
    # Pab = np.einsum('mb,amef->abef', t1, MO[v, o, v, v])

    Wabef -= Pab
    Wabef += Pab.swapaxes(0, 1)

    tmp_tau = build_tau(t1, t2)

    Wabef += 0.25 * np.tensordot(tmp_tau, MO[v, v, o, o], axes=((0, 1), (2, 3)))
    # Wabef += 0.25 * np.einsum('mnab,mnef->abef', tmp_tau, MO[o, o, v, v])
    return Wabef


#Build Eqn 8:
def build_Wmbej(t1, t2):
    """Builds [Stanton:1991:4334] Eqn. 8"""
    Wmbej = MO[o, v, v, o].copy()
    Wmbej += contract('jf,mbef->mbej', t1, MO[o, v, v, v])
    Wmbej -= contract('nb,mnej->mbej', t1, MO[o, o, v, o])

    tmp = (0.5 * t2) + contract('jf,nb->jnfb', t1, t1)

    Wmbej -= contract('jbme->mbej', np.tensordot(tmp, MO[o, o, v, v], axes=((1, 2), (1, 3))))
    # Wmbej -= np.einsum('jnfb,mnef->mbej', tmp, MO[o, o, v, v])
    return Wmbej


### Build so Fock matirx

# Update H, transform to MO basis and tile for alpha/beta spin
H = np.einsum('uj,vi,uv', C, C, H)
H = np.repeat(H, 2, axis=0)
H = np.repeat(H, 2, axis=1)

# Make H block diagonal
spin_ind = np.arange(H.shape[0], dtype=np.int) % 2
H *= (spin_ind.reshape(-1, 1) == spin_ind)

# Compute Fock matrix
F = H + np.einsum('pmqm->pq', MO[:, o, :, o])
F64 = H + np.einsum('pmqm->pq', MO[:, o, :, o])
### Build D matrices: [Stanton:1991:4334] Eqns. 12 & 13
Focc = F[np.arange(nocc), np.arange(nocc)].flatten()
Fvirt = F[np.arange(nocc, nvirt + nocc), np.arange(nocc, nvirt + nocc)].flatten()

Dia = Focc.reshape(-1, 1) - Fvirt
Dijab = Focc.reshape(-1, 1, 1, 1) + Focc.reshape(-1, 1, 1) - Fvirt.reshape(-1, 1) - Fvirt

### Construct initial guess

# t^a_i
t1 = np.zeros((nocc, nvirt))
# t^{ab}_{ij}
MOijab = MO[o, o, v, v]
t2 = MOijab / Dijab

### Compute MP2 in MO basis set to make sure the transformation was correct
MP2corr_E = np.einsum('ijab,ijab->', MOijab, t2) / 4
MP2_E = SCF_E + MP2corr_E

print('MO based MP2 correlation energy: %.8f' % MP2corr_E)
print('MP2 total energy:       %.8f' % MP2_E)
psi4.compare_values(psi4.energy('mp2'), MP2_E, 6, 'MP2 Energy')

### Start CCSD iterations
print('\nStarting CCSD iterations')
ccsd_tstart = time.time()
CCSDcorr_E_old = 0.0
### Convert the initial values to single precision to build the intermediates
MO = np.float32(MO)
F = np.float32(F)

for CCSD_iter in range(1, maxiter + 1):
    ### Convert to single precision to build the intermediates
    t1_sp = np.float32(t1)
    t2_sp = np.float32(t2)

    ### Build intermediates: [Stanton:1991:4334] Eqns. 3-8 (in single precision)
    Fae = build_Fae(t1_sp, t2_sp)
    Fmi = build_Fmi(t1_sp, t2_sp)
    Fme = build_Fme(t1_sp, t2_sp)


    time2 = time.time()
    Wmnij = build_Wmnij(t1_sp, t2_sp)
    Wabef = build_Wabef(t1_sp, t2_sp)
    Wmbej = build_Wmbej(t1_sp, t2_sp)


    tmp1 = contract('ie,ae->ia', t1_sp, Fae)
    tmp2 = contract('ma,mi->ia', t1_sp, Fmi)
    tmp3 = contract('imae,me->ia', t2_sp, Fme)
    tmp4 = contract('nf,naif->ia', t1_sp, MO[o, v, o, v])
    tmp5 = contract('imef,maef->ia', t2_sp, MO[o, v, v, v])
    tmp6 = contract('mnae,nmei->ia', t2_sp, MO[o, o, v, o])

    ### Convert back to double precision

    tmp1 = np.float64(tmp1)
    tmp2 = np.float64(tmp2)
    tmp3 = np.float64(tmp3)
    tmp4 = np.float64(tmp4)
    tmp5 = np.float64(tmp5)
    tmp6 = np.float64(tmp6)

    #### Build RHS side of t1 equations, [Stanton:1991:4334] Eqn. 1
    rhs_T1  = F64[o, v].copy()
    rhs_T1 += tmp1
    rhs_T1 -= tmp2
    rhs_T1 += tmp3
    rhs_T1 -= tmp4
    rhs_T1 -= 0.5 * tmp5
    rhs_T1 -= 0.5 * tmp6


    ### Build RHS side of t2 equations, [Stanton:1991:4334] Eqn. 2
    rhs_T2 = MO64[o, o, v, v].copy()

    # P_(ab) t_ijae (F_be - 0.5 t_mb F_me)
    tmp21 = contract('mb,me->be', t1_sp, Fme)
    tmp21 = np.float64(tmp21)
    Fae = np.float64(Fae)
    tmp = Fae - 0.5 * tmp21
    tmp = np.float32(tmp)
    Pab = contract('ijae,be->ijab', t2_sp, tmp)
    Pab = np.float64(Pab)
    #Pab = contract('ijae,be->ijab', t2, tmp)
    rhs_T2 += Pab
    rhs_T2 -= Pab.swapaxes(2, 3)

    # P_(ij) t_imab (F_mj + 0.5 t_je F_me)
    tmp22 = contract('je,me->mj', t1_sp, Fme)
    tmp22 = np.float64(tmp22)
    Fmi = np.float64(Fmi)
    tmp = Fmi + 0.5 * tmp22
    tmp = np.float32(tmp)
    Pij = contract('imab,mj->ijab', t2_sp, tmp)
    Pij = np.float64(Pij)
    rhs_T2 -= Pij
    rhs_T2 += Pij.swapaxes(0, 1)

    tmp_tau = build_tau(t1_sp, t2_sp)
    tmp23 = contract('mnab,mnij->ijab', tmp_tau, Wmnij)
    tmp23 = np.float64(tmp23)
    rhs_T2 += 0.5 * tmp23
    tmp24 = contract('ijef,abef->ijab', tmp_tau, Wabef)
    tmp24 = np.float64(tmp24)
    rhs_T2 += 0.5 * tmp24


    # P_(ij) * P_(ab)
    # (ij - ji) * (ab - ba)
    # ijab - ijba -jiab + jiba
    tmp = contract('ie,ma,mbej->ijab', t1_sp, t1_sp, MO[o, v, v, o])
    Pijab = contract('imae,mbej->ijab', t2_sp, Wmbej)
    Pijab = np.float64(Pijab)
    Pijab -= np.float64(tmp)

    rhs_T2 += Pijab
    rhs_T2 -= Pijab.swapaxes(2, 3)
    rhs_T2 -= Pijab.swapaxes(0, 1)
    rhs_T2 += Pijab.swapaxes(0, 1).swapaxes(2, 3)


    Pij = np.float64(contract('ie,abej->ijab', t1_sp, MO[v, v, v, o]))
    rhs_T2 += Pij
    rhs_T2 -= Pij.swapaxes(0, 1)

    Pab = np.float64(contract('ma,mbij->ijab', t1_sp, MO[o, v, o, o]))
    rhs_T2 -= Pab
    rhs_T2 += Pab.swapaxes(2, 3)


    ### Update t1 and t2 amplitudes
    t1 = np.float64(t1_sp)
    t2 = np.float64(t2_sp)
    t1 = rhs_T1 / Dia
    t2 = rhs_T2 / Dijab


    ### Compute CCSD correlation energy
    CCSDcorr_E = np.einsum('ia,ia->', F64[o, v], t1)
    tmpE = np.einsum('ijab,ijab->', MO64[o, o, v, v], t2)
    CCSDcorr_E += 0.25 * tmpE
    tmpE = np.einsum('ijab,ia,jb->', MO64[o, o, v, v], t1, t1)
    CCSDcorr_E += 0.5 * tmpE

    ### Print CCSD correlation energy
    print('CCSD Iteration %3d: CCSD correlation = %3.12f  '\
          'dE = %3.5E' % (CCSD_iter, CCSDcorr_E, (CCSDcorr_E - CCSDcorr_E_old)))
    if (abs(CCSDcorr_E - CCSDcorr_E_old) < E_conv):
        break

    CCSDcorr_E_old = CCSDcorr_E

print('CCSD iterations took %.2f seconds.\n' % (time.time() - ccsd_tstart))

CCSD_E = SCF_E + CCSDcorr_E

print('\nFinal CCSD correlation energy:     % 16.10f' % CCSDcorr_E)
print('Total CCSD energy:                 % 16.10f' % CCSD_E)
if compare_psi4:
    psi4.compare_values(psi4.energy('CCSD'), CCSD_E, 6, 'CCSD Energy')

if print_amps:
    # [::4] take every 4th, [-5:] take last 5, [::-1] reverse order
    t2_args = np.abs(t2).ravel().argsort()[::2][-5:][::-1]
    t1_args = np.abs(t1).ravel().argsort()[::4][-5:][::-1]

    print('\nLargest t1 amplitudes')
    for pos in t1_args:
        value = t1.flat[pos]
        inds = np.unravel_index(pos, t1.shape)
        print('%4d  %4d |   % 5.10f' % (inds[0], inds[1], value))

    print('\nLargest t2 amplitudes')
    for pos in t2_args:
        value = t2.flat[pos]
        inds = np.unravel_index(pos, t2.shape)
        print('%4d  %4d  %4d  %4d |   % 5.10f' % (inds[0], inds[1], inds[2], inds[3], value))


"""

# Check the magnitudes of r_T1


print("1: ", F[o, v].copy())
print("2: ", np.einsum('ie,ae->ia', t1, Fae))
print("3: ", np.einsum('ma,mi->ia', t1, Fmi))
print("4: ", np.einsum('imae,me->ia', t2, Fme))
print("5: ", np.einsum('nf,naif->ia', t1, MO[o, v, o, v]))
print("6: ", 0.5 * np.einsum('imef,maef->ia', t2, MO[o, v, v, v]))
print("7: ", 0.5 * np.einsum('mnae,nmei->ia', t2, MO[o, o, v, o]))

print("norm: ")
print("1: ", norm(F[o, v].copy()))
print("2: ", norm(np.einsum('ie,ae->ia', t1, Fae)))
print("3: ", norm(np.einsum('ma,mi->ia', t1, Fmi)))
print("4: ", norm(np.einsum('imae,me->ia', t2, Fme)))
print("5: ", norm(np.einsum('nf,naif->ia', t1, MO[o, v, o, v])))
print("6: ", norm(0.5 * np.einsum('imef,maef->ia', t2, MO[o, v, v, v])))
print("7: ", norm(0.5 * np.einsum('mnae,nmei->ia', t2, MO[o, o, v, o])))

# Components of Fae
tmp1 = F[v, v].copy()
tmp1[np.diag_indices_from(Fae)] = 0
print("Fae1: ", np.einsum('ie,ae->ia', t1, tmp1))

tmp2 = F[v, v].copy()
tmp2 = -0.5 * np.einsum('me,ma->ae', F[o, v], t1)
print("Fae2: ", np.einsum('ie,ae->ia', t1, tmp2))

tmp3 = F[v, v].copy()
tmp3 = np.einsum('mf,mafe->ae', t1, MO[o, v, v, v])
print("Fae3: ", np.einsum('ie,ae->ia', t1, tmp3))

tmp_tau = build_tilde_tau(t1, t2)
tmp4 = F[v, v].copy()
tmp4 = -0.5 * np.einsum('mnaf,mnef->ae', tmp_tau, MO[o, o, v, v])
print("Fae4: ", np.einsum('ie,ae->ia', t1, tmp4))

print("norm of Fae3: ", norm(np.einsum('ie,ae->ia', t1, tmp3)))
print("norm of Fae4: ", norm(np.einsum('ie,ae->ia', t1, tmp4)))

print("E1: ", np.einsum('ia,ia->', F[o, v], t1))
print("E2: ", 0.25 * np.einsum('ijab,ijab->', MO[o, o, v, v], t2))
print("E3: ", 0.5 * np.einsum('ijab,ia,jb->', MO[o, o, v, v], t1, t1))


# Check the magnitudes for r_T2

tmp = Fae - 0.5 * np.einsum('mb,me->be', t1, Fme)
tmp_a = Fae
tmp_b = -0.5 * np.einsum('mb,me->be', t1, Fme)
Pab = np.einsum('ijae,be->ijab', t2, tmp)
Pab_a = np.einsum('ijae,be->ijab', t2, tmp_a)
Pab_b = np.einsum('ijae,be->ijab', t2, tmp_b)
tmp2 = Pab
tmp2 -= Pab.swapaxes(2, 3)
tmp2a = Pab_a
tmp2a -= Pab_a.swapaxes(2, 3)
tmp2b = Pab_b
tmp2b -= Pab_b.swapaxes(2, 3)

tmp = Fmi + 0.5 * np.einsum('je,me->mj', t1, Fme)
tmp_a = Fmi
tmp_b = 0.5 * np.einsum('je,me->mj', t1, Fme)
Pij = np.einsum('imab,mj->ijab', t2, tmp)
Pij_a = np.einsum('imab,mj->ijab', t2, tmp_a)
Pij_b = np.einsum('imab,mj->ijab', t2, tmp_b)
tmp3 = Pij
tmp3 += Pij.swapaxes(0, 1)
tmp3a = Pij_a
tmp3a += Pij_a.swapaxes(0, 1)
tmp3b = Pij_b
tmp3b += Pij_b.swapaxes(0, 1)

tmp_tau = build_tau(t1, t2)
tmp4 = 0.5 * np.einsum('mnab,mnij->ijab', tmp_tau, Wmnij)
tmp5 = 0.5 * np.einsum('ijef,abef->ijab', tmp_tau, Wabef)

tmp_b = np.einsum('ie,ma,mbej->ijab', t1, t1, MO[o, v, v, o])
tmp_a = np.einsum('imae,mbej->ijab', t2, Wmbej)
#tmp6a = tmp
#tmp6b = Pijab
Pijab = tmp_a - tmp_b
tmp6 = Pijab
tmp6 -= Pijab.swapaxes(2, 3)
tmp6 -= Pijab.swapaxes(0, 1)
tmp6 += Pijab.swapaxes(0, 1).swapaxes(2, 3)
tmp6a = tmp_a
tmp6a -= tmp6a.swapaxes(2, 3)
tmp6a -= tmp6a.swapaxes(0, 1)
tmp6a += tmp6a.swapaxes(0, 1).swapaxes(2, 3)
tmp6b = tmp_b
tmp6b -= tmp6b.swapaxes(2, 3)
tmp6b -= tmp6b.swapaxes(0, 1)
tmp6b += tmp6b.swapaxes(0, 1).swapaxes(2, 3)


Pij = np.einsum('ie,abej->ijab', t1, MO[v, v, v, o])
tmp7 = Pij
tmp7 -= Pij.swapaxes(0, 1)

Pab = np.einsum('ma,mbij->ijab', t1, MO[o, v, o, o])
tmp8 = Pab
tmp8 += Pab.swapaxes(2, 3)

print("1: ", norm(MO[o, o, v, v].copy()))

print("2: ", norm(tmp2))
print("2a: ", norm(tmp2a))
print("2b: ", norm(tmp2b))

print("3: ", norm(tmp3))
print("3a: ", norm(tmp3a))
print("3b: ", norm(tmp3b))

print("4: ", norm(tmp4))

print("5: ", norm(tmp5))

print("6: ", norm(tmp6))
print("6a: ", norm(tmp6a))
print("6b: ", norm(tmp6b))

print("7: ", norm(tmp7))

print("8: ", norm(tmp8))

"""
"""
ATT
0 1
N  4.648954  0.062237 -2.370046
C  5.222661  1.044161 -1.595124
N  4.354894  1.743856 -0.892815
C  3.124382  1.191641 -1.234691
C  1.796538  1.473710 -0.820053
N  1.477058  2.412334  0.063341
N  0.802690  0.730580 -1.347517
C  1.118897 -0.233683 -2.222372
N  2.315056 -0.597666 -2.682467
C  3.287489  0.159535 -2.145551
N -4.119880  1.175240 -1.668947
C -2.788993  0.852210 -1.845154
O -2.385787  0.134394 -2.748531
N -1.937862  1.401121 -0.916872
C -2.275607  2.205332  0.157303
O -1.389232  2.644961  0.899530
C -3.691851  2.470201  0.309209
C -4.156175  3.291113  1.475440
C -4.531960  1.955432 -0.607897
N -3.093123 -2.936903 -1.004080
C -1.789262 -2.710900 -0.630457
O -0.840457 -3.167912 -1.243704
N -1.636359 -1.922963  0.479694
C -2.636971 -1.336214  1.225506
O -2.346142 -0.645842  2.195441
C -3.987755 -1.598884  0.766076
C -5.137909 -0.995328  1.511021
C -4.143924 -2.378097 -0.316323
H  6.291440  1.207255 -1.586397
H  2.216641  2.881413  0.561742
H  0.519215  2.485870  0.406134
H  0.269720 -0.796087 -2.601331
H -0.930607  1.148159 -1.047409
H -5.240856  3.424662  1.455065
H -3.680063  4.275568  1.470584
H -3.878536  2.804296  2.415249
H -5.602410  2.125849 -0.555843
H -0.639080 -1.781127  0.770889
H -6.090057 -1.247239  1.035981
H -5.040838  0.094030  1.551165
H -5.161876 -1.348358  2.546381
H -5.126930 -2.611972 -0.712652
H  5.101053 -0.601273 -2.978111
H -4.769876  0.803247 -2.342853
H -3.229023 -3.459136 -1.855586

Benzene
0 1
C  1.4059535336  0.0000000000  0.0000000000
C  0.7029767633  1.2175914744  0.0000000000
C  0.7029767633 -1.2175914744  0.0000000000
C -0.7029767633  1.2175914744  0.0000000000
C -0.7029767633 -1.2175914744  0.0000000000
C -1.4059535336  0.0000000000  0.0000000000
H  2.5018761033  0.0000000000  0.0000000000
H  1.2509380519  2.1666882634  0.0000000000
H  1.2509380519 -2.1666882634  0.0000000000
H -1.2509380519  2.1666882634  0.0000000000
H -1.2509380519 -2.1666882634  0.0000000000
H -2.5018761033  0.0000000000  0.0000000000

Uracil
0 1
C  1.1965439001  1.1059975965  0.0000000000
C -0.0106406622  1.6974586473  0.0000000000
N -1.1759657685  0.9706327770  0.0000000000
C  1.2905520182 -0.3511811123  0.0000000000
N  0.0394261039 -0.9922273002  0.0000000000
C -1.2164849902 -0.4175249683  0.0000000000
O -2.2534478773 -1.0446838400  0.0000000000
O  2.3153668717 -1.0016569295  0.0000000000
H  2.1145824324  1.6760396163  0.0000000000
H -0.1369773173  2.7740930054  0.0000000000
H -2.0769371453  1.4242304202  0.0000000000
H  0.0555272212 -2.0045027192  0.0000000000

Formaldehyde
0 1
C -0.2581670178  0.0631333004  0.0000000000
O  0.9398801946 -0.1450331544  0.0000000000
H -0.8478178295  0.1654099608 -0.9426389633
H -0.8478178295  0.1654099608  0.9426389633
"""
