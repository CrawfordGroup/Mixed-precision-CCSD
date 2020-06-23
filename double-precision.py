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
from utils import ndot
from opt_einsum import contract
from numpy.linalg import norm

# Set memory
psi4.set_memory('9 GB')
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

### Water cluster
## Number of water molecules
## 2

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

## 3

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

## 4

#mol = psi4.geometry(
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
#)

## 5

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

## 6

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

## 7

mol = psi4.geometry(
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
)


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
for CCSD_iter in range(1, maxiter + 1):
    ### Build intermediates: [Stanton:1991:4334] Eqns. 3-8
    time1 = time.time()
    Fae = build_Fae(t1, t2)
    Fmi = build_Fmi(t1, t2)
    Fme = build_Fme(t1, t2)
    time2 = time.time()
    Wmnij = build_Wmnij(t1, t2)
    Wabef = build_Wabef(t1, t2)
    Wmbej = build_Wmbej(t1, t2)

    #### Build RHS side of t1 equations, [Stanton:1991:4334] Eqn. 1
    rhs_T1  = F[o, v].copy()
    rhs_T1 += contract('ie,ae->ia', t1, Fae)
    rhs_T1 -= contract('ma,mi->ia', t1, Fmi)
    rhs_T1 += contract('imae,me->ia', t2, Fme)
    rhs_T1 -= contract('nf,naif->ia', t1, MO[o, v, o, v])
    rhs_T1 -= 0.5 * contract('imef,maef->ia', t2, MO[o, v, v, v])
    rhs_T1 -= 0.5 * contract('mnae,nmei->ia', t2, MO[o, o, v, o])

    ### Build RHS side of t2 equations, [Stanton:1991:4334] Eqn. 2
    rhs_T2 = MO[o, o, v, v].copy()
    # P_(ab) t_ijae (F_be - 0.5 t_mb F_me)
    tmp = Fae - 0.5 * contract('mb,me->be', t1, Fme)
    Pab = contract('ijae,be->ijab', t2, tmp)
    rhs_T2 += Pab
    rhs_T2 -= Pab.swapaxes(2, 3)

    # P_(ij) t_imab (F_mj + 0.5 t_je F_me)
    tmp = Fmi + 0.5 * contract('je,me->mj', t1, Fme)
    Pij = contract('imab,mj->ijab', t2, tmp)
    rhs_T2 -= Pij
    rhs_T2 += Pij.swapaxes(0, 1)

    tmp_tau = build_tau(t1, t2)
    rhs_T2 += 0.5 * contract('mnab,mnij->ijab', tmp_tau, Wmnij)
    rhs_T2 += 0.5 * contract('ijef,abef->ijab', tmp_tau, Wabef)
    # P_(ij) * P_(ab)
    # (ij - ji) * (ab - ba)
    # ijab - ijba -jiab + jiba
    tmp = contract('ie,ma,mbej->ijab', t1, t1, MO[o, v, v, o])
    Pijab = contract('imae,mbej->ijab', t2, Wmbej)
    Pijab -= tmp

    rhs_T2 += Pijab
    rhs_T2 -= Pijab.swapaxes(2, 3)
    rhs_T2 -= Pijab.swapaxes(0, 1)
    rhs_T2 += Pijab.swapaxes(0, 1).swapaxes(2, 3)
    Pij = contract('ie,abej->ijab', t1, MO[v, v, v, o])
    rhs_T2 += Pij
    rhs_T2 -= Pij.swapaxes(0, 1)

    Pab = contract('ma,mbij->ijab', t1, MO[o, v, o, o])
    rhs_T2 -= Pab
    rhs_T2 += Pab.swapaxes(2, 3)
    ### Update t1 and t2 amplitudes
    t1 = rhs_T1 / Dia
    t2 = rhs_T2 / Dijab
    ### Compute CCSD correlation energy
    CCSDcorr_E = np.einsum('ia,ia->', F[o, v], t1)
    CCSDcorr_E += 0.25 * np.einsum('ijab,ijab->', MO[o, o, v, v], t2)
    CCSDcorr_E += 0.5 * np.einsum('ijab,ia,jb->', MO[o, o, v, v], t1, t1)

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




