# -*- coding: utf-8 -*-
"""
A simple python script to calculate RHF-CCSD specific rotation in length,
velocity and modified velocity gauge using coupled cluster linear response theory.
References:
1. H. Koch and P. Jørgensen, J. Chem. Phys. Volume 93, pp. 3333-3344 (1991).
2. T. B. Pedersen and H. Koch, J. Chem. Phys. Volume 106, pp. 8059-8072 (1997).
3. T. Daniel Crawford, Theor. Chem. Acc., Volume 115, pp. 227-245 (2006).
4. T. B. Pedersen, H. Koch, L. Boman, and A. M. J. Sánchez de Merás, Chem. Phys. Lett.,
   Volime 393, pp. 319, (2004).
5. A Whirlwind Introduction to Coupled Cluster Response Theory, T.D. Crawford, Private Notes,
   (pdf in the current directory).
"""

__authors__ = "Zhe Wang"
__credits__ = [
    "Ashutosh Kumar", "Daniel G. A. Smith", "Lori A. Burns", "T. D. Crawford"
]

__copyright__ = "(c) 2014-2018, The Psi4NumPy Developers"
__license__ = "BSD-3-Clause"
__date__ = "2018-02-20"

import os.path
import sys
dirname = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(dirname, '../../../Coupled-Cluster/RHF'))
import numpy as np
np.set_printoptions(precision=15, linewidth=200, suppress=True)
from ccenergy import *
from cclambda import *
from ccpert import *

import psi4
from psi4 import constants as pc

psi4.set_memory(int(2e9), False)
psi4.core.set_output_file('output.dat', False)

# can only handle C1 symmetry
#mol = psi4.geometry(
"""
 O     -0.028962160801    -0.694396279686    -0.049338350190                                                                  
 O      0.028962160801     0.694396279686    -0.049338350190                                                                  
 H      0.350498145881    -0.910645626300     0.783035421467                                                                  
 H     -0.350498145881     0.910645626300     0.783035421467                                                                  
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

# Uracil

#mol = psi4.geometry(
"""
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
symmetry c1
"""
#)

# Benzene
#mol = psi4.geometry(
"""
0 1
C  1.4059535336  0.0000000000 0.0000000000
C  0.7029767668  1.2175914766 0.0000000000
C  0.7029767668 -1.2175914766 0.0000000000 
C -0.7029767668  1.2175914766 0.0000000000
C -0.7029767668 -1.2175914766 0.0000000000
C -1.4059535336  0.0000000000 0.0000000000 
H  2.5018761059  0.0000000000 0.0000000000 
H  1.2509380530  2.1666882648 0.0000000000
H  1.2509380530 -2.1666882648 0.0000000000 
H -1.2509380530  2.1666882648 0.0000000000
H -1.2509380530 -2.1666882648 0.0000000000
H -2.5018761059  0.0000000000 0.0000000000 
symmetry c1
"""
#)

# Chiral molecules
# (S)-1,3-dimethylallene
mol = psi4.geometry(
"""
C  0.000000  0.000000  0.414669
C  0.000000  1.309935  0.412904
C  0.000000 -1.309935  0.412904
H -0.677576  1.833097  1.091213
H  0.677576 -1.833097  1.091213
C  0.875978  2.177784 -0.460206
C -0.875978 -2.177784 -0.460206
H  1.519837  1.571995 -1.104163
H -1.519837 -1.571995 -1.104163
H  0.267559  2.832375 -1.098077
H -0.267559 -2.832375 -1.098077
H  1.514280  2.829499  0.150828
H -1.514280 -2.829499  0.150828
symmetry c1
"""
)

# (S)-2-chloropropionitrile
#mol = psi4.geometry(
"""
C  -0.074850  0.536809  0.165803
C  -0.131322  0.463480  1.692375
H   0.576771  1.182613  2.118184
C   1.260194  0.235581 -0.354931
N   2.332284  0.017558 -0.742030
CL -1.285147 -0.612976 -0.578450
H  -0.370843  1.528058 -0.186624
H  -1.138978  0.710124  2.035672
H   0.130424 -0.538322  2.041140
symmetry c1
"""
#)

# (R)-methylthiirane
#mol = psi4.geometry(
"""
C    0.364937   1.156173  -0.153047
C   -0.521930   0.166108   0.502066
C   -1.785526  -0.309717  -0.178180
H   -1.645392  -0.376658  -1.261546
S    1.101404  -0.526707  -0.054082
H    0.106264   1.485766  -1.157365
H    0.850994   1.912823   0.458508
H   -0.587772   0.237291   1.586932
H   -2.083855  -1.297173   0.188561
H   -2.607593   0.389884   0.025191
symetry c1
"""
#)

# (S)-methyloxirane:
#mol = psi4.geometry(
"""
 zmat = (
 C
 C 1 B1
 O 2 B2 1 A1
 H 2 B3 1 A2 3 D1
 H 2 B4 1 A3 3 D2
 H 1 B5 2 A4 3 D3
 C 1 B6 2 A5 3 D4
 H 7 B7 1 A6 2 D5
 H 7 B8 1 A7 2 D6
 H 7 B9 1 A8 2 D7
  )
  zvars = (
 A1 59.24053923
 A2 119.43315591
 A3 120.16908378
 A4 117.21842257
 A5 122.37833117
 A6 110.6489744
 A7 110.71820821
 A8 110.62534492
 D1 -103.61051046
 D2 103.47749463
 D3 -102.30478838
 D4 103.75862518
 D5 -24.51750626
 D6 96.01547667
 D7 -144.34513611
 B1 1.47013445
 B2 1.43318871
 B3 1.0909973
 B4 1.09072868
 B5 1.09287012
 B6 1.50793568
 B7 1.09469424
 B8 1.09673641
 B9 1.09609604
  )
symmetry c1
"""
#)

# setting up SCF options
psi4.set_options({
    'basis': '3-21g',
    'scf_type': 'PK',
    'd_convergence': 1e-10,
    'e_convergence': 1e-10,
})
rhf_e, rhf_wfn = psi4.energy('SCF', return_wfn=True)
print('RHF Final Energy                          % 16.10f\n' % rhf_e)

# Calculate Ground State CCSD energy
ccsd = ccEnergy(mol, rhf_e, rhf_wfn, numpy_memory=2)
CCSDcorr_E = ccsd.compute_energy(e_conv=1e-7, r_conv=1e-7, maxiter=50, max_diis=8, start_diis=100)
CCSD_E = CCSDcorr_E + ccsd.SCF_E
#CCSDcorr_E = ccsd.ccsd_corr_e
#CCSD_E = ccsd.ccsd_e

print('\nFinal CCSD correlation energy:          % 16.15f' % CCSDcorr_E)
print('Total CCSD energy:                      % 16.15f' % CCSD_E)

# Now that we have T1 and T2 amplitudes, we can construct
# the pieces of the similarity transformed hamiltonian (Hbar).
#cchbar = ccHbar(ccsd)

# Calculate Lambda amplitudes using Hbar
cclambda = ccLambda(ccsd)
cclambda.compute_lambda(e_conv=1e-7, r_conv=1e-7, maxiter=50, max_diis=8, start_diis=100)

# frequency of calculation
omega_nm = 589

# convert from nm into hartree
omega = (pc.c * pc.h * 1e9) / (pc.hartree2J * omega_nm)
Om = str(omega)
Om_0 = str(0)

cart = ['X', 'Y', 'Z']
pert = {}
ccpert = {}
tensor = {}
cclinresp = {}
optrot_lg = np.zeros(9)
optrot_vg_om = np.zeros(9)
optrot_vg_0 = np.zeros(9)

###############################################   Length Gauge   ###############################################################

# In length gauge the representation of electric dipole operator is mu i.e. r. So, optical rotation tensor in this gauge
# representation can be given by -Im <<mu;L>>, where L is the angular momemtum operator, refer to Eqn. 5 of [Crawford:2006:227].
# For general form of a response function, refer to Eqn. 94 of [Koch:1991:3333].

print("\n\n Length Gauge Calculations Starting ..\n\n")

# Obtain the required AO Perturabtion Matrices From Mints

# Electric Dipole
dipole_array = ccsd.mints.ao_dipole()

# Angular Momentum
angmom_array = ccsd.mints.ao_angular_momentum()

for i in range(0, 3):
    Mu = "MU_" + cart[i]
    L = "L_" + cart[i]

    # Transform perturbations from AO to MO basis
    pert[Mu] = np.einsum('uj,vi,uv', ccsd.npC, ccsd.npC,
                         np.asarray(dipole_array[i]))
    pert[L] = -0.5 * np.einsum('uj,vi,uv', ccsd.npC, ccsd.npC,
                               np.asarray(angmom_array[i]))

    # Initializing the perturbation class corresponding to each perturabtion at the given omega
    ccpert[Mu + Om] = CCPert(Mu, pert[Mu], ccsd, cclambda, omega)
    ccpert[L + Om] = CCPert(L, pert[L], ccsd, cclambda, omega)

    # Solve X and Y amplitudes corresponding to each perturabtion at the given omega
    print(
        '\nsolving right hand perturbed amplitudes for %s @ omega = %s a.u.\n'
        % (Mu, Om))
    ccpert[Mu + Om].solve('right', r_conv=1e-7)

    print(
        '\nsolving left hand perturbed amplitudes for %s @ omega = %s a.u.\n' %
        (Mu, Om))
    ccpert[Mu + Om].solve('left', r_conv=1e-7)

    print(
        '\nsolving right hand perturbed amplitudes for %s @ omega = %s a.u.\n'
        % (L, Om))
    ccpert[L + Om].solve('right', r_conv=1e-7)

    print(
        '\nsolving left hand perturbed amplitudes for %s @ omega = %s a.u.\n' %
        (L, Om))
    ccpert[L + Om].solve('left', r_conv=1e-7)

for A in range(0, 3):
    str_A = "MU_" + cart[A]
    for B in range(0, 3):
        str_B = "L_" + cart[B]
        str_AB = "<<" + str_A + ";" + str_B + ">>"
        str_BA = "<<" + str_B + ";" + str_A + ">>"

        # constructing the linear response functions <<MU;L>> and <<L;MU>> @ given omega
        # The optical rotation tensor beta can be written in length gauge as:
        # beta_pq = 0.5 * (<<MU_p;L_q>>  - <<L_q;MU_p>), Please refer to eq. 49 of
        # [Pedersen:1997:8059].

        cclinresp[str_AB] = CCLinresp(cclambda, ccpert[str_A + Om],
                                            ccpert[str_B + Om])
        cclinresp[str_BA] = CCLinresp(cclambda, ccpert[str_B + Om],
                                            ccpert[str_A + Om])

        tensor[str_AB] = cclinresp[str_AB].linresp()
        tensor[str_BA] = cclinresp[str_BA].linresp()

        optrot_lg[3 * A + B] = 0.5 * (tensor[str_AB] - tensor[str_BA])

# Isotropic optical rotation in length gauge @ given omega
rlg_au = optrot_lg[0] + optrot_lg[4] + optrot_lg[8]
rlg_au /= 3

print('\n CCSD Optical Rotation Tensor (Length Gauge) @ %d nm' % omega_nm)
print("\t\t%s\t             %s\t                  %s\n" % (cart[0], cart[1],
                                                           cart[2]))

for a in range(0, 3):
    print(" %s %20.10lf %20.10lf %20.10lf\n" %
          (cart[a], optrot_lg[3 * a + 0], optrot_lg[3 * a + 1],
           optrot_lg[3 * a + 2]))

# convert from a.u. into deg/[dm (g/cm^3)]
# refer to eq. 4 of [Crawford:1996:189].
Mass = 0
for atom in range(mol.natom()):
    Mass += mol.mass(atom)
m2a = pc.bohr2angstroms * 1e-10
hbar = pc.h / (2.0 * np.pi)
prefactor = 1e-2 * hbar / (pc.c * 2.0 * np.pi * pc.me * (m2a**2))
prefactor *= prefactor
prefactor *= 288e-30 * (np.pi**2) * pc.na * (pc.bohr2angstroms**4)
prefactor *= -1
specific_rotation_lg = prefactor * rlg_au * omega / Mass
print("Specific rotation @ %d nm (Length Gauge): %10.5lf deg/[dm (g/cm^3)]" %
      (omega_nm, specific_rotation_lg))

###############################################     Velocity Gauge      #########################################################

# In length gauge the representation of electric dipole operator is in terms of p, i,e. the momentum operator.
# So, optical rotation tensor in this gauge representation can be given by -Im <<P;L>>.

print("\n\n Velocity Gauge Calculations Starting ..\n\n")

# Grabbing the momentum integrals from mints
nabla_array = ccsd.mints.ao_nabla()

for i in range(0, 3):
    P = "P_" + cart[i]

    # Transform momentum from AO to MO basis
    pert[P] = np.einsum('uj,vi,uv', ccsd.npC, ccsd.npC,
                        np.asarray(nabla_array[i]))

    # Initializing the perturbation class
    ccpert[P + Om] = CCPert(P, pert[P], ccsd, cclambda, omega)

    # Solve X and Y amplitudes corresponding to the perturabtion at the given omega
    print(
        '\nsolving right hand perturbed amplitudes for %s @ omega = %s a.u.\n'
        % (P, Om))
    ccpert[P + Om].solve('right', r_conv=1e-7)

    print(
        '\nsolving left hand perturbed amplitudes for %s @ omega = %s a.u.\n' %
        (P, Om))
    ccpert[P + Om].solve('left', r_conv=1e-7)

for A in range(0, 3):
    str_A = "P_" + cart[A]
    for B in range(0, 3):
        str_B = "L_" + cart[B]
        str_AB = "<<" + str_A + ";" + str_B + ">>"
        str_BA = "<<" + str_B + ";" + str_A + ">>"

        # constructing the linear response functions <<P;L>> and <<L;P>> @ given omega
        # The optical rotation tensor beta can be written in velocity gauge as:
        # beta_pq = 0.5 * (<<MU_p;L_q>> + <<L_q;MU_p>), Please refer to eq. 49 of
        # [Pedersen:1991:8059].

        cclinresp[str_AB] = CCLinresp(cclambda, ccpert[str_A + Om],
                                            ccpert[str_B + Om])
        cclinresp[str_BA] = CCLinresp(cclambda, ccpert[str_B + Om],
                                            ccpert[str_A + Om])
        tensor[str_AB] = cclinresp[str_AB].linresp()
        tensor[str_BA] = cclinresp[str_BA].linresp()
        optrot_vg_om[3 * A + B] = 0.5 * (tensor[str_AB] + tensor[str_BA])

# Isotropic optical rotation in velocity gauge @ given omega
rvg_om_au = optrot_vg_om[0] + optrot_vg_om[4] + optrot_vg_om[8]
rvg_om_au /= 3

print('\n CCSD Optical Rotation Tensor (Velocity Gauge) @ %d nm' % omega_nm)
print("\t\t%s\t             %s\t                  %s\n" % (cart[0], cart[1],
                                                           cart[2]))

for a in range(0, 3):
    print(" %s %20.10lf %20.10lf %20.10lf\n" %
          (cart[a], optrot_vg_om[3 * a + 0], optrot_vg_om[3 * a + 1],
           optrot_vg_om[3 * a + 2]))

specific_rotation_vg_om = prefactor * rvg_om_au / Mass
print("Specific rotation @ %d nm (Velocity Gauge): %10.5lf deg/[dm (g/cm^3)]" %
      (omega_nm, specific_rotation_vg_om))

###############################################   Modified Velocity Gauge   ######################################################
#
# Velocity gauge (VG) representation gives a non-zero optical rotation at zero frequency,
# which is clearly an unphysical result. [Pedersen:319:2004] proposed the modified
# velocity gauge (MVG) representation where the VG optical rotation at # zero frequency is subtracted from VG results at a given frequency.

print("\n\nModified Velocity Gauge Calculations Starting ..\n\n")

Om_0 = str(0)
for i in range(0, 3):
    L = "L_" + cart[i]
    P = "P_" + cart[i]
    Om_0 = str(0)

    # Initializing perturbation classes at zero frequency
    ccpert[L + Om_0] = CCPert(L, pert[L], ccsd, cclambda, 0)
    ccpert[P + Om_0] = CCPert(P, pert[P], ccsd, cclambda, 0)

    # Solving X and Y amplitudes of the perturbation classes at zero frequency

    print(
        '\nsolving right hand perturbed amplitudes for %s @ omega = %s (a.u.)\n'
        % (L, Om_0))
    ccpert[L + Om_0].solve('right', r_conv=1e-7)

    print(
        '\nsolving left hand perturbed amplitudes for %s @ omega = %s (a.u.)\n'
        % (L, Om_0))
    ccpert[L + Om_0].solve('left', r_conv=1e-7)

    print(
        '\nsolving right hand perturbed amplitudes for %s @ omega = %s (a.u.)\n'
        % (P, Om_0))
    ccpert[P + Om_0].solve('right', r_conv=1e-7)

    print(
        '\nsolving left hand perturbed amplitudes for %s @ omega = %s (a.u.)\n'
        % (P, Om_0))
    ccpert[P + Om_0].solve('left', r_conv=1e-7)

for A in range(0, 3):
    str_A = "P_" + cart[A]
    for B in range(0, 3):
        str_B = "L_" + cart[B]
        str_AB = "<<" + str_A + ";" + str_B + ">>"
        str_BA = "<<" + str_B + ";" + str_A + ">>"

        # constructing the linear response functions <<P;L>> and <<L;P>> @ zero frequency)

        cclinresp[str_AB] = CCLinresp(cclambda, ccpert[str_A + Om_0],
                                            ccpert[str_B + Om_0])
        cclinresp[str_BA] = CCLinresp(cclambda, ccpert[str_B + Om_0],
                                            ccpert[str_A + Om_0])

        tensor[str_AB] = cclinresp[str_AB].linresp()
        tensor[str_BA] = cclinresp[str_BA].linresp()

        optrot_vg_0[3 * A + B] = 0.5 * (tensor[str_AB] + tensor[str_BA])

#  MVG(omega) = VG(omega) - VG(0)
optrot_mvg = optrot_vg_om - optrot_vg_0

# Isotropic optical rotation in modified velocity gauge @ given omega
rmvg_au = optrot_mvg[0] + optrot_mvg[4] + optrot_mvg[8]
rmvg_au /= 3

print('\n CCSD Optical Rotation Tensor (Modified Velocity Gauge) @ %d nm' %
      omega_nm)
print("\t\t%s\t             %s\t                  %s\n" % (cart[0], cart[1],
                                                           cart[2]))
for a in range(0, 3):
    print(" %s %20.10lf %20.10lf %20.10lf\n" %
          (cart[a], optrot_vg_0[3 * a + 0], optrot_vg_0[3 * a + 1],
           optrot_vg_0[3 * a + 2]))

specific_rotation_mvg = prefactor * rmvg_au / Mass
print(
    "Specific rotation @ %d nm (Modified Velocity Gauge): %10.5lf deg/[dm (g/cm^3)]"
    % (omega_nm, specific_rotation_mvg))

#  Comaprison with PSI4 (if you have near to latest version of psi4)
psi4.set_options({'d_convergence': 1e-10,
                  'e_convergence': 1e-10,
                  'r_convergence': 1e-10,
                  'omega': [589, 'nm'],  
                  'gauge': 'both'})  
psi4.properties('ccsd', properties=['rotation'])
psi4.compare_values(specific_rotation_lg, psi4.variable("CCSD SPECIFIC ROTATION (LEN) @ 589NM"), \
 5, "CCSD SPECIFIC ROTATION (LENGTH GAUGE) 589 nm") #TEST
psi4.compare_values(specific_rotation_mvg, psi4.variable("CCSD SPECIFIC ROTATION (MVG) @ 589NM"), \
  5, "CCSD SPECIFIC ROTATION (MODIFIED VELOCITY GAUGE) 589 nm") #TEST

"""
psi4.compare_values(specific_rotation_lg, 7.03123, 3,
                    "CCSD SPECIFIC ROTATION (LENGTH GAUGE) 589 nm")  #TEST
psi4.compare_values(
    specific_rotation_mvg, -81.44742, 5,
    "CCSD SPECIFIC ROTATION (MODIFIED VELOCITY GAUGE) 589 nm")  #TEST
"""
